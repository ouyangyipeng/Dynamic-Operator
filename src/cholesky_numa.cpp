/**
 * NUMA感知的分块Cholesky分解算法
 * 2026年毕昇杯编译系统挑战赛 - 动态算子图编译与并行调度
 * 
 * 优化策略：
 * 1. NUMA感知内存分配
 * 2. 线程绑定到NUMA节点
 * 3. 数据局部性优化
 * 4. NEON向量化
 * 5. OpenMP并行优化
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <numa.h>
#include <numaif.h>
#include <pthread.h>
#include <sched.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#define USE_NEON 1
#else
#define USE_NEON 0
#endif

// 缓存行大小
#define CACHE_LINE_SIZE 64

// NUMA配置
struct NumaConfig {
    int num_nodes;
    int num_cpus;
    std::vector<std::vector<int>> node_cpus;  // 每个NUMA节点的CPU列表
    
    static NumaConfig detect() {
        NumaConfig config;
        config.num_nodes = numa_num_configured_nodes();
        config.num_cpus = numa_num_configured_cpus();
        
        struct bitmask* mask = numa_allocate_cpumask();
        for (int node = 0; node < config.num_nodes; node++) {
            numa_node_to_cpus(node, mask);
            std::vector<int> cpus;
            for (int i = 0; i < config.num_cpus; i++) {
                if (numa_bitmask_isbitset(mask, i)) {
                    cpus.push_back(i);
                }
            }
            config.node_cpus.push_back(cpus);
        }
        numa_free_cpumask(mask);
        
        return config;
    }
    
    void print() const {
        printf("NUMA Configuration:\n");
        printf("  Nodes: %d, CPUs: %d\n", num_nodes, num_cpus);
        for (int i = 0; i < num_nodes; i++) {
            printf("  Node %d: %zu CPUs [", i, node_cpus[i].size());
            for (size_t j = 0; j < std::min(node_cpus[i].size(), (size_t)8); j++) {
                printf("%d ", node_cpus[i][j]);
            }
            if (node_cpus[i].size() > 8) printf("...");
            printf("]\n");
        }
    }
};

// NUMA感知的内存分配
class NumaMemory {
public:
    NumaMemory(size_t size, int node = -1) : size_(size), node_(node), ptr_(nullptr) {
        if (node >= 0) {
            // 绑定到特定NUMA节点
            ptr_ = numa_alloc_onnode(size, node);
        } else {
            // 在当前线程的NUMA节点上分配
            ptr_ = numa_alloc_local(size);
        }
        
        if (!ptr_) {
            fprintf(stderr, "Failed to allocate NUMA memory\n");
            exit(1);
        }
    }
    
    ~NumaMemory() {
        if (ptr_) {
            numa_free(ptr_, size_);
        }
    }
    
    void* get() const { return ptr_; }
    size_t size() const { return size_; }
    
private:
    size_t size_;
    int node_;
    void* ptr_;
};

// 线程绑定工具
void bind_thread_to_cpu(int cpu) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

void bind_thread_to_node(int node, const NumaConfig& config) {
    if (node < 0 || node >= config.num_nodes) return;
    
    // 绑定到节点的第一个CPU
    if (!config.node_cpus[node].empty()) {
        bind_thread_to_cpu(config.node_cpus[node][0]);
    }
}

// NEON优化的向量点积
static inline double dot_product_neon(const double* a, const double* b, int n) {
#if USE_NEON
    float64x2_t sum_vec = vdupq_n_f64(0.0);
    int i = 0;
    
    for (; i + 1 < n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        sum_vec = vmlaq_f64(sum_vec, va, vb);
    }
    
    double sum = vaddvq_f64(sum_vec);
    
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    
    return sum;
#else
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

// 朴素Cholesky分解（用于对角块）
int cholesky_numa(double* A, double* L, int n, int lda) {
    for (int i = 0; i < n; i++) {
        double diag = A[i * lda + i];
        diag -= dot_product_neon(&L[i * lda], &L[i * lda], i);
        
        if (diag <= 0) {
            return -1;
        }
        
        L[i * lda + i] = sqrt(diag);
        double inv_diag = 1.0 / L[i * lda + i];
        
        for (int j = i + 1; j < n; j++) {
            double sum = A[j * lda + i];
            sum -= dot_product_neon(&L[j * lda], &L[i * lda], i);
            L[j * lda + i] = sum * inv_diag;
        }
    }
    return 0;
}

// 三角方程求解: X = A * L^{-1}
void trsm_numa(double* A, double* L, double* X, int m, int n, int lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = A[i * lda + j];
            for (int k = 0; k < j; k++) {
                sum -= X[i * lda + k] * L[j * lda + k];
            }
            X[i * lda + j] = sum / L[j * lda + j];
        }
    }
}

// 矩阵乘加: C = C - A * B^T
void madd_numa(double* A, double* B, double* C, int m, int n, int k, int lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = C[i * lda + j];
            sum -= dot_product_neon(&A[i * lda], &B[j * lda], k);
            C[i * lda + j] = sum;
        }
    }
}

// 验证结果
double verify_result(double* A, double* L, int n) {
    double max_residual = 0.0;
    double max_A = 0.0;
    
    #pragma omp parallel for reduction(max:max_residual, max_A) schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            int k_end = std::min(i, j) + 1;
            sum = dot_product_neon(&L[i * n], &L[j * n], k_end);
            
            double residual = std::abs(A[i * n + j] - sum);
            max_residual = std::max(max_residual, residual);
            max_A = std::max(max_A, std::abs(A[i * n + j]));
        }
    }
    
    double eps = 2.220446049250313e-16;
    return max_residual / (max_A * n * eps);
}

// 生成正定矩阵 A = L * L^T
void generate_spd_matrix(double* A, int n, unsigned int seed) {
    srand(seed);
    std::vector<double> L_temp(n * n, 0.0);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            if (i == j) {
                L_temp[i * n + j] = (double)rand() / RAND_MAX * 0.5 + (double)(n + 1) / (i + 1);
            } else {
                L_temp[i * n + j] = (double)rand() / RAND_MAX * 0.1;
            }
        }
    }
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += L_temp[i * n + k] * L_temp[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
}

// 单核基准测试
int cholesky_single_thread(double* A, double* L, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[i * n + j] = 0.0;
        }
    }
    
    for (int i = 0; i < n; i++) {
        double sum = A[i * n + i];
        for (int k = 0; k < i; k++) {
            sum -= L[i * n + k] * L[i * n + k];
        }
        if (sum <= 0) return -1;
        L[i * n + i] = sqrt(sum);
        
        for (int j = i + 1; j < n; j++) {
            sum = A[j * n + i];
            for (int k = 0; k < i; k++) {
                sum -= L[j * n + k] * L[i * n + k];
            }
            L[j * n + i] = sum / L[i * n + i];
        }
    }
    return 0;
}

// NUMA感知的并行分块Cholesky分解
int block_cholesky_numa(double* A, double* L, int n, int b, const NumaConfig& config) {
    // 初始化
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < n * n; idx++) {
        L[idx] = 0.0;
    }
    
    // 工作矩阵 - 使用NUMA本地分配
    double* S = (double*)numa_alloc_local(n * n * sizeof(double));
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n * n; i++) {
        S[i] = A[i];
    }
    // 设置OpenMP线程绑定 - 通过环境变量控制
    // OMP_PROC_BIND=close OMP_PLACES=cores
    
    
    for (int i = 0; i < n; i += b) {
        int ib = std::min(b, n - i);
        
        // 步骤1: 对角块Cholesky分解
        int ret = cholesky_numa(&S[i * n + i], &L[i * n + i], ib, n);
        if (ret != 0) {
            numa_free(S, n * n * sizeof(double));
            return ret;
        }
        
        // 步骤2: 三角求解 (并行)
        #pragma omp parallel for schedule(dynamic, 2)
        for (int j = i + b; j < n; j += b) {
            int jb = std::min(b, n - j);
            trsm_numa(&S[j * n + i], &L[i * n + i], &L[j * n + i], jb, ib, n);
        }
        
        // 步骤3: Schur补更新 (并行)
        int num_j = (n - i - b + b - 1) / b;
        
        #pragma omp parallel for schedule(dynamic, 1)
        for (int j_idx = 0; j_idx < num_j; j_idx++) {
            int j = i + b + j_idx * b;
            int jb = std::min(b, n - j);
            
            for (int k = i + b; k <= j; k += b) {
                int kb = std::min(b, n - k);
                madd_numa(&L[j * n + i], &L[k * n + i], &S[j * n + k], jb, kb, ib, n);
            }
        }
    }
    
    // 清零上三角
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            L[i * n + j] = 0.0;
        }
    }
    
    numa_free(S, n * n * sizeof(double));
    return 0;
}

// 使用numactl绑定到特定NUMA节点的版本
int block_cholesky_numa_bound(double* A, double* L, int n, int b, const NumaConfig& config, int num_nodes_to_use) {
    // 设置内存策略： interleaved across selected nodes
    struct bitmask* nodemask = numa_allocate_nodemask();
    for (int i = 0; i < num_nodes_to_use && i < config.num_nodes; i++) {
        numa_bitmask_setbit(nodemask, i);
    }
    set_mempolicy(MPOL_INTERLEAVE, nodemask->maskp, nodemask->size);
    numa_free_nodemask(nodemask);
    
    return block_cholesky_numa(A, L, n, b, config);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s --test <matrix_size> [block_size]\n", argv[0]);
        return 1;
    }
    
    int block_size = 64;
    if (argc >= 3) {
        block_size = atoi(argv[2]);
    }
    
    // 检测NUMA配置
    NumaConfig numa_config = NumaConfig::detect();
    numa_config.print();
    
    std::vector<double> A, L, A_copy;
    int n = 0;
    
    if (strcmp(argv[1], "--test") == 0) {
        if (argc < 3) {
            printf("Error: --test requires matrix size\n");
            return 1;
        }
        n = atoi(argv[2]);
        if (argc >= 4) {
            block_size = atoi(argv[3]);
        }
        
        A.resize(n * n);
        A_copy.resize(n * n);
        L.resize(n * n);
        
        generate_spd_matrix(A.data(), n, 42);
        for (int i = 0; i < n * n; i++) {
            A_copy[i] = A[i];
        }
        printf("Generated %dx%d SPD matrix for testing\n", n, n);
    } else {
        printf("Error: Only --test mode supported\n");
        return 1;
    }
    
    int num_threads = omp_get_max_threads();
    printf("Using %d OpenMP threads\n", num_threads);
    printf("NEON optimization: %s\n", USE_NEON ? "enabled" : "disabled");
    
    // 测试NUMA优化版本
    printf("\n=== NUMA-Optimized Block Cholesky ===\n");
    auto start = std::chrono::high_resolution_clock::now();
    int result = block_cholesky_numa(A.data(), L.data(), n, block_size, numa_config);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (result != 0) {
        printf("NUMA-Optimized failed!\n");
        return 1;
    }
    
    double elapsed = std::chrono::duration<double>(end - start).count();
    printf("NUMA-Optimized time: %.6f seconds\n", elapsed);
    
    double scaled_residual = verify_result(A_copy.data(), L.data(), n);
    printf("Scaled residual: %.6e\n", scaled_residual);
    printf("Result: %s\n", scaled_residual < 16.0 ? "PASS" : "FAIL");
    
    // 单线程基准
    printf("\n=== Single-thread Cholesky (T0 baseline) ===\n");
    std::vector<double> L_single(n * n, 0.0);
    start = std::chrono::high_resolution_clock::now();
    int result_single = cholesky_single_thread(A_copy.data(), L_single.data(), n);
    end = std::chrono::high_resolution_clock::now();
    
    if (result_single == 0) {
        elapsed = std::chrono::duration<double>(end - start).count();
        printf("Single-thread time (T0): %.6f seconds\n", elapsed);
        double scaled_residual_single = verify_result(A_copy.data(), L_single.data(), n);
        printf("Single-thread scaled residual: %.6e\n", scaled_residual_single);
    }
    
    return 0;
}