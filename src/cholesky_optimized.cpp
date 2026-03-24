/**
 * 高度优化的分块Cholesky分解算法
 * 2026年毕昇杯编译系统挑战赛 - 动态算子图编译与并行调度
 * 
 * 优化策略：
 * 1. NEON向量化 (ARM SIMD)
 * 2. 缓存优化 (分块、数据对齐)
 * 3. OpenMP并行优化 (任务调度、NUMA感知)
 * 4. 循环展开和优化
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#define USE_NEON 1
#else
#define USE_NEON 0
#endif

// 缓存行大小（鲲鹏920 L1缓存行64字节）
#define CACHE_LINE_SIZE 64

// 对齐宏
#define ALIGNED(x) __attribute__((aligned(x)))

// NEON优化的向量点积
static inline double dot_product_neon(const double* a, const double* b, int n) {
#if USE_NEON
    float64x2_t sum_vec = vdupq_n_f64(0.0);
    int i = 0;
    
    // 主循环，每次处理2个元素
    for (; i + 1 < n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        sum_vec = vmlaq_f64(sum_vec, va, vb);
    }
    
    // 水平求和
    double sum = vaddvq_f64(sum_vec);
    
    // 处理剩余元素
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

// NEON优化的向量减法
static inline void vec_sub_neon(double* c, const double* a, const double* b, int n) {
#if USE_NEON
    int i = 0;
    for (; i + 1 < n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        float64x2_t vc = vsubq_f64(va, vb);
        vst1q_f64(c + i, vc);
    }
    for (; i < n; i++) {
        c[i] = a[i] - b[i];
    }
#else
    for (int i = 0; i < n; i++) {
        c[i] = a[i] - b[i];
    }
#endif
}

// NEON优化的向量乘加: c = a * b + c
static inline void vec_madd_neon(double* c, const double* a, double b, int n) {
#if USE_NEON
    float64x2_t vb = vdupq_n_f64(b);
    int i = 0;
    for (; i + 1 < n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vc = vld1q_f64(c + i);
        vc = vmlaq_f64(vc, va, vb);
        vst1q_f64(c + i, vc);
    }
    for (; i < n; i++) {
        c[i] += a[i] * b;
    }
#else
    for (int i = 0; i < n; i++) {
        c[i] += a[i] * b;
    }
#endif
}

// 朴素Cholesky分解（用于对角块）- 优化版本
int cholesky_opt(double* A, double* L, int n, int lda) {
    for (int i = 0; i < n; i++) {
        double diag = A[i * lda + i];
        
        // 使用NEON优化的点积
        diag -= dot_product_neon(&L[i * lda], &L[i * lda], i);
        
        if (diag <= 0) {
            printf("Error: Matrix is not positive definite at row %d (diag=%e)\n", i, diag);
            return -1;
        }
        
        double inv_sqrt_diag = 1.0 / sqrt(diag);
        L[i * lda + i] = sqrt(diag);
        
        // 并行化内层循环
        #pragma omp parallel for schedule(static) if(n - i - 1 > 32)
        for (int j = i + 1; j < n; j++) {
            double sum = A[j * lda + i];
            sum -= dot_product_neon(&L[j * lda], &L[i * lda], i);
            L[j * lda + i] = sum * inv_sqrt_diag;
        }
    }
    return 0;
}

// 三角方程求解: X = A * L^{-1} - 优化版本
void trsm_opt(double* A, double* L, double* X, int m, int n, int lda) {
    // 逐列求解
    for (int j = 0; j < n; j++) {
        double inv_ljj = 1.0 / L[j * lda + j];
        
        #pragma omp parallel for schedule(static) if(m > 64)
        for (int i = 0; i < m; i++) {
            double sum = A[i * lda + j] - X[i * lda + j];
            
            // 减去已知项
            for (int k = 0; k < j; k++) {
                sum -= X[i * lda + k] * L[j * lda + k];
            }
            
            X[i * lda + j] = sum * inv_ljj;
        }
    }
}

// 矩阵乘加: C = C - A * B^T - 优化版本
void madd_opt(double* A, double* B, double* C, int m, int n, int k, int lda) {
    // 使用分块优化缓存
    const int BLOCK_SIZE = 64;
    
    #pragma omp parallel for schedule(dynamic, 1) collapse(2) if(m > 32)
    for (int ii = 0; ii < m; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
            int i_end = std::min(ii + BLOCK_SIZE, m);
            int j_end = std::min(jj + BLOCK_SIZE, n);
            
            for (int i = ii; i < i_end; i++) {
                for (int j = jj; j < j_end; j++) {
                    double sum = C[i * lda + j];
                    
                    // 使用NEON优化的点积
                    sum -= dot_product_neon(&A[i * lda], &B[j * lda], k);
                    
                    C[i * lda + j] = sum;
                }
            }
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
            
            // 使用NEON优化的点积
            sum = dot_product_neon(&L[i * n], &L[j * n], k_end);
            
            double residual = std::abs(A[i * n + j] - sum);
            max_residual = std::max(max_residual, residual);
            max_A = std::max(max_A, std::abs(A[i * n + j]));
        }
    }
    
    double eps = 2.220446049250313e-16;
    double scaled_residual = max_residual / (max_A * n * eps);
    
    return scaled_residual;
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
        for (int j = i + 1; j < n; j++) {
            L_temp[i * n + j] = 0.0;
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
        if (sum <= 0) {
            printf("Error: Matrix is not positive definite at row %d (diag=%e)\n", i, sum);
            return -1;
        }
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

// 串行分块Cholesky分解
int block_cholesky_serial(double* A, double* L, int n, int b) {
    for (int idx = 0; idx < n * n; idx++) {
        L[idx] = 0.0;
    }
    
    std::vector<double> S(n * n, 0.0);
    for (int i = 0; i < n * n; i++) {
        S[i] = A[i];
    }
    
    for (int i = 0; i < n; i += b) {
        int ib = std::min(b, n - i);
        
        int ret = cholesky_opt(&S[i * n + i], &L[i * n + i], ib, n);
        if (ret != 0) return ret;
        
        for (int j = i + b; j < n; j += b) {
            int jb = std::min(b, n - j);
            trsm_opt(&S[j * n + i], &L[i * n + i], &L[j * n + i], jb, ib, n);
        }
        
        for (int j = i + b; j < n; j += b) {
            int jb = std::min(b, n - j);
            for (int k = i + b; k <= j; k += b) {
                int kb = std::min(b, n - k);
                madd_opt(&L[j * n + i], &L[k * n + i], &S[j * n + k], jb, kb, ib, n);
            }
        }
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            L[i * n + j] = 0.0;
        }
    }
    
    return 0;
}

// 高度优化的并行分块Cholesky分解 - 使用任务依赖
int block_cholesky_parallel_opt(double* A, double* L, int n, int b) {
    // 初始化
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < n * n; idx++) {
        L[idx] = 0.0;
    }
    
    // 工作矩阵 - 使用对齐分配
    double* S = (double*)aligned_alloc(CACHE_LINE_SIZE, n * n * sizeof(double));
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n * n; i++) {
        S[i] = A[i];
    }
    
    int num_blocks = (n + b - 1) / b;
    
    // 使用任务依赖图进行并行调度
    // 每个块列需要等待前一个块列的更新完成
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int bi = 0; bi < num_blocks; bi++) {
                int i = bi * b;
                int ib = std::min(b, n - i);
                
                // 等待当前块列的所有更新完成
                #pragma omp taskwait
                
                // 任务1: 对角块Cholesky分解
                #pragma omp task priority(100) firstprivate(i, ib, b, n) shared(S, L)
                {
                    cholesky_opt(&S[i * n + i], &L[i * n + i], ib, n);
                }
                
                // 等待cholesky完成
                #pragma omp taskwait
                
                // 任务2: TRSM
                for (int bj = bi + 1; bj < num_blocks; bj++) {
                    int j = bj * b;
                    int jb = std::min(b, n - j);
                    
                    #pragma omp task priority(50) firstprivate(i, j, ib, jb, b, n) shared(S, L)
                    {
                        trsm_opt(&S[j * n + i], &L[i * n + i], &L[j * n + i], jb, ib, n);
                    }
                }
                
                // 等待所有TRSM完成
                #pragma omp taskwait
                
                // 任务3: MADDS
                for (int bj = bi + 1; bj < num_blocks; bj++) {
                    int j = bj * b;
                    int jb = std::min(b, n - j);
                    
                    for (int bk = bi + 1; bk <= bj; bk++) {
                        int k = bk * b;
                        int kb = std::min(b, n - k);
                        
                        #pragma omp task priority(10) firstprivate(i, j, k, ib, jb, kb, b, n) shared(S, L)
                        {
                            madd_opt(&L[j * n + i], &L[k * n + i], &S[j * n + k], jb, kb, ib, n);
                        }
                    }
                }
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
    
    free(S);
    return 0;
}

// 简化并行版本 - 使用parallel for
int block_cholesky_parallel_simple(double* A, double* L, int n, int b) {
    // 初始化
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < n * n; idx++) {
        L[idx] = 0.0;
    }
    
    // 工作矩阵
    double* S = (double*)aligned_alloc(CACHE_LINE_SIZE, n * n * sizeof(double));
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n * n; i++) {
        S[i] = A[i];
    }
    
    for (int i = 0; i < n; i += b) {
        int ib = std::min(b, n - i);
        
        // 步骤1: 对角块Cholesky分解 (串行)
        int ret = cholesky_opt(&S[i * n + i], &L[i * n + i], ib, n);
        if (ret != 0) {
            free(S);
            return ret;
        }
        
        // 步骤2: 三角求解 (并行)
        #pragma omp parallel for schedule(dynamic, 1)
        for (int j = i + b; j < n; j += b) {
            int jb = std::min(b, n - j);
            trsm_opt(&S[j * n + i], &L[i * n + i], &L[j * n + i], jb, ib, n);
        }
        
        // 步骤3: Schur补更新 (并行)
        // 收集所有需要更新的(j,k)对
        std::vector<std::pair<int,int>> jk_pairs;
        for (int j = i + b; j < n; j += b) {
            for (int k = i + b; k <= j; k += b) {
                jk_pairs.push_back({j, k});
            }
        }
        
        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t idx = 0; idx < jk_pairs.size(); idx++) {
            int j = jk_pairs[idx].first;
            int k = jk_pairs[idx].second;
            int jb = std::min(b, n - j);
            int kb = std::min(b, n - k);
            madd_opt(&L[j * n + i], &L[k * n + i], &S[j * n + k], jb, kb, ib, n);
        }
    }
    
    // 清零上三角
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            L[i * n + j] = 0.0;
        }
    }
    
    free(S);
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input_file> [block_size]\n", argv[0]);
        printf("   or: %s --test <matrix_size> [block_size]\n", argv[0]);
        return 1;
    }
    
    int block_size = 64;
    if (argc >= 3) {
        block_size = atoi(argv[2]);
    }
    
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
        const char* input_file = argv[1];
        std::ifstream fin(input_file, std::ios::binary);
        
        if (!fin) {
            printf("Error: Cannot open input file %s\n", input_file);
            return 1;
        }
        
        int num_matrices;
        fin.read(reinterpret_cast<char*>(&num_matrices), sizeof(int));
        printf("Reading %d matrices from %s\n", num_matrices, input_file);
        
        fin.read(reinterpret_cast<char*>(&n), sizeof(int));
        A.resize(n * n);
        A_copy.resize(n * n);
        L.resize(n * n);
        fin.read(reinterpret_cast<char*>(A.data()), n * n * sizeof(double));
        
        for (int i = 0; i < n * n; i++) {
            A_copy[i] = A[i];
        }
        
        fin.close();
        printf("Read %dx%d matrix\n", n, n);
    }
    
    // 设置OpenMP线程数
    int num_threads = omp_get_max_threads();
    printf("Using %d OpenMP threads\n", num_threads);
    printf("NEON optimization: %s\n", USE_NEON ? "enabled" : "disabled");
    
    // 测试串行版本
    printf("\n=== Serial Block Cholesky ===\n");
    std::vector<double> L_serial(n * n);
    auto start = std::chrono::high_resolution_clock::now();
    int result = block_cholesky_serial(A.data(), L_serial.data(), n, block_size);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (result != 0) {
        printf("Serial Block Cholesky failed!\n");
        return 1;
    }
    
    double elapsed = std::chrono::duration<double>(end - start).count();
    printf("Serial Block Cholesky time: %.6f seconds\n", elapsed);
    
    double scaled_residual = verify_result(A_copy.data(), L_serial.data(), n);
    printf("Scaled residual: %.6e\n", scaled_residual);
    printf("Result: %s\n", scaled_residual < 16.0 ? "PASS" : "FAIL");
    
    // 测试优化并行版本
    printf("\n=== Optimized Parallel Block Cholesky ===\n");
    std::fill(L.begin(), L.end(), 0.0);
    start = std::chrono::high_resolution_clock::now();
    result = block_cholesky_parallel_opt(A.data(), L.data(), n, block_size);
    end = std::chrono::high_resolution_clock::now();
    
    if (result != 0) {
        printf("Optimized Parallel Block Cholesky failed!\n");
        return 1;
    }
    
    elapsed = std::chrono::duration<double>(end - start).count();
    printf("Optimized Parallel Block Cholesky time: %.6f seconds\n", elapsed);
    
    scaled_residual = verify_result(A_copy.data(), L.data(), n);
    printf("Scaled residual: %.6e\n", scaled_residual);
    printf("Result: %s\n", scaled_residual < 16.0 ? "PASS" : "FAIL");
    
    // 测试简化并行版本
    printf("\n=== Simple Parallel Block Cholesky ===\n");
    std::fill(L.begin(), L.end(), 0.0);
    start = std::chrono::high_resolution_clock::now();
    result = block_cholesky_parallel_simple(A.data(), L.data(), n, block_size);
    end = std::chrono::high_resolution_clock::now();
    
    if (result != 0) {
        printf("Simple Parallel Block Cholesky failed!\n");
        return 1;
    }
    
    elapsed = std::chrono::duration<double>(end - start).count();
    printf("Simple Parallel Block Cholesky time: %.6f seconds\n", elapsed);
    
    scaled_residual = verify_result(A_copy.data(), L.data(), n);
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
        printf("Single-thread Cholesky time (T0): %.6f seconds\n", elapsed);
        double scaled_residual_single = verify_result(A_copy.data(), L_single.data(), n);
        printf("Single-thread scaled residual: %.6e\n", scaled_residual_single);
    }
    
    return 0;
}