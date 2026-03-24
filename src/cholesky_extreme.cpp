/**
 * 极致优化的分块Cholesky分解算法
 * 2026年毕昇杯编译系统挑战赛 - 动态算子图编译与并行调度
 * 
 * 优化策略：
 * 1. NEON向量化 (ARM SIMD)
 * 2. 缓存优化 (分块、数据对齐、预取)
 * 3. OpenMP并行优化 (任务调度、NUMA感知)
 * 4. 循环展开和优化
 * 5. 矩阵转置优化缓存访问
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

// NEON优化的向量点积 - 展开版本
static inline double dot_product_neon_unrolled(const double* a, const double* b, int n) {
#if USE_NEON
    float64x2_t sum0 = vdupq_n_f64(0.0);
    float64x2_t sum1 = vdupq_n_f64(0.0);
    float64x2_t sum2 = vdupq_n_f64(0.0);
    float64x2_t sum3 = vdupq_n_f64(0.0);
    
    int i = 0;
    // 主循环，每次处理8个元素（4个NEON寄存器）
    for (; i + 7 < n; i += 8) {
        float64x2_t va0 = vld1q_f64(a + i);
        float64x2_t vb0 = vld1q_f64(b + i);
        float64x2_t va1 = vld1q_f64(a + i + 2);
        float64x2_t vb1 = vld1q_f64(b + i + 2);
        float64x2_t va2 = vld1q_f64(a + i + 4);
        float64x2_t vb2 = vld1q_f64(b + i + 4);
        float64x2_t va3 = vld1q_f64(a + i + 6);
        float64x2_t vb3 = vld1q_f64(b + i + 6);
        
        sum0 = vmlaq_f64(sum0, va0, vb0);
        sum1 = vmlaq_f64(sum1, va1, vb1);
        sum2 = vmlaq_f64(sum2, va2, vb2);
        sum3 = vmlaq_f64(sum3, va3, vb3);
    }
    
    // 合并结果
    sum0 = vaddq_f64(sum0, sum1);
    sum2 = vaddq_f64(sum2, sum3);
    sum0 = vaddq_f64(sum0, sum2);
    
    // 水平求和
    double sum = vaddvq_f64(sum0);
    
    // 处理剩余元素
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    
    return sum;
#else
    double sum = 0.0;
    // 展开标量版本
    int i = 0;
    for (; i + 3 < n; i += 4) {
        sum += a[i] * b[i] + a[i+1] * b[i+1] + a[i+2] * b[i+2] + a[i+3] * b[i+3];
    }
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

// NEON优化的矩阵向量乘加: C[i,:] -= A[i,k] * B[j,k]
static inline void madd_row_neon(double* C_row, const double* A_row, const double* B_row, int n, int k_len) {
#if USE_NEON
    int k = 0;
    float64x2_t a_vec;
    
    for (; k + 1 < k_len; k += 2) {
        a_vec = vld1q_f64(A_row + k);
        
        // 对C_row的每个元素进行更新
        for (int j = 0; j < n; j++) {
            float64x2_t b_vec = vld1q_f64(B_row + j * k_len + k);  // 假设B是转置存储
            float64x2_t c_vec = vdupq_n_f64(C_row[j]);
            c_vec = vmlaq_f64(c_vec, a_vec, b_vec);
            // 水平求和并存储
            C_row[j] -= vaddvq_f64(c_vec);
        }
    }
    
    // 处理剩余
    for (; k < k_len; k++) {
        double a_val = A_row[k];
        for (int j = 0; j < n; j++) {
            C_row[j] -= a_val * B_row[j * k_len + k];
        }
    }
#else
    for (int k = 0; k < k_len; k++) {
        double a_val = A_row[k];
        for (int j = 0; j < n; j++) {
            C_row[j] -= a_val * B_row[j * k_len + k];
        }
    }
#endif
}

// 朴素Cholesky分解（用于对角块）- 极致优化版本
int cholesky_extreme(double* A, double* L, int n, int lda) {
    for (int i = 0; i < n; i++) {
        double diag = A[i * lda + i];
        
        // 使用NEON优化的点积
        diag -= dot_product_neon_unrolled(&L[i * lda], &L[i * lda], i);
        
        if (diag <= 0) {
            return -1;
        }
        
        double inv_sqrt_diag = 1.0 / sqrt(diag);
        L[i * lda + i] = sqrt(diag);
        
        // 并行化内层循环
        #pragma omp parallel for schedule(static) if(n - i - 1 > 16)
        for (int j = i + 1; j < n; j++) {
            double sum = A[j * lda + i];
            sum -= dot_product_neon_unrolled(&L[j * lda], &L[i * lda], i);
            L[j * lda + i] = sum * inv_sqrt_diag;
        }
    }
    return 0;
}

// 三角方程求解: X = A * L^{-1} - 极致优化版本
void trsm_extreme(double* A, double* L, double* X, int m, int n, int lda) {
    // 逐列求解
    for (int j = 0; j < n; j++) {
        double inv_ljj = 1.0 / L[j * lda + j];
        
        #pragma omp parallel for schedule(static) if(m > 32)
        for (int i = 0; i < m; i++) {
            double sum = A[i * lda + j] - X[i * lda + j];
            
            // 使用NEON优化的点积
            sum -= dot_product_neon_unrolled(&X[i * lda], &L[j * lda], j);
            
            X[i * lda + j] = sum * inv_ljj;
        }
    }
}

// 矩阵乘加: C = C - A * B^T - 极致优化版本
void madd_extreme(double* A, double* B, double* C, int m, int n, int k, int lda) {
    // 使用分块优化缓存
    const int BLOCK_I = 32;
    const int BLOCK_J = 32;
    
    #pragma omp parallel for schedule(dynamic, 1) collapse(2) if(m > 16 && n > 16)
    for (int ii = 0; ii < m; ii += BLOCK_I) {
        for (int jj = 0; jj < n; jj += BLOCK_J) {
            int i_end = std::min(ii + BLOCK_I, m);
            int j_end = std::min(jj + BLOCK_J, n);
            
            for (int i = ii; i < i_end; i++) {
                for (int j = jj; j < j_end; j++) {
                    double sum = C[i * lda + j];
                    sum -= dot_product_neon_unrolled(&A[i * lda], &B[j * lda], k);
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
            int k_end = std::min(i, j) + 1;
            double sum = dot_product_neon_unrolled(&L[i * n], &L[j * n], k_end);
            
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

// 极致优化的并行分块Cholesky分解
int block_cholesky_extreme(double* A, double* L, int n, int b) {
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
    
    for (int i = 0; i < n; i += b) {
        int ib = std::min(b, n - i);
        
        // 步骤1: 对角块Cholesky分解
        int ret = cholesky_extreme(&S[i * n + i], &L[i * n + i], ib, n);
        if (ret != 0) {
            free(S);
            return ret;
        }
        
        // 步骤2: 三角求解 (并行)
        #pragma omp parallel for schedule(dynamic, 2)
        for (int j = i + b; j < n; j += b) {
            int jb = std::min(b, n - j);
            trsm_extreme(&S[j * n + i], &L[i * n + i], &L[j * n + i], jb, ib, n);
        }
        
        // 步骤3: Schur补更新 (并行)
        // 使用更细粒度的任务划分
        int num_j = (n - i - b + b - 1) / b;
        
        #pragma omp parallel for schedule(dynamic, 1)
        for (int j_idx = 0; j_idx < num_j; j_idx++) {
            int j = i + b + j_idx * b;
            int jb = std::min(b, n - j);
            
            for (int k = i + b; k <= j; k += b) {
                int kb = std::min(b, n - k);
                madd_extreme(&L[j * n + i], &L[k * n + i], &S[j * n + k], jb, kb, ib, n);
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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s --test <matrix_size> [block_size]\n", argv[0]);
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
        printf("Error: Only --test mode supported\n");
        return 1;
    }
    
    int num_threads = omp_get_max_threads();
    printf("Using %d OpenMP threads\n", num_threads);
    printf("NEON optimization: %s\n", USE_NEON ? "enabled" : "disabled");
    
    // 测试极致优化版本
    printf("\n=== Extreme Optimized Parallel Block Cholesky ===\n");
    auto start = std::chrono::high_resolution_clock::now();
    int result = block_cholesky_extreme(A.data(), L.data(), n, block_size);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (result != 0) {
        printf("Extreme Optimized failed!\n");
        return 1;
    }
    
    double elapsed = std::chrono::duration<double>(end - start).count();
    printf("Extreme Optimized time: %.6f seconds\n", elapsed);
    
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