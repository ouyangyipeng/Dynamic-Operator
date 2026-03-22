/**
 * 测试脚本 - 200组测试矩阵验证
 * 2026年毕昇杯编译系统挑战赛
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <iostream>
#include <omp.h>

// 朴素Cholesky分解（用于对角块）
int cholesky(double* A, double* L, int n, int lda) {
    for (int i = 0; i < n; i++) {
        double diag = A[i * lda + i];
        for (int k = 0; k < i; k++) {
            diag -= L[i * lda + k] * L[i * lda + k];
        }
        if (diag <= 0) {
            return -1;
        }
        L[i * lda + i] = sqrt(diag);
        
        for (int j = i + 1; j < n; j++) {
            double sum = A[j * lda + i];
            for (int k = 0; k < i; k++) {
                sum -= L[j * lda + k] * L[i * lda + k];
            }
            L[j * lda + i] = sum / L[i * lda + i];
        }
    }
    return 0;
}

// 三角方程求解: X = A * L^{-1}
void trsm(double* A, double* L, double* X, int m, int n, int lda) {
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
void madd(double* A, double* B, double* C, int m, int n, int k, int lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = C[i * lda + j];
            for (int l = 0; l < k; l++) {
                sum -= A[i * lda + l] * B[j * lda + l];
            }
            C[i * lda + j] = sum;
        }
    }
}

// 验证结果
double verify_result(double* A, double* L, int n) {
    double max_residual = 0.0;
    double max_A = 0.0;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k <= std::min(i, j); k++) {
                sum += L[i * n + k] * L[j * n + k];
            }
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
    }
    
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

// 并行分块Cholesky分解
int block_cholesky_parallel(double* A, double* L, int n, int b) {
    for (int idx = 0; idx < n * n; idx++) {
        L[idx] = 0.0;
    }
    
    std::vector<double> S(n * n, 0.0);
    for (int i = 0; i < n * n; i++) {
        S[i] = A[i];
    }
    
    for (int i = 0; i < n; i += b) {
        int ib = std::min(b, n - i);
        
        int ret = cholesky(&S[i * n + i], &L[i * n + i], ib, n);
        if (ret != 0) return ret;
        
        #pragma omp parallel for schedule(dynamic)
        for (int j = i + b; j < n; j += b) {
            int jb = std::min(b, n - j);
            trsm(&S[j * n + i], &L[i * n + i], &L[j * n + i], jb, ib, n);
        }
        
        std::vector<std::pair<int,int>> jk_pairs;
        for (int j = i + b; j < n; j += b) {
            for (int k = i + b; k <= j; k += b) {
                jk_pairs.push_back({j, k});
            }
        }
        
        #pragma omp parallel for schedule(dynamic)
        for (size_t idx = 0; idx < jk_pairs.size(); idx++) {
            int j = jk_pairs[idx].first;
            int k = jk_pairs[idx].second;
            int jb = std::min(b, n - j);
            int kb = std::min(b, n - k);
            madd(&L[j * n + i], &L[k * n + i], &S[j * n + k], jb, kb, ib, n);
        }
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            L[i * n + j] = 0.0;
        }
    }
    
    return 0;
}

int main(int argc, char* argv[]) {
    int num_tests = 200;
    int matrix_size = 1024;
    int block_size = 64;
    
    if (argc >= 2) num_tests = atoi(argv[1]);
    if (argc >= 3) matrix_size = atoi(argv[2]);
    if (argc >= 4) block_size = atoi(argv[3]);
    
    printf("Running %d tests with %dx%d matrices, block size %d\n", 
           num_tests, matrix_size, matrix_size, block_size);
    printf("Using %d OpenMP threads\n\n", omp_get_max_threads());
    
    int pass_count = 0;
    int fail_count = 0;
    double total_time = 0.0;
    double max_residual = 0.0;
    
    for (int test = 0; test < num_tests; test++) {
        std::vector<double> A(matrix_size * matrix_size);
        std::vector<double> L(matrix_size * matrix_size);
        std::vector<double> A_copy(matrix_size * matrix_size);
        
        generate_spd_matrix(A.data(), matrix_size, test + 42);
        for (int i = 0; i < matrix_size * matrix_size; i++) {
            A_copy[i] = A[i];
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        int result = block_cholesky_parallel(A.data(), L.data(), matrix_size, block_size);
        auto end = std::chrono::high_resolution_clock::now();
        
        double elapsed = std::chrono::duration<double>(end - start).count();
        total_time += elapsed;
        
        if (result != 0) {
            printf("Test %d: FAIL (decomposition error)\n", test + 1);
            fail_count++;
            continue;
        }
        
        double scaled_residual = verify_result(A_copy.data(), L.data(), matrix_size);
        max_residual = std::max(max_residual, scaled_residual);
        
        if (scaled_residual < 16.0) {
            pass_count++;
            printf("Test %d: PASS (residual=%.6e, time=%.4fs)\n", 
                   test + 1, scaled_residual, elapsed);
        } else {
            fail_count++;
            printf("Test %d: FAIL (residual=%.6e, time=%.4fs)\n", 
                   test + 1, scaled_residual, elapsed);
        }
    }
    
    printf("\n========================================\n");
    printf("Summary:\n");
    printf("  Total tests: %d\n", num_tests);
    printf("  Passed: %d\n", pass_count);
    printf("  Failed: %d\n", fail_count);
    printf("  Pass rate: %.1f%%\n", 100.0 * pass_count / num_tests);
    printf("  Total time: %.2f seconds\n", total_time);
    printf("  Average time: %.4f seconds\n", total_time / num_tests);
    printf("  Max residual: %.6e\n", max_residual);
    printf("========================================\n");
    
    return (fail_count > 0) ? 1 : 0;
}