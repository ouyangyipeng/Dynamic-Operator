/**
 * 并行分块Cholesky分解算法
 * 2026年毕昇杯编译系统挑战赛 - 动态算子图编译与并行调度
 * 
 * 使用运行时库实现任务并行
 */

#include "runtime/runtime.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <mutex>

// 朴素Cholesky分解（用于对角块）
int cholesky(double* A, double* L, int n, int lda) {
    for (int i = 0; i < n; i++) {
        double diag = A[i * lda + i];
        for (int k = 0; k < i; k++) {
            diag -= L[i * lda + k] * L[i * lda + k];
        }
        if (diag <= 0) {
            printf("Error: Matrix is not positive definite at row %d (diag=%e)\n", i, diag);
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
        for (int j = i + 1; j < n; j++) {
            L_temp[i * n + j] = 0.0;
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
        
        int ret = cholesky(&S[i * n + i], &L[i * n + i], ib, n);
        if (ret != 0) return ret;
        
        for (int j = i + b; j < n; j += b) {
            int jb = std::min(b, n - j);
            trsm(&S[j * n + i], &L[i * n + i], &L[j * n + i], jb, ib, n);
        }
        
        for (int j = i + b; j < n; j += b) {
            int jb = std::min(b, n - j);
            for (int k = i + b; k <= j; k += b) {
                int kb = std::min(b, n - k);
                madd(&L[j * n + i], &L[k * n + i], &S[j * n + k], jb, kb, ib, n);
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

// 并行分块Cholesky分解
int block_cholesky_parallel(double* A, double* L, int n, int b, int num_threads = 0) {
    // 初始化
    for (int idx = 0; idx < n * n; idx++) {
        L[idx] = 0.0;
    }
    
    // 工作矩阵
    std::vector<double> S(n * n, 0.0);
    for (int i = 0; i < n * n; i++) {
        S[i] = A[i];
    }
    
    int num_blocks = (n + b - 1) / b;
    
    // 初始化运行时
    runtime::init_runtime(num_threads);
    auto scheduler = runtime::get_scheduler();
    
    // 用于同步的互斥锁
    std::mutex mtx;
    std::atomic<int> error_flag{0};
    
    // 创建任务依赖图
    // 任务ID映射
    std::vector<std::vector<runtime::TaskId>> chol_tasks(num_blocks, std::vector<runtime::TaskId>(num_blocks, -1));
    std::vector<std::vector<runtime::TaskId>> trsm_tasks(num_blocks, std::vector<runtime::TaskId>(num_blocks, -1));
    std::vector<std::vector<std::vector<runtime::TaskId>>> madd_tasks(num_blocks, 
        std::vector<std::vector<runtime::TaskId>>(num_blocks, std::vector<runtime::TaskId>(num_blocks, -1)));
    
    // 第一步：创建所有任务
    for (int i = 0; i < num_blocks; i++) {
        int ib = std::min(b, n - i * b);
        
        // Cholesky任务
        chol_tasks[i][i] = scheduler->create_task(
            runtime::TaskType::CHOLESKY,
            [&, i, ib, b, n]() {
                int ret = cholesky(&S[i * b * n + i * b], &L[i * b * n + i * b], ib, n);
                if (ret != 0) error_flag = ret;
            },
            i, i, -1
        );
        
        // TRSM任务
        for (int j = i + 1; j < num_blocks; j++) {
            int jb = std::min(b, n - j * b);
            trsm_tasks[j][i] = scheduler->create_task(
                runtime::TaskType::TRSM,
                [&, i, j, ib, jb, b, n]() {
                    trsm(&S[j * b * n + i * b], &L[i * b * n + i * b], &L[j * b * n + i * b], jb, ib, n);
                },
                j, i, -1
            );
        }
        
        // MADDS任务
        for (int j = i + 1; j < num_blocks; j++) {
            int jb = std::min(b, n - j * b);
            for (int k = i + 1; k <= j; k++) {
                int kb = std::min(b, n - k * b);
                madd_tasks[i][j][k] = scheduler->create_task(
                    runtime::TaskType::MADDS,
                    [&, i, j, k, ib, jb, kb, b, n]() {
                        madd(&L[j * b * n + i * b], &L[k * b * n + i * b], &S[j * b * n + k * b], jb, kb, ib, n);
                    },
                    j, k, i
                );
            }
        }
    }
    
    // 第二步：建立依赖关系
    for (int i = 0; i < num_blocks; i++) {
        // Cholesky(i) 依赖于之前所有更新到块(i,i)的madd任务
        if (i > 0) {
            for (int prev_i = 0; prev_i < i; prev_i++) {
                // madd(prev_i, i, i) 更新了 S(i,i)
                if (madd_tasks[prev_i][i][i] != -1) {
                    scheduler->add_dependency(chol_tasks[i][i], madd_tasks[prev_i][i][i]);
                }
            }
        }
        
        // TRSM(j, i) 依赖于 Cholesky(i)
        for (int j = i + 1; j < num_blocks; j++) {
            scheduler->add_dependency(trsm_tasks[j][i], chol_tasks[i][i]);
            
            // TRSM(j, i) 也依赖于之前更新到 S(j, i) 的madd任务
            if (i > 0) {
                for (int prev_i = 0; prev_i < i; prev_i++) {
                    if (madd_tasks[prev_i][j][i] != -1) {
                        scheduler->add_dependency(trsm_tasks[j][i], madd_tasks[prev_i][j][i]);
                    }
                }
            }
        }
        
        // MADDS(j, k, i) 依赖于 TRSM(j, i) 和 TRSM(k, i)
        for (int j = i + 1; j < num_blocks; j++) {
            for (int k = i + 1; k <= j; k++) {
                scheduler->add_dependency(madd_tasks[i][j][k], trsm_tasks[j][i]);
                if (k != j) {
                    scheduler->add_dependency(madd_tasks[i][j][k], trsm_tasks[k][i]);
                }
            }
        }
    }
    
    // 执行任务并等待完成
    scheduler->execute_and_wait();
    
    // 清理
    runtime::shutdown_runtime();
    
    // 清零上三角
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            L[i * n + j] = 0.0;
        }
    }
    
    return error_flag.load();
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
    
    // 测试并行版本
    printf("\n=== Parallel Block Cholesky ===\n");
    std::fill(L.begin(), L.end(), 0.0);
    start = std::chrono::high_resolution_clock::now();
    result = block_cholesky_parallel(A.data(), L.data(), n, block_size);
    end = std::chrono::high_resolution_clock::now();
    
    if (result != 0) {
        printf("Parallel Block Cholesky failed!\n");
        return 1;
    }
    
    elapsed = std::chrono::duration<double>(end - start).count();
    printf("Parallel Block Cholesky time: %.6f seconds\n", elapsed);
    
    scaled_residual = verify_result(A_copy.data(), L.data(), n);
    printf("Scaled residual: %.6e\n", scaled_residual);
    printf("Result: %s\n", scaled_residual < 16.0 ? "PASS" : "FAIL");
    
    // 单线程基准
    printf("\n=== Single-thread Cholesky ===\n");
    std::vector<double> L_single(n * n, 0.0);
    start = std::chrono::high_resolution_clock::now();
    int result_single = cholesky_single_thread(A_copy.data(), L_single.data(), n);
    end = std::chrono::high_resolution_clock::now();
    
    if (result_single == 0) {
        elapsed = std::chrono::duration<double>(end - start).count();
        printf("Single-thread Cholesky time: %.6f seconds\n", elapsed);
        double scaled_residual_single = verify_result(A_copy.data(), L_single.data(), n);
        printf("Single-thread scaled residual: %.6e\n", scaled_residual_single);
    }
    
    return 0;
}