/**
 * 分块Cholesky分解算法
 * 2026年毕昇杯编译系统挑战赛 - 动态算子图编译与并行调度
 * 
 * 本文件包含cholesky、trsm、madd三个核心算子的实现
 * 以及分块Cholesky分解的主函数
 * 
 * 算法说明：
 * 标准的分块Cholesky分解算法（右看版本）：
 * 对于每个块列i：
 * 1. 对角块分解：L_ii = chol(A_ii)
 * 2. 三角求解：L_ji = A_ji * L_ii^{-1}
 * 3. Schur补更新：A_jk = A_jk - L_ji * L_ki^T (对于所有j, k >= i+b)
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>

// 朴素Cholesky分解
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

// 分块Cholesky分解 - 正确版本
// 使用右看算法（Right-looking algorithm）
int block_cholesky(double* A, double* L, int n, int b) {
    // 初始化L为0
    for (int idx = 0; idx < n * n; idx++) {
        L[idx] = 0.0;
    }
    
    // 创建工作矩阵用于Schur补，只存储下三角部分
    std::vector<double> S(n * n, 0.0);
    for (int i = 0; i < n * n; i++) {
        S[i] = A[i];
    }
    
    // 逐块处理
    for (int i = 0; i < n; i += b) {
        int ib = std::min(b, n - i);
        
        // 步骤1: 对角块Cholesky分解
        // L_ii = chol(S_ii)
        // 注意：cholesky函数从S读取，写入L
        int ret = cholesky(&S[i * n + i], &L[i * n + i], ib, n);
        if (ret != 0) {
            return ret;
        }
        
        // 步骤2: 三角求解（计算当前块列的所有非对角块）
        // L_ji = S_ji * L_ii^{-T}，即解 L_ji * L_ii^T = S_ji
        for (int j = i + b; j < n; j += b) {
            int jb = std::min(b, n - j);
            trsm(&S[j * n + i], &L[i * n + i], &L[j * n + i], jb, ib, n);
        }
        
        // 步骤3: Schur补更新（更新所有后续块）
        // S_jk = S_jk - L_ji * L_ki^T，对于 j, k >= i+b 且 j >= k
        for (int j = i + b; j < n; j += b) {
            int jb = std::min(b, n - j);
            for (int k = i + b; k <= j; k += b) {  // 只更新下三角部分 (j >= k)
                int kb = std::min(b, n - k);
                madd(&L[j * n + i], &L[k * n + i], &S[j * n + k], jb, kb, ib, n);
            }
        }
    }
    
    // 将L的上三角部分清零
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            L[i * n + j] = 0.0;
        }
    }
    
    return 0;
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
    
    // 生成下三角矩阵L，确保对角元素足够大以保证正定性
    // 使用更好的数值稳定性策略
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            if (i == j) {
                // 对角元素取较大值，确保正定
                // 使用 n+1 作为对角值的基础，确保条件数不会太差
                L_temp[i * n + j] = (double)rand() / RAND_MAX * 0.5 + (double)(n + 1) / (i + 1);
            } else {
                // 非对角元素取较小值
                L_temp[i * n + j] = (double)rand() / RAND_MAX * 0.1;
            }
        }
        // 上三角部分保持为0
        for (int j = i + 1; j < n; j++) {
            L_temp[i * n + j] = 0.0;
        }
    }
    
    // 计算 A = L * L^T
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

// 单核基准测试 - 标准Cholesky分解
int cholesky_single_thread(double* A, double* L, int n) {
    // 初始化L为0
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[i * n + j] = 0.0;
        }
    }
    
    for (int i = 0; i < n; i++) {
        // 计算对角元素
        double sum = A[i * n + i];
        for (int k = 0; k < i; k++) {
            sum -= L[i * n + k] * L[i * n + k];
        }
        if (sum <= 0) {
            printf("Error: Matrix is not positive definite at row %d (diag=%e)\n", i, sum);
            return -1;
        }
        L[i * n + i] = sqrt(sum);
        
        // 计算下三角元素
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
        
        // 先测试单线程版本
        printf("\n=== Testing single-thread Cholesky first ===\n");
        std::vector<double> L_single(n * n, 0.0);
        int result_single = cholesky_single_thread(A_copy.data(), L_single.data(), n);
        if (result_single == 0) {
            double scaled_residual_single = verify_result(A_copy.data(), L_single.data(), n);
            printf("Single-thread scaled residual: %.6e\n", scaled_residual_single);
            if (scaled_residual_single < 16.0) {
                printf("Single-thread: PASS\n");
            } else {
                printf("Single-thread: FAIL\n");
            }
        } else {
            printf("Single-thread Cholesky failed!\n");
        }
        
        printf("\n=== Testing block Cholesky ===\n");
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
    
    auto start = std::chrono::high_resolution_clock::now();
    int result = block_cholesky(A.data(), L.data(), n, block_size);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (result != 0) {
        printf("Block Cholesky decomposition failed!\n");
        return 1;
    }
    
    double elapsed = std::chrono::duration<double>(end - start).count();
    printf("Block Cholesky decomposition time: %.6f seconds\n", elapsed);
    
    double scaled_residual = verify_result(A_copy.data(), L.data(), n);
    printf("Scaled residual: %.6e\n", scaled_residual);
    
    if (scaled_residual < 16.0) {
        printf("Result: PASS\n");
    } else {
        printf("Result: FAIL (scaled residual too large)\n");
    }
    
    std::vector<double> L_single(n * n, 0.0);
    auto start_single = std::chrono::high_resolution_clock::now();
    int result_single = cholesky_single_thread(A_copy.data(), L_single.data(), n);
    auto end_single = std::chrono::high_resolution_clock::now();
    
    if (result_single == 0) {
        double elapsed_single = std::chrono::duration<double>(end_single - start_single).count();
        printf("Single-thread Cholesky time: %.6f seconds\n", elapsed_single);
        printf("Speedup: %.2fx\n", elapsed_single / elapsed);
        
        double scaled_residual_single = verify_result(A_copy.data(), L_single.data(), n);
        printf("Single-thread scaled residual: %.6e\n", scaled_residual_single);
    }
    
    return 0;
}