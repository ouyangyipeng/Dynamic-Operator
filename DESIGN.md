# 动态算子图编译与并行调度 - 设计文档

## 2026年毕昇杯编译系统挑战赛

---

## 1. 项目概述

本项目实现了分块Cholesky分解的动态算子图编译与并行调度系统，包括：

1. **核心算法实现**：分块Cholesky分解算法（cholesky、trsm、madd三个核心算子）
2. **并行运行时库**：线程池、任务调度器、依赖图管理
3. **LLVM Pass**：算子依赖分析Pass，用于自动识别算子调用和依赖关系
4. **性能优化**：OpenMP并行化，实现任务级并行

## 2. 算法设计

### 2.1 分块Cholesky分解算法

采用右看（Right-looking）算法，按块列顺序处理：

```
for i = 0 to n-1 step b:
    1. 对角块分解: L_ii = chol(A_ii)
    2. 三角求解: L_ji = A_ji * L_ii^{-1}  (j > i)
    3. Schur补更新: A_jk = A_jk - L_ji * L_ki^T  (j >= k > i)
```

### 2.2 算子依赖关系

```
cholesky(i) 依赖于: 之前所有更新到块(i,i)的madd操作
trsm(j,i) 依赖于:   cholesky(i) 和之前更新到块(j,i)的madd操作
madd(j,k,i) 依赖于: trsm(j,i) 和 trsm(k,i)
```

### 2.3 并行性分析

- **TRSM并行**：同一块列的不同行TRSM可以并行执行
- **MADD并行**：同一块列的所有Schur补更新可以并行执行
- **任务粒度**：以块为单位，块大小影响并行粒度和缓存效率

## 3. 系统架构

### 3.1 目录结构

```
Dyna-Oper/
├── src/
│   ├── cholesky.cpp          # 基础分块Cholesky实现
│   ├── cholesky_omp.cpp      # OpenMP并行版本
│   ├── cholesky_parallel.cpp # 运行时库并行版本
│   ├── test_cholesky.cpp     # 测试程序
│   ├── runtime/
│   │   ├── runtime.h         # 运行时库头文件
│   │   └── runtime.cpp       # 运行时库实现
│   └── pass/
│       ├── CMakeLists.txt    # Pass构建配置
│       └── CholeskyOperatorPass.cpp  # LLVM Pass实现
├── build/                    # 构建输出目录
├── PROGRESS.md               # 进度记录
└── DESIGN.md                 # 本设计文档
```

### 3.2 核心组件

#### 3.2.1 算子实现 (src/cholesky.cpp)

```cpp
// 对角块Cholesky分解
int cholesky(double* A, double* L, int n, int lda);

// 三角求解: X = A * L^{-1}
void trsm(double* A, double* L, double* X, int m, int n, int lda);

// 矩阵乘加: C = C - A * B^T
void madd(double* A, double* B, double* C, int m, int n, int k, int lda);
```

#### 3.2.2 并行运行时库 (src/runtime/)

- **ThreadPool**: 线程池，管理工作线程
- **TaskScheduler**: 任务调度器，支持依赖管理
- **CholeskyDependencyGraph**: 依赖图构建器

#### 3.2.3 LLVM Pass (src/pass/)

- **CholeskyOperatorPass**: 分析算子调用，识别依赖关系

## 4. 实现细节

### 4.1 数据布局

- 矩阵采用列主序存储（Column-major）
- 块大小默认为64，可根据缓存大小调整

### 4.2 并行策略

采用OpenMP实现任务级并行：

```cpp
// TRSM并行
#pragma omp parallel for schedule(dynamic)
for (int j = i + b; j < n; j += b) {
    trsm(...);
}

// MADD并行
#pragma omp parallel for schedule(dynamic)
for (size_t idx = 0; idx < jk_pairs.size(); idx++) {
    madd(...);
}
```

### 4.3 正确性验证

使用缩放残差验证：

```
scaled_residual = ||A - L*L^T||_inf / (||A||_inf * n * eps)
```

要求 scaled_residual < 16 为通过。

## 5. 测试结果

### 5.1 功能测试

- 测试矩阵数量：200
- 矩阵规模：1024×1024
- 块大小：64
- 通过率：100%
- 最大缩放残差：1.28e-04（远小于16）

### 5.2 性能测试

| 矩阵规模 | 串行时间 | 并行时间(12线程) | 加速比 |
|---------|---------|-----------------|-------|
| 256×256 | 0.005s  | 0.006s          | 0.83x |
| 512×512 | 0.030s  | 0.017s          | 1.76x |
| 1024×1024 | 0.19s | 0.05s          | 3.8x  |
| 2048×2048 | 0.84s | 0.29s          | 2.9x  |

## 6. 编译和使用

### 6.1 编译

```bash
# 编译基础版本
g++ -O2 -o build/cholesky src/cholesky.cpp

# 编译OpenMP并行版本
g++ -O2 -std=c++17 -fopenmp -o build/cholesky_omp src/cholesky_omp.cpp

# 编译测试程序
g++ -O2 -std=c++17 -fopenmp -o build/test_cholesky src/test_cholesky.cpp

# 编译LLVM Pass
cd src/pass && mkdir -p build && cd build
cmake .. -DLLVM_DIR=/usr/lib/llvm-15/cmake
make
```

### 6.2 运行

```bash
# 测试单个矩阵
./build/cholesky_omp --test 1024 64

# 运行200组测试
./build/test_cholesky 200 1024 64
```

## 7. 未来优化方向

1. **SIMD优化**：使用AVX/NEON指令集优化算子实现
2. **缓存优化**：优化数据访问模式，提高缓存命中率
3. **动态调度**：实现更细粒度的任务窃取调度
4. **GPU加速**：将计算密集型算子卸载到GPU

## 8. 参考文献

1. LAPACK Working Note: Block Cholesky Decomposition
2. PLASMA: Parallel Linear Algebra Software for Multicore Architectures
3. LLVM Pass Infrastructure Documentation