# 动态算子图编译与并行调度

## 2026年毕昇杯编译系统挑战赛

本项目实现了分块Cholesky分解的动态算子图编译与并行调度系统。

### 项目结构

```
Dyna-Oper/
├── docs/                    # 文档目录
│   └── GUIDE.md             # 队员指南
├── src/
│   ├── cholesky.cpp         # 基础分块Cholesky实现
│   ├── cholesky_omp.cpp     # OpenMP并行版本
│   ├── cholesky_parallel.cpp# 运行时库并行版本
│   ├── test_cholesky.cpp    # 测试程序
│   ├── runtime/             # 并行运行时库
│   │   ├── runtime.h
│   │   └── runtime.cpp
│   └── pass/                # LLVM Pass
│       ├── CMakeLists.txt
│       └── CholeskyOperatorPass.cpp
├── DESIGN.md                # 设计文档
├── PROGRESS.md              # 进度记录
└── README.md                # 本文件
```

### 快速开始

#### 编译

```bash
# 编译OpenMP并行版本
g++ -O3 -std=c++17 -fopenmp -march=armv8-a -o build/cholesky_omp src/cholesky_omp.cpp

# 编译测试程序
g++ -O3 -std=c++17 -fopenmp -march=armv8-a -o build/test_cholesky src/test_cholesky.cpp

# 编译LLVM Pass
cd src/pass && mkdir -p build && cd build
cmake .. -DLLVM_DIR=/usr/lib/llvm-15/cmake
make
```

#### 运行测试

```bash
# 单个测试
./build/cholesky_omp --test 1024 64

# 200组测试
OMP_NUM_THREADS=64 ./build/test_cholesky 200 1024 64
```

### 测试结果

#### 鲲鹏920平台测试 (2026-03-24)

**环境配置**:
- CPU: 鲲鹏920 (192核, 4插槽×48核, 8个NUMA节点)
- 内存: 1.5TB
- 操作系统: Ubuntu 22.04 LTS (aarch64)
- 编译器: g++ 11.4.0

**功能测试结果**:
```
Running 200 tests with 1024x1024 matrices, block size 64
Using 64 OpenMP threads

========================================
Summary:
  Total tests: 200
  Passed: 200
  Failed: 0
  Pass rate: 100.0%
  Total time: 5.66 seconds
  Average time: 0.0283 seconds
  Max residual: 1.218316e-04
========================================
```

### 性能数据

#### 不同矩阵规模 (64线程, 块大小64)

| 矩阵规模 | 串行时间(T0) | 并行时间(T) | 加速比(η) |
|---------|-------------|------------|----------|
| 1024×1024 | 0.35s | 0.03s | **11.7x** |
| 2048×2048 | 3.46s | 0.17s | **20.3x** |
| 4096×4096 | 26.6s | 0.60s | **44.3x** |
| 8192×8192 | 186.2s | 4.99s | **37.3x** |

#### 不同线程数 (4096×4096矩阵)

| 线程数 | 并行时间 | 加速比 |
|-------|---------|-------|
| 1 | 27.95s | 1.0x |
| 8 | 3.83s | 7.3x |
| 16 | 1.91s | 14.6x |
| 32 | 0.93s | 30.1x |
| 48 | 0.79s | 35.5x |
| 64 | 0.60s | 46.5x |
| 96 | 0.59s | 47.3x |
| 128 | 0.53s | **52.7x** |

#### 最优配置

- **最优块大小**: 64
- **最优线程数**: 64-128 (大矩阵推荐128线程)
- **最大加速比**: 52.7x (4096矩阵, 128线程)

### 文档

- [设计文档](DESIGN.md)
- [进度记录](PROGRESS.md)
- [队员指南](docs/GUIDE.md)

### 许可证

本项目仅用于2026年毕昇杯编译系统挑战赛。