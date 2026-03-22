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
g++ -O2 -std=c++17 -fopenmp -o build/cholesky_omp src/cholesky_omp.cpp

# 编译测试程序
g++ -O2 -std=c++17 -fopenmp -o build/test_cholesky src/test_cholesky.cpp

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
./build/test_cholesky 200 1024 64
```

### 测试结果

```
Running 200 tests with 1024x1024 matrices, block size 64
========================================
Summary:
  Total tests: 200
  Passed: 200
  Failed: 0
  Pass rate: 100.0%
  Max residual: 1.28e-04 (threshold: 16)
========================================
```

### 性能数据

| 矩阵规模 | 串行时间 | 并行时间(12核) | 加速比 |
|---------|---------|---------------|-------|
| 512×512 | 0.030s | 0.017s | 1.76x |
| 1024×1024 | 0.19s | 0.05s | 3.8x |
| 2048×2048 | 0.84s | 0.29s | 2.9x |

### 文档

- [设计文档](DESIGN.md)
- [进度记录](PROGRESS.md)
- [队员指南](docs/GUIDE.md)

### 许可证

本项目仅用于2026年毕昇杯编译系统挑战赛。