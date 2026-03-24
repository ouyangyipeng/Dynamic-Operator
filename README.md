# 动态算子图编译与并行调度

## 2026年毕昇杯编译系统挑战赛

本项目实现了分块Cholesky分解的动态算子图编译与并行调度系统。

### 项目结构

```
Dynamic-Operator/
├── docs/                    # 文档目录
│   └── GUIDE.md             # 队员指南
├── src/
│   ├── cholesky.cpp         # 基础分块Cholesky实现
│   ├── cholesky_omp.cpp     # OpenMP并行版本
│   ├── cholesky_optimized.cpp # 高度优化版本
│   ├── cholesky_extreme.cpp # 极致优化版本（NEON向量化）
│   ├── cholesky_numa.cpp    # NUMA感知优化版本
│   ├── cholesky_npu.cpp     # 昇腾NPU加速版本（开发中）
│   ├── test_cholesky.cpp    # 测试程序
│   ├── runtime/             # 并行运行时库
│   │   ├── runtime.h
│   │   └── runtime.cpp
│   └── pass/                # LLVM Pass
│       ├── CMakeLists.txt
│       └── CholeskyOperatorPass.cpp
├── plans/                   # 优化计划文档
│   └── optimization_plan.md
├── DESIGN.md                # 设计文档
├── PROGRESS.md              # 进度记录
└── README.md                # 本文件
```

### 快速开始

#### 编译

```bash
# 编译所有版本
mkdir -p build

# 基础OpenMP版本
g++ -O3 -std=c++17 -fopenmp -march=armv8-a -o build/cholesky_omp src/cholesky_omp.cpp

# 极致优化版本（推荐）
g++ -O3 -std=c++17 -fopenmp -march=armv8-a -o build/cholesky_extreme src/cholesky_extreme.cpp

# NUMA感知优化版本
g++ -O3 -std=c++17 -fopenmp -march=armv8-a -o build/cholesky_numa src/cholesky_numa.cpp -lnuma

# 测试程序
g++ -O3 -std=c++17 -fopenmp -march=armv8-a -o build/test_cholesky src/test_cholesky.cpp
```

#### 运行测试

```bash
# 单个测试（推荐使用极致优化版本）
OMP_NUM_THREADS=64 ./build/cholesky_extreme --test 4096 64

# NUMA优化版本（192核全开）
OMP_NUM_THREADS=192 OMP_PROC_BIND=close OMP_PLACES=cores ./build/cholesky_numa --test 4096 64

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

#### 极致优化版本 (cholesky_extreme)

**不同矩阵规模 (64线程, 块大小64)**:

| 矩阵规模 | 串行时间(T0) | 并行时间(T) | 加速比(η) |
|---------|-------------|------------|----------|
| 1024×1024 | 0.35s | 0.03s | **11.7x** |
| 2048×2048 | 3.46s | 0.17s | **20.3x** |
| 4096×4096 | 44.4s | 0.77s | **57.7x** |
| 8192×8192 | 186.2s | 4.99s | **37.3x** |

**不同线程数 (4096×4096矩阵)**:

| 线程数 | 并行时间 | 加速比 |
|-------|---------|-------|
| 1 | 50.0s | 1.0x |
| 8 | 15.0s | 3.3x |
| 16 | 2.04s | 24.5x |
| 32 | 0.91s | 54.9x |
| 48 | 0.87s | 57.5x |
| 64 | 0.77s | **64.9x** |
| 96 | 0.79s | 63.3x |

#### NUMA优化版本 (cholesky_numa)

**192线程全开 (4096×4096矩阵)**:

| 版本 | 并行时间 | 加速比 |
|------|---------|-------|
| 极致优化版本 | 4.19s | 11.9x |
| NUMA优化版本 | 2.17s | **26.5x** |

**NUMA优化效果**: 在192核全开时，NUMA感知优化将性能提升了约2倍！

#### 最优配置

- **最优块大小**: 64
- **最优线程数**: 64-96（大矩阵推荐64线程）
- **最大加速比**: **64.9x** (4096矩阵, 64线程)
- **NUMA优化后**: **26.5x** (192线程全开)

### 优化技术

| 技术 | 实现位置 | 效果 |
|------|---------|------|
| 分块算法 | `src/cholesky.cpp` | 提高缓存命中率 |
| OpenMP并行 | `src/cholesky_omp.cpp` | 多核加速 |
| 动态调度 | `schedule(dynamic)` | 负载均衡 |
| NEON向量化 | `src/cholesky_extreme.cpp` | ARM SIMD加速 |
| NUMA感知 | `src/cholesky_numa.cpp` | 减少远程内存访问 |
| 任务依赖分析 | `src/pass/` | 自动识别并行机会 |

### 文档

- [设计文档](DESIGN.md)
- [进度记录](PROGRESS.md)
- [队员指南](docs/GUIDE.md)
- [优化计划](plans/optimization_plan.md)

### 许可证

本项目仅用于2026年毕昇杯编译系统挑战赛。