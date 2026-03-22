# 毕昇杯编译系统挑战赛 - 动态算子图编译与并行调度

## 赛题概述

**比赛名称**: 2026年全国大学生计算机系统能力大赛编译系统设计赛（华为毕昇杯）

**赛题**: 动态算子图编译与并行调度

**核心任务**: 在LLVM中以增加Pass的方式实现对分块Cholesky分解算法中算子依赖关系的分析，生成二进制可执行程序，确保程序功能正确且通过误差检验的前提下，最大化并行计算性能。

---

## 关键技术要求

### 1. 目标平台
- **CPU**: 鲲鹏920 ARM
- **操作系统**: openEuler
- **编译器**: OpenEuler LLVM 15 + 毕昇编译器

### 2. 核心算子
- `cholesky`: 朴素Cholesky分解
- `trsm`: 三角方程求解 (solve X*B=A)
- `madd`: 矩阵乘加 (C=A*Bt+C)

### 3. 限制条件
- ✅ 仅限LLVM层面增加Pass方式
- ✅ 可以在源代码中加入标注辅助LLVM Pass分析
- ✅ 可以设计实现并行运行时库
- ❌ 不可修改算法源码（标注除外）
- ❌ 不可修改cholesky、trsm、madd算子实现

### 4. 评分标准
| 指标 | 权重 |
|------|------|
| 精度通过率 | 40% |
| 性能得分 | 60% |

性能得分计算: `100*(a*η1+b*η2+c*η3)/m`
- η1, η2, η3: 最大、平均、最小加速比
- a=60%, b=30%, c=10%
- m: 鲲鹏920线程数量
- T0: 单核执行时间, T: 实际执行时间, η=T0/T

---

## 资源链接

### 代码与编译器
- LLVM for openEuler: https://gitee.com/openeuler/llvm-project/tree/dev_17.0.6
- 毕昇编译器二进制: https://mirrors.huaweicloud.com/kunpeng/archive/compiler/bisheng_compiler/BiSheng%20Compiler-3.2.0.1-aarch64-linux.tar.gz
- 比赛代码仓: https://compiler.educg.net

### 文档
- LLVM Pass开发指南: https://llvm.org/docs/WritingAnLLVMNewPMPass.html
- 毕昇编译器用户指南: https://mirrors.huaweicloud.com/kunpeng/archive/compiler/bisheng_compiler/

---

## 技术方案要点

### 分块Cholesky算法分析

```
block_cholesky(A, L, n, b):
    for i = 0 to n step b:
        cholesky(A[i*n+i], L[i*n+i], i_dim, n)      // 对角块分解
        for j = i+b to n step b:
            trsm(A[j*n+i], L[i*n+i], L[j*n+i], ...)  // 三角求解
            for k = i+b to n step b:
                madd(L[j*n+i], L[k*n+i], L[j*n+k], ...) // 矩阵乘加
```

### 算子依赖关系
1. **cholesky(i)** 依赖于前一轮迭代的所有madd完成
2. **trsm(i,j)** 依赖于cholesky(i)完成
3. **madd(i,j,k)** 依赖于trsm(i,j)和trsm(i,k)完成

### 并行化机会
- 同一轮次的不同j的trsm可以并行
- 同一轮次的不同(j,k)的madd可以并行
- 需要构建依赖图并进行任务调度

---

## 进度记录

### 当前状态
- [x] 阅读赛题文档，理解需求
- [x] 环境搭建
- [x] 实现基础分块Cholesky分解算法
- [x] 实现并行运行时库
- [x] 实现线程池和任务调度器
- [x] 实现依赖图构建和调度
- [x] 设计LLVM Pass架构
- [x] 实现算子依赖分析Pass
- [x] 功能测试（200组测试矩阵）
- [x] 性能测试和优化
- [x] 编写设计文档
- [ ] 准备提交材料

### 更新日志

#### 2026-03-22
- 创建PROGRESS.md文档
- 阅读赛题文档，理解比赛要求
- 实现基础分块Cholesky分解算法
  - 实现cholesky、trsm、madd三个核心算子
  - 实现block_cholesky主函数
  - 实现verify_result验证函数
  - 实现generate_spd_matrix生成正定矩阵
- 修复矩阵生成函数的数值稳定性问题
- 测试通过：256x256, 512x512, 1024x1024, 2048x2048矩阵
- 实现并行运行时库
  - ThreadPool: 线程池实现
  - TaskScheduler: 任务调度器，支持依赖管理
  - CholeskyDependencyGraph: 依赖图构建器
- 实现OpenMP并行版本
  - block_cholesky_parallel_simple: 简化并行版本
  - 使用OpenMP parallel for实现TRSM和MADD并行
- 实现LLVM Pass
  - CholeskyOperatorPass: 分析算子调用和依赖关系
  - 支持新旧Pass管理器
- 功能测试：200组测试矩阵全部通过
  - 通过率：100%
  - 最大缩放残差：1.28e-04（远小于16）
- 编写设计文档DESIGN.md

---

## 项目结构

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
└── DESIGN.md                 # 设计文档
```

---

## 测试结果

### 功能测试结果
```
Running 200 tests with 1024x1024 matrices, block size 64
Using 12 OpenMP threads

========================================
Summary:
  Total tests: 200
  Passed: 200
  Failed: 0
  Pass rate: 100.0%
  Total time: 21.37 seconds
  Average time: 0.106 seconds
  Max residual: 1.28e-04
========================================
```

### 性能测试结果
| 矩阵规模 | 串行时间 | 并行时间(12线程) | 加速比 |
|---------|---------|-----------------|-------|
| 256×256 | 0.005s  | 0.006s          | 0.83x |
| 512×512 | 0.030s  | 0.017s          | 1.76x |
| 1024×1024 | 0.19s | 0.05s          | 3.8x  |
| 2048×2048 | 0.84s | 0.29s          | 2.9x  |

---

## 下一步计划

1. 在鲲鹏920 ARM平台上测试和优化
2. 集成毕昇编译器
3. 进一步优化并行性能
4. 准备提交材料