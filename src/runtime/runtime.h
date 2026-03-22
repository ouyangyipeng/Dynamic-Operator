/**
 * 动态算子图并行运行时库
 * 2026年毕昇杯编译系统挑战赛
 *
 * 提供线程池、任务调度和依赖图管理功能
 */

#ifndef RUNTIME_H
#define RUNTIME_H

#include <functional>
#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <unordered_set>

namespace runtime {

// 任务ID类型
using TaskId = int;

// 任务状态
enum class TaskStatus {
    PENDING,    // 等待中
    READY,      // 就绪（依赖已满足）
    RUNNING,    // 执行中
    COMPLETED   // 已完成
};

// 任务类型
enum class TaskType {
    CHOLESKY,   // 对角块Cholesky分解
    TRSM,       // 三角求解
    MADDS,      // Schur补更新（对称更新）
    UNKNOWN
};

// 任务结构
struct Task {
    TaskId id;
    TaskType type;
    std::function<void()> func;          // 任务函数
    std::vector<TaskId> dependencies;   // 依赖的任务ID列表
    std::atomic<int> ref_count{0};       // 剩余依赖计数
    TaskStatus status = TaskStatus::PENDING;
    
    // 任务元数据（用于调试和分析）
    int block_i = -1;  // 块行索引
    int block_j = -1;  // 块列索引
    int block_k = -1;  // 块索引（用于madd）
};

// 线程池
class ThreadPool {
public:
    explicit ThreadPool(int num_threads = 0);
    ~ThreadPool();
    
    // 提交任务（返回任务ID）
    TaskId submit(std::function<void()> func);
    
    // 等待所有任务完成
    void wait_all();
    
    // 获取线程数
    int get_num_threads() const { return num_threads_; }
    
private:
    void worker_thread();
    
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::atomic<bool> stop_{false};
    std::atomic<int> active_tasks_{0};
    int num_threads_;
};

// 任务调度器（支持依赖管理）
class TaskScheduler {
public:
    explicit TaskScheduler(int num_threads = 0);
    ~TaskScheduler();
    
    // 创建任务
    TaskId create_task(TaskType type, std::function<void()> func,
                       int block_i = -1, int block_j = -1, int block_k = -1);
    
    // 添加依赖关系
    void add_dependency(TaskId from, TaskId to);
    
    // 执行所有任务并等待完成
    void execute_and_wait();
    
    // 重置调度器
    void reset();
    
    // 获取任务信息
    const Task* get_task(TaskId id) const;
    
    // 获取任务数量
    int get_task_count() const { return tasks_.size(); }
    
    // 获取已完成任务数量
    int get_completed_count() const { return completed_count_.load(); }
    
    // 获取线程数
    int get_num_threads() const { return pool_->get_num_threads(); }

private:
    void schedule_task(Task* task);
    void on_task_completed(TaskId id);
    void try_schedule_ready_tasks();
    
    std::unique_ptr<ThreadPool> pool_;
    std::vector<std::unique_ptr<Task>> tasks_;
    std::unordered_map<TaskId, std::vector<TaskId>> dependents_;  // 被依赖关系
    std::mutex mutex_;
    std::atomic<int> completed_count_{0};
    std::condition_variable done_cv_;
};

// 依赖图构建器（专门用于分块Cholesky分解）
class CholeskyDependencyGraph {
public:
    CholeskyDependencyGraph(int n, int block_size);
    
    // 构建依赖图
    void build(TaskScheduler& scheduler);
    
    // 获取块数量
    int get_num_blocks() const { return num_blocks_; }
    
private:
    int n_;           // 矩阵维度
    int block_size_;  // 块大小
    int num_blocks_;  // 块数量
    
    // 任务ID映射：task_map_[i][j] 存储块(i,j)对应的cholesky/trsm任务ID
    std::vector<std::vector<TaskId>> task_map_;
    // madd任务映射：madd_map_[i][j][k] 存储更新块(j,k)在第i步的madd任务ID
    std::vector<std::vector<std::vector<TaskId>>> madd_map_;
};

// 全局运行时初始化
void init_runtime(int num_threads = 0);
void shutdown_runtime();
TaskScheduler* get_scheduler();

} // namespace runtime

#endif // RUNTIME_H