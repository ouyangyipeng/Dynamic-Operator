/**
 * 动态算子图并行运行时库实现
 * 2026年毕昇杯编译系统挑战赛
 */

#include "runtime.h"
#include <iostream>
#include <algorithm>

namespace runtime {

// ==================== ThreadPool 实现 ====================

ThreadPool::ThreadPool(int num_threads) {
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
    }
    num_threads_ = num_threads;
    
    threads_.reserve(num_threads_);
    for (int i = 0; i < num_threads_; i++) {
        threads_.emplace_back(&ThreadPool::worker_thread, this);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    
    for (auto& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

TaskId ThreadPool::submit(std::function<void()> func) {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.push(std::move(func));
    active_tasks_++;
    cv_.notify_one();
    return 0;  // 简化版本，不返回具体ID
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this] { return tasks_.empty() && active_tasks_ == 0; });
}

void ThreadPool::worker_thread() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            
            if (stop_ && tasks_.empty()) {
                return;
            }
            
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        
        task();
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_tasks_--;
            if (active_tasks_ == 0 && tasks_.empty()) {
                done_cv_.notify_all();
            }
        }
    }
}

// ==================== TaskScheduler 实现 ====================

TaskScheduler::TaskScheduler(int num_threads)
    : pool_(std::make_unique<ThreadPool>(num_threads)) {
}

TaskScheduler::~TaskScheduler() = default;

TaskId TaskScheduler::create_task(TaskType type, std::function<void()> func,
                                   int block_i, int block_j, int block_k) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    TaskId id = static_cast<TaskId>(tasks_.size());
    auto task = std::make_unique<Task>();
    task->id = id;
    task->type = type;
    task->func = std::move(func);
    task->block_i = block_i;
    task->block_j = block_j;
    task->block_k = block_k;
    task->status = TaskStatus::PENDING;
    task->ref_count = 0;  // 初始化为0，add_dependency时增加
    
    tasks_.push_back(std::move(task));
    return id;
}

void TaskScheduler::add_dependency(TaskId from, TaskId to) {
    // from 依赖于 to (to 必须先完成)
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (from >= 0 && from < static_cast<TaskId>(tasks_.size()) &&
        to >= 0 && to < static_cast<TaskId>(tasks_.size())) {
        tasks_[from]->dependencies.push_back(to);
        tasks_[from]->ref_count++;
        dependents_[to].push_back(from);
    }
}

void TaskScheduler::execute_and_wait() {
    // 初始化：找出所有没有依赖的任务
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& task : tasks_) {
            if (task->dependencies.empty() && task->status == TaskStatus::PENDING) {
                task->status = TaskStatus::READY;
                schedule_task(task.get());
            }
        }
    }
    
    // 等待所有任务完成
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this] {
        return completed_count_ == static_cast<int>(tasks_.size());
    });
}

void TaskScheduler::schedule_task(Task* task) {
    // 注意：调用此函数时应该已经持有mutex_
    task->status = TaskStatus::RUNNING;
    
    // 捕获this和task->id用于回调
    TaskId task_id = task->id;
    auto func = task->func;
    
    pool_->submit([this, task_id, func]() {
        func();
        on_task_completed(task_id);
    });
}

void TaskScheduler::on_task_completed(TaskId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    tasks_[id]->status = TaskStatus::COMPLETED;
    completed_count_++;
    
    // 更新依赖此任务的其他任务
    auto it = dependents_.find(id);
    if (it != dependents_.end()) {
        for (TaskId dep_id : it->second) {
            Task* dep_task = tasks_[dep_id].get();
            dep_task->ref_count--;
            
            if (dep_task->ref_count == 0 && dep_task->status == TaskStatus::PENDING) {
                dep_task->status = TaskStatus::READY;
                schedule_task(dep_task);
            }
        }
    }
    
    // 检查是否所有任务都完成
    if (completed_count_ == static_cast<int>(tasks_.size())) {
        done_cv_.notify_all();
    }
}

void TaskScheduler::try_schedule_ready_tasks() {
    for (auto& task : tasks_) {
        if (task->ref_count == 0 && task->status == TaskStatus::PENDING) {
            task->status = TaskStatus::READY;
            schedule_task(task.get());
        }
    }
}

void TaskScheduler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.clear();
    dependents_.clear();
    completed_count_ = 0;
}

const Task* TaskScheduler::get_task(TaskId id) const {
    if (id >= 0 && id < static_cast<TaskId>(tasks_.size())) {
        return tasks_[id].get();
    }
    return nullptr;
}

// ==================== CholeskyDependencyGraph 实现 ====================

CholeskyDependencyGraph::CholeskyDependencyGraph(int n, int block_size)
    : n_(n), block_size_(block_size) {
    num_blocks_ = (n + block_size - 1) / block_size;
}

void CholeskyDependencyGraph::build(TaskScheduler& scheduler) {
    // 初始化任务映射
    task_map_.resize(num_blocks_, std::vector<TaskId>(num_blocks_, -1));
    madd_map_.resize(num_blocks_, 
                     std::vector<std::vector<TaskId>>(num_blocks_, 
                     std::vector<TaskId>(num_blocks_, -1)));
    
    // 按照分块Cholesky算法的依赖关系创建任务
    // 右看算法（Right-looking）：
    // 对于每个块列 i：
    //   1. cholesky(i): 对角块分解
    //   2. trsm(j, i): 三角求解，j > i
    //   3. madd(j, k, i): Schur补更新，j >= k > i
    
    for (int i = 0; i < num_blocks_; i++) {
        // 步骤1: 对角块Cholesky分解
        // 依赖于之前所有步骤的madd更新
        TaskId chol_task = scheduler.create_task(
            TaskType::CHOLESKY,
            []() { /* 实际操作由外部设置 */ },
            i, i, -1
        );
        task_map_[i][i] = chol_task;
        
        // cholesky(i) 依赖于所有 madd(j, i, k) 其中 k < i 且 j >= i
        if (i > 0) {
            for (int k = 0; k < i; k++) {
                for (int j = i; j < num_blocks_; j++) {
                    if (madd_map_[k][j][i] != -1) {
                        scheduler.add_dependency(chol_task, madd_map_[k][j][i]);
                    }
                }
            }
        }
        
        // 步骤2: 三角求解
        for (int j = i + 1; j < num_blocks_; j++) {
            TaskId trsm_task = scheduler.create_task(
                TaskType::TRSM,
                []() { /* 实际操作由外部设置 */ },
                j, i, -1
            );
            task_map_[j][i] = trsm_task;
            
            // trsm(j, i) 依赖于 cholesky(i)
            scheduler.add_dependency(trsm_task, chol_task);
            
            // trsm(j, i) 也依赖于之前的madd更新
            if (i > 0) {
                for (int k = 0; k < i; k++) {
                    if (madd_map_[k][j][i] != -1) {
                        scheduler.add_dependency(trsm_task, madd_map_[k][j][i]);
                    }
                }
            }
        }
        
        // 步骤3: Schur补更新
        for (int j = i + 1; j < num_blocks_; j++) {
            for (int k = i + 1; k <= j; k++) {
                TaskId madd_task = scheduler.create_task(
                    TaskType::MADDS,
                    []() { /* 实际操作由外部设置 */ },
                    j, k, i
                );
                madd_map_[i][j][k] = madd_task;
                
                // madd(j, k, i) 依赖于 trsm(j, i) 和 trsm(k, i)
                scheduler.add_dependency(madd_task, task_map_[j][i]);
                if (k != j) {
                    scheduler.add_dependency(madd_task, task_map_[k][i]);
                }
            }
        }
    }
}

// ==================== 全局运行时 ====================

static TaskScheduler* g_scheduler = nullptr;

void init_runtime(int num_threads) {
    if (g_scheduler == nullptr) {
        g_scheduler = new TaskScheduler(num_threads);
    }
}

void shutdown_runtime() {
    delete g_scheduler;
    g_scheduler = nullptr;
}

TaskScheduler* get_scheduler() {
    return g_scheduler;
}

} // namespace runtime