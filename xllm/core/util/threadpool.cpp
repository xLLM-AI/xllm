/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "threadpool.h"

#include <glog/logging.h>

#include <thread>

#include "core/util/cpu_affinity.h"

namespace xllm {
namespace {

void log_threadpool_creation(size_t num_threads,
                             const std::string& pool_name,
                             const std::string& extra_info) {
  const std::string display_pool_name =
      pool_name.empty() ? "<unnamed>" : pool_name;
  LOG(INFO) << "ThreadPool created, thread count " << num_threads
            << ", pool name " << display_pool_name << extra_info << ".";
}

}  // namespace

ThreadPool::ThreadPool(size_t num_threads,
                       bool cpu_binding,
                       const std::string& pool_name)
    : ThreadPool(num_threads, nullptr, cpu_binding, pool_name) {}

ThreadPool::ThreadPool(size_t num_threads,
                       Runnable init_func,
                       bool cpu_binding,
                       //  const char* fn,
                       //  int32_t ln,
                       const std::string& pool_name)
    : queues_(num_threads), pool_name_(pool_name) {
  log_threadpool_creation(
      num_threads,
      pool_name,
      cpu_binding ? ", cpu_binding true" : ", cpu_binding false");

  std::shared_ptr<Runnable> shared_init;
  if (init_func) {
    shared_init = std::make_shared<Runnable>(std::move(init_func));
  }
  auto counter =
      std::make_shared<BlockingCounter>(static_cast<int32_t>(num_threads));
  for (size_t i = 0; i < num_threads; ++i) {
    int32_t cpu_core =
        cpu_binding ? CpuAffinity::get_instance().next_cpu_core() : -1;
    threads_.emplace_back([this, i, cpu_core, shared_init, counter]() {
      internal_loop(i, shared_init, counter, cpu_core);
    });
  }
  counter->wait();
}

ThreadPool::ThreadPool(size_t num_threads,
                       Runnable init_func,
                       std::vector<int32_t> cpu_cores,
                       const std::string& pool_name)
    : queues_(num_threads), pool_name_(pool_name) {
  log_threadpool_creation(num_threads,
                          pool_name,
                          ", cpu_cores " + std::to_string(cpu_cores.size()));

  // if (!cpu_cores.empty() && cpu_cores.size() != num_threads) {
  //   LOG(WARNING) << "ThreadPool: cpu_cores.size() (" << cpu_cores.size()
  //                << ") != num_threads (" << num_threads
  //                << "), CPU core binding will be skipped";
  //   // cpu_cores.clear();
  // }
  std::shared_ptr<Runnable> shared_init;
  if (init_func) {
    shared_init = std::make_shared<Runnable>(std::move(init_func));
  }
  auto counter =
      std::make_shared<BlockingCounter>(static_cast<int32_t>(num_threads));
  for (size_t i = 0; i < num_threads; ++i) {
    int32_t cpu_core = cpu_cores.empty() ? -1 : cpu_cores[i % cpu_cores.size()];
    threads_.emplace_back([this, i, cpu_core, shared_init, counter]() {
      internal_loop(i, shared_init, counter, cpu_core);
    });
  }
  counter->wait();
}

ThreadPool::~ThreadPool() {
  // push nullptr to the queue to signal threads to exit
  for (size_t i = 0; i < threads_.size(); ++i) {
    queues_[i].push(nullptr);
  }
  // wait for all threads to finish
  for (auto& thread : threads_) {
    thread.join();
  }
}

// schedule a runnable to be executed
int32_t ThreadPool::schedule(Runnable runnable) {
  if (runnable == nullptr) {
    return -1;
  }

  // LOG(INFO) << "ThreadPool " << pool_name_ << " schedule.";

  size_t current;
  size_t next;
  do {
    current = index_.load(std::memory_order_relaxed);
    next = (current + 1) % queues_.size();
  } while (!index_.compare_exchange_weak(
      current, next, std::memory_order_relaxed, std::memory_order_relaxed));
  queues_[current].push(std::move(runnable));
  return current;
}

void ThreadPool::schedule_with_tid(Runnable runnable, size_t tid) {
  if (runnable == nullptr) {
    return;
  }

  // LOG(INFO) << "ThreadPool " << pool_name_ << " schedule.";

  queues_[tid].push(std::move(runnable));
}

void ThreadPool::internal_loop(size_t index,
                               std::shared_ptr<Runnable> init_func,
                               std::shared_ptr<BlockingCounter> block_counter,
                               int32_t cpu_core) {
  if (cpu_core >= 0 && bind_thread_to_cpu_core(cpu_core) != 0) {
    LOG(WARNING) << "Thread " << index << " CPU binding to core " << cpu_core
                 << " failed, running unbound";
  }
  if (init_func && *init_func) {
    (*init_func)();
  }
  block_counter->decrement_count();

  while (true) {
    // if (queues_[index].size() >= 2) {
    //   LOG(INFO) << "=== ThreadPool " << pool_name_ << ", thread " << index
    //             << " has tasks " << queues_[index].size();
    // }
    Runnable runnable = queues_[index].pop();

    if (runnable == nullptr) {
      // nullptr is a signal to exit
      break;
    }
    runnable();
  }
}

MPMCThreadPool::MPMCThreadPool(size_t num_threads,
                               bool cpu_binding,
                               const char* fn,
                               int32_t ln,
                               const std::string& pool_name)
    : MPMCThreadPool(num_threads, nullptr, cpu_binding, fn, ln, pool_name) {}

MPMCThreadPool::MPMCThreadPool(size_t num_threads,
                               Runnable init_func,
                               bool cpu_binding,
                               const char* fn,
                               int32_t ln,
                               const std::string& pool_name)
    : pool_name_(pool_name) {
  if (fn != nullptr) {
    LOG(INFO) << "ThreadPool thread count " << num_threads << " in file " << fn
              << " line " << ln << ", pool name " << pool_name
              << ", cpu_binding " << (cpu_binding ? "true" : "false");
  } else {
    LOG(INFO) << "ThreadPool thread count " << num_threads << ", pool name "
              << pool_name << ", cpu_binding "
              << (cpu_binding ? "true" : "false");
  }

  sems_.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    sems_.emplace_back(std::make_unique<moodycamel::LightweightSemaphore>());
  }

  BlockingCounter counter(num_threads);
  threads_.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    int32_t cpu_core =
        cpu_binding ? CpuAffinity::get_instance().next_cpu_core() : -1;
    threads_.emplace_back([this,
                           i,
                           cpu_core,
                           init_func_ptr = &init_func,
                           counter_ptr = &counter]() mutable {
      internal_loop(i, init_func_ptr, counter_ptr, cpu_core);
    });
  }
  counter.wait();
}

MPMCThreadPool::MPMCThreadPool(size_t num_threads,
                               Runnable init_func,
                               std::vector<int32_t> cpu_cores,
                               const char* fn,
                               int32_t ln,
                               const std::string& pool_name)
    : pool_name_(pool_name) {
  if (fn != nullptr) {
    LOG(INFO) << "ThreadPool thread count " << num_threads << " in file " << fn
              << " line " << ln << ", pool name " << pool_name;
  } else {
    LOG(INFO) << "ThreadPool thread count " << num_threads << ", pool name "
              << pool_name;
  }

  sems_.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    sems_.emplace_back(std::make_unique<moodycamel::LightweightSemaphore>());
  }

  BlockingCounter counter(num_threads);
  threads_.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    int32_t cpu_core = cpu_cores.empty() ? -1 : cpu_cores[i % cpu_cores.size()];
    threads_.emplace_back([this,
                           i,
                           cpu_core,
                           init_func_ptr = &init_func,
                           counter_ptr = &counter]() mutable {
      internal_loop(i, init_func_ptr, counter_ptr, cpu_core);
    });
  }
  counter.wait();
}

MPMCThreadPool::~MPMCThreadPool() {
  // Mark shutdown first so workers see it once they wake up.
  stopped_.store(true, std::memory_order_release);
  // Push a nullptr sentinel into each private queue to guarantee FIFO
  // shutdown semantics (any tasks queued via `schedule_with_tid` before
  // destruction are still drained in order). Also wake every worker.
  // for (size_t i = 0; i < private_queues_.size(); ++i) {
  //   private_queues_[i].push(nullptr);
  //   sems_[i]->signal();
  // }
  for (auto& thread : threads_) {
    thread.join();
  }
}

void MPMCThreadPool::schedule(Runnable runnable) {
  if (runnable == nullptr) {
    return;
  }
  global_queue_.enqueue(std::move(runnable));
  // Broadcast wake to all workers. For workers that are already running,
  // `signal()` is a single atomic increment (no syscall); for idle workers
  // it triggers a futex wake so the first one to wake up steals the task.
  const size_t n = sems_.size();
  for (size_t i = 0; i < n; ++i) {
    sems_[i]->signal();
  }
}

bool MPMCThreadPool::drain_global_queue() {
  bool ran = false;
  Runnable task;
  while (global_queue_.try_dequeue(task)) {
    if (task != nullptr) {
      task();
    }
    ran = true;
  }
  return ran;
}

void MPMCThreadPool::internal_loop(size_t index,
                                   Runnable* init_func,
                                   BlockingCounter* block_counter,
                                   int32_t cpu_core) {
  if (cpu_core >= 0 && bind_thread_to_cpu_core(cpu_core) != 0) {
    LOG(WARNING) << "Thread " << index << " CPU binding to core " << cpu_core
                 << " failed, running unbound";
  }
  if (init_func != nullptr && *init_func != nullptr) {
    (*init_func)();
  }
  block_counter->decrement_count();

  auto& sem = *sems_[index];
  // auto& private_q = private_queues_[index];

  while (true) {
    // 1) Steal from the global queue.
    drain_global_queue();

    // 2) Re-check shutdown before blocking. The destructor sets `stopped_`
    //    *and* pushes a sentinel into every private queue, so this branch
    //    is mostly belt-and-suspenders.
    if (stopped_.load(std::memory_order_acquire)) {
      drain_global_queue();
      return;
    }

    // 3) Block until someone signals us. A stray signal (e.g. the task we
    //    were woken for was stolen by another worker) is harmless: we just
    //    loop back, find both queues empty, and re-block.
    sem.wait();
  }
}

}  // namespace xllm
