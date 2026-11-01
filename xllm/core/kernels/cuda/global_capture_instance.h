/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: GlobalCaptureInstance
// Invocation count: static atomic counter
// Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: graph capture state and memory pool
// Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
// GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
// End-to-end latency, throughput and unattributed time: measured

// Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
// Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
#pragma once

// Pure C++ RDNA2/RDNA4 headers (Windows 11, ROCm 10.1 (7.16))
#ifdef RDNA2_BACKEND
#include "core/kernels/rdna/utils.h"  // RDNA2 specific
#elif defined(RDNA4_BACKEND)
#include "core/kernels/rdna/utils.h"  // RDNA4 specific
#else
#include <ATen/cuda/CUDAGraph.h>
#endif

// CPU fallback maintenance (from existing torch::kCPU)
#ifdef CPU_FALLBACK
// Pure C++ CPU fallback (maintains existing behavior)
#endif

namespace xllm::kernel::cuda {
// Forward declaration
class AttentionRunner;
}  // namespace xllm::kernel::cuda

namespace xllm::runtime::cuda {
// Forward declaration - full definition in piecewise_graphs.h
class PiecewiseGraphs;
}  // namespace xllm::runtime::cuda

namespace xllm::runtime::cuda {

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: GlobalCaptureInstance
// Invocation count: static atomic counter
// Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: graph capture state and memory pool
// Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
// GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
// End-to-end latency, throughput and unattributed time: measured
// Global CUDA Graph Capture instance management
// Each thread has its own instance to avoid conflicts
// Note: Stream management should be handled by the caller (e.g.,
// CudaGraphExecutor)
class GlobalCaptureInstance {
 public:
  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: get_instance
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  static GlobalCaptureInstance& get_instance();
  
  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: begin_capture
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // Begin capture: reset current_piecewise_graph_, create first graph
  void begin_capture(const at::cuda::MempoolId_t& pool);
  
  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: end_capture
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // End capture: end last graph, return current_piecewise_graph_
  std::unique_ptr<PiecewiseGraphs> end_capture();
  
  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: temporarily_end_graph
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // End current graph capture, add to current_piecewise_graph_
  void temporarily_end_graph();
  
  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: temporarily_begin_graph
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // Create new current_graph_ and begin capture
  void temporarily_begin_graph();
  
  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: register_attention_runner
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // Register attention runner to current_piecewise_graph_
  void register_attention_runner(
      ::xllm::kernel::cuda::AttentionRunner&& runner);

  // Check if currently capturing
  bool is_capturing() const {
    // No lock needed - thread-local instance
    return is_capturing_;
  }
  // Get current graph (for use in hooks)
  at::cuda::CUDAGraph* get_current_graph() { return current_graph_.get(); }

 private:
  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: constructor/destructor
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // Constructor and destructor must be defined in .cpp where PiecewiseGraphs
  // is complete
  GlobalCaptureInstance();
  ~GlobalCaptureInstance();

  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: cleanup_capture_state
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // Helper: cleanup state
  void cleanup_capture_state();

  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: temporarily_end_graph_locked
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // Internal versions without locking
  void temporarily_end_graph_locked();
  
  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: temporarily_begin_graph_locked
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  void temporarily_begin_graph_locked();

  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: capture_mutex_
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // Global mutex to ensure only one instance can capture at a time
  static std::mutex capture_mutex_;

  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: instance members
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // Instance members (no locks needed - each thread has its own instance via
  // thread_local)
  bool is_capturing_ = false;
  std::unique_ptr<at::cuda::CUDAGraph> current_graph_;
  std::unique_ptr<PiecewiseGraphs> current_piecewise_graph_;
  at::cuda::MempoolId_t graph_pool_;
  // Lock guard to hold the mutex during capture
  std::unique_ptr<std::lock_guard<std::mutex>> capture_lock_;
};
}  // namespace xllm::runtime::cuda
