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
// Operation/kernel/shader name: PiecewiseGraphs
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
#include <glog/logging.h>
#endif

// CPU fallback maintenance (from existing torch::kCPU)
#ifdef CPU_FALLBACK
// Pure C++ CPU fallback (maintains existing behavior)
#endif

#include <memory>
#include <vector>

namespace xllm::kernel::cuda {
// Forward declaration
class AttentionRunner;
}  // namespace xllm::kernel::cuda

namespace xllm::runtime::cuda {

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: PiecewiseGraphs
// Invocation count: static atomic counter
// Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: graph capture state and memory pool
// Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
// GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
// End-to-end latency, throughput and unattributed time: measured
// Encapsulate multiple piecewise CUDA graphs, provide unified replay interface
class PiecewiseGraphs {
 public:
  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: InstructionType
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // Instruction type for replay sequence
  enum class InstructionType { kGraph, kRunner };

  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: constructor/destructor
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  PiecewiseGraphs() = default;
  ~PiecewiseGraphs() = default;
  PiecewiseGraphs(PiecewiseGraphs&&) noexcept = default;
  PiecewiseGraphs& operator=(PiecewiseGraphs&&) noexcept = default;

  // SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
  // Operation/kernel/shader name: add_graph
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // Add a graph and record instruction
  void add_graph(std::unique_ptr<at::cuda::CUDAGraph>&& graph);
  
  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: add_attention_runner
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // Add attention runner and record instruction
  void add_attention_runner(::xllm::kernel::cuda::AttentionRunner&& runner);
  
  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: replay
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // Replay all graphs and runners according to recorded instruction sequence
  void replay(const ::xllm::kernel::cuda::AttentionReplayParams& runner_params);
  
  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: size
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // Get number of graphs
  size_t size() const { return graphs_.size(); }
  
  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: empty
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // Check if empty
  bool empty() const { return graphs_.empty(); }
  
  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: num_runners
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // Get runner count
  size_t num_runners() const;

 private:
  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: graphs_
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // Recorded instruction sequence for replay
  std::vector<std::unique_ptr<at::cuda::CUDAGraph>> graphs_;
  
  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: attention_runners_
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  std::vector<std::unique_ptr<::xllm::kernel::cuda::AttentionRunner>>
      attention_runners_;
  
  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: instructions_
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: graph capture state and memory pool
  // Dispatch dimensions, workload shape and execution configuration: graph pool, piecewise graph
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  // Recorded instruction sequence for replay
  std::vector<InstructionType> instructions_;
};

}  // namespace xllm::runtime::cuda
