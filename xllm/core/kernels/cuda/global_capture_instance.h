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

#pragma once
#include <ATen/cuda/CUDAGraph.h>

#include <memory>
#include <mutex>

namespace xllm::kernel::cuda {
// Forward declaration
class AttentionRunner;
}  // namespace xllm::kernel::cuda

namespace xllm::runtime::cuda {
// Forward declaration - full definition in piecewise_graphs.h
class PiecewiseGraphs;
}  // namespace xllm::runtime::cuda

namespace xllm::runtime::cuda {

// Global CUDA Graph Capture instance management
// Each thread has its own instance to avoid conflicts
// Note: Stream management should be handled by the caller (e.g.,
// CudaGraphExecutor)
class GlobalCaptureInstance {
 public:
  static GlobalCaptureInstance& get_instance();
  // Begin capture: reset current_piecewise_graph_, create first graph
  void begin_capture(const at::cuda::MempoolId_t& pool);
  // End capture: end last graph, return current_piecewise_graph_
  std::unique_ptr<PiecewiseGraphs> end_capture();
  // End current graph capture, add to current_piecewise_graph_
  void temporarily_end_graph();
  // Create new current_graph_ and begin capture
  void temporarily_begin_graph();
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
  // Constructor and destructor must be defined in .cpp where PiecewiseGraphs
  // is complete
  GlobalCaptureInstance();
  ~GlobalCaptureInstance();

  // Helper: cleanup state
  void cleanup_capture_state();

  // Internal versions without locking
  void temporarily_end_graph_locked();
  void temporarily_begin_graph_locked();

  // Global mutex to ensure only one instance can capture at a time
  static std::mutex capture_mutex_;

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
