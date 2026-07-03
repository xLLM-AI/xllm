/* Copyright 2025-2026 The xLLM Authors.

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

#pragma once
#include <ATen/cuda/CUDAGraph.h>

#include <memory>
#include <mutex>

namespace xllm::kernel::cuda {
class AttentionRunner;
}

namespace xllm::runtime::cuda {
class PiecewiseGraphs;
}

namespace xllm::runtime::cuda {

class GlobalCaptureInstance final {
 public:
  static GlobalCaptureInstance& get_instance();
  void begin_capture(const at::cuda::MempoolId_t& pool);
  std::unique_ptr<PiecewiseGraphs> end_capture();
  void temporarily_end_graph();
  void temporarily_begin_graph();
  void register_attention_runner(
      ::xllm::kernel::cuda::AttentionRunner&& runner);

  bool is_capturing() const { return is_capturing_; }
  at::cuda::CUDAGraph* get_current_graph() { return current_graph_.get(); }

 private:
  GlobalCaptureInstance();
  ~GlobalCaptureInstance();

  void cleanup_capture_state();

  void temporarily_end_graph_locked();
  void temporarily_begin_graph_locked();

  static std::mutex capture_mutex_;

  bool is_capturing_ = false;
  std::unique_ptr<at::cuda::CUDAGraph> current_graph_;
  std::unique_ptr<PiecewiseGraphs> current_piecewise_graph_;
  at::cuda::MempoolId_t graph_pool_;
  std::unique_lock<std::mutex> capture_lock_;
};
}  // namespace xllm::runtime::cuda
