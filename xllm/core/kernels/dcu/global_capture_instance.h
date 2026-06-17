/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <ATen/hip/HIPGraph.h>
#include <c10/hip/HIPCachingAllocator.h>

#include <memory>
#include <mutex>

#include "kernels/dcu/piecewise_graphs.h"

namespace xllm::runtime::dcu {

using DcuGraphSegment = at::cuda::CUDAGraph;
using DcuMempoolId = c10::hip::MempoolId_t;

class GlobalCaptureInstance final {
 public:
  static GlobalCaptureInstance& get_instance();

  void begin_capture(const DcuMempoolId& pool);
  std::unique_ptr<PiecewiseGraphs> end_capture();

  void temporarily_end_graph();
  void temporarily_begin_graph();

  void register_attention_runner(::xllm::kernel::dcu::AttentionRunner&& runner);

  bool is_capturing() const { return is_capturing_; }

 private:
  GlobalCaptureInstance();
  ~GlobalCaptureInstance();

  void temporarily_end_graph_locked();
  void temporarily_begin_graph_locked();

  static std::mutex capture_mutex_;

  bool is_capturing_ = false;
  DcuMempoolId graph_pool_{0, 0};

  std::unique_ptr<DcuGraphSegment> current_graph_;
  std::unique_ptr<PiecewiseGraphs> current_piecewise_graph_;

  std::unique_ptr<std::lock_guard<std::mutex>> capture_lock_;
};

}  // namespace xllm::runtime::dcu
