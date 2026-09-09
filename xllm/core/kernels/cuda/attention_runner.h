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

#include <glog/logging.h>
#include <torch/torch.h>

#include <optional>
#include <string>

#include "core/kernels/cuda/cuda_ops_api.h"
#include "core/kernels/param.h"

namespace xllm::kernel::cuda {

// Attention replay parameters shared across all layers in a prefill batch
struct AttentionReplayParams {
  ffi::Array<int64_t> plan_info;
  torch::Tensor q_cu_seq_lens;
  torch::Tensor kv_cu_seq_lens;
  uint32_t actual_num_tokens;  // All layers share the same actual_num_tokens
};

// AttentionRunner encapsulates batch_prefill for piecewise CUDA Graph
class AttentionRunner {
 public:
  AttentionRunner() = default;

  // Piecewise mode: capture phase
  void run_capture(const std::string& uri,
                   ffi::Array<int64_t> plan_info,
                   torch::Tensor float_workspace_buffer,
                   torch::Tensor int_workspace_buffer,
                   torch::Tensor page_locked_int_workspace_buffer,
                   torch::Tensor query,  // shape: [padded_num_tokens, ...]
                   torch::Tensor key,
                   torch::Tensor value,
                   torch::Tensor q_cu_seq_lens,
                   torch::Tensor kv_cu_seq_lens,
                   int64_t window_left,
                   double sm_scale,
                   torch::Tensor output,  // shape: [padded_num_tokens, ...]
                   std::optional<torch::Tensor>& output_lse,
                   uint32_t padded_num_tokens);

  // Piecewise mode: replay phase
  void run_replay(const AttentionReplayParams& params);

 private:
  // Captured flashiner workspace buffers
  torch::Tensor float_workspace_buffer_;
  torch::Tensor int_workspace_buffer_;
  torch::Tensor page_locked_int_workspace_buffer_;

  // Captured tensors (padded shape)
  torch::Tensor query_;
  torch::Tensor key_;
  torch::Tensor value_;
  torch::Tensor output_;

  // Captured parameters
  std::string uri_;
  int64_t window_size_left_;
  double scale_;
  uint32_t padded_num_tokens_;
};

}  // namespace xllm::kernel::cuda
