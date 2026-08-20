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
#include <torch/torch.h>

namespace xllm {
namespace layer {

class AttentionMask : public torch::nn::Module {
 public:
  AttentionMask() = default;

  explicit AttentionMask(at::Device device,
                         torch::Dtype dtype,
                         float mask_value = -9984);

  torch::Tensor get_decode_attn_mask(torch::Tensor input_lengths,
                                     int64_t max_s,
                                     torch::Dtype dtype,
                                     torch::Device device);

  torch::Tensor get_attn_mask(int64_t max_s,
                              torch::Dtype dtype,
                              torch::Device device);

  torch::Tensor gen_free_mask(int32_t q_len,
                              torch::Dtype dtype,
                              torch::Device device);

  torch::Tensor gen_append_mask(int32_t q_len,
                                int32_t kv_len,
                                int32_t max_kv_len,
                                torch::Dtype dtype,
                                torch::Device device);

 private:
  void update_attn_cache(torch::Dtype dtype,
                         torch::Device device,
                         int64_t seqlen);

  int seq_len_cached_;
  float mask_value_;
  at::Tensor atten_mask_cache_;
};

}  // namespace layer
}  // namespace xllm
