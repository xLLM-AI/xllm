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

#pragma once

#include <torch/torch.h>

#include <vector>

#include "framework/model/model_args.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "linear.h"

namespace xllm {
namespace layer {

class Qwen2VisionAttentionImpl : public torch::nn::Module {
 public:
  Qwen2VisionAttentionImpl() = default;
  Qwen2VisionAttentionImpl(const ModelContext& context, bool has_bias = true);

  virtual torch::Tensor forward(torch::Tensor& hidden_states,
                                torch::Tensor& m_cos_pos,
                                torch::Tensor& m_sin_pos,
                                torch::Tensor& cu_seq_len,
                                std::vector<int32_t>& cu_seq_len_vec);

  void load_state_dict(const StateDict& state_dict);

 protected:
  std::vector<torch::Tensor> split_qkv(const torch::Tensor& qkv);
  static int32_t get_max_sequence_length(
      const std::vector<int32_t>& cu_seq_len_vec);

  int64_t hidden_size_per_attention_head_;
  int64_t num_attention_heads_per_partition_;
  float scale_;

  ProcessGroup* tp_group_;

  QKVParallelLinear qkv_proj_{nullptr};
  RowParallelLinear proj_{nullptr};
};
TORCH_MODULE(Qwen2VisionAttention);

}  // namespace layer
}  // namespace xllm
