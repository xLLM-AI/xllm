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

#include <torch/torch.h>

#include "common/attention.h"
#include "common/linear.h"
#include "common/rotary_embedding.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "minimax_m2_tensor_parallel_rms_norm.h"

namespace xllm {
namespace layer {

class MiniMaxM2AttentionImpl : public torch::nn::Module {
 public:
  MiniMaxM2AttentionImpl() = default;
  explicit MiniMaxM2AttentionImpl(const ModelContext& context);

  torch::Tensor forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache);

  void load_state_dict(const StateDict& state_dict);

 private:
  int64_t num_heads_ = 0;
  int64_t num_kv_heads_ = 0;
  int64_t num_kv_head_replicas_ = 0;
  int64_t head_dim_ = 0;
  int64_t q_size_ = 0;
  int64_t kv_size_ = 0;
  float scaling_ = 1.0f;

  QKVParallelLinear qkv_proj_{nullptr};
  RowParallelLinear o_proj_{nullptr};
  MiniMaxM2TensorParallelRMSNorm q_norm_tp_{nullptr};
  MiniMaxM2TensorParallelRMSNorm k_norm_tp_{nullptr};
  RotaryEmbedding rotary_emb_{nullptr};
  Attention attn_{nullptr};
};
TORCH_MODULE(MiniMaxM2Attention);

}  // namespace layer
}  // namespace xllm
