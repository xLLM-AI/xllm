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

#include <string>
#include <vector>

#include "core/framework/model_context.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/linear.h"
#include "core/layers/npu/rotary_embedding.h"
#include "core/layers/npu_torch/attention.h"
#include "core/layers/npu_torch/minimax_rms_norm.h"

namespace xllm {
namespace layer {

class MiniMaxM2AttentionImpl : public torch::nn::Module {
 public:
  explicit MiniMaxM2AttentionImpl(const ModelContext& context);

  torch::Tensor forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const layer::AttentionMetadata& attn_metadata,
                        KVCache& kv_cache);

  void load_state_dict(const StateDict& state_dict);

 private:
  int64_t num_heads_ = 0;
  int64_t num_kv_heads_ = 0;
  int64_t num_kv_head_replicas_ = 0;
  int64_t head_dim_ = 0;
  int64_t q_size_ = 0;
  int64_t kv_size_ = 0;
  int64_t sliding_window_ = -1;
  float scaling_ = 1.0f;
  layer::QKVParallelLinear qkv_proj_{nullptr};
  layer::RowParallelLinear o_proj_{nullptr};
  MiniMaxTensorParallelRMSNorm q_norm_tp_{nullptr};
  MiniMaxTensorParallelRMSNorm k_norm_tp_{nullptr};
  std::shared_ptr<NpuRotaryEmbedding> rotary_emb_;
  layer::Attention attn_{nullptr};
};
TORCH_MODULE(MiniMaxM2Attention);

}  // namespace layer
}  // namespace xllm
