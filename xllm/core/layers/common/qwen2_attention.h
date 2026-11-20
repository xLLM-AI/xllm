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

#include "attention.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quantization/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "linear.h"
#include "rotary_embedding.h"

namespace xllm {
namespace layer {

class Qwen2AttentionImpl : public torch::nn::Module {
 public:
  Qwen2AttentionImpl(const ModelContext& context);

  torch::Tensor forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache);

  virtual void load_state_dict(const StateDict& state_dict);

  virtual void verify_loaded_weights(const std::string& prefix) const;

  // Get FP8 input scale from qkv_proj for fused RMSNorm+FP8 quantization
  std::optional<torch::Tensor> get_fp8_input_scale() const;

 protected:
  virtual void apply_qk_transform(torch::Tensor& qkv,
                                  torch::Tensor& q,
                                  torch::Tensor& k,
                                  const torch::Tensor& positions,
                                  const AttentionMetadata& attn_metadata);

  int64_t num_heads_;
  int64_t num_kv_heads_;
  int64_t head_dim_;
  MRotaryEmbedding rotary_emb_{nullptr};

 private:
  int64_t q_size_;
  int64_t kv_size_;
  // The fused QKV loader only exposes an aggregate loaded flag. Preserve the
  // logical checkpoint names across shards so verification can identify the
  // missing projection precisely.
  bool q_proj_weight_seen_ = false;
  bool k_proj_weight_seen_ = false;
  bool v_proj_weight_seen_ = false;

  QKVParallelLinear qkv_proj_{nullptr};
  RowParallelLinear o_proj_{nullptr};
  Attention attn_{nullptr};
};
TORCH_MODULE(Qwen2Attention);

}  // namespace layer
}  // namespace xllm
