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

#include <torch/torch.h>

#include <memory>
#include <optional>
#include <string>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/dense_mlp.h"
#include "layers/common/linear.h"
#include "layers/common/qwen3_next_rms_norm.h"
#include "layers/common/rotary_embedding.h"
#include "layers/dcu/attention.h"
#include "layers/dcu/fused_moe.h"
#include "layers/dcu/qwen3_5_gated_delta_net.h"

namespace xllm {
namespace layer {

class Qwen3_5DecoderLayerImpl final : public torch::nn::Module {
 public:
  Qwen3_5DecoderLayerImpl(const ModelContext& context, int32_t layer_id);

  void load_state_dict(const StateDict& state_dict);

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 private:
  // Full attention members (only initialized for full_attention layers)
  QKVParallelLinear qkv_proj_{nullptr};
  RowParallelLinear o_proj_{nullptr};
  Qwen3NextRMSNorm q_norm_{nullptr};
  Qwen3NextRMSNorm k_norm_{nullptr};
  Attention attn_{nullptr};
  MRotaryEmbedding rotary_emb_{nullptr};

  // Linear attention (only initialized for linear_attention layers)
  Qwen3_5GatedDeltaNet linear_attn_{nullptr};

  // Shared across both paths
  DenseMLP mlp_{nullptr};
  FusedMoE moe_mlp_{nullptr};
  Qwen3NextRMSNorm input_norm_{nullptr};
  Qwen3NextRMSNorm post_norm_{nullptr};

  bool is_full_attention_ = true;
  int64_t num_heads_ = 0;
  int64_t num_kv_heads_ = 0;
  int64_t num_kv_head_replicas_ = 0;
  int64_t head_dim_ = 0;
  int64_t q_size_ = 0;
  int64_t kv_size_ = 0;
  float scaling_ = 0.0f;
  bool attn_output_gate_ = false;
  ParallelArgs parallel_args_;
};

TORCH_MODULE(Qwen3_5DecoderLayer);

}  // namespace layer
}  // namespace xllm
