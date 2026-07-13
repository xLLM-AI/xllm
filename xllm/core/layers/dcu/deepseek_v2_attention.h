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

// DCU DeepSeek-V2 MLA attention. Prefill expands MLA to MHA with
// FlashAttention varlen; decode uses FlashMLA.

#pragma once

#include <torch/torch.h>

#include <memory>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm.h"
#include "layers/common/rotary_embedding.h"

namespace xllm {
namespace layer {

class DeepseekV2AttentionImpl final : public torch::nn::Module {
 public:
  DeepseekV2AttentionImpl(const ModelArgs& args,
                          const QuantArgs& quant_args,
                          const ParallelArgs& parallel_args,
                          const torch::TensorOptions& options);

  // positions:      [num_tokens] token positions
  // hidden_states:  [num_tokens, hidden_size]
  // Returns attention output [num_tokens, hidden_size].
  torch::Tensor forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache);

  void load_state_dict(const StateDict& state_dict);

 private:
  torch::Tensor prepare_query(const torch::Tensor& hidden_states);

  void store_latent_cache(const torch::Tensor& latent_cache,
                          const torch::Tensor& slot_mapping,
                          const torch::Tensor& k_cache);

  torch::Tensor decode_flash_mla(const torch::Tensor& q_nope_absorbed,
                                 const torch::Tensor& q_pe,
                                 const AttentionMetadata& attn_metadata,
                                 KVCache& kv_cache);

  torch::Tensor prefill_mha(const torch::Tensor& q_nope,
                            const torch::Tensor& q_pe,
                            const torch::Tensor& c_kv_normed,
                            const torch::Tensor& k_pe,
                            const AttentionMetadata& attn_metadata);

  torch::Tensor prefill_mha_with_full_context(
      const torch::Tensor& q_nope,
      const torch::Tensor& q_pe,
      const AttentionMetadata& attn_metadata,
      const torch::Tensor& k_cache);

  torch::Tensor project_output(const torch::Tensor& attn_latent);
  void refresh_kv_b_proj_weights();

  int64_t q_lora_rank_;
  int64_t kv_lora_rank_;
  int64_t qk_nope_head_dim_;
  int64_t qk_rope_head_dim_;
  int64_t qk_head_dim_;
  int64_t v_head_dim_;
  int64_t tp_heads_;
  double eps_;
  float softmax_scale_;
  bool interleaved_;
  torch::ScalarType runtime_dtype_;

  ReplicatedLinear q_a_proj_{nullptr};
  RMSNorm q_a_layernorm_{nullptr};
  ColumnParallelLinear q_b_proj_{nullptr};
  ColumnParallelLinear q_proj_{nullptr};  // used when q_lora_rank_ == 0

  ReplicatedLinear kv_a_proj_with_mqa_{nullptr};
  RMSNorm kv_a_layernorm_{nullptr};
  ColumnParallelLinear kv_b_proj_{nullptr};
  RowParallelLinear o_proj_{nullptr};

  torch::Tensor w_kc_;  // [tp_heads, qk_nope, kv_lora]
  torch::Tensor w_vc_;  // [tp_heads, kv_lora, v_head_dim]
  bool has_trans_ = false;

  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;
};

TORCH_MODULE(DeepseekV2Attention);

}  // namespace layer
}  // namespace xllm
