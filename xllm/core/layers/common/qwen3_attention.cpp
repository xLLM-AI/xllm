/* Copyright 2026 The xLLM Authors.

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

#include "core/layers/common/qwen3_attention.h"

#include <tuple>

#if defined(USE_CUDA) || defined(USE_DCU)
#include "core/kernels/cuda/cuda_ops_api.h"
#endif

namespace xllm {
namespace layer {

Qwen3AttentionImpl::Qwen3AttentionImpl(const ModelContext& context)
    : Qwen2AttentionImpl(context) {
  const auto& args = context.get_model_args();
  const auto& options = context.get_tensor_options();
  q_norm_ = register_module(
      /*name=*/"q_norm", RMSNorm(head_dim_, args.rms_norm_eps(), options));
  k_norm_ = register_module(
      /*name=*/"k_norm", RMSNorm(head_dim_, args.rms_norm_eps(), options));
#if defined(USE_CUDA) || defined(USE_DCU)
  can_use_fused_qk_norm_rope_ =
      args.rope_scaling_mrope_section().empty() &&
      (head_dim_ == 64 || head_dim_ == 128 || head_dim_ == 256);
#endif
}

void Qwen3AttentionImpl::load_state_dict(const StateDict& state_dict) {
  Qwen2AttentionImpl::load_state_dict(state_dict);
  q_norm_->load_state_dict(
      state_dict.get_dict_with_prefix(/*prefix=*/"q_norm."));
  k_norm_->load_state_dict(
      state_dict.get_dict_with_prefix(/*prefix=*/"k_norm."));
}

void Qwen3AttentionImpl::verify_loaded_weights(
    const std::string& prefix) const {
  Qwen2AttentionImpl::verify_loaded_weights(prefix);
  q_norm_->verify_loaded_weights(prefix + "q_norm.");
  k_norm_->verify_loaded_weights(prefix + "k_norm.");
}

torch::Tensor Qwen3AttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  auto qkv = qkv_proj_->forward(hidden_states);
  auto [q, k, v] = split_qkv(qkv);
#if defined(USE_CUDA) || defined(USE_DCU)
  if (can_use_fused_qk_norm_rope_ && positions.dim() == 1) {
    xllm::kernel::cuda::fused_qk_norm_rope(
        qkv,
        num_heads_,
        num_kv_heads_,
        num_kv_heads_,
        head_dim_,
        q_norm_->eps(),
        q_norm_->weight(),
        k_norm_->weight(),
        rotary_emb_->precomputed_cos_sin_cache(),
        /*interleaved=*/false,
        positions.to(torch::kInt64).contiguous());
    return forward_attention(attn_metadata, q, k, v, kv_cache);
  }
#endif
  q = std::get<0>(q_norm_->forward(q));
  k = std::get<0>(k_norm_->forward(k));
  rotary_emb_->forward(q, k, positions, attn_metadata);
  return forward_attention(attn_metadata, q, k, v, kv_cache);
}

}  // namespace layer
}  // namespace xllm
