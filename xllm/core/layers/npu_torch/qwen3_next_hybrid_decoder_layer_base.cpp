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

#include "qwen3_next_hybrid_decoder_layer_base.h"

#include <algorithm>
#include <optional>
#include <tuple>

#include "common/flash_comm1_context.h"

namespace xllm {
namespace layer {

Qwen3HybridDecoderLayerImplBase::Qwen3HybridDecoderLayerImplBase(
    const ModelContext& context,
    int32_t layer_id,
    std::shared_ptr<Qwen3GatedDeltaNetBaseImpl> linear_attention_module) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();
  const bool use_full_attention = is_full_attention_layer(model_args, layer_id);

  // Initialize attention layers
  if (use_full_attention) {
    attention_ = register_module(
        "self_attn",
        Qwen3NextAttention(
            model_args, quant_args, parallel_args, options, layer_id));
  } else {
    linear_attention_ =
        register_module("linear_attn", std::move(linear_attention_module));
  }

  // Initialize norm layers
  input_norm_ = register_module(
      "input_layernorm",
      Qwen3NextRMSNorm(
          model_args.hidden_size(), model_args.rms_norm_eps(), options));

  post_norm_ = register_module(
      "post_attention_layernorm",
      Qwen3NextRMSNorm(
          model_args.hidden_size(), model_args.rms_norm_eps(), options));

  // Initialize mlp
  auto mlp_only_layers = model_args.mlp_only_layers();
  if ((std::count(mlp_only_layers.begin(), mlp_only_layers.end(), layer_id) ==
       0) &&
      model_args.n_routed_experts() > 0 &&
      (layer_id + 1) % model_args.decoder_sparse_step() == 0) {
    moe_mlp_ = register_module("mlp",
                               FusedMoE(model_args,
                                        FusedMoEArgs{.is_gated = true},
                                        quant_args,
                                        parallel_args,
                                        options));
  } else {
    mlp_ = register_module("mlp",
                           DenseMLP(model_args.hidden_size(),
                                    model_args.intermediate_size(),
                                    true,
                                    false,
                                    model_args.hidden_act(),
                                    /*enable_result_reduction=*/true,
                                    quant_args,
                                    parallel_args.tp_group_,
                                    options));
  }
}

void Qwen3HybridDecoderLayerImplBase::load_state_dict(
    const StateDict& state_dict) {
  if (attention_) {
    attention_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
  } else {
    linear_attention_->load_state_dict(
        state_dict.get_dict_with_prefix("linear_attn."));
  }
  input_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("input_layernorm."));
  post_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("post_attention_layernorm."));
  if (moe_mlp_) {
    moe_mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  } else {
    mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  }
}

void Qwen3HybridDecoderLayerImplBase::verify_loaded_weights(
    const std::string& prefix) const {
  if (linear_attention_) {
    linear_attention_->verify_loaded_weights(prefix + "linear_attn.");
  }
}

torch::Tensor Qwen3HybridDecoderLayerImplBase::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    const torch::Tensor& mrope_cos_sin) {
  const FlashComm1Context* fc1_ctx = get_current_flash_comm1_context();
  // Pre-attention norm
  if (!residual.has_value()) {
    residual = x;
    x = std::get<0>(input_norm_->forward(x));
  } else {
    if (fc1_ctx && is_sequence_sharded(*fc1_ctx) &&
        residual.value().size(0) != x.size(0)) {
      residual = maybe_shard_residual(residual.value(), *fc1_ctx);
    }
    if (fc1_ctx && is_sequence_sharded(*fc1_ctx)) {
      CHECK_EQ(residual.value().size(0), x.size(0))
          << "FC1 input residual and hidden states must share the same "
          << "padded local sequence layout.";
    }
    std::tie(x, residual) = input_norm_->forward(x, residual);
  }

  // Attention
  if (attention_) {
    x = attention_->forward(
        positions, x, attn_metadata, kv_cache, mrope_cos_sin);
  } else {
    x = linear_attention_->forward(x, attn_metadata, kv_cache, input_params);
  }

  // Post-attention norm
  // Ensure the residual layout matches the attention output before post_norm.
  if (fc1_ctx && is_sequence_sharded(*fc1_ctx) && residual.has_value() &&
      residual.value().size(0) != x.size(0)) {
    residual = maybe_shard_residual(residual.value(), *fc1_ctx);
    CHECK_EQ(residual.value().size(0), x.size(0))
        << "FC1 post-attention residual and hidden states must share the same "
        << "padded local sequence layout.";
  }

  std::tie(x, residual) = post_norm_->forward(x, residual);

  // MLP forward
  if (moe_mlp_) {
    x = moe_mlp_(x, input_params);
  } else {
    x = mlp_(x);
  }

  return x;
}

torch::Tensor Qwen3HybridDecoderLayerImplBase::build_mrope_cos_sin(
    const torch::Tensor& positions) const {
  if (attention_) {
    return attention_->build_mrope_cos_sin(positions);
  }
  return {};
}

}  // namespace layer
}  // namespace xllm
