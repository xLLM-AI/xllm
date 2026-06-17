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

#include "minimax_m2_decode_layer.h"

namespace xllm {
namespace layer {

MiniMaxM2DecoderLayerImpl::MiniMaxM2DecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id) {
  const auto& model_args = context.get_model_args();
  auto quant_args = context.get_quant_args();
  if (quant_args.quant_method() == "fp8") {
    quant_args.quant_method("");
  }
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();

  attention_ = register_module("self_attn", MiniMaxM2Attention(context));
  input_norm_ = register_module(
      "input_layernorm",
      layer::RMSNorm(
          model_args.hidden_size(), model_args.rms_norm_eps(), options));
  post_norm_ = register_module(
      "post_attention_layernorm",
      layer::RMSNorm(
          model_args.hidden_size(), model_args.rms_norm_eps(), options));

  layer::FusedMoEArgs moe_args;
  moe_ = register_module(
      "mlp",
      layer::FusedMoE(
          model_args, moe_args, quant_args, parallel_args, options));
}

torch::Tensor MiniMaxM2DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const layer::AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  if (x.numel() == 0) {
    return moe_->forward(x, input_params);
  }

  if (!residual.has_value()) {
    residual = x;
    x = std::get<0>(input_norm_->forward(x));
  } else {
    std::tie(x, residual) = input_norm_->forward(x, residual);
  }

  x = attention_->forward(positions, x, attn_metadata, kv_cache);
  std::tie(x, residual) = post_norm_->forward(x, residual);
  x = moe_->forward(x, input_params);
  return x;
}

void MiniMaxM2DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  attention_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
  input_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("input_layernorm."));
  post_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("post_attention_layernorm."));
  moe_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
}

}  // namespace layer
}  // namespace xllm
