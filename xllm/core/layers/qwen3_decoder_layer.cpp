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

#include "core/layers/qwen3_decoder_layer.h"

#include <tuple>

namespace xllm {
namespace layer {

Qwen3DecoderLayerImpl::Qwen3DecoderLayerImpl(const ModelContext& context,
                                             int32_t layer_id) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& options = context.get_tensor_options();
  const std::string mlp_module_prefix =
      layer_id >= 0 ? "model.layers." + std::to_string(layer_id) + ".mlp" : "";

  input_norm_ = register_module(
      /*name=*/"input_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  post_norm_ = register_module(
      /*name=*/"post_attention_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  attention_ = register_module(/*name=*/"self_attn", Qwen3Attention(context));
  mlp_ = register_module(/*name=*/"mlp",
                         DenseMLP(model_args.hidden_size(),
                                  model_args.intermediate_size(),
                                  /*is_gated=*/true,
                                  /*has_bias=*/false,
                                  model_args.hidden_act(),
                                  /*enable_result_reduction=*/true,
                                  quant_args,
                                  context.get_parallel_args().tp_group_,
                                  options,
                                  mlp_module_prefix));
}

void Qwen3DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  attention_->load_state_dict(
      state_dict.get_dict_with_prefix(/*prefix=*/"self_attn."));
  input_norm_->load_state_dict(
      state_dict.get_dict_with_prefix(/*prefix=*/"input_layernorm."));
  post_norm_->load_state_dict(
      state_dict.get_dict_with_prefix(/*prefix=*/"post_attention_layernorm."));
  mlp_->load_state_dict(state_dict.get_dict_with_prefix(/*prefix=*/"mlp."));
}

void Qwen3DecoderLayerImpl::verify_loaded_weights(
    const std::string& prefix) const {
  attention_->verify_loaded_weights(prefix + "self_attn.");
  mlp_->verify_loaded_weights(prefix + "mlp.");
  input_norm_->verify_loaded_weights(prefix + "input_layernorm.");
  post_norm_->verify_loaded_weights(prefix + "post_attention_layernorm.");
}

torch::Tensor Qwen3DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  const auto attention_scale = attention_->get_fp8_input_scale();
  if (residual.has_value()) {
    std::tie(x, residual) = input_norm_->forward(
        x, residual, /*inplace_output=*/std::nullopt, attention_scale);
  } else {
    residual = x;
    x = std::get<0>(input_norm_->forward(x,
                                         /*residual=*/std::nullopt,
                                         /*inplace_output=*/std::nullopt,
                                         attention_scale));
  }
  x = attention_->forward(positions, x, attn_metadata, kv_cache);
  std::tie(x, residual) = post_norm_->forward(x,
                                              residual,
                                              /*inplace_output=*/std::nullopt,
                                              mlp_->get_fp8_input_scale());
  x = mlp_->forward(x);
  return x;
}

}  // namespace layer
}  // namespace xllm
