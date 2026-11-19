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

#include "core/layers/qwen2_decoder_layer.h"

#include "core/layers/common/qwen3_attention.h"

namespace xllm {
namespace layer {

template <typename AttentionType>
QwenDecoderLayerImpl<AttentionType>::QwenDecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id)
    : parallel_args_(context.get_parallel_args()) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& options = context.get_tensor_options();
  const std::string mlp_module_prefix =
      layer_id >= 0 ? "model.layers." + std::to_string(layer_id) + ".mlp" : "";

  attention_ = register_module(/*name=*/"self_attn", AttentionType(context));

  // Initialize norm layers
  input_norm_ = register_module(
      "input_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  post_norm_ = register_module(
      "post_attention_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  // Initialize mlp
  mlp_ = register_module("mlp",
                         DenseMLP(model_args.hidden_size(),
                                  model_args.intermediate_size(),
                                  true,
                                  false,
                                  model_args.hidden_act(),
                                  /*enable_result_reduction=*/true,
                                  quant_args,
                                  parallel_args_.tp_group_,
                                  options,
                                  mlp_module_prefix));

  if (model_args.model_type() == "qwen3_dflash2") {
    attention_conv_ = register_module(
        "attention_conv",
        DFlash2GroupedConv(model_args.hidden_size(),
                           model_args.dflash2_conv_kernel_size(),
                           model_args.dflash2_conv_group_size(),
                           context.num_speculative_tokens(),
                           options));
    mlp_conv_ = register_module(
        "mlp_conv",
        DFlash2GroupedConv(model_args.hidden_size(),
                           model_args.dflash2_conv_kernel_size(),
                           model_args.dflash2_conv_group_size(),
                           context.num_speculative_tokens(),
                           options));
  }
}

template <typename AttentionType>
void QwenDecoderLayerImpl<AttentionType>::load_state_dict(
    const StateDict& state_dict) {
  attention_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
  input_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("input_layernorm."));
  post_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("post_attention_layernorm."));
  mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  if (attention_conv_) {
    attention_conv_->load_state_dict(
        state_dict.get_dict_with_prefix("attention_conv."));
    mlp_conv_->load_state_dict(state_dict.get_dict_with_prefix("mlp_conv."));
  }
}

template <typename AttentionType>
void QwenDecoderLayerImpl<AttentionType>::verify_loaded_weights(
    const std::string& prefix) const {
  attention_->verify_loaded_weights(prefix + "self_attn.");
  mlp_->verify_loaded_weights(prefix + "mlp.");
  input_norm_->verify_loaded_weights(prefix + "input_layernorm.");
  post_norm_->verify_loaded_weights(prefix + "post_attention_layernorm.");
  if (attention_conv_) {
    attention_conv_->verify_loaded_weights(prefix + "attention_conv.");
    mlp_conv_->verify_loaded_weights(prefix + "mlp_conv.");
  }
}

template <typename AttentionType>
std::tuple<torch::Tensor, std::optional<torch::Tensor>>
QwenDecoderLayerImpl<AttentionType>::apply_norm(
    RMSNorm& norm,
    torch::Tensor& input,
    std::optional<torch::Tensor>& residual,
    const std::optional<torch::Tensor>& fp8_scale) {
  const bool use_fp8_fusion = fp8_scale.has_value();

  if (!residual.has_value()) {
    // First layer: initialize residual from input
    auto new_residual = input;
    auto output = use_fp8_fusion
                      ? std::get<0>(norm->forward_fp8(input, fp8_scale.value()))
                      : std::get<0>(norm->forward(input));
    return {output, new_residual};
  }

  // Subsequent layers: fused add + norm
  return use_fp8_fusion ? norm->forward_fp8(input, fp8_scale.value(), residual)
                        : norm->forward(input, residual);
}

template <typename AttentionType>
torch::Tensor QwenDecoderLayerImpl<AttentionType>::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  auto pre_fp8_scale = attention_->get_fp8_input_scale();
  auto post_fp8_scale = mlp_->get_fp8_input_scale();

  // Pre-attention norm
  std::tie(x, residual) = apply_norm(input_norm_, x, residual, pre_fp8_scale);

  torch::Tensor attention_coefficients;
  if (attention_conv_) {
    std::tie(x, attention_coefficients) = attention_conv_->prepare(x);
  }

  // Attention
  x = attention_->forward(positions, x, attn_metadata, kv_cache);
  if (attention_conv_) {
    x = attention_conv_->finish(x, attention_coefficients);
  }

  // Post-attention norm
  std::tie(x, residual) = apply_norm(post_norm_, x, residual, post_fp8_scale);

  torch::Tensor mlp_coefficients;
  if (mlp_conv_) {
    std::tie(x, mlp_coefficients) = mlp_conv_->prepare(x);
  }

  // MLP
  x = mlp_->forward(x);
  if (mlp_conv_) {
    x = mlp_conv_->finish(x, mlp_coefficients);
  }

  return x;
}

template class QwenDecoderLayerImpl<Qwen2Attention>;
template class QwenDecoderLayerImpl<Qwen3Attention>;

}  // namespace layer
}  // namespace xllm
