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

#include "layers/dcu/deepseek_v2_decoder_layer_impl.h"

#include <glog/logging.h>

#include <string>

#include "framework/parallel_state/parallel_state.h"

namespace xllm {
namespace layer {

namespace {

// DeepSeek: layers [0, first_k_dense_replace) are dense MLP; the rest are MoE.
bool is_moe_layer(const ModelArgs& model_args, int32_t layer_id) {
  return model_args.n_routed_experts() > 0 &&
         layer_id >= model_args.first_k_dense_replace();
}

}  // namespace

DeepseekV2DecoderLayerImpl::DeepseekV2DecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id)
    : parallel_args_(context.get_parallel_args()),
      is_moe_layer_(is_moe_layer(context.get_model_args(), layer_id)) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& options = context.get_tensor_options();

  attention_ = register_module(
      "self_attn",
      DeepseekV2Attention(model_args, quant_args, parallel_args_, options));

  input_norm_ = register_module(
      "input_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));
  post_norm_ = register_module(
      "post_attention_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  if (is_moe_layer_) {
    moe_mlp_ = register_module("mlp",
                               FusedMoE(model_args,
                                        FusedMoEArgs{.is_gated = true},
                                        quant_args,
                                        parallel_args_,
                                        options));
  } else {
    const std::string mlp_module_prefix =
        "model.layers." + std::to_string(layer_id) + ".mlp";
    mlp_ = register_module("mlp",
                           DenseMLP(model_args.hidden_size(),
                                    model_args.intermediate_size(),
                                    /*is_gated=*/true,
                                    /*has_bias=*/false,
                                    model_args.hidden_act(),
                                    /*enable_result_reduction=*/true,
                                    quant_args,
                                    parallel_args_.tp_group_,
                                    options,
                                    mlp_module_prefix));
  }
}

void DeepseekV2DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  attention_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
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

torch::Tensor DeepseekV2DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  // Pre-attention norm
  if (!residual.has_value()) {
    residual = x;
    x = std::get<0>(input_norm_->forward(x));
  } else {
    std::tie(x, residual) = input_norm_->forward(x, residual);
  }

  // MLA attention
  x = attention_->forward(positions, x, attn_metadata, kv_cache);
  if (parallel_args_.tp_group_->world_size() > 1) {
    x = parallel_state::reduce(x, parallel_args_.tp_group_);
  }

  // Post-attention norm
  std::tie(x, residual) = post_norm_->forward(x, residual);

  // MLP / MoE
  if (moe_mlp_) {
    x = moe_mlp_(x, input_params);
  } else {
    x = mlp_(x);
  }

  return x;
}

}  // namespace layer
}  // namespace xllm
