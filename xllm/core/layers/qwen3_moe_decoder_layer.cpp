/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "qwen3_moe_decoder_layer.h"

#include <glog/logging.h>

#include "common/global_flags.h"
#include "core/framework/config/eplb_config.h"
#include "layers/common/dp_utils.h"

namespace xllm {
namespace layer {

namespace {

#if defined(USE_MLU)
bool use_moe_all2all(bool enable_deep_ep,
                     const ModelInputParams& input_params) {
  return enable_deep_ep && all_dp_ranks_are_decode(input_params);
}
#endif
bool is_moe_layer(const ModelArgs& model_args, int32_t layer_id) {
  const auto& mlp_only_layers = model_args.mlp_only_layers();
  return std::count(mlp_only_layers.begin(), mlp_only_layers.end(), layer_id) ==
             0 &&
         model_args.n_routed_experts() > 0 &&
         (layer_id + 1) % model_args.decoder_sparse_step() == 0;
}

}  // namespace

Qwen3MoeDecoderLayerImpl::Qwen3MoeDecoderLayerImpl(const ModelContext& context,
                                                   int32_t layer_id)
    : parallel_args_(context.get_parallel_args()) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& options = context.get_tensor_options();
  const bool use_moe = is_moe_layer(model_args, layer_id);

  // Qwen3 only support deep ep all2all
  //  when dp_size == ep_size && dp_size == world_size for now
  enable_deep_ep_ =
      use_moe &&
      ::xllm::EPLBConfig::get_instance().expert_parallel_degree() == 2;
  if (enable_deep_ep_) {
    CHECK_EQ(parallel_args_.dp_size(), parallel_args_.world_size())
        << "Qwen3 MoE only support deep ep all2all when dp_size == world_size";
    CHECK_EQ(parallel_args_.dp_size(), parallel_args_.ep_size())
        << "Qwen3 MoE only support deep ep all2all when dp_size == ep_size";
  }

  // Initialize attention layers
  attention_ = register_module("self_attn", Qwen2Attention(context));

  // Initialize norm layers
  input_norm_ = register_module(
      "input_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  post_norm_ = register_module(
      "post_attention_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  // Initialize mlp
  if (use_moe) {
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
                                    true,
                                    false,
                                    model_args.hidden_act(),
                                    /*enable_result_reduction=*/true,
                                    quant_args,
                                    parallel_args_.tp_group_,
                                    options,
                                    mlp_module_prefix));
  }
}

torch::Tensor Qwen3MoeDecoderLayerImpl::run_moe(
    torch::Tensor x,
    const ModelInputParams& input_params) {
#if defined(USE_MLU)
  const bool enable_moe_all2all =
      use_moe_all2all(enable_deep_ep_, input_params);
  if (need_dp_moe_gather(parallel_args_, enable_moe_all2all)) {
    x = gather_dp_tokens(x, input_params, parallel_args_);
    x = moe_mlp_->forward_experts(x, enable_moe_all2all);
    return get_dp_local_slice(x, input_params, parallel_args_);
  }
  return moe_mlp_->forward_experts(x, enable_moe_all2all);
#else
  return moe_mlp_(x, input_params);
#endif
}

void Qwen3MoeDecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
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

torch::Tensor Qwen3MoeDecoderLayerImpl::forward(
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

  // Attention
  x = attention_->forward(positions, x, attn_metadata, kv_cache);

  // Post-attention norm
  std::tie(x, residual) = post_norm_->forward(x, residual);

  // MLP forward
  if (moe_mlp_) {
    x = run_moe(x, input_params);
  } else {
    x = mlp_(x);
  }

  return x;
}

}  // namespace layer
}  // namespace xllm
