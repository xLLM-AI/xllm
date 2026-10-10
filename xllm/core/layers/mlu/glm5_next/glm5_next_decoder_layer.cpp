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

#include "layers/mlu/glm5_next/glm5_next_decoder_layer.h"

#include <glog/logging.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "core/kv_cache/storage/kv_cache_utils.h"
#include "framework/parallel_state/parallel_state.h"
#include "layers/mlu/dsa_topk_relay.h"

namespace xllm {
namespace layer {
namespace {

std::string layer_type_at(const std::vector<std::string>& types,
                          int32_t layer_id,
                          const std::string& fallback) {
  CHECK_GE(layer_id, 0);
  if (layer_id < static_cast<int32_t>(types.size())) {
    return types[static_cast<size_t>(layer_id)];
  }
  return fallback;
}

}  // namespace

Glm5NextDecoderLayerImpl::Glm5NextDecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id,
    const DsaTopkSharePlan& topk_share_plan)
    : topk_share_decision_(topk_share_plan.decision_for(layer_id)) {
  const ModelArgs& args = context.get_model_args();
  const std::string attention_type =
      layer_type_at(args.layer_types(), layer_id, "deepseek_sparse_attention");
  CHECK(attention_type == "linear_attention" ||
        attention_type == "deepseek_sparse_attention" ||
        attention_type == "attention" || attention_type == "full_attention")
      << "Unsupported GLM5-Next attention layer type at layer " << layer_id
      << ": " << attention_type;

  const std::string mlp_type = layer_type_at(
      args.mlp_layer_types(),
      layer_id,
      layer_id < args.first_k_dense_replace() ? "dense" : "sparse");
  CHECK(mlp_type == "dense" || mlp_type == "sparse")
      << "Unsupported GLM5-Next MLP layer type at layer " << layer_id << ": "
      << mlp_type;

  const QuantArgs& quant_args = context.get_quant_args();
  const ParallelArgs& parallel_args = context.get_parallel_args();
  const torch::TensorOptions& options = context.get_tensor_options();
  tp_group_ = parallel_args.tp_group_;

  const int64_t hidden_size = args.hidden_size();
  const double norm_eps = static_cast<double>(args.rms_norm_eps());
  attn_hc_pre_ = register_module("attn_hc_pre",
                                 MHCPre(args.hc_mult(),
                                        hidden_size,
                                        args.hc_sinkhorn_iters(),
                                        static_cast<double>(args.hc_eps()),
                                        norm_eps,
                                        options));
  ffn_hc_pre_ = register_module("ffn_hc_pre",
                                MHCPre(args.hc_mult(),
                                       hidden_size,
                                       args.hc_sinkhorn_iters(),
                                       static_cast<double>(args.hc_eps()),
                                       norm_eps,
                                       options));
  hc_post_ = register_module("hc_post", MHCPost(norm_eps));
  input_norm_ = register_module("input_layernorm", RMSNorm(context));
  post_norm_ = register_module("post_attention_layernorm", RMSNorm(context));

  if (is_linear_attention_layer(
          layer_id, /*full_attention_interval=*/1, args.layer_types())) {
    kda_ = register_module("self_attn", Glm5NextKDA(context));
  } else {
    dsa_ = register_module(
        "self_attn",
        DeepseekV2Attention(
            context,
            /*enable_indexer=*/!topk_share_decision_.reuse_topk));
  }

  if (mlp_type == "dense") {
    dense_mlp_ =
        register_module("mlp",
                        DenseMLP(hidden_size,
                                 args.intermediate_size(),
                                 /*is_gated=*/true,
                                 /*has_bias=*/false,
                                 args.hidden_act(),
                                 /*enable_result_reduction=*/true,
                                 quant_args,
                                 parallel_args.tp_group_,
                                 options,
                                 /*module_prefix=*/"",
                                 static_cast<double>(args.swiglu_limit())));
  } else {
    sparse_moe_ = register_module(
        "mlp", DeepseekV4SparseMoEBlock(context, /*use_hash=*/false));
  }
}

void Glm5NextDecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  CHECK(weight_prefix_.empty() || weight_prefix_ == state_dict.prefix())
      << "GLM5-Next layer checkpoint prefix changed between shards";
  weight_prefix_ = state_dict.prefix();
  const StateDict attention_state = state_dict.get_dict_with_prefix(
      std::vector<std::string>{"self_attn.", "attn."});
  if (kda_) {
    kda_->load_state_dict(attention_state);
  } else {
    dsa_->load_state_dict(attention_state);
  }
  input_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("input_layernorm."));
  post_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("post_attention_layernorm."));
  const StateDict mlp_state = state_dict.get_dict_with_prefix("mlp.");
  if (dense_mlp_) {
    dense_mlp_->load_state_dict(mlp_state);
  } else if (mlp_state.size() > 0) {
    sparse_moe_->load_state_dict(mlp_state);
  }
  attn_hc_pre_->load_state_dict(
      get_hc_state(state_dict, "attn_hc_pre.", "hc_attn_"));
  ffn_hc_pre_->load_state_dict(
      get_hc_state(state_dict, "ffn_hc_pre.", "hc_ffn_"));
}

void Glm5NextDecoderLayerImpl::verify_loaded_weights() const {
  if (kda_) {
    kda_->verify_loaded_weights(weight_prefix_ + "self_attn.");
  } else {
    dsa_->verify_loaded_weights();
  }
  input_norm_->verify_loaded_weights(weight_prefix_ + "input_layernorm.");
  post_norm_->verify_loaded_weights(weight_prefix_ +
                                    "post_attention_layernorm.");
  if (dense_mlp_) {
    dense_mlp_->verify_loaded_weights(weight_prefix_ + "mlp.");
  } else {
    sparse_moe_->verify_loaded_weights();
  }
  attn_hc_pre_->verify_loaded_weights(weight_prefix_ + "attn_hc_pre.");
  ffn_hc_pre_->verify_loaded_weights(weight_prefix_ + "ffn_hc_pre.");
}

torch::Tensor Glm5NextDecoderLayerImpl::forward(
    torch::Tensor& hidden_states,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    DsaTopkRelay* topk_relay) {
  residual = std::nullopt;

  const torch::Tensor residual_attention = hidden_states;
  MHCPreOutput attention_hc = attn_hc_pre_->forward(hidden_states);
  torch::Tensor attention_input =
      std::get<0>(input_norm_->forward(attention_hc.output));

  torch::Tensor attention_output;
  if (kda_) {
    attention_output =
        kda_->forward(attention_input, attn_metadata, kv_cache, input_params);
  } else {
    std::optional<DsaTopkTransfer> topk_transfer;
    if (topk_relay != nullptr && !attn_metadata.is_dummy) {
      topk_transfer = topk_relay->prepare_layer(topk_share_decision_);
    }
    attention_output =
        dsa_->forward(
                positions,
                attention_input,
                attn_metadata,
                kv_cache,
                /*sp_ctx=*/nullptr,
                topk_transfer.has_value() ? &topk_transfer.value() : nullptr)
            .output;
    if (topk_transfer.has_value()) {
      topk_relay->finish_layer(topk_share_decision_, topk_transfer.value());
    }
    if (tp_group_ != nullptr && tp_group_->world_size() > 1) {
      attention_output = parallel_state::reduce(attention_output, tp_group_);
    }
  }

  std::tie(hidden_states, std::ignore) = hc_post_->forward(attention_output,
                                                           residual_attention,
                                                           attention_hc.post,
                                                           attention_hc.comb);

  const torch::Tensor residual_ffn = hidden_states;
  MHCPreOutput ffn_hc = ffn_hc_pre_->forward(hidden_states);
  torch::Tensor ffn_input = std::get<0>(post_norm_->forward(ffn_hc.output));
  torch::Tensor ffn_output;
  if (dense_mlp_) {
    ffn_output = dense_mlp_->forward(ffn_input);
  } else {
    FusedMoEImpl::RouteInfo route = sparse_moe_->prep_route(ffn_input);
    ffn_output = sparse_moe_->forward_selected(
        ffn_input, route.reduce_weight, route.expert_id, input_params);
  }
  std::tie(hidden_states, std::ignore) =
      hc_post_->forward(ffn_output, residual_ffn, ffn_hc.post, ffn_hc.comb);
  return hidden_states;
}

}  // namespace layer
}  // namespace xllm
