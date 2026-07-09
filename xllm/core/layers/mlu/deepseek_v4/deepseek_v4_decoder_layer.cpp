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

#include "layers/mlu/deepseek_v4/deepseek_v4_decoder_layer.h"

#include <glog/logging.h>

#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace xllm {
namespace layer {
namespace {

StateDict get_alias_dict(const StateDict& state_dict,
                         const std::vector<std::string>& prefixes) {
  return state_dict.get_dict_with_prefix(prefixes);
}

StateDict get_hc_state(const StateDict& state_dict,
                       const std::string& module_prefix,
                       const std::string& legacy_prefix) {
  StateDict module_state = state_dict.get_dict_with_prefix(module_prefix);
  if (module_state.size() > 0) {
    return module_state;
  }

  std::unordered_map<std::string, torch::Tensor> tensors;
  torch::Tensor fn = state_dict.get_tensor(legacy_prefix + "fn");
  torch::Tensor base = state_dict.get_tensor(legacy_prefix + "base");
  torch::Tensor scale = state_dict.get_tensor(legacy_prefix + "scale");
  if (fn.defined()) {
    tensors.emplace("hc_fn", fn);
  }
  if (base.defined()) {
    tensors.emplace("hc_base", base);
  }
  if (scale.defined()) {
    tensors.emplace("hc_scale", scale);
  }
  return StateDict(tensors);
}

}  // namespace

DeepseekV4DecoderLayerImpl::DeepseekV4DecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id)
    : layer_id_(layer_id),
      use_hash_(layer_id < context.get_model_args().n_hash_layers()) {
  const ModelArgs& model_args = context.get_model_args();
  const QuantArgs& quant_args = context.get_quant_args();
  const ParallelArgs& parallel_args = context.get_parallel_args();
  const torch::TensorOptions& options = context.get_tensor_options();

  const int64_t hidden_size = model_args.hidden_size();
  const int64_t hc_mult = model_args.hc_mult();
  const int64_t hc_sinkhorn_iters = model_args.hc_sinkhorn_iters();
  const double hc_eps = static_cast<double>(model_args.hc_eps());
  const double norm_eps = static_cast<double>(model_args.rms_norm_eps());

  attn_hc_pre_ = register_module(
      "attn_hc_pre",
      DeepseekV4HCPre(
          hc_mult, hidden_size, hc_sinkhorn_iters, hc_eps, norm_eps, options));
  ffn_hc_pre_ = register_module(
      "ffn_hc_pre",
      DeepseekV4HCPre(
          hc_mult, hidden_size, hc_sinkhorn_iters, hc_eps, norm_eps, options));
  hc_post_ = register_module("hc_post", DeepseekV4HCPost(norm_eps));

  attn_norm_ =
      register_module("input_layernorm",
                      RMSNorm(hidden_size, model_args.rms_norm_eps(), options));
  ffn_norm_ =
      register_module("post_attention_layernorm",
                      RMSNorm(hidden_size, model_args.rms_norm_eps(), options));
  attention_ = register_module(
      "self_attn",
      DeepseekV4Attention(
          model_args, quant_args, parallel_args, options, layer_id_));

  sparse_moe_ = register_module(
      "mlp",
      DeepseekV4SparseMoEBlock(
          model_args, quant_args, parallel_args, options, use_hash_));
}

void DeepseekV4DecoderLayerImpl::set_cache_mapping(
    const DSACacheMapping& mapping) {
  attention_->set_cache_mapping(mapping);
}

void DeepseekV4DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  StateDict attn_state = get_alias_dict(state_dict, {"self_attn.", "attn."});
  if (attn_state.size() > 0) {
    attention_->load_state_dict(attn_state);
  }

  StateDict attn_norm_state =
      get_alias_dict(state_dict, {"input_layernorm.", "attn_norm."});
  if (attn_norm_state.size() > 0) {
    attn_norm_->load_state_dict(attn_norm_state);
  }

  StateDict ffn_norm_state =
      get_alias_dict(state_dict, {"post_attention_layernorm.", "ffn_norm."});
  if (ffn_norm_state.size() > 0) {
    ffn_norm_->load_state_dict(ffn_norm_state);
  }

  StateDict moe_state = get_alias_dict(state_dict, {"mlp.", "ffn."});
  if (moe_state.size() > 0) {
    sparse_moe_->load_state_dict(moe_state);
  }

  attn_hc_pre_->load_state_dict(
      get_hc_state(state_dict, "attn_hc_pre.", "hc_attn_"));
  ffn_hc_pre_->load_state_dict(
      get_hc_state(state_dict, "ffn_hc_pre.", "hc_ffn_"));
}

void DeepseekV4DecoderLayerImpl::verify_loaded_weights() const {
  sparse_moe_->verify_loaded_weights();
}

std::optional<torch::Tensor> DeepseekV4DecoderLayerImpl::route_input_ids(
    const torch::Tensor& ffn_input,
    const std::optional<torch::Tensor>& input_ids) const {
  if (!use_hash_) {
    return std::nullopt;
  }

  CHECK(input_ids.has_value() && input_ids.value().defined())
      << "DeepseekV4 hash routing requires input_ids.";
  torch::Tensor flat_ids =
      input_ids.value().reshape({-1}).to(ffn_input.device()).contiguous();
  const int64_t id_count = flat_ids.size(0);
  const int64_t token_count =
      ffn_input.reshape({-1, ffn_input.size(-1)}).size(0);
  if (id_count == token_count) {
    return flat_ids;
  }
  CHECK_GT(id_count, 0) << "id_count must be greater than 0.";
  CHECK_EQ(token_count % id_count, 0)
      << "token_count must be divisible by id_count.";
  const int64_t repeat_factor = token_count / id_count;
  return flat_ids.unsqueeze(1)
      .repeat({1, repeat_factor})
      .reshape({token_count});
}

torch::Tensor DeepseekV4DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    const std::optional<torch::Tensor>& input_ids) {
  (void)positions;

  residual = std::nullopt;

  torch::Tensor residual_attn = x;
  DeepseekV4HCPreOutput attn_hc = attn_hc_pre_->forward(x);
  torch::Tensor attn_input = attn_hc.output;
  attn_input = std::get<0>(attn_norm_->forward(attn_input));
  torch::Tensor attn_output;
  std::tie(attn_output, std::ignore) =
      attention_->forward(attn_metadata, attn_input, kv_cache);
  std::tie(x, std::ignore) =
      hc_post_->forward(attn_output, residual_attn, attn_hc.post, attn_hc.comb);

  torch::Tensor residual_ffn = x;
  DeepseekV4HCPreOutput ffn_hc = ffn_hc_pre_->forward(x);
  torch::Tensor ffn_input = ffn_hc.output;
  ffn_input = std::get<0>(ffn_norm_->forward(ffn_input));
  std::optional<torch::Tensor> ids = route_input_ids(ffn_input, input_ids);
  FusedMoEImpl::RouteInfo route_info = sparse_moe_->prep_route(ffn_input, ids);
  torch::Tensor ffn_output = sparse_moe_->forward_selected(
      ffn_input, route_info.reduce_weight, route_info.expert_id, input_params);
  std::tie(x, std::ignore) =
      hc_post_->forward(ffn_output, residual_ffn, ffn_hc.post, ffn_hc.comb);
  return x;
}

}  // namespace layer
}  // namespace xllm
