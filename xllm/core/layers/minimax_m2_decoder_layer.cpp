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

#include "minimax_m2_decoder_layer.h"

#include <string>
#include <tuple>
#include <unordered_map>

namespace xllm {
namespace layer {
namespace {

void replace_all(std::string& text,
                 const std::string& from,
                 const std::string& to) {
  size_t pos = 0;
  while ((pos = text.find(from, pos)) != std::string::npos) {
    text.replace(pos, from.size(), to);
    pos += to.size();
  }
}

StateDict get_minimax_moe_state_dict(const StateDict& state_dict) {
  StateDict mlp_state_dict = state_dict.get_dict_with_prefix("mlp.");
  if (mlp_state_dict.size() > 0) {
    return mlp_state_dict;
  }

  StateDict sparse_moe_state_dict =
      state_dict.get_dict_with_prefix("block_sparse_moe.");
  std::unordered_map<std::string, torch::Tensor> remapped;
  remapped.reserve(sparse_moe_state_dict.size());
  for (const auto& [name, tensor] : sparse_moe_state_dict) {
    std::string mapped_name = name;
    if (mapped_name == "e_score_correction_bias") {
      mapped_name = "gate.e_score_correction_bias";
    }
    replace_all(mapped_name, ".w1.", ".gate_proj.");
    replace_all(mapped_name, ".w2.", ".down_proj.");
    replace_all(mapped_name, ".w3.", ".up_proj.");
    remapped.emplace(std::move(mapped_name), tensor);
  }
  return StateDict(std::move(remapped));
}

}  // namespace

MiniMaxM2DecoderLayerImpl::MiniMaxM2DecoderLayerImpl(
    const ModelContext& context,
    int32_t /*layer_id*/) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();

  attention_ = register_module("self_attn", MiniMaxM2Attention(context));
  input_norm_ = register_module(
      "input_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));
  post_norm_ = register_module(
      "post_attention_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  moe_ = register_module("mlp",
                         FusedMoE(model_args,
                                  FusedMoEArgs{.is_gated = true},
                                  quant_args,
                                  parallel_args,
                                  options));
}

torch::Tensor MiniMaxM2DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
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
  moe_->load_state_dict(get_minimax_moe_state_dict(state_dict));
}

}  // namespace layer
}  // namespace xllm
