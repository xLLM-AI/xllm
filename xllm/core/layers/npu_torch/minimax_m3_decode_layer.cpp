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

#include "core/layers/npu_torch/minimax_m3_decode_layer.h"

#include <absl/strings/match.h>
#include <absl/strings/str_replace.h>

#include <array>
#include <string>
#include <unordered_map>

namespace xllm {
namespace layer {

namespace {

constexpr char kQuantMethodMxfp8[] = "mxfp8";
constexpr int64_t kE8m0ExponentBias = 127;
constexpr int64_t kWeightScaleInvSuffixSize = 10;

bool is_moe_layer(const ModelArgs& model_args, int32_t layer_id) {
  return model_args.n_routed_experts() > 0 &&
         layer_id >= model_args.first_k_dense_replace();
}

bool is_load_time_dequant_method(const std::string& quant_method) {
  return quant_method == kQuantMethodFp8 || quant_method == kQuantMethodMxfp8;
}

bool is_fp8_dtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat8_e5m2 || dtype == torch::kFloat8_e4m3fn;
}

std::string remap_moe_weight_name(const std::string& name) {
  std::string mapped_name = name;
  if (absl::StartsWith(mapped_name, "block_sparse_moe.")) {
    mapped_name =
        absl::StrReplaceAll(mapped_name, {{"block_sparse_moe.", "mlp."}});
  }
  if (mapped_name == "mlp.e_score_correction_bias") {
    return "mlp.gate.e_score_correction_bias";
  }
  mapped_name = absl::StrReplaceAll(mapped_name,
                                    {{".w1.", ".gate_proj."},
                                     {".w2.", ".down_proj."},
                                     {".w3.", ".up_proj."}});
  return mapped_name;
}

torch::Tensor build_e8m0_scale_lut(const torch::Device& device) {
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const torch::Tensor exponent = torch::arange(/*start=*/0,
                                               /*end=*/256,
                                               options) -
                                 static_cast<float>(kE8m0ExponentBias);
  return torch::pow(2.0f, exponent).to(torch::kBFloat16);
}

torch::Tensor convert_e8m0_scale_to_bf16(const torch::Tensor& scale) {
  CHECK_EQ(scale.scalar_type(), torch::kUInt8)
      << "MiniMax-M3 MXFP8 scale must use E8M0 uint8 storage, got "
      << scale.scalar_type();
  const torch::Tensor lut = build_e8m0_scale_lut(scale.device());
  return lut.index_select(/*dim=*/0, scale.reshape({-1}).to(torch::kInt64))
      .reshape(scale.sizes());
}

torch::Tensor convert_scale_to_bf16(const torch::Tensor& scale,
                                    bool use_e8m0_scale) {
  if (use_e8m0_scale) {
    return convert_e8m0_scale_to_bf16(scale);
  }
  return scale.to(torch::kBFloat16);
}

torch::Tensor dequantize_fp8_block_weight(
    const torch::Tensor& fp8_weight,
    const torch::Tensor& weight_scale_inv,
    const std::array<int64_t, 2>& block_size,
    bool use_e8m0_scale) {
  CHECK_EQ(fp8_weight.dim(), 2)
      << "Only 2D MiniMax-M3 fp8 weights are supported, got shape "
      << fp8_weight.sizes();
  CHECK_EQ(weight_scale_inv.dim(), 2)
      << "MiniMax-M3 fp8 scale tensor must be 2D, got shape "
      << weight_scale_inv.sizes();

  const int64_t block_n = block_size[0];
  const int64_t block_k = block_size[1];
  const int64_t n = fp8_weight.size(0);
  const int64_t k = fp8_weight.size(1);
  const int64_t n_tiles = (n + block_n - 1) / block_n;
  const int64_t k_tiles = (k + block_k - 1) / block_k;

  CHECK_EQ(weight_scale_inv.size(0), n_tiles)
      << "Unexpected MiniMax-M3 fp8 scale shape " << weight_scale_inv.sizes()
      << " for weight shape " << fp8_weight.sizes();
  CHECK_EQ(weight_scale_inv.size(1), k_tiles)
      << "Unexpected MiniMax-M3 fp8 scale shape " << weight_scale_inv.sizes()
      << " for weight shape " << fp8_weight.sizes();

  if (n % block_n == 0 && k % block_k == 0) {
    torch::Tensor weight_bf16 =
        fp8_weight.to(torch::kBFloat16)
            .reshape({n_tiles, block_n, k_tiles, block_k});
    torch::Tensor scale_bf16 =
        convert_scale_to_bf16(weight_scale_inv, use_e8m0_scale)
            .reshape({n_tiles, 1, k_tiles, 1});
    return (weight_bf16 * scale_bf16).reshape({n, k}).contiguous();
  }

  torch::Tensor expanded_scale =
      convert_scale_to_bf16(weight_scale_inv, use_e8m0_scale)
          .repeat_interleave(block_n, 0)
          .repeat_interleave(block_k, 1);
  expanded_scale = expanded_scale.slice(/*dim=*/0, /*start=*/0, /*end=*/n)
                       .slice(/*dim=*/1, /*start=*/0, /*end=*/k);
  return (fp8_weight.to(torch::kBFloat16) * expanded_scale).contiguous();
}

StateDict prepare_m3_layer_state_dict(
    const StateDict& state_dict,
    bool remap_moe_names,
    bool enable_weight_dequant,
    bool use_e8m0_scale,
    const std::array<int64_t, 2>& weight_block_size) {
  if (!remap_moe_names && !enable_weight_dequant) {
    return state_dict;
  }

  torch::NoGradGuard no_grad;
  std::unordered_map<std::string, torch::Tensor> remapped;
  remapped.reserve(state_dict.size());
  std::unordered_map<std::string, torch::Tensor> pending_fp8_weights;
  std::unordered_map<std::string, torch::Tensor> pending_fp8_scales;

  for (const auto& [name, tensor] : state_dict) {
    const std::string mapped_name =
        remap_moe_names ? remap_moe_weight_name(name) : name;
    if (enable_weight_dequant) {
      if (absl::EndsWith(mapped_name, ".weight_scale_inv")) {
        const std::string paired_weight_name = mapped_name.substr(
            0, mapped_name.size() - kWeightScaleInvSuffixSize);
        auto pending_weight = pending_fp8_weights.find(paired_weight_name);
        if (pending_weight != pending_fp8_weights.end()) {
          remapped.emplace(paired_weight_name,
                           dequantize_fp8_block_weight(pending_weight->second,
                                                       tensor,
                                                       weight_block_size,
                                                       use_e8m0_scale));
          pending_fp8_weights.erase(pending_weight);
        } else {
          pending_fp8_scales.emplace(mapped_name, tensor);
        }
        continue;
      }

      if (absl::EndsWith(mapped_name, ".weight") &&
          is_fp8_dtype(tensor.scalar_type())) {
        const std::string scale_name = mapped_name + "_scale_inv";
        auto pending_scale = pending_fp8_scales.find(scale_name);
        if (pending_scale != pending_fp8_scales.end()) {
          remapped.emplace(mapped_name,
                           dequantize_fp8_block_weight(tensor,
                                                       pending_scale->second,
                                                       weight_block_size,
                                                       use_e8m0_scale));
          pending_fp8_scales.erase(pending_scale);
        } else {
          pending_fp8_weights.emplace(mapped_name, tensor);
        }
        continue;
      }
    }

    remapped.emplace(mapped_name, tensor);
  }

  if (enable_weight_dequant) {
    CHECK(pending_fp8_weights.empty() && pending_fp8_scales.empty())
        << "Unpaired MiniMax-M3 fp8 weight/scale tensors detected: "
        << "pending_weights=" << pending_fp8_weights.size()
        << ", pending_scales=" << pending_fp8_scales.size();
  }

  return StateDict(std::move(remapped), std::string(state_dict.prefix()));
}

}  // namespace

MiniMaxM3DecoderLayerImpl::MiniMaxM3DecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id) {
  const ModelArgs& model_args = context.get_model_args();
  QuantArgs quant_args = context.get_quant_args();
  enable_weight_dequant_ =
      is_load_time_dequant_method(quant_args.quant_method());
  use_e8m0_scale_ = quant_args.quant_method() == kQuantMethodMxfp8;
  if (enable_weight_dequant_) {
    if (quant_args.weight_block_size().size() == 2 &&
        quant_args.weight_block_size()[0] > 0 &&
        quant_args.weight_block_size()[1] > 0) {
      weight_block_size_ = {quant_args.weight_block_size()[0],
                            quant_args.weight_block_size()[1]};
    }
    quant_args.quant_method("");
  }
  const ParallelArgs& parallel_args = context.get_parallel_args();
  const torch::TensorOptions& options = context.get_tensor_options();

  is_moe_layer_ = is_moe_layer(model_args, layer_id);

  attention_ =
      register_module("self_attn", MiniMaxM3Attention(context, layer_id));
  input_norm_ = register_module(
      "input_layernorm",
      layer::Qwen3NextRMSNorm(
          model_args.hidden_size(), model_args.rms_norm_eps(), options));
  post_norm_ = register_module(
      "post_attention_layernorm",
      layer::Qwen3NextRMSNorm(
          model_args.hidden_size(), model_args.rms_norm_eps(), options));

  if (is_moe_layer_) {
    layer::FusedMoEArgs moe_args;
    moe_ = register_module(
        "mlp",
        layer::FusedMoE(
            model_args, moe_args, quant_args, parallel_args, options));
  } else {
    const std::string mlp_module_prefix =
        "language_model.model.layers." + std::to_string(layer_id) + ".mlp";
    mlp_ = register_module("mlp",
                           layer::DenseMLP(model_args.hidden_size(),
                                           model_args.intermediate_size(),
                                           /*is_gated=*/true,
                                           /*has_bias=*/false,
                                           model_args.hidden_act(),
                                           /*enable_result_reduction=*/true,
                                           quant_args,
                                           parallel_args.tp_group_,
                                           options,
                                           mlp_module_prefix));
  }
}

torch::Tensor MiniMaxM3DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const layer::AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  if (x.numel() == 0) {
    return is_moe_layer_ ? moe_->forward(x, input_params) : mlp_->forward(x);
  }

  if (!residual.has_value()) {
    residual = x;
    x = std::get<0>(input_norm_->forward(x));
  } else {
    std::tie(x, residual) = input_norm_->forward(x, residual);
  }

  x = attention_->forward(positions, x, attn_metadata, kv_cache);
  std::tie(x, residual) = post_norm_->forward(x, residual);
  if (is_moe_layer_) {
    x = moe_->forward(x, input_params);
  } else {
    x = mlp_->forward(x);
  }
  return x;
}

void MiniMaxM3DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  StateDict prepared_state_dict =
      prepare_m3_layer_state_dict(state_dict,
                                  is_moe_layer_,
                                  enable_weight_dequant_,
                                  use_e8m0_scale_,
                                  weight_block_size_);

  attention_->load_state_dict(
      prepared_state_dict.get_dict_with_prefix("self_attn."));
  input_norm_->load_state_dict(
      prepared_state_dict.get_dict_with_prefix("input_layernorm."));
  post_norm_->load_state_dict(
      prepared_state_dict.get_dict_with_prefix("post_attention_layernorm."));
  if (is_moe_layer_) {
    moe_->load_state_dict(prepared_state_dict.get_dict_with_prefix("mlp."));
  } else {
    mlp_->load_state_dict(prepared_state_dict.get_dict_with_prefix("mlp."));
  }
}

}  // namespace layer
}  // namespace xllm
