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

#include "core/layers/qwen3_vision_layer.h"

namespace xllm {
namespace layer {

Qwen3_VisionLayerImpl::Qwen3_VisionLayerImpl(const ModelContext& context)
    : Qwen2_5_VisionLayerImpl(context, true) {}

void Qwen3_VisionLayerImpl::load_state_dict(const StateDict& state_dict) {
  const auto attention_dict = state_dict.get_dict_with_prefix("attn.");
  const auto mlp_dict = state_dict.get_dict_with_prefix("mlp.");
  const auto norm1_dict = state_dict.get_dict_with_prefix("norm1.");
  const auto norm2_dict = state_dict.get_dict_with_prefix("norm2.");

  attention_->load_state_dict(attention_dict);
  mlp_->load_state_dict(mlp_dict, {"linear_fc1."}, "linear_fc2.");
  norm1_->load_state_dict(norm1_dict);
  norm2_->load_state_dict(norm2_dict);

  attention_qkv_weight_loaded_ |= attention_dict.has("qkv.weight");
  attention_qkv_bias_loaded_ |= attention_dict.has("qkv.bias");
  attention_proj_weight_loaded_ |= attention_dict.has("proj.weight");
  attention_proj_bias_loaded_ |= attention_dict.has("proj.bias");
  mlp_fc1_weight_loaded_ |= mlp_dict.has("linear_fc1.weight");
  mlp_fc1_bias_loaded_ |= mlp_dict.has("linear_fc1.bias");
  mlp_fc2_weight_loaded_ |= mlp_dict.has("linear_fc2.weight");
  mlp_fc2_bias_loaded_ |= mlp_dict.has("linear_fc2.bias");
  norm1_weight_loaded_ |= norm1_dict.has("weight");
  norm1_bias_loaded_ |= norm1_dict.has("bias");
  norm2_weight_loaded_ |= norm2_dict.has("weight");
  norm2_bias_loaded_ |= norm2_dict.has("bias");
}

void Qwen3_VisionLayerImpl::verify_loaded_weights(
    const std::string& prefix) const {
  CHECK(attention_qkv_weight_loaded_)
      << "weight is not loaded for " << prefix + "attn.qkv.weight";
  CHECK(attention_qkv_bias_loaded_)
      << "bias is not loaded for " << prefix + "attn.qkv.bias";
  CHECK(attention_proj_weight_loaded_)
      << "weight is not loaded for " << prefix + "attn.proj.weight";
  CHECK(attention_proj_bias_loaded_)
      << "bias is not loaded for " << prefix + "attn.proj.bias";
  CHECK(mlp_fc1_weight_loaded_)
      << "weight is not loaded for " << prefix + "mlp.linear_fc1.weight";
  CHECK(mlp_fc1_bias_loaded_)
      << "bias is not loaded for " << prefix + "mlp.linear_fc1.bias";
  CHECK(mlp_fc2_weight_loaded_)
      << "weight is not loaded for " << prefix + "mlp.linear_fc2.weight";
  CHECK(mlp_fc2_bias_loaded_)
      << "bias is not loaded for " << prefix + "mlp.linear_fc2.bias";
  CHECK(norm1_weight_loaded_)
      << "weight is not loaded for " << prefix + "norm1.weight";
  CHECK(norm1_bias_loaded_)
      << "bias is not loaded for " << prefix + "norm1.bias";
  CHECK(norm2_weight_loaded_)
      << "weight is not loaded for " << prefix + "norm2.weight";
  CHECK(norm2_bias_loaded_)
      << "bias is not loaded for " << prefix + "norm2.bias";
}

}  // namespace layer
}  // namespace xllm
