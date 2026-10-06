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

#include "core/layers/npu/loader/qwen3_audio_encoder_loader.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xllm::layer {

namespace {

constexpr int32_t kInputNormWeight = 0;
constexpr int32_t kInputNormBias = 1;
constexpr int32_t kPostNormWeight = 2;
constexpr int32_t kPostNormBias = 3;
constexpr int32_t kQkvWeight = 4;
constexpr int32_t kQkvBias = 5;
constexpr int32_t kAttentionOutWeight = 6;
constexpr int32_t kAttentionOutBias = 7;
constexpr int32_t kLinearFc1Weight = 8;
constexpr int32_t kLinearFc1Bias = 9;
constexpr int32_t kLinearFc2Weight = 10;
constexpr int32_t kLinearFc2Bias = 11;
constexpr int32_t kQueryWeight = 12;
constexpr int32_t kQueryBias = 13;
constexpr int32_t kKeyWeight = 14;
constexpr int32_t kKeyBias = 15;
constexpr int32_t kValueWeight = 16;
constexpr int32_t kValueBias = 17;

const std::vector<std::pair<int32_t, std::string>> kWeightMapping = {
    {kInputNormWeight, "self_attn_layer_norm.weight"},
    {kInputNormBias, "self_attn_layer_norm.bias"},
    {kPostNormWeight, "final_layer_norm.weight"},
    {kPostNormBias, "final_layer_norm.bias"},
    {kAttentionOutWeight, "self_attn.out_proj.weight"},
    {kAttentionOutBias, "self_attn.out_proj.bias"},
    {kLinearFc1Weight, "fc1.weight"},
    {kLinearFc1Bias, "fc1.bias"},
    {kLinearFc2Weight, "fc2.weight"},
    {kLinearFc2Bias, "fc2.bias"},
    {kQueryWeight, "self_attn.q_proj.weight"},
    {kQueryBias, "self_attn.q_proj.bias"},
    {kKeyWeight, "self_attn.k_proj.weight"},
    {kKeyBias, "self_attn.k_proj.bias"},
    {kValueWeight, "self_attn.v_proj.weight"},
    {kValueBias, "self_attn.v_proj.bias"}};

const std::unordered_map<int32_t, int32_t> kWeightShard = {
    {kAttentionOutWeight, 1},
    {kLinearFc1Weight, 0},
    {kLinearFc1Bias, 0},
    {kLinearFc2Weight, 1},
};

}  // namespace

Qwen3AudioEncoderLoader::Qwen3AudioEncoderLoader(uint64_t weight_count,
                                                 const ModelContext& context)
    : BaseLoader(weight_count, context) {
  const ParallelArgs& parallel_args = context.get_parallel_args();
  const torch::TensorOptions options = context.get_tensor_options();
  encode_param_rank_ = parallel_args.rank();
  encode_param_world_size_ = parallel_args.world_size();
  at_weight_tensors_.resize(weight_count);
  dtype_ = torch::typeMetaToScalarType(options.dtype());
  for (uint64_t index = 0; index < weight_count; ++index) {
    at_weight_tensors_[index] = torch::zeros({1}).to(options);
  }
}

void Qwen3AudioEncoderLoader::load_state_dict(const StateDict& state_dict) {
  for (const auto& [index, name] : kWeightMapping) {
    auto shard = kWeightShard.find(index);
    if (shard != kWeightShard.end()) {
      set_weight(state_dict, name, index, shard->second);
    } else {
      set_weight(state_dict, name, index);
    }
  }
}

void Qwen3AudioEncoderLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : kWeightMapping) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Qwen3AudioEncoderLoader::merge_loaded_weights() {
  // Split packed QKV weights when tensor parallelism is enabled.
  get_weights_col_packed_qkv();

  const torch::Tensor new_qkv_weight =
      torch::cat({at_weight_tensors_[kQueryWeight],
                  at_weight_tensors_[kKeyWeight],
                  at_weight_tensors_[kValueWeight]},
                 0)
          .to(device_);
  at_weight_tensors_[kQkvWeight] = new_qkv_weight;
  at_weight_tensors_[kQueryWeight] = torch::zeros({1}).to(device_);
  at_weight_tensors_[kKeyWeight] = torch::zeros({1}).to(device_);
  at_weight_tensors_[kValueWeight] = torch::zeros({1}).to(device_);

  const torch::Tensor new_qkv_bias =
      torch::cat({at_weight_tensors_[kQueryBias],
                  at_weight_tensors_[kKeyBias],
                  at_weight_tensors_[kValueBias]},
                 0)
          .to(device_);
  at_weight_tensors_[kQkvBias] = new_qkv_bias;
  at_weight_tensors_[kQueryBias] = torch::zeros({1}).to(device_);
  at_weight_tensors_[kKeyBias] = torch::zeros({1}).to(device_);
  at_weight_tensors_[kValueBias] = torch::zeros({1}).to(device_);
}

void Qwen3AudioEncoderLoader::get_weights_col_packed_qkv() {
  const int32_t rank = encode_param_rank_;
  const int32_t world_size = encode_param_world_size_;
  at_weight_tensors_[kQueryWeight] =
      at_weight_tensors_[kQueryWeight].chunk(world_size, 0)[rank].to(device_);
  at_weight_tensors_[kKeyWeight] =
      at_weight_tensors_[kKeyWeight].chunk(world_size, 0)[rank].to(device_);
  at_weight_tensors_[kValueWeight] =
      at_weight_tensors_[kValueWeight].chunk(world_size, 0)[rank].to(device_);
  at_weight_tensors_[kQueryBias] =
      at_weight_tensors_[kQueryBias].chunk(world_size, 0)[rank].to(device_);
  at_weight_tensors_[kKeyBias] =
      at_weight_tensors_[kKeyBias].chunk(world_size, 0)[rank].to(device_);
  at_weight_tensors_[kValueBias] =
      at_weight_tensors_[kValueBias].chunk(world_size, 0)[rank].to(device_);
}

}  // namespace xllm::layer
