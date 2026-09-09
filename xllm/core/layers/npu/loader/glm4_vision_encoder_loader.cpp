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

#include "glm4_vision_encoder_loader.h"

namespace xllm {
namespace layer {

enum Glm4VisionEncoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_POST_NORM_WEIGHT,
  IN_QKV_WEIGHT,
  IN_ATTN_PROJ_WEIGHT,
  IN_LINEAR_GATE_UP_WEIGHT,
  IN_LINEAR_DOWN_WEIGHT,
  IN_LINEAR_UP_WEIGHT,
  IN_LINEAR_GATE_WEIGHT
};

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {IN_INPUT_NORM_WEIGHT, "norm1.weight"},
    {IN_POST_NORM_WEIGHT, "norm2.weight"},
    {IN_QKV_WEIGHT, "attn.qkv.weight"},
    {IN_ATTN_PROJ_WEIGHT, "attn.proj.weight"},
    {IN_LINEAR_GATE_WEIGHT, "mlp.gate_proj.weight"},
    {IN_LINEAR_UP_WEIGHT, "mlp.up_proj.weight"},
    {IN_LINEAR_DOWN_WEIGHT, "mlp.down_proj.weight"}};

// IN_QKV_WEIGHT is sharded explicitly in merge_host_at_weights.
static std::map<int, int> WEIGHT_SHARD = {{IN_ATTN_PROJ_WEIGHT, 1},
                                          {IN_LINEAR_UP_WEIGHT, 0},
                                          {IN_LINEAR_GATE_WEIGHT, 0},
                                          {IN_LINEAR_DOWN_WEIGHT, 1}};

Glm4VisionEncoderLoader::Glm4VisionEncoderLoader(uint64_t weight_count,
                                                 const ModelContext& context,
                                                 LoadMode mode)
    : BaseLoader(weight_count, context, mode) {
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();
  encode_param_rank_ = parallel_args.rank();
  encode_param_world_size_ = parallel_args.world_size();
  working_tensors().resize(weight_count);
  if (load_to_host()) {
    auto host_options =
        torch::TensorOptions().dtype(options.dtype()).device(torch::kCPU);
    for (int i = 0; i < weight_count; ++i) {
      working_tensors()[i] = torch::zeros({1}, host_options);
    }
  } else {
    for (int i = 0; i < weight_count; ++i) {
      working_tensors()[i] = torch::zeros({1}).to(options);
    }
  }
}

void Glm4VisionEncoderLoader::load_state_dict(const StateDict& state_dict) {
  const bool to_host = load_to_host();
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index], to_host);
    } else {
      set_weight(state_dict, name, index, to_host);
    }
  }
}

void Glm4VisionEncoderLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(working_tensors()[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Glm4VisionEncoderLoader::merge_host_at_weights() {
  if (encode_param_world_size_ > 1) {
    get_weights_col_packed_qkv();
  }
  auto& w = working_tensors();
  w[IN_LINEAR_GATE_UP_WEIGHT] =
      torch::cat({w[IN_LINEAR_GATE_WEIGHT], w[IN_LINEAR_UP_WEIGHT]}, 0);

  if (load_to_host()) {
    auto make_zero_like = [](const torch::Tensor& ref) {
      return torch::zeros(
          {1},
          torch::TensorOptions().dtype(ref.scalar_type()).device(torch::kCPU));
    };
    w[IN_LINEAR_GATE_WEIGHT] = make_zero_like(w[IN_LINEAR_GATE_WEIGHT]);
    w[IN_LINEAR_UP_WEIGHT] = make_zero_like(w[IN_LINEAR_UP_WEIGHT]);
  } else {
    w[IN_LINEAR_GATE_WEIGHT] = torch::empty({}, device_);
    w[IN_LINEAR_UP_WEIGHT] = torch::empty({}, device_);
  }
}

void Glm4VisionEncoderLoader::get_weights_col_packed_qkv() {
  auto& w = working_tensors();
  auto qkv_weight = torch::chunk(w[IN_QKV_WEIGHT], 3, 0);
  w[IN_QKV_WEIGHT] = torch::cat(
      {qkv_weight[0].chunk(encode_param_world_size_, 0)[encode_param_rank_],
       qkv_weight[1].chunk(encode_param_world_size_, 0)[encode_param_rank_],
       qkv_weight[2].chunk(encode_param_world_size_, 0)[encode_param_rank_]},
      0);
}

}  // namespace layer
}  // namespace xllm
