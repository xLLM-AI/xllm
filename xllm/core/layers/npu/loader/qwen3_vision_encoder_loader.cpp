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

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <torch_npu/csrc/libs/init_npu.h>

#include "qwen3_vision_encoder_loader.h"
#include "qwen_loader_constants.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace xllm {
namespace layer {

using namespace qwen3_vision_encoder_constants;

Qwen3VisionEncoderLoader::Qwen3VisionEncoderLoader(uint64_t weight_count,
                                                   const ModelContext& context,
                                                   LoadMode mode)
    : BaseLoader(weight_count, context, mode) {
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();
  encode_param_rank_ = parallel_args.rank();
  encode_param_world_size_ = parallel_args.world_size();
  dtype_ = torch::typeMetaToScalarType(options.dtype());
  device_id_ = options.device().index();
  working_tensors().resize(weight_count);
  if (load_to_host()) {
    auto host_options =
        torch::TensorOptions().dtype(options.dtype()).device(torch::kCPU);
    for (int i = 0; i < weight_count; ++i) {
      working_tensors()[i] = torch::zeros({1}, host_options);
    }
  } else {
    at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
    for (int i = 0; i < weight_count; ++i) {
      working_tensors()[i] = torch::zeros({1}).to(options);
    }
  }
}

void Qwen3VisionEncoderLoader::load_state_dict(const StateDict& state_dict) {
  const bool to_host = load_to_host();
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index], to_host);
    } else {
      set_weight(state_dict, name, index, to_host);
    }
  }
}

void Qwen3VisionEncoderLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(working_tensors()[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Qwen3VisionEncoderLoader::merge_host_at_weights() {
  get_weights_col_packed_qkv();
  if (encode_param_world_size_ > 1) {
    auto& w = working_tensors();
    w[IN_QKV_WEIGHT] = torch::cat(
        {w[IN_VISION_Q_WEIGHT], w[IN_VISION_K_WEIGHT], w[IN_VISION_V_WEIGHT]},
        0);
    w[IN_VISION_Q_WEIGHT] = torch::zeros({1}).to(target_device());
    w[IN_VISION_K_WEIGHT] = torch::zeros({1}).to(target_device());
    w[IN_VISION_V_WEIGHT] = torch::zeros({1}).to(target_device());

    w[IN_QKV_BIAS] = torch::cat(
        {w[IN_VISION_Q_BIAS], w[IN_VISION_K_BIAS], w[IN_VISION_V_BIAS]}, 0);
    w[IN_VISION_Q_BIAS] = torch::zeros({1}).to(target_device());
    w[IN_VISION_K_BIAS] = torch::zeros({1}).to(target_device());
    w[IN_VISION_V_BIAS] = torch::zeros({1}).to(target_device());
  }
}

void Qwen3VisionEncoderLoader::get_weights_col_packed_qkv() {
  auto& w = working_tensors();
  qkv_weight_ = torch::chunk(w[IN_QKV_WEIGHT], 3, 0);
  qkv_bias_ = torch::chunk(w[IN_QKV_BIAS], 3, 0);
  w[IN_VISION_Q_WEIGHT] =
      (qkv_weight_[0].chunk(encode_param_world_size_, 0))[encode_param_rank_];
  w[IN_VISION_K_WEIGHT] =
      (qkv_weight_[1].chunk(encode_param_world_size_, 0))[encode_param_rank_];
  w[IN_VISION_V_WEIGHT] =
      (qkv_weight_[2].chunk(encode_param_world_size_, 0))[encode_param_rank_];
  w[IN_VISION_Q_BIAS] =
      (qkv_bias_[0].chunk(encode_param_world_size_, 0))[encode_param_rank_];
  w[IN_VISION_K_BIAS] =
      (qkv_bias_[1].chunk(encode_param_world_size_, 0))[encode_param_rank_];
  w[IN_VISION_V_BIAS] =
      (qkv_bias_[2].chunk(encode_param_world_size_, 0))[encode_param_rank_];
}

}  // namespace layer
}  // namespace xllm
