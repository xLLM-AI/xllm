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

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <torch_npu/csrc/libs/init_npu.h>

#include "mistral3_vision_encoder_loader.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace xllm {
namespace layer {

// Pixtral ViT per-layer weight mapping (no bias in QKV/out_proj/MLP)
static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {kInputNormWeight, "attention_norm.weight"},
    {kQWeight, "attention.q_proj.weight"},
    {kKWeight, "attention.k_proj.weight"},
    {kVWeight, "attention.v_proj.weight"},
    {kAttentionOutWeight, "attention.o_proj.weight"},
    {kPostNormWeight, "ffn_norm.weight"},
    {kMlpGateWeight, "feed_forward.gate_proj.weight"},
    {kMlpUpWeight, "feed_forward.up_proj.weight"},
    {kMlpDownWeight, "feed_forward.down_proj.weight"},
};

// {weight, dim} — column-parallel split dim, 0=row, 1=col
static std::map<int, int> WEIGHT_SHARD = {
    {kQWeight, 0},
    {kKWeight, 0},
    {kVWeight, 0},
    {kAttentionOutWeight, 1},
    {kMlpGateWeight, 0},
    {kMlpUpWeight, 0},
    {kMlpDownWeight, 1},
};

Mistral3VisionEncoderLoader::Mistral3VisionEncoderLoader(
    uint64_t weight_count,
    const ModelContext& context,
    LoadMode mode)
    : BaseLoader(weight_count, context, mode) {
  auto options = context.get_tensor_options();
  dtype_ = torch::typeMetaToScalarType(options.dtype());
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

void Mistral3VisionEncoderLoader::load_state_dict(const StateDict& state_dict) {
  const bool to_host = load_to_host();
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index], to_host);
    } else {
      set_weight(state_dict, name, index, to_host);
    }
  }
}

void Mistral3VisionEncoderLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(working_tensors()[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Mistral3VisionEncoderLoader::merge_host_at_weights() {
  auto& w = working_tensors();

  // Pack separate Q/K/V into a single QKV tensor
  w[kQWeight] = torch::cat({w[kQWeight], w[kKWeight], w[kVWeight]}, 0);
  w[kKWeight] = zero_like_working(kKWeight);
  w[kVWeight] = zero_like_working(kVWeight);

  // Concatenate gate + up weights for SwiGLU
  w[kMlpGateWeight] = torch::cat({w[kMlpGateWeight], w[kMlpUpWeight]}, 0);
  w[kMlpUpWeight] = zero_like_working(kMlpUpWeight);
}

}  // namespace layer
}  // namespace xllm
