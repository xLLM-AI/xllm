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

#include "llama_decoder_loader.h"

#include "llama_loader_constants.h"

namespace xllm {
namespace layer {

using namespace llama_decoder_constants;

LlamaDecoderLoader::LlamaDecoderLoader(uint64_t weight_count,
                                       const ModelContext& context,
                                       LoadMode mode)
    : BaseLoader(weight_count, context, mode) {
  auto options = context.get_tensor_options();
  dtype_ = torch::typeMetaToScalarType(options.dtype());

  auto& targets = working_tensors();
  auto target_options = options.device(target_device());
  for (uint64_t i = 0; i < weight_count; ++i) {
    targets[i] = torch::zeros({1}, target_options);
  }
}

void LlamaDecoderLoader::load_state_dict(const StateDict& state_dict) {
  const bool to_host = load_to_host();
  for (const auto& [name, index] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index], to_host);
    } else {
      set_weight(state_dict, name, index, to_host);
    }
  }
}

void LlamaDecoderLoader::verify_loaded_weights() const {
  const auto& t = working_tensors();
  for (const auto& [name, index] : WEIGHT_MAPPING) {
    CHECK(t[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void LlamaDecoderLoader::merge_host_at_weights() {
  auto& t = working_tensors();

  t[IN_Q_WEIGHT] =
      torch::cat({t[IN_Q_WEIGHT], t[IN_K_WEIGHT], t[IN_V_WEIGHT]}, 0);
  t[IN_K_WEIGHT] = zero_like_working(IN_K_WEIGHT);
  t[IN_V_WEIGHT] = zero_like_working(IN_V_WEIGHT);

  t[IN_MLP_W2_WEIGHT] =
      torch::cat({t[IN_MLP_W2_WEIGHT], t[IN_MLP_W1_WEIGHT]}, 0);
  t[IN_MLP_W1_WEIGHT] = zero_like_working(IN_MLP_W1_WEIGHT);
}

}  // namespace layer
}  // namespace xllm
