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

#include "glm4_decoder_loader.h"

#include "glm_loader_constants.h"

namespace xllm {
namespace layer {

using namespace glm4_decoder_constants;

Glm4DecoderLoader::Glm4DecoderLoader(uint64_t weight_count,
                                     const ModelContext& context,
                                     LoadMode mode)
    : BaseLoader(weight_count, context, mode) {
  auto options = context.get_tensor_options();
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

void Glm4DecoderLoader::load_state_dict(const StateDict& state_dict) {
  const bool to_host = load_to_host();
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index], to_host);
    } else {
      set_weight(state_dict, name, index, to_host);
    }
  }
}

void Glm4DecoderLoader::merge_host_at_weights() {
  std::vector<at::Tensor>& w = working_tensors();

  merge_gate_up();

  w[IN_Q_WEIGHT] =
      torch::cat({w[IN_Q_WEIGHT], w[IN_K_WEIGHT], w[IN_V_WEIGHT]}, 0)
          .contiguous();

  w[IN_Q_BIAS] =
      torch::cat({w[IN_Q_BIAS], w[IN_K_BIAS], w[IN_V_BIAS]}, 0).contiguous();

  for (auto idx :
       {IN_MLP_W1_WEIGHT, IN_K_WEIGHT, IN_V_WEIGHT, IN_K_BIAS, IN_V_BIAS}) {
    w[idx] = zero_like_working(idx);
  }
}

void Glm4DecoderLoader::merge_gate_up() {
  std::vector<at::Tensor>& w = working_tensors();

  const int32_t world_size = parallel_args_.world_size();
  if (world_size <= 1) {
    return;
  }

  torch::Tensor gate_up_weight = w[IN_MLP_GATEUP_WEIGHT];
  const int64_t gate_up_size = gate_up_weight.size(0);
  const int64_t intermediate_size = gate_up_size / 2;

  const int32_t rank = parallel_args_.rank();
  torch::Tensor gate_weight = gate_up_weight.slice(0, 0, intermediate_size);
  torch::Tensor up_weight =
      gate_up_weight.slice(0, intermediate_size, gate_up_size);
  w[IN_MLP_GATEUP_WEIGHT] = torch::cat({gate_weight.chunk(world_size, 0)[rank],
                                        up_weight.chunk(world_size, 0)[rank]},
                                       0)
                                .contiguous();
}

void Glm4DecoderLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(working_tensors()[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

}  // namespace layer
}  // namespace xllm
