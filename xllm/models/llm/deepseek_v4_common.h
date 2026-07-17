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

#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>

#include "core/framework/model/model_output.h"
#include "core/layers/common/dsa_metadata.h"

namespace xllm {

inline torch::Tensor maybe_to_device(const torch::Tensor& tensor,
                                     const torch::Device& device) {
  if (!tensor.defined() || tensor.device() == device) {
    return tensor;
  }
  return tensor.to(device);
}

inline int64_t deepseek_v4_next_power_of_two(int64_t n) {
  int64_t value = 1;
  while (value < n) {
    value <<= 1;
  }
  return value;
}

inline torch::Tensor deepseek_v4_create_hadamard_matrix(
    int64_t n,
    torch::ScalarType dtype,
    const torch::Device& device) {
  auto options = torch::TensorOptions().dtype(dtype).device(device);
  torch::Tensor matrix = torch::ones({1, 1}, options);
  for (int64_t m = 1; m < n; m <<= 1) {
    auto top = torch::cat({matrix, matrix}, 1);
    auto bottom = torch::cat({matrix, -matrix}, 1);
    matrix = torch::cat({top, bottom}, 0);
  }
  return matrix;
}

inline int32_t deepseek_v4_normalize_compress_ratio(int32_t ratio) {
  return ratio <= 1 ? 1 : ratio;
}

// Group key: (ratio, type, block_size) -> group_id
class DSAGroupKey {
 public:
  int32_t ratio;
  DSACacheType type;
  int32_t block_size;

  bool operator==(const DSAGroupKey& o) const {
    return ratio == o.ratio && type == o.type && block_size == o.block_size;
  }
};

class DSAGroupKeyHash {
 public:
  size_t operator()(const DSAGroupKey& k) const {
    size_t h = std::hash<int32_t>()(k.ratio);
    h ^= std::hash<int32_t>()(static_cast<int32_t>(k.type)) << 16;
    h ^= std::hash<int32_t>()(k.block_size) << 8;
    return h;
  }
};

inline ModelOutput make_deepseek_v4_mtp_target_output(
    const torch::Tensor& hidden_states,
    const std::optional<torch::Tensor>& residual,
    const torch::Tensor& pre_hc_head_hidden_states) {
  ModelOutput output(hidden_states, residual);
  output.aux_hidden_states = pre_hc_head_hidden_states.flatten(1);
  return output;
}

}  // namespace xllm
