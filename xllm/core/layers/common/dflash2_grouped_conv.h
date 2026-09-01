/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <tuple>

#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/add_matmul.h"

namespace xllm::layer {

torch::Tensor dflash2_grouped_conv(const torch::Tensor& hidden_states,
                                   const torch::Tensor& delta,
                                   const torch::Tensor& base,
                                   int32_t block_size,
                                   int32_t num_groups,
                                   int32_t group_size,
                                   int32_t taps);

class DFlash2GroupedConvImpl final : public torch::nn::Module {
 public:
  DFlash2GroupedConvImpl(int64_t hidden_size,
                         int32_t taps,
                         int32_t group_size,
                         int32_t block_size,
                         const torch::TensorOptions& options);

  std::tuple<torch::Tensor, torch::Tensor> prepare(
      const torch::Tensor& hidden_states);

  torch::Tensor finish(const torch::Tensor& hidden_states,
                       const torch::Tensor& coefficients);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights(const std::string& prefix) const;

 private:
  torch::Tensor convolve(const torch::Tensor& hidden_states,
                         const torch::Tensor& delta,
                         int32_t side) const;

  AddMatmul kernel_projection_{nullptr};
  torch::Tensor base_kernel_;
  bool base_kernel_is_loaded_ = false;
  int32_t block_size_ = 0;
  int32_t taps_ = 0;
  int32_t group_size_ = 0;
  int32_t num_groups_ = 0;
};
TORCH_MODULE(DFlash2GroupedConv);

}  // namespace xllm::layer
