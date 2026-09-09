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

#include "core/layers/common/dflash2_grouped_conv.h"

#include <glog/logging.h>

#include "core/framework/state_dict/utils.h"

namespace xllm::layer {

torch::Tensor dflash2_grouped_conv(const torch::Tensor& hidden_states,
                                   const torch::Tensor& delta,
                                   const torch::Tensor& base,
                                   int32_t block_size,
                                   int32_t num_groups,
                                   int32_t group_size,
                                   int32_t taps) {
  CHECK_EQ(hidden_states.dim(), 2);
  CHECK_EQ(delta.dim(), 3);
  CHECK_EQ(base.dim(), 2);
  CHECK_GT(block_size, 0);
  CHECK_EQ(hidden_states.size(1),
           static_cast<int64_t>(num_groups) * group_size);
  CHECK_EQ(delta.size(0), hidden_states.size(0));
  CHECK_EQ(delta.size(1), taps);
  CHECK_EQ(delta.size(2), num_groups);
  CHECK_EQ(base.size(0), taps);
  CHECK_EQ(base.size(1), hidden_states.size(1));

  const int64_t num_tokens = hidden_states.size(0);
  // Rows must be laid out as contiguous, fixed-width blocks per sequence.
  // The modulo position below is only valid when every sequence begins on a
  // block_size boundary.
  CHECK_EQ(num_tokens % block_size, 0)
      << "DFlash2 convolution rows must contain complete sequence blocks.";
  torch::Tensor blocks =
      hidden_states.view({num_tokens, num_groups, group_size});
  torch::Tensor coefficients =
      base.view({1, taps, num_groups, group_size}) + delta.unsqueeze(-1);
  torch::Tensor output = coefficients.select(/*dim=*/1, /*index=*/0) * blocks;
  torch::Tensor positions = torch::arange(num_tokens,
                                          torch::TensorOptions()
                                              .dtype(torch::kLong)
                                              .device(hidden_states.device())) %
                            block_size;

  for (int32_t tap = 1; tap < taps; ++tap) {
    CHECK_GT(num_tokens, tap)
        << "DFlash2 convolution token count must exceed its tap offset.";
    // Aligned-slice add_ replaces the previous zeros+cat shift: the first
    // `tap` output rows would receive padding-block contributions (zero) and
    // are skipped, which the block-boundary mask below already guarantees.
    torch::Tensor valid =
        positions.slice(/*dim=*/0, /*start=*/tap, /*end=*/num_tokens)
            .ge(tap)
            .view({num_tokens - tap, 1, 1})
            .to(hidden_states.dtype());
    output.slice(/*dim=*/0, /*start=*/tap, /*end=*/num_tokens)
        .add_(coefficients.select(/*dim=*/1, /*index=*/tap)
                  .slice(/*dim=*/0, /*start=*/tap, /*end=*/num_tokens) *
              blocks.slice(/*dim=*/0, /*start=*/0, /*end=*/num_tokens - tap) *
              valid);
  }
  return output.flatten(/*start_dim=*/1);
}

DFlash2GroupedConvImpl::DFlash2GroupedConvImpl(
    int64_t hidden_size,
    int32_t taps,
    int32_t group_size,
    int32_t block_size,
    const torch::TensorOptions& options)
    : block_size_(block_size), taps_(taps), group_size_(group_size) {
  CHECK_GT(hidden_size, 0);
  CHECK_GT(taps_, 0);
  CHECK_GT(group_size_, 0);
  CHECK_GT(block_size_, 0);
  CHECK_EQ(hidden_size % group_size_, 0)
      << "DFlash2 conv_group_size must divide hidden_size.";
  num_groups_ = static_cast<int32_t>(hidden_size / group_size_);
  base_kernel_ = register_parameter(
      "base_kernel", torch::empty({2, taps_, hidden_size}, options), false);
  kernel_projection_ = register_module("kernel_projection",
                                       AddMatmul(hidden_size,
                                                 2LL * taps_ * num_groups_,
                                                 /*with_bias=*/false,
                                                 options));
}

std::tuple<torch::Tensor, torch::Tensor> DFlash2GroupedConvImpl::prepare(
    const torch::Tensor& hidden_states) {
  torch::Tensor coefficients =
      kernel_projection_->forward(hidden_states)
          .view({hidden_states.size(0), 2, taps_, num_groups_});
  return {convolve(hidden_states,
                   coefficients.select(/*dim=*/1, /*index=*/0),
                   /*side=*/0),
          coefficients.select(/*dim=*/1, /*index=*/1)};
}

torch::Tensor DFlash2GroupedConvImpl::finish(
    const torch::Tensor& hidden_states,
    const torch::Tensor& coefficients) {
  return convolve(hidden_states, coefficients, /*side=*/1);
}

void DFlash2GroupedConvImpl::load_state_dict(const StateDict& state_dict) {
  weight::load_weight(
      state_dict, "base_kernel", base_kernel_, base_kernel_is_loaded_);
  kernel_projection_->load_state_dict(
      state_dict.get_dict_with_prefix("kernel_projection."));
}

void DFlash2GroupedConvImpl::verify_loaded_weights(
    const std::string& prefix) const {
  CHECK(base_kernel_is_loaded_)
      << "weight is not loaded for " << prefix + "base_kernel";
  kernel_projection_->verify_loaded_weights(prefix + "kernel_projection.");
}

torch::Tensor DFlash2GroupedConvImpl::convolve(
    const torch::Tensor& hidden_states,
    const torch::Tensor& delta,
    int32_t side) const {
  return dflash2_grouped_conv(hidden_states,
                              delta,
                              base_kernel_.select(/*dim=*/0, side),
                              block_size_,
                              num_groups_,
                              group_size_,
                              taps_);
}

}  // namespace xllm::layer
