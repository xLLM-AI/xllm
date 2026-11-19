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

#include <algorithm>

#include "core/framework/state_dict/utils.h"

namespace xllm::layer {

DFlash2GroupedConvImpl::DFlash2GroupedConvImpl(
    int64_t hidden_size,
    int32_t taps,
    int32_t group_size,
    int32_t num_speculative_tokens,
    const torch::TensorOptions& options)
    : num_speculative_tokens_(num_speculative_tokens),
      taps_(taps),
      group_size_(group_size) {
  CHECK_GT(hidden_size, 0);
  CHECK_GT(taps_, 0);
  CHECK_GT(group_size_, 0);
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
          .view({hidden_states.size(/*dim=*/0), 2, taps_, num_groups_});
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
  const int64_t num_tokens = hidden_states.size(/*dim=*/0);
  const int64_t num_decoding_tokens =
      static_cast<int64_t>(num_speculative_tokens_) + 1;
  const int64_t num_sequences = num_tokens / num_decoding_tokens;
  const torch::Tensor blocks = hidden_states.view(
      {num_sequences, num_decoding_tokens, num_groups_, group_size_});
  const torch::Tensor coefficients =
      (base_kernel_.select(/*dim=*/0, side)
           .view({1, taps_, num_groups_, group_size_}) +
       delta.unsqueeze(/*dim=*/-1))
          .view({num_sequences,
                 num_decoding_tokens,
                 taps_,
                 num_groups_,
                 group_size_});
  torch::Tensor output = coefficients.select(/*dim=*/2, /*index=*/0) * blocks;

  const int64_t active_taps =
      std::min<int64_t>({taps_, num_decoding_tokens, num_tokens});
  for (int64_t tap = 1; tap < active_taps; ++tap) {
    output.slice(/*dim=*/1, /*start=*/tap, /*end=*/num_decoding_tokens)
        .add_(
            coefficients.select(/*dim=*/2, /*index=*/tap)
                .slice(/*dim=*/1, /*start=*/tap, /*end=*/num_decoding_tokens) *
            blocks.slice(
                /*dim=*/1, /*start=*/0, /*end=*/num_decoding_tokens - tap));
  }
  return output.reshape_as(hidden_states);
}

}  // namespace xllm::layer
