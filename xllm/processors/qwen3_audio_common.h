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

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>

namespace xllm::qwen3_audio {

inline constexpr char kInputFeaturesKey[] = "input_features";
inline constexpr char kFeatureLengthKey[] = "feat_length";
inline constexpr char kFeatureOriginLengthsKey[] = "feat_origin_lens";
inline constexpr char kMaskKey[] = "audio|mask";

inline torch::Tensor get_feature_output_lengths(
    const torch::Tensor& input_lengths,
    int64_t window_length) {
  CHECK_GT(window_length, 0);

  torch::Tensor output_lengths = input_lengths % window_length;
  int64_t full_window_output_length = window_length;
  constexpr int32_t kConvolutionLayerCount = 3;
  for (int32_t index = 0; index < kConvolutionLayerCount; ++index) {
    output_lengths = torch::floor_divide(output_lengths - 1, 2) + 1;
    full_window_output_length = (full_window_output_length - 1) / 2 + 1;
  }
  output_lengths += torch::floor_divide(input_lengths, window_length) *
                    full_window_output_length;
  return output_lengths.to(torch::kInt64);
}

}  // namespace xllm::qwen3_audio
