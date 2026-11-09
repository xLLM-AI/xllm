/* Copyright 2026 The xLLM Authors.

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

#include <cstdint>
#include <type_traits>

#include "core/framework/model/domain_attention_input.h"

namespace xllm::layer {

class NpuDecoderInputView final {
 public:
  template <typename InputParams,
            std::enable_if_t<!std::is_same_v<std::remove_cv_t<InputParams>,
                                             NpuDecoderInputView>,
                             int32_t> = 0>
  explicit NpuDecoderInputView(InputParams& params)
      : meta(params.meta),
        attention(params.attention),
        parallel(params.parallel),
        expert(params.expert),
        tiling_data(params.graph.tiling_data) {}

  const BatchInputMeta& meta;
  AttentionInputView attention;
  const ParallelInput& parallel;
  const ExpertInput& expert;
  const torch::Tensor& tiling_data;
};

}  // namespace xllm::layer
