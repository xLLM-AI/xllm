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

#include "framework/sampling/sampling_params.h"

namespace xllm {

struct BlockDraftSampleOutput {
  torch::Tensor token_ids;
  torch::Tensor probs;
};

class BlockDraftSampler {
 public:
  virtual ~BlockDraftSampler() = default;

  virtual BlockDraftSampleOutput sample(
      const torch::Tensor& base_logits,
      const torch::Tensor& anchor_token_ids,
      const SamplingParameters& sampling_params) = 0;
};

}  // namespace xllm
