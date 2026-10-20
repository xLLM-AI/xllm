/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/framework/sampling/sampling_params.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm {

TEST(SamplingParamsTest, NormalizesGreedyAndSmallTemperatureRows) {
  RequestSamplingParam greedy;
  greedy.temperature = 0;
  greedy.top_k = 1;
  greedy.top_p = 0.2;
  greedy.min_p = 0.9;
  RequestSamplingParam random;
  random.temperature = 0.001;
  random.min_p = 0.2;
  SamplingParameters params;
  params.init({&greedy, &random}, {0, 1}, {}, {}, {});
  EXPECT_TRUE(
      torch::allclose(params.temperatures, torch::tensor({0.0F, 0.01F})));
  EXPECT_TRUE(torch::allclose(params.min_p, torch::tensor({0.0F, 0.2F})));
  EXPECT_TRUE(torch::equal(params.do_sample, torch::tensor({false, true})));
  EXPECT_FALSE(params.top_k.defined());
  EXPECT_FALSE(params.top_p.defined());
  EXPECT_FALSE(params.sampling_mode == SamplingMode::RANDOM);
  EXPECT_FALSE(params.sampling_mode == SamplingMode::GREEDY);
}

}  // namespace xllm
