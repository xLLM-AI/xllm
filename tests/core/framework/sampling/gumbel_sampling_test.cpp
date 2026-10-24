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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/framework/sampling/sampler.h"
#include "tests/core/framework/sampling/sampling_test_utils.h"

namespace xllm {

using testutil::make_sampling_params;

TEST(GumbelSamplingTest, MasksGreedyRowsInMixedBatch) {
  torch::manual_seed(2026);

  torch::Tensor noise = gumbel_noise(
      make_sampling_params(torch::tensor({false, true}, torch::kBool)),
      {2, 3, 4},
      torch::Device(torch::kCPU));

  EXPECT_TRUE(noise.select(/*dim=*/0, /*index=*/0).eq(0).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(noise).all().item<bool>());
  EXPECT_TRUE(noise.select(/*dim=*/0, /*index=*/1).ne(0).any().item<bool>());
}

}  // namespace xllm
