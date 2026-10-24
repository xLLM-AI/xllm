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

#include <gtest/gtest.h>

#include "core/framework/sampling/sampler.h"
#include "tests/core/framework/sampling/sampling_test_utils.h"

namespace xllm {
namespace {

using testutil::get_test_device;
using testutil::make_sampling_params;

TEST(SamplerFilterMaskTest, GreedySamplingHonorsMixedRows) {
  SamplingParameters params = make_sampling_params(/*batch_size=*/2);
  params.filter_mask =
      torch::tensor({{0.0F, -1.0e9F, 0.0F}, {0.0F, 0.0F, -1.0e9F}});
  torch::Tensor logits =
      torch::tensor({{1.0F, 100.0F, 2.0F}, {1.0F, 2.0F, 100.0F}});
  SampleOutput output = sample(logits, params);

  EXPECT_EQ(output.next_tokens.index({0}).item<int64_t>(), 2);
  EXPECT_EQ(output.next_tokens.index({1}).item<int64_t>(), 1);
}

TEST(SamplerFilterMaskTest, RandomSamplingCannotSelectDisallowedToken) {
  SamplingParameters params =
      make_sampling_params(torch::ones({1}, torch::kBool));
  params.filter_mask = torch::tensor({{-1.0e9F, 0.0F, -1.0e9F}});
  torch::Tensor logits = torch::tensor({{100.0F, 1.0F, 100.0F}});

  const torch::Device device = get_test_device();
  params = params.to(device, torch::kFloat32);
  logits = logits.to(device);
  SampleOutput output = sample(logits, params);

  EXPECT_EQ(output.next_tokens.index({0}).item<int64_t>(), 1);
}

TEST(SamplerFilterMaskTest, PackedMaskFiltersCallerLogitsInPlace) {
  SamplingParameters params = make_sampling_params(/*batch_size=*/1);
  params.filter_bitmask =
      torch::tensor({{static_cast<int32_t>(0b0101)}},
                    torch::TensorOptions().dtype(torch::kInt32));
  torch::Tensor logits = torch::tensor({{1.0F, 100.0F, 2.0F, 100.0F}});
  SampleOutput output = sample(logits, params);

  EXPECT_EQ(output.next_tokens.index({0}).item<int64_t>(), 2);
  EXPECT_EQ(logits.index({0, 0}).item<float>(), 1.0F);
  EXPECT_LT(logits.index({0, 1}).item<float>(), -1.0F);
  EXPECT_EQ(logits.index({0, 2}).item<float>(), 2.0F);
  EXPECT_LT(logits.index({0, 3}).item<float>(), -1.0F);
}

}  // namespace
}  // namespace xllm
