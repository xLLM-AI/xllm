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

#include <gtest/gtest.h>

#include "core/framework/sampling/sampler.h"

namespace xllm {
namespace {

SamplingParameters make_greedy_params(int64_t batch_size) {
  SamplingParameters params;
  params.selected_token_idxes =
      torch::arange(batch_size, torch::TensorOptions().dtype(torch::kInt));
  params.sample_idxes =
      torch::arange(batch_size, torch::TensorOptions().dtype(torch::kInt));
  params.do_sample = torch::zeros({batch_size}, torch::kBool);
  params.all_greedy_sample = true;
  params.all_random_sample = false;
  return params;
}

TEST(SamplerFilterMaskTest, GreedySamplingHonorsMixedRows) {
  SamplingParameters params = make_greedy_params(/*batch_size=*/2);
  params.filter_mask =
      torch::tensor({{0.0F, -1.0e9F, 0.0F}, {0.0F, 0.0F, -1.0e9F}});
  torch::Tensor logits =
      torch::tensor({{1.0F, 100.0F, 2.0F}, {1.0F, 2.0F, 100.0F}});

  Sampler sampler;
  SampleOutput output = sampler.forward(logits, params);

  EXPECT_EQ(output.next_tokens.index({0}).item<int64_t>(), 2);
  EXPECT_EQ(output.next_tokens.index({1}).item<int64_t>(), 1);
}

TEST(SamplerFilterMaskTest, RandomSamplingCannotSelectDisallowedToken) {
  SamplingParameters params = make_greedy_params(/*batch_size=*/1);
  params.do_sample = torch::ones({1}, torch::kBool);
  params.all_greedy_sample = false;
  params.all_random_sample = true;
  params.filter_mask = torch::tensor({{-1.0e9F, 0.0F, -1.0e9F}});
  torch::Tensor logits = torch::tensor({{100.0F, 1.0F, 100.0F}});

  Sampler sampler;
  SampleOutput output = sampler.forward(logits, params);

  EXPECT_EQ(output.next_tokens.index({0}).item<int64_t>(), 1);
}

TEST(SamplerFilterMaskTest, PackedMaskFiltersCallerLogitsInPlace) {
  SamplingParameters params = make_greedy_params(/*batch_size=*/1);
  params.filter_bitmask =
      torch::tensor({{static_cast<int32_t>(0b0101)}},
                    torch::TensorOptions().dtype(torch::kInt32));
  torch::Tensor logits = torch::tensor({{1.0F, 100.0F, 2.0F, 100.0F}});

  Sampler sampler;
  SampleOutput output = sampler.forward(logits, params);

  EXPECT_EQ(output.next_tokens.index({0}).item<int64_t>(), 2);
  EXPECT_EQ(logits.index({0, 0}).item<float>(), 1.0F);
  EXPECT_LT(logits.index({0, 1}).item<float>(), -1.0F);
  EXPECT_EQ(logits.index({0, 2}).item<float>(), 2.0F);
  EXPECT_LT(logits.index({0, 3}).item<float>(), -1.0F);
}

}  // namespace
}  // namespace xllm
