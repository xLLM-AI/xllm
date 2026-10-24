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

TEST(SamplingParamsTest, MinimumConstraintsExpandForVerifyAndStayOffDrafts) {
  SamplingParameters params;
  params.selected_token_idxes = torch::arange(2, torch::kInt32);
  params.min_token_positions = torch::tensor({100, 200}, torch::kInt64);
  params.min_tokens_remaining = torch::tensor({3, 4}, torch::kInt64);
  params.min_token_ids = torch::tensor({{7, -1}, {8, 9}}, torch::kInt64);
  EXPECT_TRUE(params.has_sampling_fields());
  for (const auto& field : kSamplingFields) {
    if (!field.per_request) {
      repeat_interleave_if_defined(params.*field.member, /*repeats=*/3);
    }
  }
  params.selected_token_idxes = torch::arange(6, torch::kInt32);
  EXPECT_TRUE(torch::equal(
      params.min_token_positions,
      torch::tensor({100, 100, 100, 200, 200, 200}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(params.min_tokens_remaining,
                           torch::tensor({3, 3, 3, 4, 4, 4}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      params.min_token_ids,
      torch::tensor({{7, -1}, {7, -1}, {7, -1}, {8, 9}, {8, 9}, {8, 9}},
                    torch::kInt64)));
  for (const bool probabilistic : {false, true}) {
    SamplingParameters draft = params;
    prepare_draft_sampling(probabilistic, draft);
    EXPECT_FALSE(draft.min_token_positions.defined());
    EXPECT_FALSE(draft.min_tokens_remaining.defined());
    EXPECT_FALSE(draft.min_token_ids.defined());
    EXPECT_FALSE(draft.has_sampling_fields());
  }
  EXPECT_TRUE(params.min_token_ids.defined());
}

}  // namespace xllm
