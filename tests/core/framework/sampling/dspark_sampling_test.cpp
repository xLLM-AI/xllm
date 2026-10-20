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

#include "core/framework/sampling/dspark_sampling.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <vector>

namespace xllm {
namespace {

TEST(DSparkSamplingTest, BatchedMarkovChainUsesConsensusBeforeNextStep) {
  const auto logits = torch::zeros({2, 3, 8});
  const auto anchors = torch::tensor({0, 4}, torch::kInt64);
  SamplingParameters params;
  params.sampling_mode = SamplingMode::GREEDY;
  params.do_sample = torch::zeros({2}, torch::kBool);
  std::vector<torch::Tensor> predecessors;
  predecessors.reserve(3);
  int32_t broadcasts = 0;
  auto output = torch::full({2, 3}, -1, torch::kInt64);
  const auto result = sample_dspark_block(
      logits,
      anchors,
      params,
      [&predecessors](const torch::Tensor& previous) {
        predecessors.emplace_back(previous.clone());
        auto bias = torch::zeros({2, 8});
        bias.scatter_(1, (previous + 1).remainder(8).view({2, 1}), 20.0);
        return bias;
      },
      [&broadcasts](torch::Tensor& tokens) {
        // Simulate the authoritative rank choosing a different first token.
        if (broadcasts++ == 0) {
          tokens.copy_(torch::tensor({2, 5}, torch::kInt64));
        }
      },
      output);
  EXPECT_EQ(result.token_ids().data_ptr(), output.data_ptr());
  EXPECT_TRUE(
      torch::equal(result.token_ids(),
                   torch::tensor({{2, 3, 4}, {5, 6, 7}}, torch::kInt64)));
  EXPECT_TRUE(
      torch::equal(predecessors.at(1), torch::tensor({2, 5}, torch::kInt64)));
  EXPECT_TRUE(
      torch::equal(predecessors.at(2), torch::tensor({3, 6}, torch::kInt64)));
  EXPECT_FALSE(result.logits().has_value());
  EXPECT_EQ(broadcasts, 3);
}

#if !defined(USE_MLU) && !defined(USE_CUDA) && !defined(USE_DCU)
// These backends' Sampler dispatches random sampling directly to a device
// kernel, so this CPU recurrence test applies to the torch sampling backend.
TEST(DSparkSamplingTest, ProbabilisticProposalKeepsConditionalDistributions) {
  const auto logits = torch::zeros({2, 2, 4});
  const auto anchors = torch::tensor({0, 2}, torch::kInt64);
  SamplingParameters params;
  params.sampling_mode = SamplingMode::RANDOM;
  params.do_sample = torch::ones({2}, torch::kBool);
  params.return_logits = true;
  const auto bias = [](const torch::Tensor& previous) {
    return torch::arange(4, torch::kFloat32).view({1, 4}) *
           (previous + 1).view({2, 1});
  };
  const auto result = sample_dspark_block(
      logits, anchors, params, bias, [](torch::Tensor& tokens) {
        tokens.copy_(torch::tensor({1, 3}, torch::kInt64));
      });
  ASSERT_TRUE(result.logits().has_value());
  EXPECT_TRUE(torch::allclose(torch::softmax(result.logits()->select(1, 0), -1),
                              torch::softmax(bias(anchors), -1)));
  EXPECT_TRUE(torch::allclose(
      torch::softmax(result.logits()->select(1, 1), -1),
      torch::softmax(bias(torch::tensor({1, 3}, torch::kInt64)), -1)));
}
#endif

}  // namespace
}  // namespace xllm
