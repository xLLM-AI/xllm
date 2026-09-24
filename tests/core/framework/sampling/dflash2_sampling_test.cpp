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

#include "core/framework/sampling/dflash2_sampling.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm {
namespace {

DFlash2CandidateOutput make_candidates() {
  DFlash2CandidateOutput candidates;
  candidates.candidate_ids =
      torch::tensor({2, 5, 3, 7}, torch::kInt64).view({1, 2, 2});
  // Step zero chooses candidate 1. Step one's row 1 chooses candidate 0,
  // while row 0 would choose candidate 1, exposing a broken path recurrence.
  candidates.edge_logits =
      torch::tensor({0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 3.0f, 4.0f, 0.0f})
          .view({1, 2, 2, 2});
  return candidates;
}

TEST(DFlash2SamplingTest, GreedyFollowsPreviouslySelectedCandidate) {
  const auto candidates = make_candidates();
  SamplingParameters params;
  params.all_greedy_sample = true;
  params.all_random_sample = false;
  params.do_sample = torch::zeros({1}, torch::kBool);
  params.temperatures = torch::zeros({1});
  const auto result = sample_dflash2_path(candidates,
                                          params,
                                          torch::zeros({1, 2, 2}),
                                          /*vocab_size=*/8,
                                          /*need_dense_probs=*/false);
  EXPECT_TRUE(torch::equal(result.token_ids,
                           torch::tensor({5, 3}, torch::kInt64).view({1, 2})));
  EXPECT_FALSE(result.dense_probs.defined());
}

TEST(DFlash2SamplingTest, MixedRowsPreserveExactProposalDistribution) {
  auto candidates = make_candidates();
  candidates.candidate_ids = candidates.candidate_ids.repeat({2, 1, 1});
  candidates.edge_logits = candidates.edge_logits.repeat({2, 1, 1, 1});
  SamplingParameters params;
  params.all_greedy_sample = false;
  params.all_random_sample = false;
  params.do_sample = torch::tensor({false, true}, torch::kBool);
  params.temperatures = torch::tensor({0.0f, 2.0f});
  // Fixed noise forces the random row down candidate 0 then candidate 1.
  const auto noise =
      torch::tensor({0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f})
          .view({2, 2, 2});
  const auto result = sample_dflash2_path(candidates,
                                          params,
                                          noise,
                                          /*vocab_size=*/8,
                                          /*need_dense_probs=*/true);
  EXPECT_TRUE(
      torch::equal(result.token_ids,
                   torch::tensor({5, 3, 2, 7}, torch::kInt64).view({2, 2})));
  auto expected = torch::zeros({2, 2, 8});
  expected.index_put_({0, 0, 5}, 1.0f);
  expected.index_put_({0, 1, 3}, 1.0f);
  const auto first = torch::softmax(torch::tensor({0.0f, 1.0f}), 0);
  const auto second = torch::softmax(torch::tensor({0.0f, 1.5f}), 0);
  expected.index_put_({1, 0, 2}, first[0]);
  expected.index_put_({1, 0, 5}, first[1]);
  expected.index_put_({1, 1, 3}, second[0]);
  expected.index_put_({1, 1, 7}, second[1]);
  EXPECT_TRUE(torch::allclose(result.dense_probs, expected));
  EXPECT_TRUE(torch::allclose(result.dense_probs.sum(-1), torch::ones({2, 2})));
}

TEST(DFlash2SamplingTest, RandomRowsReturnConditionalSoftmaxForRejection) {
  const auto candidates = make_candidates();
  SamplingParameters params;
  params.all_greedy_sample = false;
  params.all_random_sample = true;
  params.do_sample = torch::ones({1}, torch::kBool);
  const auto result = sample_dflash2_path(candidates,
                                          params,
                                          torch::zeros({1, 2, 2}),
                                          /*vocab_size=*/8,
                                          /*need_dense_probs=*/true);
  const auto probabilities = torch::softmax(torch::tensor({4.0f, 0.0f}), 0);
  EXPECT_FLOAT_EQ(result.dense_probs.index({0, 1, 3}).item<float>(),
                  probabilities[0].item<float>());
  EXPECT_FLOAT_EQ(result.dense_probs.index({0, 1, 7}).item<float>(),
                  probabilities[1].item<float>());
  EXPECT_TRUE(torch::allclose(result.dense_probs.sum(-1), torch::ones({1, 2})));
}
}  // namespace
}  // namespace xllm
