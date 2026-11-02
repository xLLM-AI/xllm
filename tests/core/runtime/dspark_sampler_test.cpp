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
#include <torch/torch.h>

#include <vector>

#include "core/runtime/dspark_worker_impl.h"

namespace xllm {
TEST(DSparkMarkovSamplerTest,
     ReducedVocabularyPreservesFeedbackAndDistribution) {
  const torch::Tensor logits =
      torch::tensor({{{-1.0f, 0.0f, 2.0f}, {0.0f, 1.0f, -2.0f}}});
  const torch::Tensor mapping = torch::tensor({1, 4, 6}, torch::kInt64);
  const DraftVocabMapping draft_vocab(torch::tensor({1, 3, 4}, torch::kInt64),
                                      3,
                                      8,
                                      torch::Device(torch::kCPU));
  const torch::Tensor anchors = torch::tensor({7}, torch::kInt64);
  for (const bool probabilistic : {false, true}) {
    SamplingParameters sampling;
    sampling
        .sampling_mode(probabilistic ? SamplingMode::RANDOM
                                     : SamplingMode::GREEDY)
        .return_logits(probabilistic);
    std::vector<torch::Tensor> feedback;
    const DraftProposal proposal = sample_dspark_markov_block(
        [&feedback](const torch::Tensor& previous) {
          feedback.push_back(previous.clone());
          return torch::zeros({1, 3});
        },
        logits,
        anchors,
        sampling,
        [](torch::Tensor&) {},
        probabilistic,
        torch::Tensor(),
        draft_vocab);
    ASSERT_EQ(feedback.size(), 2u);
    EXPECT_TRUE(torch::equal(feedback[0], anchors));
    EXPECT_TRUE(torch::equal(feedback[1], proposal.token_ids().select(1, 0)));
    EXPECT_EQ(proposal.logits().has_value(), probabilistic);
    for (int64_t step = 0; step < 2; ++step) {
      EXPECT_TRUE(mapping.eq(proposal.token_ids()[0][step]).any().item<bool>());
      if (!probabilistic) {
        continue;
      }
      torch::Tensor expected = torch::zeros({1, 8});
      expected.index_copy_(
          1, mapping, torch::softmax(logits.select(1, step), -1));
      EXPECT_TRUE(torch::allclose(
          torch::softmax(proposal.logits()->select(1, step), -1), expected));
    }
  }
}

TEST(DraftVocabMappingTest, ElidesOnlyFullIdentity) {
  for (const bool identity : {true, false}) {
    SCOPED_TRACE(identity ? "identity" : "permutation");
    const torch::Tensor offsets = identity
                                      ? torch::zeros({2}, torch::kInt64)
                                      : torch::tensor({1, -1}, torch::kInt64);
    const DraftVocabMapping mapping(offsets, 2, 2, torch::Device(torch::kCPU));
    EXPECT_EQ(mapping.draft_to_target_ids_.defined(), !identity);
    SampleOutput output;
    output.next_tokens = torch::tensor({1}, torch::kInt64);
    output.draft_logits = torch::tensor({{0.5f, -1.0f}});
    mapping.remap(output);
    const torch::Tensor expected_tokens =
        torch::tensor({identity ? 1 : 0}, torch::kInt64);
    const torch::Tensor expected_logits = identity
                                              ? torch::tensor({{0.5f, -1.0f}})
                                              : torch::tensor({{-1.0f, 0.5f}});
    EXPECT_TRUE(torch::equal(output.next_tokens, expected_tokens));
    EXPECT_TRUE(torch::equal(output.draft_logits, expected_logits));
  }
}

}  // namespace xllm
