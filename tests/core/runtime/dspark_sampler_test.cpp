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

#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/draft_vocab_mapping.h"

namespace xllm {
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
