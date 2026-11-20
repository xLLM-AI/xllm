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

#include <limits>

#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/draft_vocab_mapping.h"

namespace xllm {
TEST(DraftVocabMappingTest, RemapsDifferentVocabulary) {
  const auto offsets = torch::tensor({2, -1}, torch::kInt64);
  const DraftVocabMapping mapping(offsets,
                                  /*draft_vocab_size=*/2,
                                  /*target_vocab_size=*/3,
                                  torch::Device(torch::kCPU));
  SampleOutput output;
  output.next_tokens = torch::tensor({0, 1}, torch::kInt64);
  output.draft_logits = torch::tensor({{0.5f, -1.0f}, {1.0f, -0.5f}});
  mapping.remap(output);
  const float missing = -std::numeric_limits<float>::infinity();
  EXPECT_TRUE(
      torch::equal(output.next_tokens, torch::tensor({2, 0}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      output.draft_logits,
      torch::tensor({{-1.0f, missing, 0.5f}, {-0.5f, missing, 1.0f}})));
}

}  // namespace xllm
