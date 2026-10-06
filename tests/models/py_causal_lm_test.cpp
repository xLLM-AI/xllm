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

#include "models/llm/py_causal_lm.h"

#include <gtest/gtest.h>

namespace xllm::detail {
namespace {

TEST(PyCausalLMMegaMoeCapacityTest, DecodeCoversGraphAndEagerFallbackLayouts) {
  EXPECT_EQ(python_mega_moe_max_num_tokens_per_rank(
                /*max_seqs_per_batch=*/16,
                /*num_speculative_tokens=*/0,
                /*dp_size=*/1),
            16);
  EXPECT_EQ(python_mega_moe_max_num_tokens_per_rank(
                /*max_seqs_per_batch=*/16,
                /*num_speculative_tokens=*/0,
                /*dp_size=*/16),
            31);
  EXPECT_EQ(python_mega_moe_max_num_tokens_per_rank(
                /*max_seqs_per_batch=*/17,
                /*num_speculative_tokens=*/0,
                /*dp_size=*/16),
            32);
}

TEST(PyCausalLMMegaMoeCapacityTest, SpeculativeDecodeScalesRowsBySpecWidth) {
  EXPECT_EQ(python_mega_moe_max_num_tokens_per_rank(
                /*max_seqs_per_batch=*/16,
                /*num_speculative_tokens=*/3,
                /*dp_size=*/16),
            79);
  EXPECT_EQ(python_mega_moe_max_num_tokens_per_rank(
                /*max_seqs_per_batch=*/17,
                /*num_speculative_tokens=*/3,
                /*dp_size=*/16),
            128);
}

TEST(PyCausalLMMegaMoeCapacityTest,
     Qwen35CoversPrefillAndSpeculativeDecodeWithoutDpGather) {
  EXPECT_EQ(python_qwen3_5_mega_moe_max_num_tokens_per_rank(
                /*max_tokens_per_batch=*/4096,
                /*max_seqs_per_batch=*/256,
                /*num_speculative_tokens=*/3),
            4096);
  EXPECT_EQ(python_qwen3_5_mega_moe_max_num_tokens_per_rank(
                /*max_tokens_per_batch=*/1024,
                /*max_seqs_per_batch=*/512,
                /*num_speculative_tokens=*/4),
            2560);
}

}  // namespace
}  // namespace xllm::detail
