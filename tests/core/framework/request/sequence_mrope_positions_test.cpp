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

#include <cstdint>
#include <vector>

#include "framework/request/incremental_decoder.h"
#include "framework/request/sequence.h"

namespace xllm {
namespace {

Sequence make_prefill_sequence(const std::vector<int32_t>& prompt_token_ids) {
  static RequestSamplingParam sampling_param;
  static StoppingChecker stopping_checker;

  SequenceParams params;
  params.seq_capacity = 64;
  params.request_id = "mrope_positions_req";
  params.sampling_param = &sampling_param;
  params.stopping_checker = &stopping_checker;

  IncrementalDecoder decoder(
      /*prompt=*/"prompt",
      /*num_prompt_tokens=*/prompt_token_ids.size(),
      /*echo=*/false,
      /*skip_special_tokens=*/true);
  return Sequence(/*index=*/0,
                  prompt_token_ids,
                  /*input_embedding=*/torch::Tensor(),
                  /*mm_data=*/MMData(),
                  decoder,
                  params);
}

// A [3, num_tokens] int32 tensor, matching the shape mRoPE position generators
// produce.
torch::Tensor make_positions(int64_t num_tokens) {
  return torch::arange(3 * num_tokens, torch::kInt32).reshape({3, num_tokens});
}

}  // namespace

TEST(SequenceMropePositionsTest, CachesFullPositionsWhenWidthMatchesTokens) {
  std::vector<int32_t> prompt = {5, 6, 7, 8, 9};
  Sequence seq = make_prefill_sequence(prompt);
  EXPECT_FALSE(seq.has_mrope_positions());

  const int64_t num_tokens = static_cast<int64_t>(seq.num_tokens());
  ASSERT_EQ(num_tokens, static_cast<int64_t>(prompt.size()));

  torch::Tensor positions = make_positions(num_tokens);
  seq.set_mrope_positions(positions);
  EXPECT_TRUE(seq.has_mrope_positions());
  EXPECT_TRUE(torch::equal(seq.mrope_positions(), positions));
}

TEST(SequenceMropePositionsTest, StaleWidthIsRejected) {
  Sequence seq = make_prefill_sequence({1, 2, 3, 4});
  const int64_t num_tokens = static_cast<int64_t>(seq.num_tokens());
  // A cache whose width does not match the current token count must be treated
  // as invalid so an interrupted/re-run sequence regenerates instead of slicing
  // a stale tensor.
  seq.set_mrope_positions(make_positions(num_tokens + 2));
  EXPECT_FALSE(seq.has_mrope_positions());
}

}  // namespace xllm
