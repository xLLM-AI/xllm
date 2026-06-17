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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "framework/request/incremental_decoder.h"
#include "framework/request/sequence.h"

namespace xllm {
namespace {

Sequence make_sequence() {
  static RequestSamplingParam sampling_param;
  static StoppingChecker stopping_checker;

  SequenceParams params;
  params.seq_capacity = 8;
  params.echo = false;
  params.skip_special_tokens = true;
  params.streaming = false;
  params.enable_schedule_overlap = false;
  params.rec_type = RecType::kNone;
  params.bos_token_id = 0;
  params.request_id = "mtp_bootstrap_req";
  params.sampling_param = &sampling_param;
  params.stopping_checker = &stopping_checker;

  std::vector<int32_t> prompt_token_ids = {1, 2, 3};
  IncrementalDecoder decoder(
      /*prompt=*/"prompt",
      /*num_prompt_tokens=*/prompt_token_ids.size(),
      /*echo=*/params.echo,
      /*skip_special_tokens=*/params.skip_special_tokens);
  return Sequence(/*index=*/0,
                  prompt_token_ids,
                  /*input_embedding=*/torch::Tensor(),
                  /*mm_data=*/MMData(),
                  decoder,
                  params);
}

}  // namespace

TEST(SequenceMtpBootstrapTest, StoresAndClearsBootstrapEmbedding) {
  Sequence sequence = make_sequence();
  torch::Tensor embedding = torch::tensor({1.0f, 2.0f});

  sequence.update_mtp_bootstrap_embedding(embedding);

  torch::Tensor stored = sequence.get_mtp_bootstrap_embedding();
  ASSERT_TRUE(stored.defined());
  EXPECT_TRUE(torch::equal(stored, embedding));

  embedding.fill_(9.0f);
  EXPECT_TRUE(torch::equal(sequence.get_mtp_bootstrap_embedding(),
                           torch::tensor({1.0f, 2.0f})));

  sequence.clear_mtp_bootstrap_embedding();
  EXPECT_FALSE(sequence.get_mtp_bootstrap_embedding().defined());
}

}  // namespace xllm
