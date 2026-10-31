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

#include <utility>

namespace xllm {

TEST(SamplingParamsTest, ForwardedAccessRetainsParameterValueSemantics) {
  SamplingParameters params;
  torch::Tensor temperatures = torch::tensor({0.5F, 1.0F});
  SamplingParameters& configured =
      params.temperatures(std::move(temperatures)).logprobs(true);
  EXPECT_EQ(&configured, &params);
  EXPECT_FALSE(temperatures.defined());

  const SamplingParameters& view = params;
  EXPECT_EQ(&view.temperatures(), &params.temperatures());
  EXPECT_TRUE(view.logprobs());

  SamplingParameters copy = params;
  copy.temperatures() = torch::tensor({0.25F});
  copy.logprobs() = false;
  EXPECT_TRUE(torch::equal(view.temperatures(), torch::tensor({0.5F, 1.0F})));
  EXPECT_TRUE(view.logprobs());
  EXPECT_TRUE(torch::equal(copy.temperatures(), torch::tensor({0.25F})));
  EXPECT_FALSE(copy.logprobs());

  torch::Tensor moved = std::move(params).temperatures();
  EXPECT_TRUE(torch::equal(moved, torch::tensor({0.5F, 1.0F})));
  EXPECT_FALSE(params.temperatures().defined());
}

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
      torch::allclose(params.temperatures(), torch::tensor({0.0F, 0.01F})));
  EXPECT_TRUE(torch::allclose(params.min_p(), torch::tensor({0.0F, 0.2F})));
  EXPECT_TRUE(torch::equal(params.do_sample(), torch::tensor({false, true})));
  EXPECT_FALSE(params.top_k().defined());
  EXPECT_FALSE(params.top_p().defined());
  EXPECT_FALSE(params.sampling_mode() == SamplingMode::RANDOM);
  EXPECT_FALSE(params.sampling_mode() == SamplingMode::GREEDY);
}

TEST(SamplingParamsTest, HostDispatchMetadataTracksOnlyRandomUnseededRows) {
  RequestSamplingParam seeded;
  seeded.temperature = 1.0F;
  seeded.seed = 42;
  RequestSamplingParam greedy;
  greedy.temperature = 0.0F;
  RequestSamplingParam unseeded;
  unseeded.temperature = 1.0F;
  SamplingParameters params;
  params.init({&seeded, &greedy}, {0, 1}, {}, {}, {});
  EXPECT_FALSE(params.has_unseeded_sample());
  EXPECT_EQ(params.num_random_rows(), 1);
  const auto copied = params.to(torch::Device(torch::kCPU), torch::kFloat32);
  EXPECT_FALSE(copied.has_unseeded_sample());
  EXPECT_EQ(copied.num_random_rows(), 1);
  params.init({&seeded, &unseeded}, {0, 1}, {}, {}, {});
  EXPECT_TRUE(params.has_unseeded_sample());
  EXPECT_EQ(params.num_random_rows(), 2);
  params.init({&greedy}, {0}, {}, {}, {});
  EXPECT_FALSE(params.has_unseeded_sample());
  EXPECT_EQ(params.num_random_rows(), 0);
}

TEST(SamplingParamsTest, MinimumConstraintsExpandForVerifyAndStayOffDrafts) {
  SamplingParameters params;
  params.selected_token_idxes() = torch::arange(2, torch::kInt32);
  params.min_token_positions() = torch::tensor({100, 200}, torch::kInt64);
  params.min_tokens_remaining() = torch::tensor({3, 4}, torch::kInt64);
  params.min_token_ids() = torch::tensor({{7, -1}, {8, 9}}, torch::kInt64);
  params.bad_words() = torch::ones({2, 1, 2}, torch::kInt64);
  params.bad_word_suffix() = torch::ones({2, 1}, torch::kInt64);
  params.seeds() = torch::ones({2}, torch::kInt64);
  EXPECT_TRUE(params.has_sampling_fields());
  for (const auto& field : kSamplingFields) {
    if (!field.per_request) {
      repeat_interleave_if_defined(field.member(params), /*repeats=*/3);
    }
  }
  params.selected_token_idxes() = torch::arange(6, torch::kInt32);
  EXPECT_TRUE(torch::equal(
      params.min_token_positions(),
      torch::tensor({100, 100, 100, 200, 200, 200}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(params.min_tokens_remaining(),
                           torch::tensor({3, 3, 3, 4, 4, 4}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      params.min_token_ids(),
      torch::tensor({{7, -1}, {7, -1}, {7, -1}, {8, 9}, {8, 9}, {8, 9}},
                    torch::kInt64)));
  for (const bool probabilistic : {false, true}) {
    SamplingParameters draft = params;
    prepare_draft_sampling(probabilistic, draft);
    EXPECT_FALSE(draft.min_token_positions().defined());
    EXPECT_FALSE(draft.min_tokens_remaining().defined());
    EXPECT_FALSE(draft.min_token_ids().defined());
    EXPECT_FALSE(draft.bad_words().defined());
    EXPECT_FALSE(draft.bad_word_suffix().defined());
    EXPECT_FALSE(draft.seeds().defined());
    EXPECT_FALSE(draft.has_sampling_fields());
  }
  EXPECT_TRUE(params.min_token_ids().defined());
}

TEST(SamplingParamsTest, DeviceTransferPreservesSamplingStateAndTensorTypes) {
  SamplingParameters params;
  params.selected_token_idxes() = torch::tensor({2, 5}, torch::kInt32);
  params.filter_mask() = torch::tensor({{0.0F, -1.0F}, {-2.0F, 0.0F}});
  params.filter_bitmask() = torch::tensor({{1}, {2}}, torch::kInt32);
  params.frequency_penalties() = torch::tensor({0.5F, 0.25F});
  params.presence_penalties() = torch::tensor({0.75F, 0.5F});
  params.repetition_penalties() = torch::tensor({1.25F, 1.5F});
  params.temperatures() = torch::tensor({0.5F, 1.0F});
  params.top_p() = torch::tensor({0.75F, 1.0F});
  params.min_p() = torch::tensor({0.25F, 0.0F});
  params.top_k() = torch::tensor({3, 0}, torch::kInt64);
  params.logits_bias() = torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}});
  params.min_token_positions() = torch::tensor({10, 20}, torch::kInt64);
  params.min_tokens_remaining() = torch::tensor({2, 3}, torch::kInt64);
  params.min_token_ids() = torch::tensor({{6, -1}, {7, 8}}, torch::kInt64);
  params.seeds() = torch::tensor({42, -1}, torch::kInt64);
  params.seed_offsets() = torch::tensor({3, 4}, torch::kInt64);
  params.bad_words() = torch::tensor({{{2, 3}}, {{4, 5}}}, torch::kInt64);
  params.bad_word_suffix() = torch::tensor({{2}, {4}}, torch::kInt64);
  params.prefix_tokens() = torch::tensor({{3}, {-1}}, torch::kInt64);
  params.unique_token_ids() = torch::tensor({{1, 2}, {3, 4}}, torch::kInt64);
  params.unique_token_counts() = torch::tensor({{2, 1}, {1, 3}}, torch::kInt32);
  params.unique_token_ids_lens() = torch::tensor({2, 2}, torch::kInt32);
  params.do_sample() = torch::tensor({true, false});
  params.acc_logprob() = torch::tensor({{-0.5F}, {-1.0F}});
  params.sampling_mode() = SamplingMode::MIXED;
  params.logprobs() = true;
  params.return_logits() = true;
  params.is_embeddings() = true;
  params.max_top_logprobs() = 3;
  params.num_return_sequences() = 2;
  params.use_beam_search() = true;

  const SamplingParameters transferred =
      params.to(torch::Device(torch::kCPU), torch::kFloat64);

  EXPECT_TRUE(torch::equal(transferred.selected_token_idxes(),
                           params.selected_token_idxes()));
  EXPECT_EQ(transferred.selected_token_idxes().scalar_type(), torch::kInt32);
  EXPECT_TRUE(torch::equal(transferred.filter_mask(), params.filter_mask()));
  EXPECT_EQ(transferred.filter_mask().scalar_type(), torch::kFloat64);
  EXPECT_TRUE(transferred.filter_mask().is_contiguous());
  EXPECT_TRUE(
      torch::equal(transferred.filter_bitmask(), params.filter_bitmask()));
  EXPECT_EQ(transferred.filter_bitmask().scalar_type(), torch::kInt32);
  EXPECT_TRUE(torch::equal(transferred.frequency_penalties(),
                           params.frequency_penalties()));
  EXPECT_EQ(transferred.frequency_penalties().scalar_type(), torch::kFloat64);
  EXPECT_TRUE(torch::equal(transferred.presence_penalties(),
                           params.presence_penalties()));
  EXPECT_TRUE(torch::equal(transferred.repetition_penalties(),
                           params.repetition_penalties()));
  EXPECT_TRUE(torch::equal(transferred.temperatures(), params.temperatures()));
  EXPECT_TRUE(torch::equal(transferred.top_p(), params.top_p()));
  EXPECT_TRUE(torch::equal(transferred.min_p(), params.min_p()));
  EXPECT_TRUE(torch::equal(transferred.top_k(), params.top_k()));
  EXPECT_EQ(transferred.top_k().scalar_type(), torch::kInt64);
  EXPECT_TRUE(torch::equal(transferred.logits_bias(), params.logits_bias()));
  EXPECT_EQ(transferred.logits_bias().scalar_type(), torch::kFloat32);
  EXPECT_TRUE(torch::equal(transferred.min_token_positions(),
                           params.min_token_positions()));
  EXPECT_TRUE(torch::equal(transferred.min_tokens_remaining(),
                           params.min_tokens_remaining()));
  EXPECT_TRUE(
      torch::equal(transferred.min_token_ids(), params.min_token_ids()));
  EXPECT_TRUE(torch::equal(transferred.seeds(), params.seeds()));
  EXPECT_TRUE(torch::equal(transferred.seed_offsets(), params.seed_offsets()));
  EXPECT_TRUE(torch::equal(transferred.bad_words(), params.bad_words()));
  EXPECT_TRUE(
      torch::equal(transferred.bad_word_suffix(), params.bad_word_suffix()));
  EXPECT_TRUE(
      torch::equal(transferred.prefix_tokens(), params.prefix_tokens()));
  EXPECT_TRUE(
      torch::equal(transferred.unique_token_ids(), params.unique_token_ids()));
  EXPECT_TRUE(torch::equal(transferred.unique_token_counts(),
                           params.unique_token_counts()));
  EXPECT_EQ(transferred.unique_token_counts().scalar_type(), torch::kInt32);
  EXPECT_TRUE(torch::equal(transferred.unique_token_ids_lens(),
                           params.unique_token_ids_lens()));
  EXPECT_TRUE(torch::equal(transferred.do_sample(), params.do_sample()));
  EXPECT_EQ(transferred.do_sample().scalar_type(), torch::kBool);
  EXPECT_TRUE(torch::equal(transferred.acc_logprob(), params.acc_logprob()));
  EXPECT_EQ(transferred.num_sample_seqs(), 2);
  EXPECT_EQ(transferred.sampling_mode(), SamplingMode::MIXED);
  EXPECT_TRUE(transferred.logprobs());
  EXPECT_TRUE(transferred.return_logits());
  EXPECT_TRUE(transferred.is_embeddings());
  EXPECT_EQ(transferred.max_top_logprobs(), 3);
  EXPECT_EQ(transferred.num_return_sequences(), 2);
  EXPECT_TRUE(transferred.use_beam_search());
}

TEST(SamplingParamsTest, ReinitializationClearsPreviousBatchState) {
  SamplingParameters params;
  for (const auto& field : kSamplingFields) {
    field.member(params) = torch::ones({2});
  }
  params.do_sample() = torch::tensor({true, false});
  params.acc_logprob() = torch::ones({2, 1});
  params.sampling_mode() = SamplingMode::MIXED;
  params.logprobs() = true;
  params.return_logits() = true;
  params.is_embeddings() = true;
  params.max_top_logprobs() = 5;
  params.num_return_sequences() = 3;
  params.use_beam_search() = true;
  RequestSamplingParam greedy;
  greedy.temperature = 0.0F;

  params.init({&greedy}, {7}, {}, {}, {});

  EXPECT_TRUE(torch::equal(params.selected_token_idxes(),
                           torch::tensor({7}, torch::kInt32)));
  for (const auto& field : kSamplingFields) {
    EXPECT_FALSE(field.member(params).defined());
  }
  EXPECT_FALSE(params.do_sample().defined());
  EXPECT_FALSE(params.acc_logprob().defined());
  EXPECT_EQ(params.sampling_mode(), SamplingMode::GREEDY);
  EXPECT_FALSE(params.logprobs());
  EXPECT_FALSE(params.return_logits());
  EXPECT_FALSE(params.is_embeddings());
  EXPECT_EQ(params.max_top_logprobs(), 0);
  EXPECT_EQ(params.num_return_sequences(), 0);
  EXPECT_FALSE(params.use_beam_search());
  EXPECT_TRUE(params.is_all_greedy());
}

}  // namespace xllm
