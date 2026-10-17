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

#include "rejection_sampler.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <cstdint>

#include "platform/device.h"
#include "platform/platform.h"
#include "sampler.h"

namespace xllm {

namespace {
RejectionSampler make_greedy_rejection_sampler(int64_t batch_size,
                                               torch::Device device) {
  return RejectionSampler(
      torch::zeros({batch_size},
                   torch::TensorOptions().dtype(torch::kBool).device(device)),
      /*all_random_sample=*/false,
      /*all_greedy_sample=*/true,
      /*logprobs=*/false,
      /*max_top_logprobs=*/0);
}

// Helper function to get test device: MLU if available, otherwise CPU
torch::Device get_test_device() {
  if ((Platform::is_mlu() || Platform::is_cuda()) &&
      Platform::device_count() > 0) {
    return torch::Device(Platform::type_torch(), 0);
  }
  return torch::Device(torch::kCPU);
}

// Helper function to get test tensor options with automatic device selection
torch::TensorOptions get_test_options(
    torch::ScalarType dtype = torch::kFloat32) {
  return torch::dtype(dtype).device(get_test_device());
}
}  // namespace

TEST(RejectionSamplerTest, Basic) {
  // test with hand-crafted example
  const auto options = get_test_options(torch::kFloat32);
  const auto device = get_test_device();

  // set random seed
  torch::manual_seed(100);

  const auto draft_token_ids =
      torch::tensor({{1, 2, 3}}, options.dtype(torch::kInt64));

  // shape: [1, 3, 5]
  auto draft_probs = torch::tensor({{0.2104, 0.2163, 0.1912, 0.1937, 0.1884},
                                    {0.2100, 0.1803, 0.2398, 0.2088, 0.1610},
                                    {0.1838, 0.2079, 0.2270, 0.2451, 0.1362}},
                                   options)
                         .reshape({1, 3, 5});
  // shape: [1, 3, 5]
  auto target_probs = torch::tensor({{0.1299, 0.2462, 0.1821, 0.1354, 0.3064},
                                     {0.1159, 0.2839, 0.1603, 0.2451, 0.1949},
                                     {0.0002, 0.0433, 0.6629, 0.1469, 0.1467}},
                                    options)
                          .reshape({1, 3, 5});

  // selected_target_probs:  [0.2462  0.1603  0.1469]
  // selected_draft_probs:   [0.2163  0.2398  0.2451]
  // acceptance_probs:       [1.1382  0.6685  0.5993]
  // uniform_rand:           [0.4785  0.6589  0.9399]
  // accepted:               [  1        1       0  ]
  auto uniform_rand = torch::tensor({{0.4785, 0.6589, 0.9399}}, options);
  auto bonus_token_ids = torch::tensor({{5}}, options.dtype(torch::kInt64));
  DraftProposal draft_proposal = DraftProposal(draft_token_ids, draft_probs);

  auto masked_output = RejectionSampler::random_sample(
      draft_proposal, target_probs, uniform_rand, bonus_token_ids);
  auto desired_masked_output =
      torch::tensor({{1, 2, 2, -1}}, options.dtype(torch::kInt64));
  EXPECT_TRUE(torch::allclose(masked_output, desired_masked_output));
}

TEST(RejectionSamplerTest, GreedyDraftUsesDeltaProposalDistribution) {
  const auto options = get_test_options(torch::kFloat32);

  const auto draft_token_ids =
      torch::tensor({{0, 1, 2, 1}}, options.dtype(torch::kInt64));

  const auto target_probs = torch::tensor({{{0.8f, 0.2f, 0.0f},
                                            {0.0f, 0.2f, 0.8f},
                                            {1.0f, 0.0f, 0.0f},
                                            {0.0f, 1.0f, 0.0f}}},
                                          options);
  const auto uniform_rand = torch::tensor({{0.5f, 0.5f, 0.5f, 0.5f}}, options);
  const auto bonus_token_ids =
      torch::tensor({{2}}, options.dtype(torch::kInt64));
  DraftProposal draft_proposal = DraftProposal(draft_token_ids);

  auto masked_output = RejectionSampler::random_sample(
      draft_proposal, target_probs, uniform_rand, bonus_token_ids);
  auto desired_masked_output =
      torch::tensor({{0, 2, -1, -1, -1}}, options.dtype(torch::kInt64));
  EXPECT_TRUE(torch::allclose(masked_output, desired_masked_output));
}

TEST(RejectionSamplerTest, Mask) {
  // test accepted mask
  const auto options = get_test_options(torch::kBool);

  // clang-format off
  auto accepted = torch::tensor({
        {0, 1, 0, 1},
        {1, 0, 1, 1},
        {1, 1, 0, 1},
        {1, 1, 1, 1}},
        options);
  auto desired_mask = torch::tensor({
        {1, 0, 0, 0, 0},
        {1, 1, 0, 0, 0},
        {1, 1, 1, 0, 0},
        {1, 1, 1, 1, 1}},
        options);
  // clang-format on
  auto mask = RejectionSampler::build_accepted_mask(accepted);
  EXPECT_TRUE(torch::allclose(mask, desired_mask));
}

TEST(RejectionSamplerTest, Greedy) {
  const auto options = get_test_options(torch::kFloat32);

  int64_t batch_size = 2;
  int64_t n_speculative_tokens = 3;
  int64_t vocab_size = 4;
  int64_t n_bonus_tokens = 1;

  torch::Tensor full_logits = torch::randn(
      {batch_size, n_speculative_tokens + n_bonus_tokens, vocab_size}, options);
  const auto draft_token_ids =
      full_logits.argmax(/*dim=*/-1)
          .slice(/*dim=*/-1, /*start=*/0, /*end=*/n_speculative_tokens);

  const RejectionSampler sampler =
      make_greedy_rejection_sampler(batch_size, torch::kCPU);
  SampleOutput output =
      sampler.forward(DraftProposal(draft_token_ids), full_logits);

  EXPECT_TRUE(
      torch::allclose(output.next_tokens, full_logits.argmax(/*dim=*/-1)));
}

TEST(RejectionSamplerTest, GreedyFromTokenIds) {
  const auto device = get_test_device();
  const auto options = torch::dtype(torch::kFloat32).device(device);
  const auto draft_token_ids = torch::tensor({{1, 2, 3}, {3, 1, 0}}, device);
  const auto target_token_ids = torch::tensor({{1, 4, 3}, {3, 2, 0}}, device);
  const auto bonus_token_ids = torch::tensor({{5}, {6}}, device);

  // One-hot [B, n + 1, V] logits whose argmax reproduces target + bonus.
  const int64_t vocab_size = 7;
  torch::Tensor target_logits = torch::zeros({2, 4, vocab_size}, options);
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t col = 0; col < 3; ++col) {
      target_logits.index_put_({row, col, target_token_ids.index({row, col})},
                               10.0f);
    }
    target_logits.index_put_({row, 3, bonus_token_ids.index({row, 0})}, 10.0f);
  }
  const RejectionSampler sampler = make_greedy_rejection_sampler(2, device);
  SampleOutput output =
      sampler.forward(DraftProposal(draft_token_ids), target_logits);

  EXPECT_TRUE(
      torch::equal(output.next_tokens,
                   torch::tensor({{1, 4, -1, -1}, {3, 2, -1, -1}}, device)));
}

TEST(RejectionSamplerTest, GreedyForwardAllowsUndefinedDraftProbs) {
  const auto options = get_test_options(torch::kFloat32);
  const auto device = get_test_device();
  RejectionSampler sampler = make_greedy_rejection_sampler(2, device);
  const auto draft_token_ids = torch::tensor({{1, 2}, {3, 4}}, device);
  const auto target_logits = torch::tensor({{{0.0f, 4.0f, 1.0f, 2.0f, 3.0f},
                                             {0.0f, 1.0f, 2.0f, 4.0f, 3.0f},
                                             {4.0f, 0.0f, 1.0f, 2.0f, 3.0f}},
                                            {{0.0f, 1.0f, 2.0f, 4.0f, 3.0f},
                                             {0.0f, 1.0f, 2.0f, 3.0f, 4.0f},
                                             {0.0f, 4.0f, 1.0f, 2.0f, 3.0f}}},
                                           options);
  DraftProposal draft_proposal = DraftProposal(draft_token_ids);
  SampleOutput output = sampler.forward(draft_proposal, target_logits);

  EXPECT_TRUE(torch::equal(output.next_tokens,
                           torch::tensor({{1, 3, -1}, {3, 4, 1}}, device)));
}

TEST(RejectionSamplerTest, GreedyRejectsGrammarMaskedDraftToken) {
  const auto options = get_test_options(torch::kFloat32);

  // The JSON grammar mask removes draft token 0 from the target distribution.
  const auto draft_token_ids =
      torch::tensor({{0}}, options.dtype(torch::kInt64));

  const torch::Tensor target_logits =
      torch::tensor({{{0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 2.0F}}}, options);
  const RejectionSampler sampler =
      make_greedy_rejection_sampler(1, torch::kCPU);
  SampleOutput output =
      sampler.forward(DraftProposal(draft_token_ids), target_logits);

  const auto expected_masked_output =
      torch::tensor({{1, -1}}, options.dtype(torch::kInt64));
  EXPECT_TRUE(torch::equal(output.next_tokens, expected_masked_output));
}

TEST(RejectionSamplerTest, RandomRejectsGrammarMaskedDraftToken) {
  const auto options = get_test_options(torch::kFloat32);

  // The JSON grammar mask removes draft token 0 from the target distribution.
  const auto draft_token_ids =
      torch::tensor({{0}}, options.dtype(torch::kInt64));
  const auto draft_probs = torch::tensor({{{1.0F, 0.0F, 0.0F}}}, options);
  const auto target_probs = torch::tensor({{{0.0F, 1.0F, 0.0F}}}, options);
  const auto uniform_rand = torch::tensor({{0.5F}}, options);
  const auto bonus_token_ids =
      torch::tensor({{2}}, options.dtype(torch::kInt64));

  DraftProposal draft_proposal = DraftProposal(draft_token_ids, draft_probs);
  auto masked_output = RejectionSampler::random_sample(
      draft_proposal, target_probs, uniform_rand, bonus_token_ids);

  const auto expected_masked_output =
      torch::tensor({{1, -1}}, options.dtype(torch::kInt64));
  EXPECT_TRUE(torch::equal(masked_output, expected_masked_output));
}

TEST(RejectionSamplerTest, RandomForwardUsesGreedyDraftWithoutDraftProbs) {
  const auto options = get_test_options(torch::kFloat32);
  const auto device = get_test_device();
  const auto do_sample = torch::tensor({true}, device);
  RejectionSampler sampler(do_sample,
                           /*all_random_sample=*/true,
                           /*all_greedy_sample=*/false,
                           /*logprobs=*/false,
                           /*max_top_logprobs=*/0);

  DraftProposal draft_proposal = DraftProposal(torch::tensor({{1, 2}}, device));
  const auto target_logits = torch::tensor({{{-1000.0f, 1000.0f, -1000.0f},
                                             {-1000.0f, -1000.0f, 1000.0f},
                                             {1000.0f, -1000.0f, -1000.0f}}},
                                           options);
  auto output = sampler.forward(draft_proposal, target_logits);

  EXPECT_TRUE(
      torch::equal(output.next_tokens, torch::tensor({{1, 2, 0}}, device)));
}

TEST(RejectionSamplerTest, RandomFallsBackToUniformOnDegenerateResidual) {
  const auto options = get_test_options(torch::kFloat32);
  const auto device = get_test_device();

  const int64_t vocab_size = 3;
  const auto draft_token_ids = torch::tensor({{0}}, device);
  const auto draft_probs = torch::tensor({{{1.0f, 0.0f, 0.0f}}}, options);
  const auto target_probs = torch::tensor({{{1.0f, 0.0f, 0.0f}}}, options);
  const auto uniform_rand = torch::tensor({{1.0f}}, options);
  DraftProposal draft_proposal = DraftProposal(draft_token_ids, draft_probs);
  const auto bonus_token_ids = torch::tensor({{5}}, device);
  auto masked_output = RejectionSampler::random_sample(
      draft_proposal, target_probs, uniform_rand, bonus_token_ids);

  const auto resampled = masked_output.index({0, 0}).item<int64_t>();
  EXPECT_GE(resampled, 0);
  EXPECT_LT(resampled, vocab_size);
  EXPECT_EQ(masked_output.index({0, 1}).item<int64_t>(), -1);
}

TEST(RejectionSamplerTest, LogProbs) {
  const auto options = get_test_options(torch::kFloat32);
  const auto device = get_test_device();
  const auto do_sample = torch::tensor({false, true, false, true}, device);
  const int64_t max_top_logprobs = 2;
  RejectionSampler sampler(do_sample,
                           do_sample.all().item<bool>(),
                           !do_sample.any().item<bool>(),
                           /*logprobs=*/true,
                           max_top_logprobs);

  int64_t batch_size = 4;
  int64_t n_speculative_tokens = 4;
  int64_t vocab_size = 8;

  const auto draft_token_ids =
      torch::randint(0,
                     vocab_size,
                     {batch_size, n_speculative_tokens},
                     torch::dtype(torch::kInt64).device(device));
  auto draft_probs =
      torch::randn({batch_size, n_speculative_tokens, vocab_size}, options)
          .softmax(/*dim=*/-1);

  auto target_logits =
      torch::randn({batch_size, n_speculative_tokens + 1, vocab_size}, options);
  const auto bonus_token_ids =
      torch::randint(0,
                     vocab_size,
                     {batch_size, 1},
                     torch::dtype(torch::kInt64).device(device));

  DraftProposal draft_proposal = DraftProposal(draft_token_ids, draft_probs);
  auto output = sampler.forward(draft_proposal, target_logits);

  const auto logprobs =
      torch::log_softmax(target_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  const auto selected_tokens = output.next_tokens;

  auto expected_logprobs =
      logprobs
          .gather(/*dim=*/-1,
                  selected_tokens.clamp_min(0).unsqueeze(/*dim=*/-1))
          .squeeze(/*dim=*/-1);
  expected_logprobs.masked_fill_(selected_tokens.lt(0), 0);
  EXPECT_TRUE(torch::equal(output.logprobs, expected_logprobs));

  auto [top_k_values, top_k_indices] = logprobs.topk(
      max_top_logprobs, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
  EXPECT_TRUE(torch::equal(output.top_logprobs, top_k_values));
  EXPECT_TRUE(torch::equal(output.top_tokens, top_k_indices));
}

TEST(RejectionSamplerTest, ConstructorDoesNotMutateDoSampleShape) {
  const auto device = get_test_device();
  auto do_sample = torch::tensor({false, true}, torch::device(device));

  ASSERT_EQ(do_sample.dim(), 1);
  ASSERT_EQ(do_sample.sizes(), torch::IntArrayRef({2}));

  RejectionSampler sampler(do_sample,
                           do_sample.all().item<bool>(),
                           !do_sample.any().item<bool>(),
                           /*logprobs=*/false,
                           /*max_top_logprobs=*/0);

  EXPECT_EQ(do_sample.dim(), 1);
  EXPECT_EQ(do_sample.sizes(), torch::IntArrayRef({2}));
}

TEST(RejectionSamplerTest,
     ReusingDoSampleAfterRejectionSamplerKeepsSamplerOutput1D) {
  const auto options = get_test_options(torch::kFloat32);
  const auto device = get_test_device();
  auto do_sample = torch::tensor({false, true}, torch::device(device));

  RejectionSampler rejection_sampler(do_sample,
                                     do_sample.all().item<bool>(),
                                     !do_sample.any().item<bool>(),
                                     /*logprobs=*/false,
                                     /*max_top_logprobs=*/0);
  (void)rejection_sampler;

  SamplingParameters params;
  params.selected_token_idxes =
      torch::tensor({0, 1}, torch::dtype(torch::kInt64).device(device));
  params.sample_idxes =
      torch::tensor({0, 1}, torch::dtype(torch::kInt64).device(device));
  params.do_sample = do_sample;
  params.all_random_sample = false;
  params.all_greedy_sample = false;

  auto logits =
      torch::tensor({{3.0f, 1.0f, 0.5f}, {0.1f, 0.2f, 4.0f}}, options);
  auto output = Sampler().forward(logits, params);

  EXPECT_EQ(output.probs.dim(), 2);
  EXPECT_EQ(output.probs.size(0), 2);
  EXPECT_EQ(output.next_tokens.dim(), 1);
  EXPECT_EQ(output.next_tokens.size(0), 2);
}

TEST(RejectionSamplerTest, Random) {
  const auto options = get_test_options(torch::kFloat32);

  // set random seed
  torch::manual_seed(100);

  int64_t vocab_size = 50;
  int64_t num_samples = 500000;

  auto target_prob = torch::randn({vocab_size}, options).softmax(/*dim=*/-1);
  auto target_probs =
      target_prob.reshape({1, 1, -1}).repeat({num_samples, 1, 1});

  auto draft_probs =
      torch::randn({num_samples, 1, vocab_size}, options).softmax(/*dim=*/-1);
  auto draft_token_ids = Sampler::random_sample(draft_probs);

  // not used
  auto bonus_token_ids =
      torch::ones({num_samples, 1}, options.dtype(torch::kInt64));

  auto uniform_rand = torch::rand(draft_token_ids.sizes(), options);
  DraftProposal draft_proposal = DraftProposal(draft_token_ids, draft_probs);
  auto output = RejectionSampler::random_sample(
      draft_proposal, target_probs, uniform_rand, bonus_token_ids);

  auto accepted_ids = output.slice(/*dim=*/-1, /*start=*/0, /*end=*/-1);
  accepted_ids = accepted_ids.masked_select(accepted_ids.ge(0));

  // calculate the probability of each sampled token
  auto bincount = accepted_ids.bincount(/*weights=*/torch::nullopt,
                                        /*minlength=*/vocab_size);
  auto sample_prob = bincount.to(torch::kFloat) / num_samples;

  EXPECT_TRUE(torch::allclose(target_prob,
                              sample_prob,
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2));
}

}  // namespace xllm
