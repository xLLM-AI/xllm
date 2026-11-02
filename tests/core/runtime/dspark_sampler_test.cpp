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
#include <string>
#include <vector>

#include "core/framework/batch/sampling_input_builder.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/speculative/rejection_sampler.h"
#include "core/runtime/dspark_worker_impl.h"

namespace xllm {
namespace {

class DSparkConstrainedProposalTest
    : public ::testing::TestWithParam<torch::Device> {
 protected:
  const torch::Device device_ = GetParam();
};

#if defined(XLLM_DSPARK_NPU_TEST)
// The request builder allocates pinned host tensors and needs the NPU runtime.
TEST_P(DSparkConstrainedProposalTest, DisjointRequestReturnsAllowedToken) {
  RequestSamplingParam request;
  request.vocab_size = 8;
  request.allowed_token_ids = std::vector<int32_t>{5};
  SamplingInputBuilder builder;
  builder.append(&request, /*token_index=*/0);
  const SamplingParameters params =
      builder.build().to(device_, torch::kFloat32);
  SamplingParameters draft_params = params;
  prepare_draft_sampling(true, draft_params);
  const DraftVocabMapping mapping(
      torch::tensor({1, 3, 4}, torch::kInt64), 3, 8, device_);
  const auto options = torch::TensorOptions().device(device_);
  const DraftProposal proposal = sample_dspark_markov_block(
      [&](const torch::Tensor&) { return torch::zeros({1, 3}, options); },
      torch::tensor({{{-1.0f, 4.0f, 8.0f}}}).to(device_),
      torch::full({1}, 7, options.dtype(torch::kInt64)),
      draft_params,
      [](torch::Tensor&) {},
      true,
      {},
      mapping);
  EXPECT_EQ(proposal.token_ids().item<int64_t>(), 1);
  EXPECT_EQ(proposal.distribution().logsumexp().item<float>(), 0);
  const SampleOutput verified =
      rejection_sample(params, proposal, torch::zeros({2, 8}, options));
  EXPECT_TRUE(torch::equal(verified.next_tokens.cpu(),
                           torch::tensor({{5, -1}}, torch::kInt64)));
}
#endif

TEST_P(DSparkConstrainedProposalTest, EmptySupportIsRepairedPerRow) {
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  const auto options = torch::TensorOptions().device(device_);
  const auto id_options = options.dtype(torch::kInt64);
  const DraftVocabMapping mapping(
      torch::tensor({1, 3, 4}, torch::kInt64), 3, 8, device_);
  const torch::Tensor logits =
      torch::tensor({{{-1.0f, 4.0f, 8.0f}}}).to(device_).repeat({3, 3, 1});
  const torch::Tensor anchors = torch::full({3}, 7, id_options);
  // Disjoint, partially overlapping, and fully contained draft support.
  torch::Tensor mask = torch::full({3, 8}, negative_infinity);
  mask[0][5] = 0;
  mask[1][4] = mask[1][5] = 0;
  mask[2][1] = mask[2][4] = mask[2][5] = mask[2][6] = 0;
  mask = mask.to(device_);
  torch::Tensor expected_q = torch::zeros({3, 8}, options);
  expected_q[0][1] = 1;
  expected_q[1][4] = 1;
  expected_q[2].index_copy_(
      0, mapping.draft_to_target_ids_, logits[2][0].softmax(-1));
  for (const std::string control :
       {"allowlist", "mask", "bitmask", "bad_words"}) {
    SCOPED_TRACE(control);
    SamplingParameters params;
    params.sampling_mode(SamplingMode::RANDOM);
    if (control == "allowlist") {
      // Request preparation represents allowed_token_ids as additive bias.
      params.logits_bias() = mask;
    } else if (control == "mask") {
      params.filter_mask() = mask;
    } else if (control == "bitmask") {
      params.filter_bitmask() =
          torch::tensor({{1 << 5},
                         {(1 << 4) | (1 << 5)},
                         {(1 << 1) | (1 << 4) | (1 << 5) | (1 << 6)}},
                        torch::kInt32)
              .to(device_);
    } else {
      params.bad_words() =
          torch::tensor({{{1}, {4}, {6}}, {{1}, {6}, {-1}}, {{-1}, {-1}, {-1}}},
                        torch::kInt64)
              .to(device_);
    }
    params.return_logits(true);
    std::vector<torch::Tensor> feedback;
    std::vector<torch::Tensor> synchronized;
    feedback.reserve(3);
    synchronized.reserve(3);
    const DraftProposal proposal = sample_dspark_markov_block(
        [&](const torch::Tensor& previous) {
          feedback.emplace_back(previous.clone());
          return torch::zeros({3, 3}, options);
        },
        logits,
        anchors,
        params,
        [&](torch::Tensor& tokens) {
          synchronized.emplace_back(tokens.clone());
        },
        true,
        {},
        mapping);
    const torch::Tensor lse = proposal.distribution().logsumexp();
    EXPECT_TRUE(lse.isfinite().all().item<bool>());
    EXPECT_TRUE(proposal.token_ids()[0].eq(1).all().item<bool>());
    EXPECT_TRUE(proposal.token_ids()[1].eq(4).all().item<bool>());
    for (int64_t step = 0; step < 3; ++step) {
      EXPECT_TRUE(torch::equal(
          feedback[step],
          step == 0 ? anchors : proposal.token_ids().select(1, step - 1)));
      EXPECT_TRUE(torch::equal(synchronized[step],
                               proposal.token_ids().select(1, step)));
      // p=1 exposes 1-q through the real residual implementation.
      torch::Tensor residual = torch::zeros({3, 8}, options);
      proposal.distribution().apply_residual(
          residual,
          torch::full({3, 1}, step, id_options),
          proposal.token_ids().narrow(1, step, 1),
          lse);
      EXPECT_TRUE(torch::allclose(residual.exp(), 1 - expected_q, 1e-5, 1e-6));
      EXPECT_TRUE(expected_q.gather(1, proposal.token_ids().narrow(1, step, 1))
                      .gt(0)
                      .all()
                      .item<bool>());
    }
    torch::Tensor acceptance = torch::zeros({3, 3}, options);
    proposal.distribution().apply_acceptance(
        acceptance, proposal.token_ids(), lse);
    EXPECT_TRUE(
        torch::allclose(acceptance,
                        -expected_q.gather(1, proposal.token_ids()).log(),
                        1e-5,
                        1e-6));

    // p has mass only at 5, outside q's support, so residual recovery must
    // return 5 in every row, including the repaired row.
    torch::Tensor target = torch::full({12, 8}, negative_infinity, options);
    target.select(1, 5).fill_(0);
    const SampleOutput verified = rejection_sample(params, proposal, target);
    EXPECT_TRUE(verified.next_tokens.select(1, 0).eq(5).all().item<bool>());
    EXPECT_TRUE(verified.next_tokens.narrow(1, 1, 3).eq(-1).all().item<bool>());
    EXPECT_TRUE(verified.num_accepted_tokens.eq(0).all().item<bool>());
  }
}

TEST_P(DSparkConstrainedProposalTest,
       LaterEmptyStepAndSentinelPreserveFeedback) {
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  const auto options = torch::TensorOptions().device(device_);
  const auto id_options = options.dtype(torch::kInt64);
  const DraftVocabMapping mapping(
      torch::tensor({1, 3, 4}, torch::kInt64), 3, 8, device_);
  const torch::Tensor logits =
      torch::tensor({{{negative_infinity, 0.0f, negative_infinity},
                      {0.0f, negative_infinity, 0.0f},
                      {negative_infinity, 0.0f, negative_infinity}}})
          .to(device_);
  SamplingParameters params;
  params.sampling_mode(SamplingMode::RANDOM);
  params.logits_bias() = torch::full({1, 8}, negative_infinity, options);
  params.logits_bias()[0][4] = params.logits_bias()[0][5] = 0;
  params.temperatures() = torch::full({1}, 0.7f, options);
  params.min_p() = torch::full({1}, 0.1f, options);
  params.top_k() = torch::full({1}, 2, id_options);
  params.top_p() = torch::full({1}, 0.9f, options);
  SamplingParameters draft_params = params;
  prepare_draft_sampling(true, draft_params);
  std::vector<torch::Tensor> feedback;
  feedback.reserve(3);
  torch::Tensor verify_tokens = torch::empty({1, 3}, id_options);
  const DraftProposal proposal = sample_dspark_markov_block(
      [&](const torch::Tensor& previous) {
        feedback.emplace_back(previous.clone());
        // Subsequent logits depend on the repaired previous token.
        return previous.to(torch::kFloat32).unsqueeze(1).expand({1, 3});
      },
      logits,
      torch::full({1}, 7, id_options),
      draft_params,
      [](torch::Tensor&) {},
      true,
      verify_tokens,
      mapping);
  EXPECT_TRUE(torch::equal(proposal.token_ids().cpu(),
                           torch::tensor({{4, 1, 4}}, torch::kInt64)));
  ASSERT_EQ(feedback.size(), 3u);
  EXPECT_EQ(feedback[1].item<int64_t>(), 4);
  EXPECT_EQ(feedback[2].item<int64_t>(), 1);
  const torch::Tensor lse = proposal.distribution().logsumexp();
  EXPECT_TRUE(lse.isfinite().all().item<bool>());
  EXPECT_EQ(lse[0][1].item<float>(), 0);
  for (const bool sentinel : {false, true}) {
    SCOPED_TRACE(sentinel);
    if (sentinel) {
      verify_tokens[0][1] = -1;
    }
    torch::Tensor target = torch::full({4, 8}, negative_infinity, options);
    target[0][4] = target[1][5] = target[2][4] = target[3][5] = 0;
    const SampleOutput verified = rejection_sample(params, proposal, target);
    EXPECT_TRUE(torch::equal(verified.next_tokens.cpu(),
                             torch::tensor({{4, 5, -1, -1}}, torch::kInt64)));
    EXPECT_EQ(verified.num_accepted_tokens.item<int32_t>(), 1);
  }
  EXPECT_TRUE(torch::equal(proposal.distribution().logsumexp(), lse));
}

TEST_P(DSparkConstrainedProposalTest, NormalRowsPreserveSeededDraws) {
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  const auto options = torch::TensorOptions().device(device_);
  const auto id_options = options.dtype(torch::kInt64);
  const DraftVocabMapping mapping(
      torch::tensor({1, 3, 4}, torch::kInt64), 3, 8, device_);
  const torch::Tensor base = torch::tensor({{{0.0f, 1.0f, 2.0f}}})
                                 .to(device_, torch::kBFloat16)
                                 .repeat({4, 1, 1});
  for (const SamplingMode mode :
       {SamplingMode::RANDOM, SamplingMode::MIXED, SamplingMode::GREEDY}) {
    SamplingParameters params;
    params.sampling_mode(mode).return_logits(true);
    params.seeds() = torch::arange(4, id_options).add(42);
    params.seed_offsets() = torch::zeros({4}, id_options);
    params.logits_bias() = torch::zeros({4, 8}, options);
    params.logits_bias()[0].fill_(negative_infinity);
    params.logits_bias()[0][5] = 0;
    if (mode == SamplingMode::MIXED) {
      params.do_sample() =
          torch::tensor({true, false, true, false}).to(device_);
    }
    torch::Tensor expected_logits =
        torch::full({4, 8}, negative_infinity, options);
    expected_logits.index_copy_(
        1, mapping.draft_to_target_ids_, base.select(1, 0).to(torch::kFloat32));
    const SampleOutput expected = sample(expected_logits, params);
    const DraftProposal proposal = sample_dspark_markov_block(
        [&](const torch::Tensor&) {
          return torch::zeros({4, 3}, base.options());
        },
        base,
        torch::zeros({4}, id_options),
        params,
        [](torch::Tensor&) {},
        true,
        {},
        mapping);
    EXPECT_EQ(proposal.token_ids()[0][0].item<int64_t>(), 1);
    EXPECT_TRUE(torch::equal(proposal.token_ids().narrow(0, 1, 3).squeeze(1),
                             expected.next_tokens.narrow(0, 1, 3)));
    const torch::Tensor lse = proposal.distribution().logsumexp();
    EXPECT_TRUE(lse.isfinite().all().item<bool>());
    EXPECT_TRUE(torch::equal(lse.narrow(0, 1, 3).squeeze(1),
                             expected_logits.narrow(0, 1, 3).logsumexp(-1)));
  }
}

#if defined(XLLM_DSPARK_NPU_TEST)
INSTANTIATE_TEST_SUITE_P(Npu,
                         DSparkConstrainedProposalTest,
                         ::testing::Values(torch::Device(torch::kPrivateUse1,
                                                         0)));
#else
INSTANTIATE_TEST_SUITE_P(Cpu,
                         DSparkConstrainedProposalTest,
                         ::testing::Values(torch::Device(torch::kCPU)));
#endif

}  // namespace

#if !defined(XLLM_DSPARK_NPU_TEST)
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
    const torch::Tensor lse = proposal.distribution().logsumexp();
    EXPECT_EQ(lse.defined(), probabilistic);
    for (int64_t step = 0; step < 2; ++step) {
      EXPECT_TRUE(mapping.eq(proposal.token_ids()[0][step]).any().item<bool>());
      if (!probabilistic) {
        continue;
      }
      torch::Tensor expected = torch::zeros({1, 8});
      expected.index_copy_(
          1, mapping, torch::softmax(logits.select(1, step), -1));
      torch::Tensor residual = torch::zeros({1, 8});
      proposal.distribution().apply_residual(
          residual,
          torch::full({1, 1}, step, torch::kInt64),
          proposal.token_ids().narrow(1, step, 1),
          lse);
      EXPECT_TRUE(torch::allclose(residual.exp(), 1 - expected));
    }
  }
}

TEST(DSparkMarkovSamplerTest, ReducedVocabularyAppliesTargetSamplingControls) {
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  const torch::Tensor logits =
      torch::tensor({{{-1.0f, 4.0f, 8.0f}, {-1.0f, 4.0f, 8.0f}}});
  const DraftVocabMapping mapping(torch::tensor({1, 3, 4}, torch::kInt64),
                                  3,
                                  8,
                                  torch::Device(torch::kCPU));
  for (const std::string control : {"bias", "mask", "bitmask", "penalties"}) {
    SCOPED_TRACE(control);
    for (const bool probabilistic : {false, true}) {
      SCOPED_TRACE(probabilistic);
      SamplingParameters sampling;
      sampling.sampling_mode(SamplingMode::RANDOM);
      torch::Tensor expected_logits = torch::full({1, 8}, negative_infinity);
      expected_logits[0][1] = -1.0f;
      expected_logits[0][4] = 4.0f;
      expected_logits[0][6] = 8.0f;
      int64_t expected_token = 4;
      if (control == "bias") {
        sampling.logits_bias() = torch::zeros({1, 8});
        sampling.logits_bias()[0][1] = 10.0f;
        sampling.logits_bias()[0][7] = 100.0f;
        expected_logits[0][1] = 9.0f;
        expected_token = 1;
      } else if (control == "mask" || control == "bitmask") {
        if (control == "mask") {
          sampling.filter_mask() = torch::full({1, 8}, negative_infinity);
          sampling.filter_mask()[0][4] = 0.0f;
        } else {
          sampling.filter_bitmask() = torch::tensor({{1 << 4}}, torch::kInt32);
        }
        expected_logits[0][1] = negative_infinity;
        expected_logits[0][6] = negative_infinity;
      } else {
        // Include a target ID absent from the draft vocabulary as well as a
        // mapped ID outside the draft logits' width.
        sampling.unique_token_ids() = torch::tensor({{6, 7}}, torch::kInt64);
        sampling.unique_token_counts() = torch::tensor({{2, 1}}, torch::kInt32);
        sampling.unique_token_ids_lens() = torch::tensor({2}, torch::kInt32);
        sampling.repetition_penalties() = torch::tensor({2.0f});
        sampling.frequency_penalties() = torch::tensor({1.0f});
        sampling.presence_penalties() = torch::tensor({1.0f});
        expected_logits[0][6] = 1.0f;
      }
      prepare_draft_sampling(probabilistic, sampling);
      const torch::Tensor anchors = torch::tensor({7}, torch::kInt64);
      torch::Tensor previous_tokens = anchors;
      torch::Tensor output = torch::empty({1, 2}, torch::kInt64);
      const DraftProposal proposal = sample_dspark_markov_block(
          [&previous_tokens](const torch::Tensor& previous) {
            previous_tokens = previous.clone();
            return torch::zeros({1, 3});
          },
          logits,
          anchors,
          sampling,
          [](torch::Tensor&) {},
          probabilistic,
          output,
          mapping);
      EXPECT_TRUE(torch::equal(output, proposal.token_ids()));
      EXPECT_TRUE(torch::equal(previous_tokens, output.select(1, 0)));
      if (!probabilistic) {
        EXPECT_TRUE(output.eq(expected_token).all().item<bool>());
        continue;
      }
      const torch::Tensor lse = proposal.distribution().logsumexp();
      for (int64_t step = 0; step < 2; ++step) {
        torch::Tensor residual = torch::zeros({1, 8});
        proposal.distribution().apply_residual(
            residual,
            torch::full({1, 1}, step, torch::kInt64),
            output.narrow(1, step, 1),
            lse);
        EXPECT_TRUE(torch::allclose(residual.exp(),
                                    1 - torch::softmax(expected_logits, -1)));
        EXPECT_TRUE(expected_logits[0][output[0][step].item<int64_t>()]
                        .isfinite()
                        .item<bool>());
      }
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

#endif

}  // namespace xllm
