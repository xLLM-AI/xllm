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
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>

#include "core/framework/sampling/rejection_sampler.h"
#include "core/framework/sampling/sampler.h"

namespace xllm {
namespace {

SamplingParameters make_seeded_params(const torch::Tensor& seeds,
                                      const torch::Tensor& offsets) {
  const int64_t rows = seeds.numel();
  SamplingParameters params;
  params.selected_token_idxes = torch::arange(rows, torch::kInt);
  params.sample_idxes = torch::arange(rows, torch::kInt);
  params.do_sample = torch::ones({rows}, torch::kBool);
  params.all_greedy_sample = false;
  params.all_random_sample = true;
  params.seeds = seeds;
  params.seed_offsets = offsets;
  return params.to(torch::Device(torch::kPrivateUse1, 0), torch::kFloat32);
}

TEST(SeededSamplerTest, NpuRequestsAreReproducibleAndIsolated) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto& global_generator = at_npu::detail::getDefaultNPUGenerator(0);
  const torch::Tensor before = global_generator.get_state().clone();
  Sampler sampler;
  for (int64_t offset = 0; offset < 8; ++offset) {
    const auto single =
        make_seeded_params(torch::tensor({42}, torch::kLong),
                           torch::tensor({offset}, torch::kLong));
    const auto batch =
        make_seeded_params(torch::tensor({7, 42}, torch::kLong),
                           torch::tensor({int64_t{1}, offset}, torch::kLong));
    torch::Tensor logits =
        torch::zeros({1, 64}, torch::TensorOptions().device(device));
    torch::Tensor batch_logits = torch::zeros({2, 64}, logits.options());
    const auto first = sampler.forward(logits, single);
    const auto repeated = sampler.forward(logits, single);
    const auto batched = sampler.forward(batch_logits, batch);
    EXPECT_EQ(first.next_tokens[0].item<int64_t>(),
              repeated.next_tokens[0].item<int64_t>());
    EXPECT_EQ(first.next_tokens[0].item<int64_t>(),
              batched.next_tokens[1].item<int64_t>());
  }
  EXPECT_TRUE(torch::equal(before, global_generator.get_state()));
}

TEST(SeededSamplerTest, NpuMtpRejectionUsesIndependentRequestStreams) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto& global = at_npu::detail::getDefaultNPUGenerator(0);
  const torch::Tensor before = global.get_state().clone();
  const torch::Tensor logits =
      torch::zeros({2, 4, 16}, torch::TensorOptions().device(device));
  const torch::Tensor ids = torch::zeros(
      {2, 3}, torch::TensorOptions().device(device).dtype(torch::kLong));
  const torch::Tensor bonus = torch::full({2, 1}, 15, ids.options());
  const torch::Tensor seeds = torch::tensor({7, 42}, torch::kLong).to(device);
  const torch::Tensor offsets = torch::tensor({3, 9}, torch::kLong).to(device);
  const RejectionSampler batch(
      torch::tensor({true, true}).to(device), true, false, false, 0);
  const RejectionSampler single(
      torch::tensor({true}).to(device), true, false, false, 0);
  const DraftProposal proposal(ids, std::nullopt);
  const auto first =
      batch.forward(proposal, logits, bonus, true, seeds, offsets);
  const auto repeated =
      batch.forward(proposal, logits, bonus, true, seeds, offsets);
  const DraftProposal one(ids.narrow(0, 1, 1), std::nullopt);
  const auto alone = single.forward(one,
                                    logits.narrow(0, 1, 1),
                                    bonus.narrow(0, 1, 1),
                                    true,
                                    seeds.narrow(0, 1, 1),
                                    offsets.narrow(0, 1, 1));
  EXPECT_TRUE(
      torch::equal(first.next_tokens.cpu(), repeated.next_tokens.cpu()));
  EXPECT_TRUE(
      torch::equal(first.next_tokens[1].cpu(), alone.next_tokens[0].cpu()));
  EXPECT_TRUE(torch::equal(before, global.get_state()));
}

}  // namespace
}  // namespace xllm
