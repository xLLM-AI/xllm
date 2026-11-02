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

#include <cmath>
#include <mutex>
#include <tuple>
#include <vector>

#include "core/platform/platform.h"
#include "core/runtime/mtp_worker_impl.h"
#include "core/util/scope_guard.h"

namespace xllm {
namespace {

class ValidationWorker final : public MTPWorkerImpl<LlmForwardInput> {
 public:
  using MTPWorkerImpl<LlmForwardInput>::MTPWorkerImpl;
  using MTPWorkerImpl<LlmForwardInput>::validate;
};

class MtpValidationTest
    : public ::testing::TestWithParam<
          std::tuple<SamplingMode, bool, torch::ScalarType>> {};

TEST_P(MtpValidationTest, MarksOnlyLocalVerificationTokens) {
  const auto [mode, probabilistic, token_dtype] = GetParam();
  const torch::Device device(Platform::type_torch(), /*index=*/0);
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  runtime::Options options;
  options.model_id("test_target")
      .block_size(4)
      .num_speculative_tokens(2)
      .world_size(1)
      .dp_size(1)
      .cp_size(1);
  ValidationWorker worker(parallel_args, device, options, WorkerType::LLM);
  const auto tensor_options = torch::TensorOptions().device(device);
  std::vector<ForwardOutput> drafts(2);
  for (int64_t step = 0; step < 2; ++step) {
    drafts[step].sample_output.next_tokens =
        torch::full({2}, step + 1, tensor_options.dtype(token_dtype));
    if (probabilistic) {
      drafts[step].sample_output.draft_logits =
          torch::full({2, 4}, -INFINITY, tensor_options);
      drafts[step].sample_output.draft_logits.select(1, step + 1).fill_(0);
    }
  }
  ForwardOutput target;
  target.logits =
      torch::full({6, 4}, -INFINITY, tensor_options)
          .scatter_(1,
                    torch::tensor({{1}, {2}, {3}, {1}, {2}, {3}},
                                  tensor_options.dtype(torch::kInt64)),
                    0.0F);
  SamplingParameters sampling;
  sampling.sampling_mode() = mode;
  if (mode == SamplingMode::MIXED) {
    sampling.do_sample() =
        torch::tensor({false, true}, tensor_options.dtype(torch::kBool));
  }

  const SampleOutput constrained = worker.validate(
      sampling, drafts, target, /*num_speculative_tokens=*/2, {1, 0, 0, 1});
  EXPECT_TRUE(
      torch::equal(constrained.next_tokens.cpu(),
                   torch::tensor({{1, -1, -1}, {1, 2, -1}}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(constrained.num_accepted_tokens.cpu(),
                           torch::tensor({{0}, {1}}, torch::kInt32)));

  // Reusing the original draft outputs must still accept the entire prefix.
  const SampleOutput unconstrained = worker.validate(
      sampling, drafts, target, /*num_speculative_tokens=*/2, {});
  EXPECT_TRUE(
      torch::equal(unconstrained.next_tokens.cpu(),
                   torch::tensor({{1, 2, 3}, {1, 2, 3}}, torch::kInt64)));
  for (int64_t step = 0; step < 2; ++step) {
    EXPECT_TRUE(torch::equal(drafts[step].sample_output.next_tokens.cpu(),
                             torch::full({2}, step + 1, token_dtype)));
  }
}

TEST(MtpValidationDistributionTest, PreservesNonPointMassDraftAcceptance) {
  constexpr int64_t kBatchSize = 32;
  const torch::Device device(Platform::type_torch(), /*index=*/0);
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  runtime::Options options;
  options.model_id("test_target")
      .block_size(4)
      .num_speculative_tokens(2)
      .world_size(1)
      .dp_size(1)
      .cp_size(1);
  ValidationWorker worker(parallel_args, device, options, WorkerType::LLM);
  const auto tensor_options =
      torch::TensorOptions().device(device).dtype(torch::kFloat32);
  std::vector<ForwardOutput> drafts(2);
  for (int64_t step = 0; step < 2; ++step) {
    drafts[step].sample_output.next_tokens = torch::full(
        {kBatchSize}, step + 1, tensor_options.dtype(torch::kInt64));
    drafts[step].sample_output.draft_logits =
        torch::zeros({kBatchSize, 4}, tensor_options);
  }
  // At both draft positions q(token) = 1/4 and p(token) = 1/2, so
  // Dense must accept every proposal while PointMass can reject it.
  torch::Tensor target_logits =
      torch::full({kBatchSize, 3, 4}, -INFINITY, tensor_options);
  target_logits.narrow(1, 0, 2).select(2, 0).fill_(0);
  target_logits.select(1, 0).select(1, 1).fill_(0);
  target_logits.select(1, 1).select(1, 2).fill_(0);
  target_logits.select(1, 2).select(1, 3).fill_(0);
  ForwardOutput target;
  target.logits = target_logits.view({kBatchSize * 3, 4});
  SamplingParameters sampling;
  sampling.sampling_mode() = SamplingMode::RANDOM;

  // Seed the device RNG, not the request: request seeds bypass p/q rejection.
  auto generator = at_npu::detail::getDefaultNPUGenerator(/*device_index=*/0);
  torch::Tensor rng_state;
  {
    std::lock_guard<std::mutex> lock(generator.mutex());
    rng_state = generator.get_state();
    generator.set_current_seed(42);
  }
  ScopeGuard restore_rng([&] {
    std::lock_guard<std::mutex> lock(generator.mutex());
    generator.set_state(rng_state);
  });

  const SampleOutput dense = worker.validate(
      sampling, drafts, target, /*num_speculative_tokens=*/2, {});
  EXPECT_TRUE(torch::equal(dense.num_accepted_tokens.cpu(),
                           torch::full({kBatchSize, 1}, 2, torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      dense.next_tokens.narrow(1, 0, 2).cpu(),
      torch::tensor({{1, 2}}, torch::kInt64).expand({kBatchSize, 2})));

  // Confirm this draw distinguishes absent logits from the Dense proposal.
  for (auto& draft : drafts) {
    draft.sample_output.draft_logits = torch::Tensor();
  }
  {
    std::lock_guard<std::mutex> lock(generator.mutex());
    generator.set_current_seed(42);
  }
  const SampleOutput point_mass = worker.validate(
      sampling, drafts, target, /*num_speculative_tokens=*/2, {});
  EXPECT_LT(point_mass.num_accepted_tokens.min().item<int32_t>(), 2);
}

INSTANTIATE_TEST_SUITE_P(
    DraftRepresentations,
    MtpValidationTest,
    ::testing::Combine(::testing::Values(SamplingMode::GREEDY,
                                         SamplingMode::RANDOM,
                                         SamplingMode::MIXED),
                       ::testing::Bool(),
                       ::testing::Values(torch::kInt32, torch::kInt64)));

}  // namespace
}  // namespace xllm
