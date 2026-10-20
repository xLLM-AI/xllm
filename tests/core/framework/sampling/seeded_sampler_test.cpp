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

#include "core/framework/sampling/sampler.h"

namespace xllm {
namespace {

SamplingParameters make_seeded_params(const torch::Tensor& seeds,
                                      const torch::Tensor& offsets) {
  const int64_t rows = seeds.numel();
  SamplingParameters params;
  params.selected_token_idxes = torch::arange(rows, torch::kInt);
  params.sampling_mode = SamplingMode::RANDOM;
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

}  // namespace
}  // namespace xllm
