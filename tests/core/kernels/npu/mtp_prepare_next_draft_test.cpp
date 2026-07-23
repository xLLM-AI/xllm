/* Copyright 2026 The xLLM Authors.

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
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

class MtpPrepareNextDraftTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

TEST_F(MtpPrepareNextDraftTest, ProducesExpectedOutputsForMixedAcceptance) {
  constexpr int32_t kBlockSize = 4;
  constexpr int64_t kHiddenSize = 16;
  const torch::Tensor accepted_tokens_cpu = torch::tensor(
      {{10, 11, 12, 13}, {20, 21, -1, -1}, {30, -1, -1, -1}}, torch::kLong);
  const torch::Tensor accepted_embeddings_cpu =
      torch::arange(3 * 4 * kHiddenSize, torch::kFloat)
          .reshape({3, 4, kHiddenSize})
          .to(torch::kBFloat16);
  const torch::Tensor placeholder_cpu =
      torch::full({kHiddenSize}, -7.0, torch::kBFloat16);
  const torch::Tensor base_positions_cpu =
      torch::tensor({4, 8, 12}, torch::kInt);
  const torch::Tensor base_kv_seq_lens_cpu =
      torch::tensor({5, 9, 13}, torch::kInt);
  const torch::Tensor block_tables_cpu = torch::tensor(
      {{10, 11, 12, 13, 14}, {20, 21, 22, 23, 24}, {30, 31, 32, 33, 34}},
      torch::kInt);
  const torch::Device npu_device("npu:0");

  const auto output =
      try_mtp_prepare_next_draft(accepted_tokens_cpu.to(npu_device),
                                 accepted_embeddings_cpu.to(npu_device),
                                 placeholder_cpu.to(npu_device),
                                 base_positions_cpu.to(npu_device),
                                 base_kv_seq_lens_cpu.to(npu_device),
                                 block_tables_cpu.to(npu_device),
                                 kBlockSize);
  ASSERT_TRUE(output.has_value());

  const torch::Tensor expected_tokens =
      torch::tensor({12, 13, 20, 21, 30, 30}, torch::kInt);
  const torch::Tensor expected_embeddings =
      torch::stack({accepted_embeddings_cpu[0][2],
                    accepted_embeddings_cpu[0][3],
                    accepted_embeddings_cpu[1][0],
                    accepted_embeddings_cpu[1][1],
                    placeholder_cpu,
                    accepted_embeddings_cpu[2][0]});
  const torch::Tensor expected_positions =
      torch::tensor({7, 8, 9, 10, 12, 13}, torch::kInt);
  const torch::Tensor expected_kv_seq_lens =
      torch::tensor({9, 11, 14}, torch::kInt);
  const torch::Tensor expected_slots =
      torch::tensor({47, 48, 91, 90, 134, 133}, torch::kInt);

  EXPECT_TRUE(torch::equal(output->token_ids.cpu(), expected_tokens));
  EXPECT_TRUE(torch::equal(output->embeddings.cpu(), expected_embeddings));
  EXPECT_TRUE(torch::equal(output->positions.cpu(), expected_positions));
  EXPECT_TRUE(torch::equal(output->kv_seq_lens.cpu(), expected_kv_seq_lens));
  EXPECT_TRUE(torch::equal(output->cache_slots.cpu(), expected_slots));
}

TEST_F(MtpPrepareNextDraftTest, RejectsUnsupportedHostInputs) {
  const torch::Tensor tokens = torch::tensor({{1, 2}}, torch::kLong);
  const torch::Tensor embeddings = torch::zeros({1, 2, 16}, torch::kBFloat16);
  const torch::Tensor placeholder = torch::zeros({16}, torch::kBFloat16);
  const torch::Tensor positions = torch::tensor({1}, torch::kInt);
  const torch::Tensor kv_seq_lens = torch::tensor({2}, torch::kInt);
  const torch::Tensor block_tables = torch::tensor({{0, 1}}, torch::kInt);

  EXPECT_FALSE(try_mtp_prepare_next_draft(tokens,
                                          embeddings,
                                          placeholder,
                                          positions,
                                          kv_seq_lens,
                                          block_tables,
                                          /*block_size=*/4)
                   .has_value());
}

}  // namespace
}  // namespace xllm::kernel::npu
