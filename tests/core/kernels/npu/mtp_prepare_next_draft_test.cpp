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

#include "core/kernels/npu/tilelang/mtp_prepare_next_draft.h"

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <vector>

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

  // Compact rank-2 inputs carry only previous/current hidden rows. The
  // rejected-first previous row must still use the placeholder, not row 0.
  const torch::Tensor compact = torch::stack({accepted_embeddings_cpu[0][2],
                                              accepted_embeddings_cpu[0][3],
                                              accepted_embeddings_cpu[1][0],
                                              accepted_embeddings_cpu[1][1],
                                              accepted_embeddings_cpu[2][0],
                                              accepted_embeddings_cpu[2][0]});
  MtpPrepareNextDraftWorkspace workspace;
  for (const torch::ScalarType dtype : {torch::kBFloat16, torch::kFloat16}) {
    const auto compact_output =
        try_mtp_prepare_next_draft(accepted_tokens_cpu.to(npu_device),
                                   compact.to(dtype).to(npu_device),
                                   placeholder_cpu.to(dtype).to(npu_device),
                                   base_positions_cpu.to(npu_device),
                                   base_kv_seq_lens_cpu.to(npu_device),
                                   block_tables_cpu.to(npu_device),
                                   kBlockSize,
                                   &workspace);
    ASSERT_TRUE(compact_output.has_value());
    EXPECT_TRUE(torch::equal(compact_output->token_ids.cpu(), expected_tokens));
    EXPECT_TRUE(torch::equal(compact_output->embeddings.cpu(),
                             expected_embeddings.to(dtype)));
    EXPECT_TRUE(
        torch::equal(compact_output->positions.cpu(), expected_positions));
    EXPECT_TRUE(
        torch::equal(compact_output->cache_slots.cpu(), expected_slots));
    EXPECT_TRUE(
        torch::equal(compact_output->kv_seq_lens.cpu(),
                     torch::tensor({8, 9, 10, 11, 13, 14}, torch::kInt)));
  }
}

TEST_F(MtpPrepareNextDraftTest, ReusesOutputsAndConvertsInt64Metadata) {
  const torch::Device device("npu:0");
  const torch::Tensor tokens =
      torch::tensor({{10, 11, 12, 13}}, torch::kLong).to(device);
  const torch::Tensor hidden = torch::arange(32, torch::kFloat)
                                   .reshape({2, 16})
                                   .to(torch::kFloat16)
                                   .to(device);
  const torch::Tensor placeholder = torch::full({16}, -7, hidden.options());
  const torch::Tensor positions = torch::tensor({126}, torch::kLong).to(device);
  const torch::Tensor lengths = torch::tensor({127}, torch::kLong).to(device);
  const torch::Tensor tables =
      torch::tensor({{10, 20}}, torch::kLong).to(device);
  MtpPrepareNextDraftWorkspace workspace;
  const auto first = try_mtp_prepare_next_draft(tokens,
                                                hidden,
                                                placeholder,
                                                positions,
                                                lengths,
                                                tables,
                                                /*block_size=*/128,
                                                &workspace);
  ASSERT_TRUE(first.has_value());
  EXPECT_TRUE(torch::equal(first->cache_slots.cpu(),
                           torch::tensor({2561, 2562}, torch::kInt)));
  // Reuse the captured output addresses while changing acceptance and
  // exercising the zero slot for an out-of-range cache position.
  tokens.slice(/*dim=*/1, /*start=*/1).fill_(-1);
  positions.fill_(256);
  const auto second = try_mtp_prepare_next_draft(tokens,
                                                 hidden,
                                                 placeholder,
                                                 positions,
                                                 lengths,
                                                 tables,
                                                 /*block_size=*/128,
                                                 &workspace);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(first->token_ids.data_ptr(), second->token_ids.data_ptr());
  EXPECT_EQ(first->embeddings.data_ptr(), second->embeddings.data_ptr());
  EXPECT_EQ(first->positions.data_ptr(), second->positions.data_ptr());
  EXPECT_EQ(first->kv_seq_lens.data_ptr(), second->kv_seq_lens.data_ptr());
  EXPECT_EQ(first->cache_slots.data_ptr(), second->cache_slots.data_ptr());
  EXPECT_TRUE(torch::equal(second->token_ids.cpu(),
                           torch::tensor({10, 10}, torch::kInt)));
  EXPECT_TRUE(torch::equal(second->embeddings[0].cpu(), placeholder.cpu()));
  EXPECT_TRUE(torch::equal(second->embeddings[1].cpu(), hidden[1].cpu()));
  EXPECT_TRUE(torch::equal(second->positions.cpu(),
                           torch::tensor({256, 257}, torch::kInt)));
  EXPECT_TRUE(torch::equal(second->kv_seq_lens.cpu(),
                           torch::tensor({127, 128}, torch::kInt)));
  EXPECT_TRUE(
      torch::equal(second->cache_slots.cpu(), torch::zeros({2}, torch::kInt)));
}

TEST_F(MtpPrepareNextDraftTest, RefreshesSharedSlotsAcrossAcceptanceAndPages) {
  const torch::Device device("npu:0");
  const torch::Tensor tokens =
      torch::tensor({{10, 11, 12}, {20, -1, -1}, {30, -1, -1}}, torch::kLong)
          .to(device);
  const torch::Tensor positions =
      torch::tensor({4, 6, 4}, torch::kInt).to(device);
  const torch::Tensor lengths =
      torch::tensor({5, 7, 5}, torch::kInt).to(device);
  const torch::Tensor tables =
      torch::tensor({{10, 11, 12}, {20, 21, 22}, {0, 0, 0}}, torch::kInt)
          .to(device);
  const torch::Tensor shared_tables =
      torch::tensor({{60, 62, 64}, {100, 102, 104}, {60, 60, 60}}, torch::kInt)
          .to(device);
  MtpPrepareNextDraftWorkspace workspace;
  for (const bool compact : {false, true}) {
    const torch::Tensor hidden = torch::ones(
        compact ? std::vector<int64_t>{6, 16} : std::vector<int64_t>{3, 3, 16},
        tokens.options().dtype(torch::kBFloat16));
    const torch::Tensor placeholder = torch::zeros({16}, hidden.options());
    const auto output = try_mtp_prepare_next_draft(tokens,
                                                   hidden,
                                                   placeholder,
                                                   positions,
                                                   lengths,
                                                   tables,
                                                   /*block_size=*/4,
                                                   &workspace,
                                                   shared_tables);
    ASSERT_TRUE(output.has_value());
    EXPECT_TRUE(
        torch::equal(output->cache_slots.cpu(),
                     torch::tensor({46, 47, 88, 87, 2, 1}, torch::kInt)));
    EXPECT_TRUE(
        torch::equal(output->shared_cache_slots.cpu(),
                     torch::tensor({250, 251, 416, 411, -1, -1}, torch::kInt)));

    // Rejection moves the repair write into the following page. Reusing the
    // output addresses must refresh physical slots rather than retain a
    // template.
    const torch::Tensor shared_slots = output->shared_cache_slots;
    positions.fill_(10);
    const auto out_of_range = try_mtp_prepare_next_draft(tokens,
                                                         hidden,
                                                         placeholder,
                                                         positions,
                                                         lengths,
                                                         tables,
                                                         /*block_size=*/4,
                                                         &workspace,
                                                         shared_tables);
    ASSERT_TRUE(out_of_range.has_value());
    EXPECT_EQ(shared_slots.data_ptr(),
              out_of_range->shared_cache_slots.data_ptr());
    EXPECT_TRUE(
        torch::equal(out_of_range->shared_cache_slots.cpu(),
                     torch::tensor({-1, -1, -1, 419, -1, -1}, torch::kInt)));
    positions.copy_(torch::tensor({4, 6, 4}, torch::kInt).to(device));
  }
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

TEST_F(MtpPrepareNextDraftTest,
       PreparesVerifierPrefixWithSharedPagesAndPadding) {
  const torch::Device device("npu:0");
  const torch::Tensor positions = torch::tensor({4, 8}, torch::kInt).to(device);
  const torch::Tensor lengths = torch::tensor({5, 9}, torch::kInt).to(device);
  const torch::Tensor tables =
      torch::tensor({{10, 11, 12}, {0, 0, 0}}, torch::kInt)
          .repeat_interleave(3, /*dim=*/0)
          .to(device);
  const torch::Tensor shared_tables =
      torch::tensor({{60, 62, 64}, {60, 60, 60}}, torch::kInt)
          .repeat_interleave(3, /*dim=*/0)
          .to(device);
  for (const torch::ScalarType token_dtype : {torch::kLong, torch::kInt}) {
    const torch::Tensor tokens =
        torch::tensor({{10, 11, -1}, {20, -1, -1}}, torch::kLong)
            .to(token_dtype)
            .to(device);
    positions.copy_(torch::tensor({4, 8}, torch::kInt).to(device));
    lengths.copy_(torch::tensor({5, 9}, torch::kInt).to(device));
    const auto output = try_mtp_prepare_draft_prefix(tokens,
                                                     positions,
                                                     lengths,
                                                     tables,
                                                     /*block_size=*/4,
                                                     shared_tables);
    ASSERT_TRUE(output.has_value());
    EXPECT_TRUE(
        torch::equal(output->token_ids.cpu(),
                     torch::tensor({10, 11, 0, 20, 0, 0}, torch::kInt)));
    EXPECT_TRUE(torch::equal(output->positions.cpu(),
                             torch::tensor({4, 5, 6, 8, 9, 10}, torch::kInt)));
    EXPECT_TRUE(torch::equal(output->kv_seq_lens.cpu(),
                             torch::tensor({5, 6, 7, 9, 10, 11}, torch::kInt)));
    EXPECT_TRUE(
        torch::equal(output->cache_slots.cpu(),
                     torch::tensor({44, 45, 46, 0, 1, 2}, torch::kInt)));
    EXPECT_TRUE(
        torch::equal(output->shared_cache_slots.cpu(),
                     torch::tensor({248, 249, 250, -1, -1, -1}, torch::kInt)));
    EXPECT_TRUE(torch::equal(output->logits_indices.cpu(),
                             torch::tensor({1, 3}, torch::kInt)));

    positions.fill_(1);
    lengths.fill_(12);
    tokens.fill_(-1);
    tokens.select(/*dim=*/1, /*index=*/0).fill_(42);
    const auto boundary = try_mtp_prepare_draft_prefix(tokens,
                                                       positions,
                                                       lengths,
                                                       tables,
                                                       /*block_size=*/4,
                                                       shared_tables);
    ASSERT_TRUE(boundary.has_value());
    EXPECT_TRUE(torch::equal(boundary->cache_slots.cpu(),
                             torch::tensor({51, 0, 0, 3, 0, 0}, torch::kInt)));
    EXPECT_TRUE(
        torch::equal(boundary->shared_cache_slots.cpu(),
                     torch::tensor({259, -1, -1, -1, -1, -1}, torch::kInt)));
    EXPECT_TRUE(torch::equal(boundary->logits_indices.cpu(),
                             torch::tensor({0, 3}, torch::kInt)));
  }
}

}  // namespace
}  // namespace xllm::kernel::npu
