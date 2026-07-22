/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <vector>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

class TileLangSpecVerifyMetadataUpdateTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }
  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

TEST_F(TileLangSpecVerifyMetadataUpdateTest, ReportsCompiledLayouts) {
  EXPECT_TRUE(has_spec_verify_metadata_update_specialization(4, 64));
  EXPECT_TRUE(has_spec_verify_metadata_update_specialization(5, 64));
  EXPECT_TRUE(has_spec_verify_metadata_update_specialization(6, 64));
  EXPECT_FALSE(has_spec_verify_metadata_update_specialization(3, 64));
  EXPECT_FALSE(has_spec_verify_metadata_update_specialization(6, 128));
}

TEST_F(TileLangSpecVerifyMetadataUpdateTest, WritesCompletePersistentState) {
  const auto device = torch::Device("npu:0");
  const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto positions = torch::arange(125, 131, i32);
  auto kv_seq_lens = torch::tensor({130}, i32);
  auto slots = torch::arange(1000, 1006, i32);
  auto block_table = torch::arange(2000, 2064, i32);
  auto linear_state = torch::tensor({17}, i32);
  auto num_accepted = torch::tensor({4}, i32);

  auto persistent_positions = torch::full({3, 8}, -1, i32);
  auto persistent_q = torch::full({1}, -1, i32);
  auto persistent_kv = torch::full({1}, -1, i32);
  auto persistent_slots = torch::full({8}, -1, i32);
  auto persistent_block = torch::full({64}, -1, i32);
  auto persistent_linear = torch::full({1}, -1, i32);
  auto persistent_accepted = torch::full({1}, -1, i32);
  auto persistent_q_cu = torch::full({2}, -1, i32);
  auto persistent_expanded_kv = torch::full({8}, -1, i32);
  auto persistent_expanded_block = torch::full({6, 64}, -1, i32);

  std::vector<torch::Tensor> position_rows;
  std::vector<torch::Tensor> expanded_block_rows;
  for (int64_t i = 0; i < 3; ++i) {
    position_rows.emplace_back(persistent_positions.select(0, i));
  }
  for (int64_t i = 0; i < 6; ++i) {
    expanded_block_rows.emplace_back(persistent_expanded_block.select(0, i));
  }

  spec_verify_metadata_update(positions,
                              kv_seq_lens,
                              slots,
                              block_table,
                              linear_state,
                              num_accepted,
                              position_rows,
                              persistent_q,
                              persistent_kv,
                              persistent_slots,
                              persistent_block,
                              persistent_linear,
                              persistent_accepted,
                              persistent_q_cu,
                              persistent_expanded_kv,
                              expanded_block_rows);
  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(torch::equal(
        persistent_positions.cpu()[i],
        torch::tensor({125, 126, 127, 128, 129, 130, 0, 0}, torch::kInt32)));
  }
  EXPECT_EQ(persistent_q.cpu()[0].item<int32_t>(), 6);
  EXPECT_EQ(persistent_kv.cpu()[0].item<int32_t>(), 130);
  EXPECT_TRUE(
      torch::equal(persistent_slots.cpu(),
                   torch::tensor({1000, 1001, 1002, 1003, 1004, 1005, 0, 0},
                                 torch::kInt32)));
  EXPECT_TRUE(torch::equal(persistent_block.cpu(), block_table.cpu()));
  EXPECT_EQ(persistent_linear.cpu()[0].item<int32_t>(), 17);
  EXPECT_EQ(persistent_accepted.cpu()[0].item<int32_t>(), 4);
  EXPECT_TRUE(torch::equal(persistent_q_cu.cpu(),
                           torch::tensor({0, 6}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      persistent_expanded_kv.cpu(),
      torch::tensor({125, 126, 127, 128, 129, 130, 0, 0}, torch::kInt32)));
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_TRUE(
        torch::equal(persistent_expanded_block.cpu()[i], block_table.cpu()));
  }
}

TEST_F(TileLangSpecVerifyMetadataUpdateTest, SupportsMtp3AndMtp4Widths) {
  const auto device = torch::Device("npu:0");
  const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
  for (const int64_t spec_width : {4, 5}) {
    auto positions = torch::arange(125, 125 + spec_width, i32);
    auto kv_seq_lens = torch::tensor({130}, i32);
    auto slots = torch::arange(1000, 1000 + spec_width, i32);
    auto block_table = torch::arange(2000, 2064, i32);
    auto linear_state = torch::tensor({17}, i32);
    auto num_accepted = torch::tensor({2}, i32);
    auto persistent_positions = torch::full({3, 8}, -1, i32);
    auto persistent_q = torch::full({1}, -1, i32);
    auto persistent_kv = torch::full({1}, -1, i32);
    auto persistent_slots = torch::full({8}, -1, i32);
    auto persistent_block = torch::full({64}, -1, i32);
    auto persistent_linear = torch::full({1}, -1, i32);
    auto persistent_accepted = torch::full({1}, -1, i32);
    auto persistent_q_cu = torch::full({2}, -1, i32);
    auto persistent_expanded_kv = torch::full({8}, -1, i32);
    auto persistent_expanded_block = torch::full({6, 64}, -1, i32);
    std::vector<torch::Tensor> position_rows;
    std::vector<torch::Tensor> expanded_block_rows;
    for (int64_t row = 0; row < 3; ++row) {
      position_rows.emplace_back(persistent_positions.select(0, row));
    }
    for (int64_t row = 0; row < 6; ++row) {
      expanded_block_rows.emplace_back(
          persistent_expanded_block.select(0, row));
    }

    spec_verify_metadata_update(positions,
                                kv_seq_lens,
                                slots,
                                block_table,
                                linear_state,
                                num_accepted,
                                position_rows,
                                persistent_q,
                                persistent_kv,
                                persistent_slots,
                                persistent_block,
                                persistent_linear,
                                persistent_accepted,
                                persistent_q_cu,
                                persistent_expanded_kv,
                                expanded_block_rows);

    auto expected_positions = torch::zeros({8}, torch::kInt32);
    expected_positions.narrow(0, 0, spec_width).copy_(positions.cpu());
    auto expected_slots = torch::zeros({8}, torch::kInt32);
    expected_slots.narrow(0, 0, spec_width).copy_(slots.cpu());
    for (int64_t row = 0; row < 3; ++row) {
      EXPECT_TRUE(
          torch::equal(persistent_positions.cpu()[row], expected_positions));
    }
    EXPECT_TRUE(torch::equal(persistent_slots.cpu(), expected_slots));
    EXPECT_EQ(persistent_q.cpu()[0].item<int32_t>(), spec_width);
    EXPECT_EQ(persistent_q_cu.cpu()[1].item<int32_t>(), spec_width);
    for (int64_t row = 0; row < 6; ++row) {
      EXPECT_TRUE(torch::equal(persistent_expanded_block.cpu()[row],
                               block_table.cpu()));
    }
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
