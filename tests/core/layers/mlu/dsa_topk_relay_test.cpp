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

#include "core/layers/mlu/dsa_topk_relay.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm::layer {
namespace {

TEST(DsaTopkRelayTest, PublishedStateIsReusedAsOneValue) {
  DsaTopkRelay relay;
  const DsaTopkShareDecision publish_decision{
      .reuse_topk = false,
      .output_topk = true,
  };
  std::optional<DsaTopkTransfer> publish_transfer =
      relay.prepare_layer(publish_decision);
  ASSERT_TRUE(publish_transfer.has_value());
  EXPECT_EQ(publish_transfer->input(), nullptr);
  EXPECT_TRUE(publish_transfer->captures_output());

  const torch::Tensor block_tables =
      torch::tensor({{1, 2}, {3, 4}}, torch::dtype(torch::kInt32));
  const torch::Tensor context_lens =
      torch::tensor({2, 1}, torch::dtype(torch::kInt32));
  publish_transfer->publish_output(DsaTopkState(block_tables, context_lens));
  relay.finish_layer(publish_decision, *publish_transfer);

  const DsaTopkShareDecision reuse_decision{
      .reuse_topk = true,
      .output_topk = false,
  };
  std::optional<DsaTopkTransfer> reuse_transfer =
      relay.prepare_layer(reuse_decision);
  ASSERT_TRUE(reuse_transfer.has_value());
  ASSERT_NE(reuse_transfer->input(), nullptr);
  EXPECT_FALSE(reuse_transfer->captures_output());
  EXPECT_TRUE(
      torch::equal(reuse_transfer->input()->block_tables(), block_tables));
  EXPECT_TRUE(
      torch::equal(reuse_transfer->input()->context_lens(), context_lens));
}

TEST(DsaTopkRelayTest, ReuseBeforePublishFails) {
  DsaTopkRelay relay;
  const DsaTopkShareDecision reuse_decision{
      .reuse_topk = true,
      .output_topk = false,
  };

  EXPECT_DEATH(
      {
        const std::optional<DsaTopkTransfer> transfer =
            relay.prepare_layer(reuse_decision);
        static_cast<void>(transfer);
      },
      "requires a previously published state");
}

TEST(DsaTopkStateTest, FlattenedStatePreservesSparseMetadataStorage) {
  const torch::Tensor block_tables =
      torch::tensor({{{1, 2}}, {{3, 4}}}, torch::dtype(torch::kInt32));
  const torch::Tensor context_lens =
      torch::tensor({2, 1}, torch::dtype(torch::kInt32));

  const DsaTopkState state(block_tables, context_lens);
  const DsaTopkState flattened = state.flattened();
  EXPECT_TRUE(
      torch::equal(flattened.block_tables(), block_tables.reshape({2, 2})));
  EXPECT_TRUE(torch::equal(flattened.context_lens(), context_lens));
  EXPECT_EQ(flattened.block_tables().data_ptr(), block_tables.data_ptr());
  EXPECT_EQ(flattened.context_lens().data_ptr(), context_lens.data_ptr());
}

TEST(DsaTopkTransferTest, EmptyMtpStateCapturesStateForNextStep) {
  DsaTopkTransfer transfer = DsaTopkTransfer::prepare_mtp_step(
      std::nullopt, torch::Device(torch::kCPU));
  EXPECT_EQ(transfer.input(), nullptr);
  EXPECT_TRUE(transfer.captures_output());
  EXPECT_FALSE(transfer.mtp_output_state().has_value());

  const torch::Tensor block_tables =
      torch::tensor({{1, 2}, {3, 4}}, torch::dtype(torch::kInt32));
  const torch::Tensor context_lens =
      torch::tensor({2, 1}, torch::dtype(torch::kInt32));
  transfer.publish_output(DsaTopkState(block_tables, context_lens));

  const std::optional<DsaTopkState> output = transfer.mtp_output_state();
  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->block_tables().data_ptr(), block_tables.data_ptr());
  EXPECT_EQ(output->context_lens().data_ptr(), context_lens.data_ptr());
}

TEST(DsaTopkTransferTest, ExistingMtpStateIsReusedAndRecaptured) {
  const torch::Tensor block_tables =
      torch::tensor({{1, 2}, {3, 4}}, torch::kInt32);
  const torch::Tensor context_lens = torch::tensor({2, 1}, torch::kInt32);
  const DsaTopkState state(block_tables, context_lens);

  DsaTopkTransfer transfer =
      DsaTopkTransfer::prepare_mtp_step(state, torch::Device(torch::kCPU));
  ASSERT_NE(transfer.input(), nullptr);
  EXPECT_TRUE(transfer.captures_output());
  EXPECT_EQ(transfer.input()->block_tables().data_ptr(),
            block_tables.data_ptr());
  EXPECT_EQ(transfer.input()->context_lens().data_ptr(),
            context_lens.data_ptr());
}

TEST(DsaTopkStateTest, RejectsStateWithNonInt32Dtype) {
  const torch::Tensor block_tables = torch::zeros({2, 2}, torch::kInt64);
  const torch::Tensor context_lens = torch::zeros({2}, torch::kInt64);
  EXPECT_DEATH(
      {
        const DsaTopkState state(block_tables, context_lens);
        static_cast<void>(state);
      },
      "must use int32");
}

TEST(DsaTopkStateTest, RejectsSparseMetadataWithMismatchedRows) {
  const torch::Tensor block_tables =
      torch::zeros({2, 4}, torch::dtype(torch::kInt32));
  const torch::Tensor context_lens =
      torch::zeros({3}, torch::dtype(torch::kInt32));
  EXPECT_DEATH(
      {
        const DsaTopkState state(block_tables, context_lens);
        static_cast<void>(state);
      },
      "row count");
}

}  // namespace
}  // namespace xllm::layer
