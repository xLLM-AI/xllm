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

#include <array>
#include <limits>

#include "core/runtime/task_execution_pipeline.h"
#include "tests/core/runtime/task_pipeline_test_peer.h"

namespace xllm {
namespace {

class MtpDecodeInputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_context(
                    /*capacity=*/4,
                    /*hidden_size=*/3,
                    torch::kBFloat16,
                    device_,
                    context_)
                    .ok());
    ASSERT_TRUE(
        TaskPipelineTestPeer::context_create(*context_, /*capacity=*/2, state_)
            .ok());
    Stream prepare(device_);
    const std::array<int32_t, 2> keys{{1, 2}};
    const std::array<std::string, 2> requests{{"request", "request"}};
    ASSERT_TRUE(
        TaskPipelineTestPeer::context_prepare(
            *state_, keys, requests, /*read_published_state=*/false, prepare)
            .ok());
    ASSERT_EQ(prepare.synchronize(), 0);
    state_->tokens_.copy_(torch::tensor({42, 52}, torch::kInt64));
    state_->state_.previous_tokens.copy_(
        torch::tensor({41, 51}, torch::kInt64));
    state_->state_.positions.copy_(torch::tensor({7, 15}, torch::kInt32));
    state_->state_.kv_seq_lens.copy_(torch::tensor({8, 16}, torch::kInt32));
    state_->state_.repair_required.copy_(
        torch::tensor({true, false}, torch::kBool));
    state_->state_.hidden.zero_();
    TaskPipelineTestPeer::context_publish(*state_);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  }

  void TearDown() override {
    EXPECT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    TaskPipelineTestPeer::context_release(*state_);
    state_.reset();
    context_.reset();
  }

  ModelInputHostView input() const {
    return {tokens_,
            positions_,
            slots_,
            q_lengths_,
            kv_lengths_,
            query_ends_,
            blocks_,
            4};
  }

  void run(SlotBuffer& binding) {
    Stream prepare(device_);
    Stream task(device_);
    ASSERT_TRUE(TaskPipelineTestPeer::prepare_mtp_decode(
                    binding, input(), batch_, prepare)
                    .ok());
    const auto ready = prepare.record_event();
    {
      auto guard = task.set_stream_guard();
      ASSERT_TRUE(task.wait_event(ready));
      TaskPipelineTestPeer::patch_mtp_decode(binding, *state_);
    }
    ASSERT_EQ(task.synchronize(), 0);
  }

  torch::Device device_{torch::kPrivateUse1, 0};
  std::unique_ptr<MtpContextStorage> context_;
  std::unique_ptr<MtpContextView> state_;
  MtpInputSpec spec_{{8, 2, 4}, 0, 8, 3, MtpInvocationKind::DRAFT, 0};
  ModelInputBatch batch_{BatchForwardType::DECODE, 2, 71, false};
  std::array<int32_t, 2> tokens_{{-1, -1}};
  std::array<int32_t, 2> positions_{{10, 18}};
  std::array<int32_t, 2> slots_{{0, 0}};
  std::array<int32_t, 2> q_lengths_{{1, 1}};
  std::array<int32_t, 2> kv_lengths_{{11, 19}};
  std::array<int32_t, 2> query_ends_{{1, 2}};
  std::array<int32_t, 8> blocks_{{2, 3, 4, 5, 6, 7, 8, 9}};
};

TEST_F(MtpDecodeInputTest, UnifiedSlotAccountsForExpandedScratch) {
  std::unique_ptr<SlotBuffer> binding;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                  spec_, torch::kBFloat16, device_, binding)
                  .ok());
  std::unique_ptr<SlotBuffer> ordinary;
  ASSERT_TRUE(
      SlotBuffer::create({4, 4, 4}, /*enable_mla=*/true, device_, ordinary)
          .ok());
  // Two int32 offsets; four rows of int64 positions/indices and int32
  // block ids/offsets; two int64 repair positions.
  EXPECT_EQ(binding->pinned_bytes(), ordinary->pinned_bytes());
  EXPECT_EQ(binding->device_bytes() - ordinary->device_bytes(), 120U);
}

TEST_F(MtpDecodeInputTest, BlockDraftKeepsNonCausalBlocksAndDeviceLengths) {
  spec_.kind = MtpInvocationKind::BLOCK_DRAFT;
  spec_.enable_mla = false;
  spec_.mask_token_id = 99;
  std::unique_ptr<SlotBuffer> binding;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                  spec_, torch::kBFloat16, device_, binding)
                  .ok());
  run(*binding);
  const auto& model = *binding;
  EXPECT_TRUE(
      model.model_params().meta.batch_forward_type.is_chunked_prefill());
  EXPECT_EQ(model.model_params().meta.num_sequences, 2);
  EXPECT_TRUE(torch::equal(
      model.tokens().cpu(),
      torch::tensor({42, 99, 99, 99, 52, 99, 99, 99}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      model.positions().cpu(),
      torch::tensor({7, 8, 9, 10, 15, 16, 17, 18}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(model.model_params().attention.device.q_seq_lens.cpu(),
                   torch::tensor({4, 4}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(model.model_params().attention.device.kv_seq_lens.cpu(),
                   torch::tensor({11, 19}, torch::kInt32)));
  EXPECT_EQ(model.model_params().attention.host.kv_seq_lens,
            std::vector<int32_t>({14, 22}));
  EXPECT_TRUE(torch::equal(
      model.model_params().attention.device.new_cache_slots.cpu(),
      torch::tensor({23, 24, 25, 26, 63, 64, 65, 66}, torch::kInt32)));
}

TEST_F(MtpDecodeInputTest, FirstDraftRepairsOnlyFullAcceptance) {
  std::unique_ptr<SlotBuffer> binding;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                  spec_, torch::kBFloat16, device_, binding)
                  .ok());
  run(*binding);
  const auto& model = *binding;
  EXPECT_EQ(TaskPipelineTestPeer::input_scratch(*binding).rows_per_sequence_,
            2U);
  EXPECT_TRUE(torch::equal(model.tokens().cpu(),
                           torch::tensor({41, 42, 51, 52}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(model.positions().cpu(),
                           torch::tensor({6, 7, 14, 15}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(model.model_params().attention.device.kv_seq_lens.cpu(),
                   torch::tensor({7, 8, 15, 16}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(model.model_params().attention.device.new_cache_slots.cpu(),
                   torch::tensor({22, 23, 64, 63}, torch::kInt32)));
  EXPECT_EQ(model.model_params().attention.host.kv_seq_lens,
            std::vector<int32_t>({10, 11, 18, 19}));
  EXPECT_TRUE(torch::equal(
      model.model_params().attention.device.block_tables.cpu(),
      torch::tensor({{2, 3, 4, 5}, {2, 3, 4, 5}, {6, 7, 8, 9}, {6, 7, 8, 9}},
                    torch::kInt32)));
}

TEST_F(MtpDecodeInputTest, ValidateUsesActualDeviceBaseAcrossBlocks) {
  spec_.kind = MtpInvocationKind::VALIDATE;
  std::unique_ptr<SlotBuffer> binding;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                  spec_, torch::kBFloat16, device_, binding)
                  .ok());
  run(*binding);
  const auto& model = *binding;
  EXPECT_EQ(TaskPipelineTestPeer::input_scratch(*binding).rows_per_sequence_,
            4U);
  EXPECT_TRUE(torch::equal(
      model.positions().cpu(),
      torch::tensor({7, 8, 9, 10, 15, 16, 17, 18}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      model.model_params().attention.device.kv_seq_lens.cpu(),
      torch::tensor({8, 9, 10, 11, 16, 17, 18, 19}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      model.model_params().attention.device.new_cache_slots.cpu(),
      torch::tensor({23, 24, 25, 26, 63, 64, 65, 66}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(model.tokens().cpu(),
                   torch::tensor({42, 0, 0, 0, 52, 0, 0, 0}, torch::kInt32)));
  const void* address = model.tokens().data_ptr();
  model.tokens().view({2, 4}).slice(1, 1).copy_(
      torch::tensor({{43, 44, 45}, {53, 54, 55}}, torch::kInt32));
  EXPECT_TRUE(torch::equal(
      model.tokens().cpu(),
      torch::tensor({42, 43, 44, 45, 52, 53, 54, 55}, torch::kInt32)));
  EXPECT_EQ(model.tokens().data_ptr(), address);
}

TEST_F(MtpDecodeInputTest, ValidatePatchesOnlyLogicalPrefixOfDpGraphBucket) {
  spec_.kind = MtpInvocationKind::VALIDATE;
  spec_.model.max_sequences = 4;
  std::unique_ptr<SlotBuffer> binding;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                  spec_, torch::kBFloat16, device_, binding)
                  .ok());
  Stream prepare(device_);
  Stream task(device_);
  ASSERT_TRUE(
      TaskPipelineTestPeer::plan_mtp_decode(*binding, input(), batch_).ok());
  ASSERT_TRUE(TaskPipelineTestPeer::prepare_planned_mtp_decode(
                  *binding, input(), batch_, prepare, /*physical_rows=*/16)
                  .ok());
  const auto ready = prepare.record_event();
  {
    auto guard = task.set_stream_guard();
    ASSERT_TRUE(task.wait_event(ready));
    TaskPipelineTestPeer::patch_mtp_decode(*binding, *state_);
  }
  ASSERT_EQ(task.synchronize(), 0);
  const auto& model = *binding;
  EXPECT_EQ(model.tokens().numel(), 16);
  EXPECT_TRUE(model.model_params().enable_graph);
  EXPECT_EQ(model.model_params().meta.actual_num_sequences, 8);
  EXPECT_TRUE(torch::equal(
      model.tokens().cpu().slice(/*dim=*/0, /*start=*/0, /*end=*/8),
      torch::tensor({42, 0, 0, 0, 52, 0, 0, 0}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(model.tokens().cpu().slice(/*dim=*/0, /*start=*/8),
                           torch::ones({8}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(model.positions().cpu().slice(/*dim=*/0, /*start=*/8),
                   torch::zeros({8}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      model.model_params().attention.device.new_cache_slots.cpu().slice(
          /*dim=*/0, /*start=*/8),
      torch::full({8}, -1, torch::kInt32)));
  ASSERT_TRUE(TaskPipelineTestPeer::prepare_mtp_decode(
                  *binding, input(), batch_, prepare)
                  .ok());
  ASSERT_EQ(prepare.synchronize(), 0);
  EXPECT_FALSE(model.model_params().enable_graph);
  EXPECT_EQ(model.tokens().numel(), 8);
}

TEST_F(MtpDecodeInputTest, LaterDraftLeavesSampledTokenToItsProducer) {
  spec_.draft_step = 2;
  std::unique_ptr<SlotBuffer> binding;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                  spec_, torch::kBFloat16, device_, binding)
                  .ok());
  run(*binding);
  const auto& model = *binding;
  EXPECT_EQ(TaskPipelineTestPeer::input_scratch(*binding).rows_per_sequence_,
            1U);
  EXPECT_TRUE(torch::equal(model.positions().cpu(),
                           torch::tensor({9, 17}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(model.model_params().attention.device.new_cache_slots.cpu(),
                   torch::tensor({25, 65}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(model.tokens().cpu(), torch::zeros({2}, torch::kInt32)));
}

TEST_F(MtpDecodeInputTest, DraftGraphPaddingSurvivesStatePatching) {
  for (uint32_t step : std::array<uint32_t, 2>{0, 2}) {
    spec_.draft_step = step;
    spec_.model.max_sequences = 4;
    std::unique_ptr<SlotBuffer> binding;
    ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                    spec_, torch::kBFloat16, device_, binding)
                    .ok());
    const int64_t logical_rows =
        2 * TaskPipelineTestPeer::input_scratch(*binding).rows_per_sequence_;
    const uint32_t padded_batch_size =
        4 * TaskPipelineTestPeer::input_scratch(*binding).rows_per_sequence_;
    Stream prepare(device_);
    Stream task(device_);
    ASSERT_TRUE(
        TaskPipelineTestPeer::plan_mtp_decode(*binding, input(), batch_).ok());
    ASSERT_TRUE(TaskPipelineTestPeer::prepare_planned_mtp_decode(
                    *binding, input(), batch_, prepare, padded_batch_size)
                    .ok());
    const auto ready = prepare.record_event();
    {
      auto guard = task.set_stream_guard();
      ASSERT_TRUE(task.wait_event(ready));
      TaskPipelineTestPeer::patch_mtp_decode(*binding, *state_);
    }
    ASSERT_EQ(task.synchronize(), 0);
    const auto& model = *binding;
    EXPECT_TRUE(model.model_params().enable_graph);
    EXPECT_EQ(model.model_params().meta.actual_num_sequences, logical_rows);
    EXPECT_TRUE(torch::equal(
        model.tokens().cpu().slice(/*dim=*/0, logical_rows),
        torch::ones({padded_batch_size - logical_rows}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(
        model.model_params().attention.device.new_cache_slots.cpu().slice(
            /*dim=*/0, logical_rows),
        torch::full({padded_batch_size - logical_rows}, -1, torch::kInt32)));
    const auto expected_positions =
        step == 0 ? torch::tensor({6, 7, 14, 15}, torch::kInt32)
                  : torch::tensor({9, 17}, torch::kInt32);
    EXPECT_TRUE(torch::equal(
        model.positions().cpu().narrow(/*dim=*/0, /*start=*/0, logical_rows),
        expected_positions));
    ASSERT_TRUE(TaskPipelineTestPeer::prepare_mtp_decode(
                    *binding, input(), batch_, prepare)
                    .ok());
    ASSERT_EQ(prepare.synchronize(), 0);
    EXPECT_FALSE(model.model_params().enable_graph);
    EXPECT_EQ(model.tokens().numel(), logical_rows);
  }
}

TEST_F(MtpDecodeInputTest, InvalidPreparationPreservesPreviousInput) {
  std::unique_ptr<SlotBuffer> binding;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                  spec_, torch::kBFloat16, device_, binding)
                  .ok());
  run(*binding);
  const auto before = binding->positions().cpu();
  const auto host_before = binding->model_params().attention.host.kv_seq_lens;
  const void* address = binding->positions().data_ptr();
  Stream prepare(device_);
  positions_[1] =
      31;  // The inactive repair row still needs a reserved scratch position.
  EXPECT_FALSE(TaskPipelineTestPeer::prepare_mtp_decode(
                   *binding, input(), batch_, prepare)
                   .ok());
  positions_[1] = 18;
  blocks_[0] = std::numeric_limits<int32_t>::max();
  EXPECT_FALSE(TaskPipelineTestPeer::prepare_mtp_decode(
                   *binding, input(), batch_, prepare)
                   .ok());
  blocks_[0] = 2;
  q_lengths_[1] = 2;
  EXPECT_FALSE(TaskPipelineTestPeer::prepare_mtp_decode(
                   *binding, input(), batch_, prepare)
                   .ok());
  EXPECT_TRUE(torch::equal(binding->positions().cpu(), before));
  EXPECT_EQ(binding->positions().data_ptr(), address);
  EXPECT_EQ(binding->model_params().attention.host.kv_seq_lens, host_before);
}

TEST_F(MtpDecodeInputTest,
       FactoryRejectsInvalidExpansionWithoutReplacingOutput) {
  std::unique_ptr<SlotBuffer> binding;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                  spec_, torch::kBFloat16, device_, binding)
                  .ok());
  const auto* original = binding.get();
  spec_.draft_step = spec_.num_speculative_tokens;
  EXPECT_FALSE(TaskPipelineTestPeer::create_mtp_input(
                   spec_, torch::kBFloat16, device_, binding)
                   .ok());
  spec_.draft_step = 0;
  spec_.num_speculative_tokens = 0;
  EXPECT_FALSE(TaskPipelineTestPeer::create_mtp_input(
                   spec_, torch::kBFloat16, device_, binding)
                   .ok());
  spec_.num_speculative_tokens = 3;
  spec_.model.max_sequences = std::numeric_limits<uint32_t>::max();
  EXPECT_FALSE(TaskPipelineTestPeer::create_mtp_input(
                   spec_, torch::kBFloat16, device_, binding)
                   .ok());
  EXPECT_EQ(binding.get(), original);
}

}  // namespace
}  // namespace xllm
