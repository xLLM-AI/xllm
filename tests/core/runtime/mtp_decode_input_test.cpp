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

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/layers/common/attention_metadata.h"
#include "core/runtime/task_execution_pipeline.h"
#include "tests/core/runtime/task_pipeline_test_peer.h"

namespace xllm {
namespace {

class MtpDecodeInputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ModelConfig::get_instance().model_impl("python");
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
    ModelConfig::get_instance().model_impl(model_impl_);
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

  void run(SlotBuffer& binding, uint32_t padded_batch_size = 0) {
    Stream prepare(device_);
    Stream task(device_);
    if (padded_batch_size == 0) {
      ASSERT_TRUE(TaskPipelineTestPeer::prepare_mtp_decode(
                      binding, input(), batch_, prepare)
                      .ok());
    } else {
      ASSERT_TRUE(
          TaskPipelineTestPeer::plan_mtp_decode(binding, input(), batch_).ok());
      ASSERT_TRUE(TaskPipelineTestPeer::prepare_planned_mtp_decode(
                      binding, input(), batch_, prepare, padded_batch_size)
                      .ok());
    }
    const auto ready = prepare.record_event();
    {
      auto guard = task.set_stream_guard();
      ASSERT_TRUE(task.wait_event(ready));
      TaskPipelineTestPeer::patch_mtp_decode(binding, *state_);
    }
    ASSERT_EQ(task.synchronize(), 0);
  }

  torch::Device device_{torch::kPrivateUse1, 0};
  const std::string model_impl_ = ModelConfig::get_instance().model_impl();
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

TEST_F(MtpDecodeInputTest, DSparkSamplesAnchorWithoutAnExtraQueryRow) {
  spec_.kind = MtpInvocationKind::BLOCK_DRAFT;
  spec_.enable_mla = false;
  spec_.mask_token_id = 99;
  spec_.sample_from_anchor = true;
  std::unique_ptr<SlotBuffer> binding;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                  spec_, torch::kBFloat16, device_, binding)
                  .ok());
  run(*binding);
  EXPECT_EQ(TaskPipelineTestPeer::input_scratch(*binding).rows_per_sequence_,
            3U);
  EXPECT_TRUE(
      torch::equal(binding->tokens().cpu(),
                   torch::tensor({42, 99, 99, 52, 99, 99}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(binding->positions().cpu(),
                   torch::tensor({7, 8, 9, 15, 16, 17}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(binding->model_params().attention.device.kv_seq_lens.cpu(),
                   torch::tensor({10, 18}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      binding->model_params().attention.device.new_cache_slots.cpu(),
      torch::tensor({23, 24, 25, 63, 64, 65}, torch::kInt32)));
  spec_.kind = MtpInvocationKind::VALIDATE;
  EXPECT_FALSE(TaskPipelineTestPeer::create_mtp_input(
                   spec_, torch::kBFloat16, device_, binding)
                   .ok());
}

TEST_F(MtpDecodeInputTest, BlockDraftGraphPadsCompleteIsolatedRequests) {
  spec_.kind = MtpInvocationKind::BLOCK_DRAFT;
  spec_.enable_mla = false;
  spec_.mask_token_id = 99;
  spec_.model.max_sequences = 4;
  for (const bool sample_from_anchor : {false, true}) {
    SCOPED_TRACE(sample_from_anchor);
    spec_.sample_from_anchor = sample_from_anchor;
    std::unique_ptr<SlotBuffer> binding;
    ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                    spec_, torch::kBFloat16, device_, binding)
                    .ok());
    const uint32_t width =
        TaskPipelineTestPeer::input_scratch(*binding).rows_per_sequence_;
    const uint64_t pinned_bytes = binding->pinned_bytes();
    const uint64_t device_bytes = binding->device_bytes();
    const void* tokens_address = nullptr;
    const void* blocks_address = nullptr;
    for (const uint32_t physical_sequences : {2U, 3U, 4U, 2U, 3U}) {
      SCOPED_TRACE(physical_sequences);
      const uint32_t physical_rows = physical_sequences * width;
      const uint32_t logical_tokens = 2 * width;
      const uint32_t padding_tokens = physical_rows - logical_tokens;
      run(*binding, physical_rows);
      const auto& params = binding->model_params();
      EXPECT_TRUE(params.enable_graph);
      EXPECT_TRUE(params.meta.batch_forward_type.is_chunked_prefill());
      EXPECT_EQ(params.meta.num_sequences, physical_sequences);
      EXPECT_EQ(params.meta.actual_num_sequences, 2);
      EXPECT_EQ(params.meta.execution_batch.num_reqs, 2);
      EXPECT_EQ(params.meta.execution_batch.num_tokens, physical_rows);
      EXPECT_EQ(params.meta.execution_batch.query_start_loc,
                (std::vector<int32_t>{0,
                                      static_cast<int32_t>(width),
                                      static_cast<int32_t>(2 * width)}));
      EXPECT_EQ(binding->tokens().numel(), physical_rows);
      EXPECT_EQ(TaskPipelineTestPeer::input_scratch(*binding).sequences_, 2U);
      EXPECT_EQ(TaskPipelineTestPeer::input_scratch(*binding)
                    .planned_.q_seq_lens.size(),
                2U);
      EXPECT_TRUE(params.attention.device.block_tables.is_contiguous());
      EXPECT_TRUE(torch::equal(
          params.attention.device.block_tables.cpu().slice(/*dim=*/0,
                                                           /*start=*/2),
          torch::zeros({physical_sequences - 2, 4}, torch::kInt32)));
      EXPECT_TRUE(
          torch::equal(binding->tokens().cpu().slice(/*dim=*/0, logical_tokens),
                       torch::full({padding_tokens}, 99, torch::kInt32)));
      EXPECT_TRUE(torch::equal(
          binding->positions().cpu().slice(/*dim=*/0, logical_tokens),
          torch::zeros({padding_tokens}, torch::kInt32)));
      EXPECT_TRUE(
          torch::equal(params.attention.device.new_cache_slots.cpu().slice(
                           /*dim=*/0, logical_tokens),
                       torch::full({padding_tokens}, -1, torch::kInt32)));
      EXPECT_TRUE(torch::equal(
          params.attention.device.q_seq_lens.cpu(),
          torch::full({physical_sequences}, width, torch::kInt32)));
      EXPECT_TRUE(torch::equal(
          params.attention.device.q_cu_seq_lens.cpu(),
          torch::arange(width, physical_rows + width, width, torch::kInt32)));
      EXPECT_TRUE(torch::equal(
          params.attention.device.kv_seq_lens.cpu().slice(/*dim=*/0,
                                                          /*start=*/2),
          torch::full({physical_sequences - 2}, width, torch::kInt32)));
      EXPECT_EQ(params.attention.host.q_cu_seq_lens.back(), physical_rows);
      EXPECT_EQ(binding->pinned_bytes(), pinned_bytes);
      EXPECT_EQ(binding->device_bytes(), device_bytes);
      if (tokens_address == nullptr) {
        tokens_address = binding->tokens().data_ptr();
        blocks_address = params.attention.device.block_tables.data_ptr();
      }
      EXPECT_EQ(binding->tokens().data_ptr(), tokens_address);
      EXPECT_EQ(params.attention.device.block_tables.data_ptr(),
                blocks_address);
    }
    Stream prepare(device_);
    ASSERT_TRUE(TaskPipelineTestPeer::prepare_mtp_decode(
                    *binding, input(), batch_, prepare)
                    .ok());
    ASSERT_EQ(prepare.synchronize(), 0);
    EXPECT_FALSE(binding->model_params().enable_graph);
    EXPECT_EQ(binding->model_params().meta.num_sequences, 2);
    EXPECT_EQ(binding->tokens().numel(), 2 * width);
  }
}

TEST_F(MtpDecodeInputTest, BlockDraftStatePatchingLeavesPaddingUnmodified) {
  spec_.kind = MtpInvocationKind::BLOCK_DRAFT;
  spec_.enable_mla = false;
  spec_.mask_token_id = 99;
  spec_.sample_from_anchor = true;
  spec_.model.max_sequences = 4;
  std::unique_ptr<SlotBuffer> binding;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                  spec_, torch::kBFloat16, device_, binding)
                  .ok());
  run(*binding, /*padded_batch_size=*/12);
  const auto tokens_before =
      binding->tokens().cpu().slice(/*dim=*/0, /*start=*/6).clone();
  const auto positions_before =
      binding->positions().cpu().slice(/*dim=*/0, /*start=*/6).clone();
  const auto slots_before = binding->model_params()
                                .attention.device.new_cache_slots.cpu()
                                .slice(
                                    /*dim=*/0, /*start=*/6)
                                .clone();
  const auto lengths_before = binding->model_params()
                                  .attention.device.kv_seq_lens.cpu()
                                  .slice(
                                      /*dim=*/0, /*start=*/2)
                                  .clone();
  state_->tokens_.copy_(torch::tensor({71, 81}, torch::kInt64));
  state_->state_.positions.copy_(torch::tensor({9, 17}, torch::kInt32));
  state_->state_.kv_seq_lens.copy_(torch::tensor({10, 18}, torch::kInt32));
  ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  Stream task(device_);
  {
    auto guard = task.set_stream_guard();
    TaskPipelineTestPeer::patch_mtp_decode(*binding, *state_);
  }
  ASSERT_EQ(task.synchronize(), 0);
  EXPECT_TRUE(torch::equal(
      binding->tokens().cpu().slice(/*dim=*/0, /*start=*/6), tokens_before));
  EXPECT_TRUE(
      torch::equal(binding->positions().cpu().slice(/*dim=*/0, /*start=*/6),
                   positions_before));
  EXPECT_TRUE(torch::equal(
      binding->model_params().attention.device.new_cache_slots.cpu().slice(
          /*dim=*/0, /*start=*/6),
      slots_before));
  EXPECT_TRUE(torch::equal(
      binding->model_params().attention.device.kv_seq_lens.cpu().slice(
          /*dim=*/0, /*start=*/2),
      lengths_before));
  EXPECT_TRUE(torch::equal(
      binding->positions().cpu().slice(/*dim=*/0, /*start=*/0, /*end=*/6),
      torch::tensor({9, 10, 11, 17, 18, 19}, torch::kInt32)));
  EXPECT_EQ(state_->tokens_.numel(), 2);
  EXPECT_EQ(binding->model_params().meta.actual_num_sequences, 2);
}

TEST_F(MtpDecodeInputTest, InvalidBlockDraftGraphBucketPreservesBinding) {
  spec_.kind = MtpInvocationKind::BLOCK_DRAFT;
  spec_.enable_mla = false;
  spec_.mask_token_id = 99;
  spec_.sample_from_anchor = true;
  spec_.model.max_sequences = 4;
  std::unique_ptr<SlotBuffer> binding;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                  spec_, torch::kBFloat16, device_, binding)
                  .ok());
  run(*binding, /*padded_batch_size=*/9);
  const auto tokens_before = binding->tokens().cpu();
  const auto positions_before = binding->positions().cpu();
  const auto host_before = binding->model_params().attention.host.kv_seq_lens;
  const void* address = binding->tokens().data_ptr();
  Stream prepare(device_);
  for (const uint32_t physical_rows : {3U, 5U, 7U, 13U, 15U}) {
    SCOPED_TRACE(physical_rows);
    EXPECT_FALSE(TaskPipelineTestPeer::prepare_planned_mtp_decode(
                     *binding, input(), batch_, prepare, physical_rows)
                     .ok());
    EXPECT_TRUE(torch::equal(binding->tokens().cpu(), tokens_before));
    EXPECT_TRUE(torch::equal(binding->positions().cpu(), positions_before));
    EXPECT_EQ(binding->tokens().data_ptr(), address);
    EXPECT_TRUE(binding->model_params().enable_graph);
    EXPECT_EQ(binding->model_params().attention.host.kv_seq_lens, host_before);
    EXPECT_EQ(binding->model_params().meta.num_sequences, 3);
    EXPECT_EQ(binding->model_params().meta.actual_num_sequences, 2);
  }
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
                           torch::tensor({5, 6, 13, 14}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(model.model_params().attention.device.kv_seq_lens.cpu(),
                   torch::tensor({6, 7, 14, 15}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(model.model_params().attention.device.new_cache_slots.cpu(),
                   torch::tensor({21, 22, 63, 62}, torch::kInt32)));
  EXPECT_EQ(model.model_params().attention.host.kv_seq_lens,
            std::vector<int32_t>({9, 10, 17, 18}));
  EXPECT_TRUE(torch::equal(
      model.model_params().attention.device.block_tables.cpu(),
      torch::tensor({{2, 3, 4, 5}, {2, 3, 4, 5}, {6, 7, 8, 9}, {6, 7, 8, 9}},
                    torch::kInt32)));
}

TEST_F(MtpDecodeInputTest, AdaptiveValidateDepthReusesMaximumStorage) {
  spec_.kind = MtpInvocationKind::VALIDATE;
  std::unique_ptr<SlotBuffer> binding;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                  spec_, torch::kBFloat16, device_, binding)
                  .ok());
  run(*binding, /*padded_batch_size=*/8);
  const uint64_t bytes = binding->device_bytes();
  const void* tokens = binding->tokens().const_data_ptr();
  for (const uint32_t depth : {1U, 2U, 3U, 1U}) {
    ASSERT_TRUE(
        TaskPipelineTestPeer::set_mtp_validate_width(*binding, depth).ok());
    run(*binding, /*padded_batch_size=*/8);
    const torch::Tensor expected =
        (torch::tensor({7, 15}, torch::kInt32).unsqueeze(1) +
         torch::arange(depth + 1, torch::kInt32).unsqueeze(0))
            .reshape({-1});
    EXPECT_TRUE(torch::equal(
        binding->positions().narrow(0, 0, 2 * (depth + 1)).cpu(), expected));
    EXPECT_EQ(binding->tokens().const_data_ptr(), tokens);
    EXPECT_EQ(binding->device_bytes(), bytes);
  }
  EXPECT_FALSE(TaskPipelineTestPeer::set_mtp_validate_width(*binding, 0).ok());
  EXPECT_FALSE(TaskPipelineTestPeer::set_mtp_validate_width(*binding, 4).ok());
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

TEST_F(MtpDecodeInputTest, DSparkVerifyBindsFixedRowsWithoutRequestWidths) {
  spec_.kind = MtpInvocationKind::VALIDATE;
  spec_.num_speculative_tokens = 7;
  spec_.model.max_sequences = 4;
  std::unique_ptr<SlotBuffer> binding;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                  spec_, torch::kBFloat16, device_, binding)
                  .ok());
  const uint64_t pinned_bytes = binding->pinned_bytes();
  const uint64_t device_bytes = binding->device_bytes();
  const auto& scratch = TaskPipelineTestPeer::input_scratch(*binding);
  const std::array<const void*, 7> scratch_addresses{
      scratch.tokens_.data(),
      scratch.positions_.data(),
      scratch.slots_.data(),
      scratch.q_lengths_.data(),
      scratch.kv_lengths_.data(),
      scratch.query_ends_.data(),
      scratch.block_tables_.data()};
  std::array<const void*, 7> device_addresses{};
  std::array<const int32_t*, 2> host_addresses{};
  Stream prepare(device_);
  for (const auto geometry : {std::array<uint32_t, 4>{3, 8, 2, 16},
                              std::array<uint32_t, 4>{7, 8, 2, 16},
                              std::array<uint32_t, 4>{9, 16, 4, 32},
                              std::array<uint32_t, 4>{13, 16, 4, 32},
                              std::array<uint32_t, 4>{31, 32, 4, 32},
                              std::array<uint32_t, 4>{32, 32, 4, 32},
                              std::array<uint32_t, 4>{3, 8, 2, 16}}) {
    const uint32_t logical_rows = geometry[0];
    const uint32_t physical_rows = geometry[1];
    const uint32_t block_table_width = geometry[2];
    const int32_t max_kv_length = static_cast<int32_t>(geometry[3]);
    SCOPED_TRACE(logical_rows);
    ASSERT_TRUE(TaskPipelineTestPeer::prepare_dspark_validate(*binding,
                                                              logical_rows,
                                                              physical_rows,
                                                              block_table_width,
                                                              max_kv_length,
                                                              /*batch_id=*/93,
                                                              prepare)
                    .ok());
    ASSERT_EQ(prepare.synchronize(), 0);
    const auto& params = binding->model_params();
    EXPECT_TRUE(params.enable_graph);
    EXPECT_FALSE(params.is_spec_verify);
    EXPECT_TRUE(params.meta.batch_forward_type.is_decode());
    EXPECT_EQ(params.meta.num_sequences, physical_rows);
    EXPECT_EQ(params.meta.actual_num_sequences, logical_rows);
    EXPECT_EQ(params.meta.batch_id, 93);
    EXPECT_EQ(params.meta.q_max_seq_len, 1);
    EXPECT_EQ(params.meta.kv_max_seq_len, max_kv_length);
    EXPECT_FALSE(params.attn_metadata->is_prefill);
    EXPECT_FALSE(params.attn_metadata->is_chunked_prefill);
    EXPECT_EQ(binding->tokens().numel(), physical_rows);
    EXPECT_TRUE(torch::equal(binding->tokens().cpu(),
                             torch::ones({physical_rows}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(binding->positions().cpu(),
                             torch::zeros({physical_rows}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(params.attention.device.new_cache_slots.cpu(),
                             torch::full({physical_rows}, -1, torch::kInt32)));
    EXPECT_TRUE(torch::equal(params.attention.device.q_seq_lens.cpu(),
                             torch::ones({physical_rows}, torch::kInt32)));
    EXPECT_TRUE(
        torch::equal(params.attention.device.q_cu_seq_lens.cpu(),
                     torch::arange(1, physical_rows + 1, torch::kInt32)));
    std::vector<int32_t> expected_kv_lengths(physical_rows, 1);
    std::fill_n(expected_kv_lengths.begin(), logical_rows, max_kv_length);
    EXPECT_EQ(params.attention.host.kv_seq_lens, expected_kv_lengths);
    EXPECT_TRUE(
        torch::equal(params.attention.device.kv_seq_lens.cpu(),
                     torch::tensor(expected_kv_lengths, torch::kInt32)));
    EXPECT_TRUE(params.attention.device.block_tables.is_contiguous());
    EXPECT_TRUE(torch::equal(params.attention.device.block_tables.cpu(),
                             torch::zeros({physical_rows, 4}, torch::kInt32)));
    std::vector<int32_t> expected_query_ends(physical_rows);
    std::iota(expected_query_ends.begin(), expected_query_ends.end(), 1);
    EXPECT_EQ(params.attention.host.q_cu_seq_lens, expected_query_ends);
    EXPECT_EQ(params.attn_metadata->q_cu_seq_lens_host_vec,
              std::vector<int64_t>(expected_query_ends.begin(),
                                   expected_query_ends.end()));
    const std::array<const void*, 7> current_device_addresses{
        binding->tokens().data_ptr(),
        binding->positions().data_ptr(),
        params.attention.device.new_cache_slots.data_ptr(),
        params.attention.device.q_seq_lens.data_ptr(),
        params.attention.device.kv_seq_lens.data_ptr(),
        params.attention.device.q_cu_seq_lens.data_ptr(),
        params.attention.device.block_tables.data_ptr()};
    const std::array<const int32_t*, 2> current_host_addresses{
        params.attention.host.graph_q_seq_lens_data,
        params.attention.host.graph_kv_seq_lens_data};
    if (device_addresses[0] == nullptr) {
      device_addresses = current_device_addresses;
      host_addresses = current_host_addresses;
    }
    EXPECT_EQ(current_device_addresses, device_addresses);
    EXPECT_EQ(current_host_addresses, host_addresses);
    EXPECT_EQ((std::array<const void*, 7>{scratch.tokens_.data(),
                                          scratch.positions_.data(),
                                          scratch.slots_.data(),
                                          scratch.q_lengths_.data(),
                                          scratch.kv_lengths_.data(),
                                          scratch.query_ends_.data(),
                                          scratch.block_tables_.data()}),
              scratch_addresses);
    EXPECT_EQ(binding->pinned_bytes(), pinned_bytes);
    EXPECT_EQ(binding->device_bytes(), device_bytes);
    EXPECT_EQ(scratch.sequences_, 0U);
    EXPECT_TRUE(scratch.planned_.token_ids.empty());
  }
}

TEST_F(MtpDecodeInputTest, DSparkVerifyClearsPreviousUniformPatchViews) {
  spec_.kind = MtpInvocationKind::VALIDATE;
  spec_.num_speculative_tokens = 7;
  spec_.model.max_sequences = 4;
  std::unique_ptr<SlotBuffer> binding;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                  spec_, torch::kBFloat16, device_, binding)
                  .ok());
  run(*binding, /*padded_batch_size=*/16);
  const auto& scratch = TaskPipelineTestPeer::input_scratch(*binding);
  ASSERT_EQ(scratch.sequences_, 2U);
  ASSERT_TRUE(scratch.positions_view_.defined());
  binding->model_params().is_spec_verify = true;
  Stream prepare(device_);
  ASSERT_TRUE(
      TaskPipelineTestPeer::prepare_dspark_validate(*binding,
                                                    /*logical_rows=*/9,
                                                    /*physical_rows=*/16,
                                                    /*block_table_width=*/4,
                                                    /*max_kv_length=*/32,
                                                    /*batch_id=*/94,
                                                    prepare)
          .ok());
  ASSERT_EQ(prepare.synchronize(), 0);
  EXPECT_FALSE(binding->model_params().is_spec_verify);
  EXPECT_EQ(scratch.sequences_, 0U);
  EXPECT_TRUE(scratch.planned_.token_ids.empty());
  for (const auto* tensor : {&scratch.positions_view_,
                             &scratch.kv_lengths_view_,
                             &scratch.tokens_view_,
                             &scratch.cache_positions_,
                             &scratch.cache_first_column_,
                             &scratch.model_first_column_,
                             &scratch.block_indices_,
                             &scratch.block_ids_,
                             &scratch.cache_offsets_,
                             &scratch.cache_offsets_flat_,
                             &scratch.future_positions_}) {
    EXPECT_FALSE(tensor->defined());
  }
  // Returning to ordinary uniform validation rebuilds its own patch views.
  run(*binding, /*padded_batch_size=*/16);
  EXPECT_EQ(scratch.sequences_, 2U);
  EXPECT_TRUE(scratch.positions_view_.defined());
}

TEST_F(MtpDecodeInputTest, InvalidDSparkVerifyGeometryPreservesBinding) {
  spec_.kind = MtpInvocationKind::VALIDATE;
  spec_.num_speculative_tokens = 7;
  spec_.model.max_sequences = 4;
  std::unique_ptr<SlotBuffer> binding;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                  spec_, torch::kBFloat16, device_, binding)
                  .ok());
  run(*binding, /*padded_batch_size=*/16);
  const auto tokens_before = binding->tokens().cpu();
  const auto positions_before = binding->positions().cpu();
  const auto host_before = binding->model_params().attention.host.kv_seq_lens;
  const auto& scratch = TaskPipelineTestPeer::input_scratch(*binding);
  const auto scratch_tokens_before = scratch.tokens_;
  const auto scratch_blocks_before = scratch.block_tables_;
  const void* address = binding->tokens().data_ptr();
  const void* patch_address = scratch.positions_view_.data_ptr();
  Stream prepare(device_);
  for (const auto geometry : {std::array<int32_t, 4>{0, 16, 4, 32},
                              std::array<int32_t, 4>{17, 16, 4, 32},
                              std::array<int32_t, 4>{1, 0, 4, 32},
                              std::array<int32_t, 4>{1, 33, 4, 32},
                              std::array<int32_t, 4>{1, 16, 0, 32},
                              std::array<int32_t, 4>{1, 16, 5, 32},
                              std::array<int32_t, 4>{1, 16, 4, 0},
                              std::array<int32_t, 4>{1, 16, 4, -1},
                              std::array<int32_t, 4>{1, 16, 2, 17}}) {
    SCOPED_TRACE(geometry[3]);
    EXPECT_FALSE(TaskPipelineTestPeer::prepare_dspark_validate(
                     *binding,
                     static_cast<uint32_t>(geometry[0]),
                     static_cast<uint32_t>(geometry[1]),
                     static_cast<uint32_t>(geometry[2]),
                     geometry[3],
                     /*batch_id=*/95,
                     prepare)
                     .ok());
    EXPECT_TRUE(torch::equal(binding->tokens().cpu(), tokens_before));
    EXPECT_TRUE(torch::equal(binding->positions().cpu(), positions_before));
    EXPECT_EQ(binding->tokens().data_ptr(), address);
    EXPECT_EQ(binding->model_params().attention.host.kv_seq_lens, host_before);
    EXPECT_EQ(binding->model_params().meta.batch_id, batch_.batch_id);
    EXPECT_EQ(scratch.tokens_, scratch_tokens_before);
    EXPECT_EQ(scratch.block_tables_, scratch_blocks_before);
    EXPECT_EQ(scratch.sequences_, 2U);
    EXPECT_EQ(scratch.positions_view_.data_ptr(), patch_address);
    EXPECT_EQ(scratch.planned_.q_seq_lens.size(), 16U);
  }
  // Reject the invocation kind without destroying a valid ordinary draft.
  spec_.kind = MtpInvocationKind::DRAFT;
  std::unique_ptr<SlotBuffer> draft;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                  spec_, torch::kBFloat16, device_, draft)
                  .ok());
  run(*draft);
  EXPECT_FALSE(
      TaskPipelineTestPeer::prepare_dspark_validate(*draft,
                                                    /*logical_rows=*/1,
                                                    /*physical_rows=*/1,
                                                    /*block_table_width=*/4,
                                                    /*max_kv_length=*/32,
                                                    /*batch_id=*/95,
                                                    prepare)
          .ok());
  EXPECT_EQ(TaskPipelineTestPeer::input_scratch(*draft).sequences_, 2U);
  EXPECT_TRUE(
      TaskPipelineTestPeer::input_scratch(*draft).positions_view_.defined());
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
                           torch::tensor({8, 16}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(model.model_params().attention.device.new_cache_slots.cpu(),
                   torch::tensor({24, 64}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(model.tokens().cpu(), torch::zeros({2}, torch::kInt32)));
}

TEST_F(MtpDecodeInputTest, OneTokenPromptKeepsUnusedRepairOutsideValidKv) {
  spec_.model.max_sequences = 4;
  positions_ = {1, 1};
  kv_lengths_ = {2, 2};
  state_->state_.positions.fill_(1);
  state_->state_.kv_seq_lens.fill_(2);
  state_->state_.repair_required.zero_();
  ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  for (const uint32_t padded_batch_size : {0U, 8U}) {
    SCOPED_TRACE(padded_batch_size);
    std::unique_ptr<SlotBuffer> binding;
    ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                    spec_, torch::kBFloat16, device_, binding)
                    .ok());
    run(*binding, padded_batch_size);
    EXPECT_TRUE(torch::equal(
        binding->positions().cpu().narrow(/*dim=*/0, /*start=*/0, 4),
        torch::zeros({4}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(
        binding->model_params().attention.device.kv_seq_lens.cpu().narrow(
            /*dim=*/0, /*start=*/0, 4),
        torch::ones({4}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(
        binding->model_params().attention.device.new_cache_slots.cpu().narrow(
            /*dim=*/0, /*start=*/0, 4),
        torch::tensor({17, 16, 49, 48}, torch::kInt32)));
  }
}

TEST_F(MtpDecodeInputTest, DcpFirstDraftRepairsAcrossLogicalPages) {
  // Physical blocks contain eight tokens on each of two DCP ranks. The
  // scheduler page table and this input builder use sixteen-token pages.
  spec_.logical_block_size = 16;
  spec_.model.max_sequences = 4;
  blocks_ = {3, 1, 2, 0, 2, 0, 3, 1};
  positions_ = {19, 18};
  kv_lengths_ = {20, 19};
  state_->state_.positions.copy_(torch::tensor({17, 16}, torch::kInt32));
  state_->state_.kv_seq_lens.copy_(torch::tensor({18, 17}, torch::kInt32));
  ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);

  for (const uint32_t padded_batch_size : {0U, 8U}) {
    SCOPED_TRACE(padded_batch_size);
    std::unique_ptr<SlotBuffer> binding;
    ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                    spec_, torch::kBFloat16, device_, binding)
                    .ok());
    run(*binding, padded_batch_size);
    const auto& params = binding->model_params();
    EXPECT_TRUE(torch::equal(
        binding->positions().cpu().narrow(/*dim=*/0, /*start=*/0, 4),
        torch::tensor({15, 16, 14, 15}, torch::kInt32)));
    // Full acceptance repairs the preceding page. The other row writes its
    // unused repair result to the future page without overwriting valid KV.
    EXPECT_TRUE(
        torch::equal(params.attention.device.new_cache_slots.cpu().narrow(
                         /*dim=*/0, /*start=*/0, 4),
                     torch::tensor({63, 16, 0, 47}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(params.attention.device.kv_seq_lens.cpu().narrow(
                                 /*dim=*/0, /*start=*/0, 4),
                             torch::tensor({16, 17, 15, 16}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(
        params.attention.device.new_cache_slots.cpu().slice(
            /*dim=*/0, /*start=*/4),
        torch::full({padded_batch_size == 0 ? 0 : 4}, -1, torch::kInt32)));
  }
}

TEST_F(MtpDecodeInputTest, DcpValidateAndLaterDraftUseLogicalPageOffsets) {
  spec_.logical_block_size = 16;
  spec_.model.max_sequences = 4;
  blocks_ = {3, 1, 2, 0, 2, 0, 3, 1};
  positions_ = {19, 18};
  kv_lengths_ = {20, 19};
  state_->state_.positions.copy_(torch::tensor({16, 15}, torch::kInt32));
  state_->state_.kv_seq_lens.copy_(torch::tensor({17, 16}, torch::kInt32));
  ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);

  for (const uint32_t padded_batch_size : {0U, 16U}) {
    SCOPED_TRACE(padded_batch_size);
    spec_.kind = MtpInvocationKind::VALIDATE;
    std::unique_ptr<SlotBuffer> binding;
    ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                    spec_, torch::kBFloat16, device_, binding)
                    .ok());
    run(*binding, padded_batch_size);
    const auto slots =
        binding->model_params().attention.device.new_cache_slots.cpu();
    EXPECT_TRUE(torch::equal(
        slots.narrow(/*dim=*/0, /*start=*/0, 8),
        torch::tensor({16, 17, 18, 19, 47, 0, 1, 2}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(
        slots.slice(/*dim=*/0, /*start=*/8),
        torch::full({padded_batch_size == 0 ? 0 : 8}, -1, torch::kInt32)));
  }

  spec_.kind = MtpInvocationKind::DRAFT;
  spec_.draft_step = 2;
  for (const uint32_t padded_batch_size : {0U, 4U}) {
    SCOPED_TRACE(padded_batch_size);
    std::unique_ptr<SlotBuffer> binding;
    ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                    spec_, torch::kBFloat16, device_, binding)
                    .ok());
    run(*binding, padded_batch_size);
    const auto slots =
        binding->model_params().attention.device.new_cache_slots.cpu();
    EXPECT_TRUE(torch::equal(slots.narrow(/*dim=*/0, /*start=*/0, 2),
                             torch::tensor({17, 0}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(
        slots.slice(/*dim=*/0, /*start=*/2),
        torch::full({padded_batch_size == 0 ? 0 : 2}, -1, torch::kInt32)));
  }
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
        step == 0 ? torch::tensor({5, 6, 13, 14}, torch::kInt32)
                  : torch::tensor({8, 16}, torch::kInt32);
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
      32;  // The inactive repair row still needs a reserved scratch position.
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
