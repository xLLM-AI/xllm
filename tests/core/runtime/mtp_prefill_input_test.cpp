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

#include <array>

#include "core/runtime/task_execution_pipeline.h"
#include "tests/core/runtime/task_pipeline_test_peer.h"

namespace xllm {
namespace {

class MtpPrefillInputTest : public ::testing::Test {
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
    ASSERT_TRUE(
        TaskPipelineTestPeer::create_mtp_input(
            {{8, 2, 2}, /*hidden_size=*/3}, torch::kBFloat16, device_, input_)
            .ok());
    sampling_.selected_token_idxes = torch::tensor({3}, torch::kInt32);
    sampling_.sample_idxes = torch::tensor({0}, torch::kInt32);
    hidden_ = torch::arange(12, torch::kFloat32)
                  .view({4, 3})
                  .to(device_, torch::kBFloat16);
    samples_ = torch::tensor({{90}}, torch::kInt64).to(device_);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  }
  void TearDown() override {
    EXPECT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    input_.reset();
    TaskPipelineTestPeer::context_release(*state_);
    state_.reset();
    context_.reset();
  }
  ModelInputHostView host() const {
    return {tokens_,
            positions_,
            slots_,
            q_lengths_,
            kv_lengths_,
            query_ends_,
            blocks_,
            2};
  }
  void run() {
    Stream prepare(device_);
    Stream task(device_);
    ASSERT_TRUE(TaskPipelineTestPeer::prepare_mtp_prefill(
                    *input_, host(), batch_, extra_, sampling_, prepare)
                    .ok());
    std::vector<int32_t> completed_ids;
    std::vector<std::string> completed_requests;
    for (uint32_t row = 0; row < extra_.size(); ++row) {
      if (extra_[row] == -1) {
        completed_ids.push_back(keys_[row]);
        completed_requests.push_back(requests_[row]);
      }
    }
    ASSERT_TRUE(
        TaskPipelineTestPeer::context_prepare_prefill(
            *state_,
            completed_ids,
            completed_requests,
            extra_,
            TaskPipelineTestPeer::input_scratch(*input_).published_rows_,
            prepare)
            .ok());
    const auto ready = prepare.record_event();
    {
      auto guard = task.set_stream_guard();
      ASSERT_TRUE(task.wait_event(ready));
      TaskPipelineTestPeer::patch_mtp_prefill(
          *input_, hidden_, samples_, *state_);
      TaskPipelineTestPeer::context_publish(*state_);
    }
    ASSERT_EQ(task.synchronize(), 0);
  }

  torch::Device device_{torch::kPrivateUse1, 0};
  std::unique_ptr<MtpContextStorage> context_;
  std::unique_ptr<MtpContextView> state_;
  std::unique_ptr<SlotBuffer> input_;
  std::array<int32_t, 2> keys_{{1, 2}};
  std::array<std::string, 2> requests_{{"request", "request"}};
  std::array<int32_t, 4> tokens_{{11, 12, 13, 24}};
  std::array<int32_t, 4> positions_{{3, 4, 5, 6}};
  std::array<int32_t, 4> slots_{{19, 20, 21, 30}};
  std::array<int32_t, 2> q_lengths_{{3, 1}};
  std::array<int32_t, 2> kv_lengths_{{6, 7}};
  std::array<int32_t, 2> query_ends_{{3, 4}};
  std::array<int32_t, 4> blocks_{{2, 3, 4, 5}};
  std::array<int32_t, 2> extra_{{14, -1}};
  ModelInputBatch batch_{BatchForwardType::CHUNKED_PREFILL, 2, 81, false};
  SamplingParameters sampling_;
  torch::Tensor hidden_;
  torch::Tensor samples_;
};

TEST_F(MtpPrefillInputTest, UnifiedSlotAccountsForPromptScratch) {
  std::unique_ptr<SlotBuffer> ordinary;
  ASSERT_TRUE(
      SlotBuffer::create({8, 2, 2}, /*enable_mla=*/true, device_, ordinary)
          .ok());
  // Only completed-row hidden is owned; prompt hidden is borrowed from the
  // target output until the draft forward completes.
  EXPECT_EQ(input_->pinned_bytes() - ordinary->pinned_bytes(), 32U);
  EXPECT_EQ(input_->device_bytes() - ordinary->device_bytes(), 52U);
  const auto* original = input_.get();
  EXPECT_FALSE(
      TaskPipelineTestPeer::create_mtp_input(
          {{8, 2, 2}, /*hidden_size=*/0}, torch::kBFloat16, device_, input_)
          .ok());
  EXPECT_FALSE(
      TaskPipelineTestPeer::create_mtp_input(
          {{8, 2, 2}, /*hidden_size=*/3}, torch::kInt32, device_, input_)
          .ok());
  EXPECT_EQ(input_.get(), original);
}

TEST_F(MtpPrefillInputTest, BlockDraftPrefillStoresOnlyCompletedRowScratch) {
  MtpInputSpec spec{{8, 2, 2}, 3};
  spec.context_only = true;
  std::unique_ptr<SlotBuffer> small;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                  spec, torch::kBFloat16, device_, small)
                  .ok());
  spec.model.max_tokens = 8192;
  std::unique_ptr<SlotBuffer> large;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_input(
                  spec, torch::kBFloat16, device_, large)
                  .ok());
  EXPECT_EQ(small->pinned_bytes(), large->pinned_bytes());
  EXPECT_EQ(small->device_bytes(), large->device_bytes());
  EXPECT_FALSE(small->tokens().defined());
  EXPECT_FALSE(small->model_params().embedding.input_embedding.defined());
}

TEST_F(MtpPrefillInputTest, MixedPrefillAndDecodePublishesCompletedRows) {
  batch_.forward_type = BatchForwardType::MIXED;
  run();
  EXPECT_TRUE(torch::equal(input_->tokens().cpu(),
                           torch::tensor({12, 13, 14, 90}, torch::kInt32)));
  EXPECT_EQ(TaskPipelineTestPeer::input_scratch(*input_).published_rows_.size(),
            1U);
  EXPECT_EQ(TaskPipelineTestPeer::input_scratch(*input_).published_rows_[0],
            1U);
  EXPECT_TRUE(input_->model_params().meta.batch_forward_type.is_mixed());
}

TEST_F(MtpPrefillInputTest, ShiftAndBootstrapRespectChunkBoundaries) {
  run();
  const auto& model = *input_;
  EXPECT_TRUE(torch::equal(model.tokens().cpu(),
                           torch::tensor({12, 13, 14, 90}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(model.positions().cpu(),
                           torch::tensor({3, 4, 5, 6}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(model.model_params().embedding.input_embedding.cpu(),
                           hidden_.cpu()));
  EXPECT_TRUE(
      torch::equal(state_->tokens_.cpu(), torch::tensor({90}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(state_->state_.positions.cpu(),
                           torch::tensor({7}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(state_->state_.kv_seq_lens.cpu(),
                           torch::tensor({8}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(state_->state_.previous_tokens.cpu(),
                           state_->tokens_.cpu()));
  EXPECT_TRUE(
      torch::equal(state_->state_.hidden.select(/*dim=*/1, /*index=*/1).cpu(),
                   hidden_.narrow(/*dim=*/0, /*start=*/3, /*length=*/1).cpu()));
  EXPECT_EQ(state_->state_.hidden.select(/*dim=*/1, /*index=*/0)
                .cpu()
                .count_nonzero()
                .item<int64_t>(),
            0);
  EXPECT_FALSE(state_->state_.repair_required.cpu().item<bool>());
  Stream prepare(device_);
  std::unique_ptr<MtpContextView> next;
  ASSERT_TRUE(
      TaskPipelineTestPeer::context_create(*context_, /*capacity=*/2, next)
          .ok());
  EXPECT_FALSE(
      TaskPipelineTestPeer::context_prepare(
          *next, keys_, requests_, /*read_published_state=*/true, prepare)
          .ok());
}

TEST_F(MtpPrefillInputTest, SamplingOrderControlsPublicationOrder) {
  extra_[0] = -1;
  sampling_.selected_token_idxes = torch::tensor({2, 3}, torch::kInt32);
  sampling_.sample_idxes = torch::tensor({1, 0}, torch::kInt32);
  samples_ = torch::tensor({{90}, {80}}, torch::kInt64).to(device_);
  run();
  const auto rows =
      TaskPipelineTestPeer::input_scratch(*input_).published_rows_;
  ASSERT_EQ(rows.size(), 2U);
  EXPECT_EQ(rows[0], 1U);
  EXPECT_EQ(rows[1], 0U);
  EXPECT_TRUE(torch::equal(input_->tokens().cpu(),
                           torch::tensor({12, 13, 80, 90}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(state_->state_.positions.cpu(),
                           torch::tensor({7, 6}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(state_->state_.kv_seq_lens.cpu(),
                           torch::tensor({8, 7}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      TaskPipelineTestPeer::input_scratch(*input_).bootstrap_hidden_.cpu(),
      hidden_
          .index_select(/*dim=*/0,
                        torch::tensor({3, 2}, torch::kInt64).to(device_))
          .cpu()));
}

TEST_F(MtpPrefillInputTest, IncompleteChunksRequireNoSampleOrPublishedState) {
  extra_[1] = 25;
  sampling_ = SamplingParameters();
  samples_ = torch::Tensor();
  run();
  EXPECT_TRUE(
      TaskPipelineTestPeer::input_scratch(*input_).published_rows_.empty());
  EXPECT_TRUE(state_->prepared_);
  EXPECT_EQ(state_->tokens_.numel(), 0);
  EXPECT_TRUE(torch::equal(input_->tokens().cpu(),
                           torch::tensor({12, 13, 14, 25}, torch::kInt32)));
}

TEST_F(MtpPrefillInputTest, RejectedSamplingLeavesActiveMappingAndStorage) {
  run();
  const void* tokens = input_->tokens().data_ptr();
  const auto before = input_->tokens().cpu();
  Stream prepare(device_);
  sampling_.selected_token_idxes = torch::tensor({1}, torch::kInt32);
  EXPECT_FALSE(TaskPipelineTestPeer::prepare_mtp_prefill(
                   *input_, host(), batch_, extra_, sampling_, prepare)
                   .ok());
  const auto rows =
      TaskPipelineTestPeer::input_scratch(*input_).published_rows_;
  ASSERT_EQ(rows.size(), 1U);
  EXPECT_EQ(rows[0], 1U);
  EXPECT_EQ(input_->tokens().data_ptr(), tokens);
  EXPECT_TRUE(torch::equal(input_->tokens().cpu(), before));
  sampling_.selected_token_idxes = torch::tensor({3}, torch::kInt32);
  sampling_.sample_idxes = torch::tensor({-1}, torch::kInt32);
  EXPECT_FALSE(TaskPipelineTestPeer::prepare_mtp_prefill(
                   *input_, host(), batch_, extra_, sampling_, prepare)
                   .ok());
  sampling_.sample_idxes = torch::tensor({0}, torch::kInt32);
  extra_[0] = -1;
  EXPECT_FALSE(TaskPipelineTestPeer::prepare_mtp_prefill(
                   *input_, host(), batch_, extra_, sampling_, prepare)
                   .ok());
}

TEST_F(MtpPrefillInputTest, DecodeRowsCanRefreshStateDuringPeerPrefill) {
  const std::array<int32_t, 2> tokens{31, 41};
  const std::array<int32_t, 2> positions{5, 6};
  const std::array<int32_t, 2> slots{21, 30};
  const std::array<int32_t, 2> q{1, 1};
  const std::array<int32_t, 2> kv{6, 7};
  const std::array<int32_t, 2> ends{1, 2};
  const std::array<int32_t, 2> extra{-1, -1};
  const ModelInputHostView view{
      tokens, positions, slots, q, kv, ends, blocks_, 2};
  const ModelInputBatch batch{BatchForwardType::DECODE, 2, 82, false};
  SamplingParameters sampling;
  sampling.selected_token_idxes = torch::tensor({0, 1}, torch::kInt32);
  sampling.sample_idxes = torch::tensor({0, 1}, torch::kInt32);
  Stream prepare(device_);
  ASSERT_TRUE(TaskPipelineTestPeer::prepare_mtp_prefill(
                  *input_, view, batch, extra, sampling, prepare)
                  .ok());
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare_prefill(
                  *state_,
                  keys_,
                  requests_,
                  extra,
                  TaskPipelineTestPeer::input_scratch(*input_).published_rows_,
                  prepare)
                  .ok());
  ASSERT_EQ(prepare.synchronize(), 0);
  auto hidden = torch::arange(6, torch::kFloat32)
                    .view({2, 3})
                    .to(device_, torch::kBFloat16);
  auto sampled = torch::tensor({{32}, {42}}, torch::kInt64).to(device_);
  TaskPipelineTestPeer::patch_mtp_prefill(*input_, hidden, sampled, *state_);
  TaskPipelineTestPeer::context_publish(*state_);
  ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  EXPECT_TRUE(torch::equal(input_->tokens().cpu(),
                           torch::tensor({32, 42}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(state_->state_.positions.cpu(),
                           torch::tensor({6, 7}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(state_->state_.kv_seq_lens.cpu(),
                           torch::tensor({7, 8}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(state_->state_.hidden.select(1, 1).cpu(), hidden.cpu()));
}

TEST_F(MtpPrefillInputTest, BorrowedHiddenSurvivesProducerReleaseAndSlotReuse) {
  run();
  const auto expected = hidden_.cpu();
  const void* hidden = hidden_.data_ptr();
  EXPECT_EQ(input_->model_params().embedding.input_embedding.data_ptr(),
            hidden);
  const void* tokens = input_->tokens().data_ptr();
  hidden_ = torch::Tensor();
  EXPECT_TRUE(torch::equal(
      input_->model_params().embedding.input_embedding.cpu(), expected));
  TaskPipelineTestPeer::context_release(*state_);
  extra_[0] = -1;
  sampling_.selected_token_idxes = torch::tensor({2, 3}, torch::kInt32);
  sampling_.sample_idxes = torch::tensor({0, 1}, torch::kInt32);
  samples_ = torch::tensor({{80}, {90}}, torch::kInt64).to(device_);
  hidden_ = torch::full(
      {4, 3},
      7.0,
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16));
  run();
  EXPECT_EQ(input_->model_params().embedding.input_embedding.data_ptr(),
            hidden_.data_ptr());
  EXPECT_TRUE(
      torch::equal(input_->model_params().embedding.input_embedding.cpu(),
                   torch::full({4, 3}, 7.0, torch::kBFloat16)));
  EXPECT_EQ(input_->tokens().data_ptr(), tokens);
}

}  // namespace
}  // namespace xllm
