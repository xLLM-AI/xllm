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

class MtpContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_context(
                    /*capacity=*/4,
                    /*hidden_size=*/3,
                    torch::kFloat32,
                    device_,
                    context_)
                    .ok());
    ASSERT_TRUE(
        TaskPipelineTestPeer::context_create(*context_, /*capacity=*/2, first_)
            .ok());
    ASSERT_TRUE(
        TaskPipelineTestPeer::context_create(*context_, /*capacity=*/2, second_)
            .ok());
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  }

  void TearDown() override {
    EXPECT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    TaskPipelineTestPeer::context_release(*first_);
    TaskPipelineTestPeer::context_release(*second_);
    first_.reset();
    second_.reset();
    context_.reset();
  }

  void seed(MtpContextView& binding) {
    binding.tokens_.copy_(torch::tensor({101, 202}, torch::kInt64));
    const auto& state = binding.state_;
    state.previous_tokens.copy_(torch::tensor({100, 201}, torch::kInt64));
    state.positions.copy_(torch::tensor({7, 13}, torch::kInt32));
    state.kv_seq_lens.copy_(torch::tensor({8, 14}, torch::kInt32));
    state.repair_required.copy_(torch::tensor({true, false}, torch::kBool));
    state.hidden.copy_(torch::arange(12, torch::kFloat32).reshape({2, 2, 3}));
    TaskPipelineTestPeer::context_publish(binding);
  }

  torch::Device device_{torch::kPrivateUse1, 0};
  std::unique_ptr<MtpContextStorage> context_;
  std::unique_ptr<MtpContextView> first_;
  std::unique_ptr<MtpContextView> second_;
  std::array<int32_t, 2> keys_{{1, 2}};
  std::array<std::string, 2> requests_{{"request", "request"}};
};

TEST_F(MtpContextTest, CapacityAndSchemaRejectWithoutReplacingOutput) {
  const MtpContextStorage* original = context_.get();
  EXPECT_FALSE(
      TaskPipelineTestPeer::create_mtp_context(
          /*capacity=*/4, /*hidden_size=*/3, torch::kInt32, device_, context_)
          .ok());
  EXPECT_FALSE(TaskPipelineTestPeer::create_mtp_context(
                   std::numeric_limits<uint32_t>::max(),
                   std::numeric_limits<uint32_t>::max(),
                   torch::kFloat32,
                   device_,
                   context_)
                   .ok());
  EXPECT_EQ(context_.get(), original);
  EXPECT_EQ(TaskPipelineTestPeer::mtp_context_bytes(*context_),
            4U * (8 + 8 + 2 * 3 * 4 + 4 + 4 + 1));
  EXPECT_EQ((first_->host_rows_.nbytes()), 2U * 8);
  EXPECT_EQ(TaskPipelineTestPeer::context_device_bytes(*first_),
            2U * (8 + 8 + 8 + 2 * 3 * 4 + 4 + 4 + 1 + 16 + 1));
  const MtpContextView* binding = first_.get();
  EXPECT_FALSE(
      TaskPipelineTestPeer::context_create(*context_, /*capacity=*/0, first_)
          .ok());
  EXPECT_EQ(first_.get(), binding);
}

TEST_F(MtpContextTest, BlockDraftAdvancesTokensWithoutAllocatingHiddenState) {
  std::unique_ptr<MtpContextStorage> storage;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_context(
                  4, 0, torch::kFloat32, device_, storage)
                  .ok());
  EXPECT_EQ(TaskPipelineTestPeer::mtp_context_bytes(*storage),
            4U * (8 + 4 + 4));
  EXPECT_FALSE(storage->mtp_state_.hidden.defined());
  EXPECT_FALSE(storage->mtp_state_.previous_tokens.defined());
  EXPECT_FALSE(storage->mtp_state_.repair_required.defined());
  std::unique_ptr<MtpContextView> view;
  ASSERT_TRUE(TaskPipelineTestPeer::context_create(*storage, 2, view).ok());
  EXPECT_FALSE(view->storage_.hidden.defined());
  EXPECT_FALSE(view->storage_.previous_tokens.defined());
  EXPECT_FALSE(view->storage_.repair_required.defined());
  EXPECT_EQ(TaskPipelineTestPeer::context_device_bytes(*view), 2U * 32);
  Stream prepare(device_);
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare(
                  *view, keys_, requests_, false, prepare)
                  .ok());
  ASSERT_EQ(prepare.synchronize(), 0);
  view->tokens_.copy_(torch::tensor({10, 20}, torch::kInt64));
  view->state_.positions.copy_(torch::tensor({3, 5}, torch::kInt32));
  view->state_.kv_seq_lens.copy_(torch::tensor({4, 6}, torch::kInt32));
  const auto accepted =
      torch::tensor({{11, -1, -1}, {21, 22, 23}}, torch::kInt64).to(device_);
  const auto lengths = torch::tensor({1, 3}, torch::kInt32).to(device_);
  TaskPipelineTestPeer::context_advance(*view, accepted, lengths, {});
  TaskPipelineTestPeer::context_publish(*view);
  ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  TaskPipelineTestPeer::context_release(*view);
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare(
                  *view, keys_, requests_, true, prepare)
                  .ok());
  ASSERT_EQ(prepare.synchronize(), 0);
  view->tokens_.zero_();
  TaskPipelineTestPeer::context_gather(*view);
  EXPECT_TRUE(torch::equal(view->tokens_.cpu(),
                           torch::tensor({11, 23}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(view->state_.positions.cpu(),
                           torch::tensor({4, 8}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(view->state_.kv_seq_lens.cpu(),
                           torch::tensor({5, 9}, torch::kInt32)));
  TaskPipelineTestPeer::context_release(*view);
}

TEST_F(MtpContextTest, ViewsSurviveDeferredEmbeddingAllocation) {
  std::unique_ptr<MtpContextStorage> storage;
  ASSERT_TRUE(TaskPipelineTestPeer::create_mtp_context(
                  0, 3, torch::kFloat32, device_, storage)
                  .ok());
  std::unique_ptr<MtpContextView> view;
  ASSERT_TRUE(TaskPipelineTestPeer::context_create(*storage, 2, view).ok());
  const void* snapshot = view->storage_.hidden.data_ptr();
  *storage = std::move(*context_);
  Stream prepare(device_);
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare(
                  *view, keys_, requests_, false, prepare)
                  .ok());
  ASSERT_EQ(prepare.synchronize(), 0);
  EXPECT_EQ(view->storage_.hidden.data_ptr(), snapshot);
  EXPECT_EQ(view->context_, storage.get());
  EXPECT_TRUE(
      torch::equal(view->host_rows_, torch::tensor({1, 2}, torch::kInt64)));
  TaskPipelineTestPeer::context_release(*view);
}

TEST_F(MtpContextTest, InvalidRowsAndRequestMismatchLeaveContextUnchanged) {
  Stream prepare(device_);
  EXPECT_FALSE(TaskPipelineTestPeer::context_prepare(
                   *first_, keys_, requests_, true, prepare)
                   .ok());
  const std::array<int32_t, 2> duplicate{1, 1};
  const std::array<int32_t, 2> padding{0, 1};
  const std::array<int32_t, 2> outside{1, 4};
  for (const auto& ids : {duplicate, padding, outside}) {
    EXPECT_FALSE(TaskPipelineTestPeer::context_prepare(
                     *first_, ids, requests_, false, prepare)
                     .ok());
  }
  EXPECT_FALSE(
      TaskPipelineTestPeer::context_prepare(
          *first_, keys_, std::span(requests_).first(1), false, prepare)
          .ok());
  EXPECT_EQ(context_->published_, (std::vector<uint8_t>{0, 0, 0, 0}));
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare(
                  *first_, keys_, requests_, false, prepare)
                  .ok());
  const auto owners = context_->request_ids_;
  const std::array<std::string, 2> other{"request", "other"};
  EXPECT_FALSE(TaskPipelineTestPeer::context_prepare(
                   *second_, keys_, other, true, prepare)
                   .ok());
  EXPECT_EQ(context_->request_ids_, owners);
  EXPECT_FALSE(second_->prepared_);
  EXPECT_FALSE(TaskPipelineTestPeer::context_prepare(
                   *first_, keys_, other, false, prepare)
                   .ok());
  ASSERT_EQ(prepare.synchronize(), 0);
}

TEST_F(MtpContextTest, ReusedEmbeddingRowsPublishAfterOlderTask) {
  Stream prepare(device_);
  Stream task(device_);
  const std::array<std::string, 2> replacement{"new-a", "new-b"};
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare(
                  *first_, keys_, requests_, false, prepare)
                  .ok());
  auto old_ready = prepare.record_event();
  // Reuse is admitted before the old task finishes. Device writes remain FIFO.
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare(
                  *second_, keys_, replacement, false, prepare)
                  .ok());
  auto new_ready = prepare.record_event();
  {
    auto guard = task.set_stream_guard();
    ASSERT_TRUE(task.wait_event(old_ready));
    seed(*first_);
    ASSERT_TRUE(task.wait_event(new_ready));
    seed(*second_);
    second_->tokens_.add_(1000);
    TaskPipelineTestPeer::context_publish(*second_);
  }
  ASSERT_EQ(task.synchronize(), 0);
  EXPECT_TRUE(torch::equal(first_->tokens_.cpu(),
                           torch::tensor({101, 202}, torch::kInt64)));
  TaskPipelineTestPeer::context_release(*first_);
  EXPECT_FALSE(TaskPipelineTestPeer::context_prepare(
                   *first_, keys_, requests_, true, prepare)
                   .ok());
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare(
                  *first_, keys_, replacement, true, prepare)
                  .ok());
  auto ready = prepare.record_event();
  {
    auto guard = task.set_stream_guard();
    ASSERT_TRUE(task.wait_event(ready));
    TaskPipelineTestPeer::context_gather(*first_);
  }
  ASSERT_EQ(task.synchronize(), 0);
  EXPECT_TRUE(torch::equal(first_->tokens_.cpu(),
                           torch::tensor({1101, 1202}, torch::kInt64)));
  TaskPipelineTestPeer::context_release(*first_);
  TaskPipelineTestPeer::context_release(*second_);
  // A completed recompute replaces the same request's old context in FIFO
  // order.
  const std::array<int32_t, 2> extra{{-1, -1}};
  const std::array<uint32_t, 2> completed{{0, 1}};
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare_prefill(
                  *first_, keys_, replacement, extra, completed, prepare)
                  .ok());
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare(
                  *second_, keys_, replacement, true, prepare)
                  .ok());
  ready = prepare.record_event();
  {
    auto guard = task.set_stream_guard();
    ASSERT_TRUE(task.wait_event(ready));
    seed(*first_);
    TaskPipelineTestPeer::context_gather(*second_);
  }
  ASSERT_EQ(task.synchronize(), 0);
  EXPECT_TRUE(torch::equal(second_->tokens_.cpu(),
                           torch::tensor({101, 202}, torch::kInt64)));
}

TEST_F(MtpContextTest, ContextSurvivesNonAdjacentBatch) {
  Stream prepare(device_);
  Stream task(device_);
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare(
                  *first_, keys_, requests_, false, prepare)
                  .ok());
  auto ready = prepare.record_event();
  {
    auto guard = task.set_stream_guard();
    ASSERT_TRUE(task.wait_event(ready));
    seed(*first_);
  }
  ASSERT_EQ(task.synchronize(), 0);
  TaskPipelineTestPeer::context_release(*first_);
  const std::array<int32_t, 1> unrelated{3};
  const std::array<std::string, 1> request{"unrelated"};
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare(
                  *first_, unrelated, request, false, prepare)
                  .ok());
  ready = prepare.record_event();
  {
    auto guard = task.set_stream_guard();
    ASSERT_TRUE(task.wait_event(ready));
    first_->tokens_.fill_(999);
    first_->state_.previous_tokens.zero_();
    first_->state_.hidden.zero_();
    first_->state_.positions.zero_();
    first_->state_.kv_seq_lens.zero_();
    first_->state_.repair_required.zero_();
    TaskPipelineTestPeer::context_publish(*first_);
  }
  ASSERT_EQ(task.synchronize(), 0);
  TaskPipelineTestPeer::context_release(*first_);
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare(
                  *first_, keys_, requests_, true, prepare)
                  .ok());
  ready = prepare.record_event();
  {
    auto guard = task.set_stream_guard();
    ASSERT_TRUE(task.wait_event(ready));
    TaskPipelineTestPeer::context_gather(*first_);
  }
  ASSERT_EQ(task.synchronize(), 0);
  EXPECT_TRUE(torch::equal(first_->tokens_.cpu(),
                           torch::tensor({101, 202}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(first_->state_.hidden.cpu(),
                           torch::arange(12, torch::kFloat32).view({2, 2, 3})));
}

TEST_F(MtpContextTest, TwoSlotsGatherReorderedStateBeforeFirstRetires) {
  Stream prepare(device_);
  Stream task(device_);
  ASSERT_TRUE(
      TaskPipelineTestPeer::context_prepare(
          *first_, keys_, requests_, /*read_published_state=*/false, prepare)
          .ok());
  const auto first_ready = prepare.record_event();
  const std::array<int32_t, 2> reversed{{keys_[1], keys_[0]}};
  ASSERT_TRUE(
      TaskPipelineTestPeer::context_prepare(
          *second_, reversed, requests_, /*read_published_state=*/true, prepare)
          .ok());
  const auto second_ready = prepare.record_event();
  {
    auto guard = task.set_stream_guard();
    ASSERT_TRUE(task.wait_event(first_ready));
    seed(*first_);
    ASSERT_TRUE(task.wait_event(second_ready));
    TaskPipelineTestPeer::context_gather(*second_);
  }
  ASSERT_EQ(task.synchronize(), 0);
  EXPECT_TRUE(torch::equal(second_->tokens_.cpu(),
                           torch::tensor({202, 101}, torch::kInt64)));
  const auto& state = second_->state_;
  EXPECT_TRUE(torch::equal(state.previous_tokens.cpu(),
                           torch::tensor({201, 100}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(state.positions.cpu(),
                           torch::tensor({13, 7}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(state.kv_seq_lens.cpu(),
                           torch::tensor({14, 8}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(state.repair_required.cpu(),
                           torch::tensor({false, true}, torch::kBool)));
  const auto expected =
      torch::arange(12, torch::kFloat32).reshape({2, 2, 3}).flip({0});
  EXPECT_TRUE(torch::equal(state.hidden.cpu(), expected));
  // Releasing or reusing one Slot cannot overwrite the other snapshot.
  TaskPipelineTestPeer::context_release(*first_);
  first_->tokens_.fill_(999);
  EXPECT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  EXPECT_TRUE(torch::equal(second_->tokens_.cpu(),
                           torch::tensor({202, 101}, torch::kInt64)));
}

TEST_F(MtpContextTest, ShapeChangesAndEmptyTasksKeepBacking) {
  Stream prepare(device_);
  ASSERT_TRUE(
      TaskPipelineTestPeer::context_prepare(
          *first_, keys_, requests_, /*read_published_state=*/false, prepare)
          .ok());
  ASSERT_EQ(prepare.synchronize(), 0);
  const void* tokens = first_->tokens_.data_ptr();
  const void* hidden = first_->state_.hidden.data_ptr();
  TaskPipelineTestPeer::context_release(*first_);
  ASSERT_TRUE(
      TaskPipelineTestPeer::context_prepare(*first_,
                                            std::span(keys_).first(1),
                                            std::span(requests_).first(1),
                                            /*read_published_state=*/true,
                                            prepare)
          .ok());
  ASSERT_EQ(prepare.synchronize(), 0);
  EXPECT_EQ(first_->tokens_.numel(), 1);
  EXPECT_EQ(first_->tokens_.data_ptr(), tokens);
  EXPECT_EQ(first_->state_.hidden.data_ptr(), hidden);
  TaskPipelineTestPeer::context_release(*first_);
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare(
                  *first_, {}, {}, /*read_published_state=*/false, prepare)
                  .ok());
  TaskPipelineTestPeer::context_gather(*first_);
  TaskPipelineTestPeer::context_publish(*first_);
  TaskPipelineTestPeer::context_release(*first_);
}

TEST_F(MtpContextTest, AcceptedPrefixesAdvanceStateEntirelyOnDevice) {
  Stream prepare(device_);
  Stream task(device_);
  ASSERT_TRUE(
      TaskPipelineTestPeer::context_prepare(
          *first_, keys_, requests_, /*read_published_state=*/false, prepare)
          .ok());
  const auto ready = prepare.record_event();
  torch::Tensor tokens =
      torch::tensor({{11, -1, -1, -1}, {21, 22, 23, 24}}, torch::kInt64)
          .to(device_);
  torch::Tensor lengths = torch::tensor({1, 4}, torch::kInt32).to(device_);
  torch::Tensor hidden =
      torch::arange(24, torch::kFloat32).view({2, 4, 3}).to(device_);
  ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  const void* state_address = first_->state_.hidden.data_ptr();
  {
    auto guard = task.set_stream_guard();
    ASSERT_TRUE(task.wait_event(ready));
    first_->state_.positions.copy_(torch::tensor({6, 12}, torch::kInt32));
    first_->state_.kv_seq_lens.copy_(torch::tensor({7, 13}, torch::kInt32));
    TaskPipelineTestPeer::context_advance(*first_, tokens, lengths, hidden);
    TaskPipelineTestPeer::context_publish(*first_);
  }
  ASSERT_EQ(task.synchronize(), 0);
  EXPECT_TRUE(torch::equal(first_->tokens_.cpu(),
                           torch::tensor({11, 24}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(first_->state_.previous_tokens.cpu(),
                           torch::tensor({11, 23}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(first_->state_.positions.cpu(),
                           torch::tensor({7, 16}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(first_->state_.kv_seq_lens.cpu(),
                           torch::tensor({8, 17}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(first_->state_.repair_required.cpu(),
                           torch::tensor({false, true}, torch::kBool)));
  EXPECT_TRUE(torch::equal(
      first_->state_.hidden.select(1, 1).cpu(),
      torch::tensor({{0., 1., 2.}, {21., 22., 23.}}, torch::kFloat32)));
  EXPECT_TRUE(torch::equal(
      first_->state_.hidden.select(1, 0).cpu(),
      torch::tensor({{0., 0., 0.}, {18., 19., 20.}}, torch::kFloat32)));
  // A partial prefix on the next Task replaces the full-accept repair flag.
  tokens.copy_(
      torch::tensor({{31, 32, -1, -1}, {41, 42, 43, -1}}, torch::kInt64));
  lengths.copy_(torch::tensor({2, 3}, torch::kInt32));
  TaskPipelineTestPeer::context_advance(*first_, tokens, lengths, hidden);
  ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  EXPECT_EQ(first_->state_.hidden.data_ptr(), state_address);
  EXPECT_TRUE(torch::equal(first_->tokens_.cpu(),
                           torch::tensor({32, 43}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(first_->state_.previous_tokens.cpu(),
                           torch::tensor({31, 42}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(first_->state_.positions.cpu(),
                           torch::tensor({9, 19}, torch::kInt32)));
  EXPECT_FALSE(first_->state_.repair_required.cpu().any().item<bool>());
}

TEST_F(MtpContextTest, ChunkedPrefillPublishesOnlyCompletedRows) {
  Stream prepare(device_);
  Stream task(device_);
  const std::array<uint32_t, 1> complete{{1}};
  const std::array<int32_t, 2> partial_extra{{99, -1}};
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare_prefill(
                  *first_,
                  std::span(keys_).last(1),
                  std::span(requests_).last(1),
                  partial_extra,
                  complete,
                  prepare)
                  .ok());
  EXPECT_EQ(first_->tokens_.numel(), 1);
  EXPECT_FALSE(
      TaskPipelineTestPeer::context_prepare(*second_,
                                            std::span(keys_).first(1),
                                            std::span(requests_).first(1),
                                            /*read_published_state=*/true,
                                            prepare)
          .ok());
  ASSERT_TRUE(
      TaskPipelineTestPeer::context_prepare(*second_,
                                            std::span(keys_).last(1),
                                            std::span(requests_).last(1),
                                            /*read_published_state=*/true,
                                            prepare)
          .ok());
  const auto ready = prepare.record_event();
  {
    auto guard = task.set_stream_guard();
    ASSERT_TRUE(task.wait_event(ready));
    first_->tokens_.fill_(202);
    first_->state_.previous_tokens.fill_(201);
    first_->state_.hidden.fill_(3);
    first_->state_.positions.fill_(13);
    first_->state_.kv_seq_lens.fill_(14);
    first_->state_.repair_required.fill_(false);
    TaskPipelineTestPeer::context_publish(*first_);
    TaskPipelineTestPeer::context_gather(*second_);
  }
  ASSERT_EQ(task.synchronize(), 0);
  EXPECT_EQ(second_->tokens_.cpu().item<int64_t>(), 202);
  TaskPipelineTestPeer::context_release(*first_);
  TaskPipelineTestPeer::context_release(*second_);

  // A later chunk completes both rows; publication follows sampling order.
  const std::array<uint32_t, 2> reversed_rows{{1, 0}};
  const std::array<int32_t, 2> complete_extra{{-1, -1}};
  ASSERT_TRUE(
      TaskPipelineTestPeer::context_prepare_prefill(
          *first_, keys_, requests_, complete_extra, reversed_rows, prepare)
          .ok());
  ASSERT_TRUE(
      TaskPipelineTestPeer::context_prepare(*second_,
                                            keys_,
                                            requests_,
                                            /*read_published_state=*/true,
                                            prepare)
          .ok());
  const auto complete_ready = prepare.record_event();
  {
    auto guard = task.set_stream_guard();
    ASSERT_TRUE(task.wait_event(complete_ready));
    seed(*first_);
    TaskPipelineTestPeer::context_gather(*second_);
  }
  ASSERT_EQ(task.synchronize(), 0);
  EXPECT_TRUE(torch::equal(second_->tokens_.cpu(),
                           torch::tensor({202, 101}, torch::kInt64)));
}

TEST_F(MtpContextTest, IncompletePrefillAndInvalidPublicationRows) {
  Stream prepare(device_);
  const std::array<uint32_t, 2> duplicate{{0, 0}};
  const std::array<uint32_t, 2> outside{{0, 2}};
  const std::array<int32_t, 2> complete_extra{{-1, -1}};
  const std::array<int32_t, 2> incomplete_extra{{99, 98}};
  EXPECT_FALSE(
      TaskPipelineTestPeer::context_prepare_prefill(
          *first_, keys_, requests_, complete_extra, duplicate, prepare)
          .ok());
  EXPECT_FALSE(TaskPipelineTestPeer::context_prepare_prefill(
                   *first_, keys_, requests_, complete_extra, outside, prepare)
                   .ok());
  EXPECT_FALSE(TaskPipelineTestPeer::context_prepare_prefill(
                   *first_, keys_, requests_, incomplete_extra, {}, prepare)
                   .ok());
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare_prefill(
                  *first_, {}, {}, incomplete_extra, {}, prepare)
                  .ok());
  ASSERT_TRUE(first_->prepared_);
  EXPECT_EQ(first_->tokens_.numel(), 0);
  TaskPipelineTestPeer::context_publish(*first_);
  TaskPipelineTestPeer::context_gather(*first_);
  EXPECT_FALSE(
      TaskPipelineTestPeer::context_prepare(*second_,
                                            keys_,
                                            requests_,
                                            /*read_published_state=*/true,
                                            prepare)
          .ok());
  ASSERT_EQ(prepare.synchronize(), 0);
  TaskPipelineTestPeer::context_release(*first_);
}

TEST_F(MtpContextTest, BootstrapRowsJoinPublishedStateAndSurviveSlotRelease) {
  Stream prepare(device_);
  Stream task(device_);
  ASSERT_TRUE(
      TaskPipelineTestPeer::context_prepare(
          *first_, keys_, requests_, /*read_published_state=*/false, prepare)
          .ok());
  auto ready = prepare.record_event();
  {
    auto guard = task.set_stream_guard();
    ASSERT_TRUE(task.wait_event(ready));
    seed(*first_);
  }
  ASSERT_EQ(task.synchronize(), 0);
  TaskPipelineTestPeer::context_release(*first_);

  const std::array<int32_t, 2> mixed{{keys_[1], 3}};
  const std::array<int32_t, 1> bootstrap{{1}};
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare_decode(
                  *second_, mixed, requests_, bootstrap, prepare)
                  .ok());
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare_decode(
                  *first_, mixed, requests_, {}, prepare)
                  .ok());
  ready = prepare.record_event();
  {
    auto guard = task.set_stream_guard();
    ASSERT_TRUE(task.wait_event(ready));
    TaskPipelineTestPeer::context_gather(*second_);
    second_->tokens_.select(/*dim=*/0, /*index=*/1).fill_(303);
    const auto& state = second_->state_;
    state.previous_tokens.select(/*dim=*/0, /*index=*/1).fill_(303);
    state.hidden.select(/*dim=*/0, /*index=*/1).fill_(7);
    state.positions.select(/*dim=*/0, /*index=*/1).fill_(21);
    state.kv_seq_lens.select(/*dim=*/0, /*index=*/1).fill_(22);
    state.repair_required.select(/*dim=*/0, /*index=*/1).fill_(false);
    TaskPipelineTestPeer::context_publish(*second_);
    TaskPipelineTestPeer::context_gather(*first_);
  }
  ASSERT_EQ(task.synchronize(), 0);
  TaskPipelineTestPeer::context_release(*second_);
  EXPECT_TRUE(torch::equal(first_->tokens_.cpu(),
                           torch::tensor({202, 303}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(first_->state_.previous_tokens.cpu(),
                           torch::tensor({201, 303}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(first_->state_.positions.cpu(),
                           torch::tensor({13, 21}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(first_->state_.kv_seq_lens.cpu(),
                           torch::tensor({14, 22}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(first_->state_.hidden.cpu().select(
                               /*dim=*/0, /*index=*/0),
                           torch::arange(6, 12, torch::kFloat32).view({2, 3})));
  EXPECT_TRUE(torch::equal(first_->state_.hidden.cpu().select(
                               /*dim=*/0, /*index=*/1),
                           torch::full({2, 3}, 7, torch::kFloat32)));
  EXPECT_FALSE(first_->state_.repair_required.cpu().any().item<bool>());
}

TEST_F(MtpContextTest, InvalidBootstrapCannotAdmitUnpublishedRows) {
  Stream prepare(device_);
  const std::array<int32_t, 2> duplicate{{0, 0}};
  const std::array<int32_t, 1> outside{{2}};
  const std::array<int32_t, 1> partial{{0}};
  EXPECT_FALSE(TaskPipelineTestPeer::context_prepare_decode(
                   *first_, keys_, requests_, duplicate, prepare)
                   .ok());
  EXPECT_FALSE(TaskPipelineTestPeer::context_prepare_decode(
                   *first_, keys_, requests_, outside, prepare)
                   .ok());
  EXPECT_FALSE(TaskPipelineTestPeer::context_prepare_decode(
                   *first_, keys_, requests_, partial, prepare)
                   .ok());
  EXPECT_FALSE(first_->prepared_);
  const std::array<int32_t, 2> all{{1, 0}};
  ASSERT_TRUE(TaskPipelineTestPeer::context_prepare_decode(
                  *first_, keys_, requests_, all, prepare)
                  .ok());
  ASSERT_EQ(prepare.synchronize(), 0);
}

}  // namespace
}  // namespace xllm
