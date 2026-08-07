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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/framework/model/model_input_params.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/platform/platform.h"
#include "core/runtime/llm_worker_impl.h"
#include "core/runtime/mtp_worker_impl.h"
#include "core/runtime/options.h"
#include "core/util/slice.h"

namespace xllm {
namespace {

class RecordingTransferWorker final : public LLMWorkerImpl {
 public:
  RecordingTransferWorker(const ParallelArgs& parallel_args,
                          const torch::Device& device,
                          const runtime::Options& options,
                          uint32_t transfer_result)
      : LLMWorkerImpl(parallel_args, device, options),
        transfer_result_(transfer_result) {}

  uint32_t transfer_kv_blocks(
      uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info) override {
    last_batch_id_ = batch_id;
    last_transfer_size_ = block_transfer_info.size();
    ++vector_transfer_count_;
    return transfer_result_;
  }

  uint32_t transfer_kv_blocks(
      uint64_t batch_id,
      Slice<BlockTransferInfo>& block_transfer_info) override {
    last_batch_id_ = batch_id;
    last_transfer_size_ = block_transfer_info.size();
    ++slice_transfer_count_;
    return transfer_result_;
  }

  uint32_t vector_transfer_count() const { return vector_transfer_count_; }
  uint32_t slice_transfer_count() const { return slice_transfer_count_; }
  uint64_t last_batch_id() const { return last_batch_id_; }
  size_t last_transfer_size() const { return last_transfer_size_; }

 private:
  uint32_t transfer_result_ = 0;
  uint32_t vector_transfer_count_ = 0;
  uint32_t slice_transfer_count_ = 0;
  uint64_t last_batch_id_ = 0;
  size_t last_transfer_size_ = 0;
};

class TestMTPWorker final : public MTPWorkerImpl {
 public:
  TestMTPWorker(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options)
      : MTPWorkerImpl(parallel_args, device, options) {}

  void replace_transfer_workers(std::unique_ptr<LLMWorkerImpl> target,
                                std::unique_ptr<LLMWorkerImpl> draft) {
    impl_ = std::move(target);
    draft_impl_ = std::move(draft);
  }
};

class MTPHostOffloadTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() < 1) {
      GTEST_SKIP() << "MLU device is required for MTP host offload tests.";
    }
  }
};

TEST_F(MTPHostOffloadTest, TransfersEveryBlockToTargetAndDraft) {
  constexpr uint64_t kBatchId = 42;
  const torch::Device device("mlu:0");
  ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  runtime::Options options;
  options.block_size(16).num_speculative_tokens(1);
  TestMTPWorker worker(parallel_args, device, options);

  const std::vector<BlockTransferInfo> transfer_info = {
      BlockTransferInfo(/*src_block_id=*/1, /*dst_block_id=*/2),
      BlockTransferInfo(/*src_block_id=*/3, /*dst_block_id=*/4)};
  auto target = std::make_unique<RecordingTransferWorker>(
      parallel_args,
      device,
      options,
      static_cast<uint32_t>(transfer_info.size()));
  auto draft = std::make_unique<RecordingTransferWorker>(
      parallel_args,
      device,
      options,
      static_cast<uint32_t>(transfer_info.size()));
  RecordingTransferWorker* target_ptr = target.get();
  RecordingTransferWorker* draft_ptr = draft.get();
  worker.replace_transfer_workers(std::move(target), std::move(draft));

  const uint32_t transferred =
      worker.transfer_kv_blocks(kBatchId, transfer_info);

  EXPECT_EQ(transferred, transfer_info.size());
  EXPECT_EQ(target_ptr->vector_transfer_count(), 1);
  EXPECT_EQ(draft_ptr->vector_transfer_count(), 1);
  EXPECT_EQ(target_ptr->last_batch_id(), kBatchId);
  EXPECT_EQ(draft_ptr->last_batch_id(), kBatchId);
  EXPECT_EQ(target_ptr->last_transfer_size(), transfer_info.size());
  EXPECT_EQ(draft_ptr->last_transfer_size(), transfer_info.size());
}

TEST_F(MTPHostOffloadTest, RejectsMismatchedTargetAndDraftTransferCounts) {
  constexpr uint64_t kBatchId = 73;
  const torch::Device device("mlu:0");
  ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  runtime::Options options;
  options.block_size(16).num_speculative_tokens(1);
  TestMTPWorker worker(parallel_args, device, options);

  const std::vector<BlockTransferInfo> transfer_info = {
      BlockTransferInfo(/*src_block_id=*/5, /*dst_block_id=*/6)};
  auto target = std::make_unique<RecordingTransferWorker>(
      parallel_args, device, options, /*transfer_result=*/1);
  auto draft = std::make_unique<RecordingTransferWorker>(
      parallel_args, device, options, /*transfer_result=*/0);
  RecordingTransferWorker* target_ptr = target.get();
  RecordingTransferWorker* draft_ptr = draft.get();
  worker.replace_transfer_workers(std::move(target), std::move(draft));
  Slice<BlockTransferInfo> transfer_slice(transfer_info);

  const uint32_t transferred =
      worker.transfer_kv_blocks(kBatchId, transfer_slice);

  EXPECT_EQ(transferred, 0);
  EXPECT_EQ(target_ptr->slice_transfer_count(), 1);
  EXPECT_EQ(draft_ptr->slice_transfer_count(), 1);
}

}  // namespace
}  // namespace xllm
