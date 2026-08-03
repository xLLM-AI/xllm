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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <torch/csrc/distributed/c10d/FakeProcessGroup.hpp>
#include <vector>

#include "core/framework/parallel_state/process_group.h"

namespace xllm {
namespace {

class RecordingWork final : public c10d::Work {
 public:
  RecordingWork(const size_t* posted_operation_count,
                std::vector<size_t>* wait_posted_counts)
      : posted_operation_count_(posted_operation_count),
        wait_posted_counts_(wait_posted_counts) {}

  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    (void)timeout;
    wait_posted_counts_->push_back(*posted_operation_count_);
    return true;
  }

 private:
  const size_t* posted_operation_count_;
  std::vector<size_t>* wait_posted_counts_;
};

class RecordingBackend final : public c10d::FakeProcessGroup {
 public:
  RecordingBackend(int32_t rank, int32_t world_size)
      : c10d::FakeProcessGroup(rank, world_size) {}

  c10::intrusive_ptr<c10d::Work> send(std::vector<torch::Tensor>& tensors,
                                      int destination_rank,
                                      int tag) override {
    storage_offsets_.push_back(tensors.front().storage_offset());
    data_ptrs_.push_back(
        reinterpret_cast<uintptr_t>(tensors.front().data_ptr()));
    (void)destination_rank;
    (void)tag;
    return record_operation();
  }

  c10::intrusive_ptr<c10d::Work> recv(std::vector<torch::Tensor>& tensors,
                                      int source_rank,
                                      int tag) override {
    storage_offsets_.push_back(tensors.front().storage_offset());
    data_ptrs_.push_back(
        reinterpret_cast<uintptr_t>(tensors.front().data_ptr()));
    (void)source_rank;
    (void)tag;
    return record_operation();
  }

  size_t posted_operation_count() const { return posted_operation_count_; }

  const std::vector<size_t>& wait_posted_counts() const {
    return wait_posted_counts_;
  }

  const std::vector<int64_t>& storage_offsets() const {
    return storage_offsets_;
  }

  const std::vector<uintptr_t>& data_ptrs() const { return data_ptrs_; }

 private:
  c10::intrusive_ptr<c10d::Work> record_operation() {
    ++posted_operation_count_;
    return c10::make_intrusive<RecordingWork>(&posted_operation_count_,
                                              &wait_posted_counts_);
  }

  size_t posted_operation_count_ = 0;
  std::vector<size_t> wait_posted_counts_;
  std::vector<int64_t> storage_offsets_;
  std::vector<uintptr_t> data_ptrs_;
};

class RecordingProcessGroup final : public ProcessGroup {
 public:
  RecordingProcessGroup(int32_t rank,
                        int32_t world_size,
                        int64_t max_wave_payload_bytes = 256 * 1024 * 1024)
      : ProcessGroup(rank, world_size, torch::Device(torch::kCPU)),
        max_wave_payload_bytes_(max_wave_payload_bytes) {
    backend_owner_ = std::make_unique<RecordingBackend>(rank, world_size);
    backend_ = backend_owner_.get();
  }

  const RecordingBackend& backend() const { return *backend_; }

 private:
  int64_t max_p2p_wave_payload_bytes() const override {
    return max_wave_payload_bytes_;
  }

  int32_t synchronize_p2p_staging() override { return 0; }

  c10::intrusive_ptr<c10d::Work> send_p2p(std::vector<torch::Tensor>& tensors,
                                          int64_t peer_rank,
                                          int32_t tag) override {
    return backend_->send(tensors, peer_rank, tag);
  }

  c10::intrusive_ptr<c10d::Work> recv_p2p(std::vector<torch::Tensor>& tensors,
                                          int64_t peer_rank,
                                          int32_t tag) override {
    return backend_->recv(tensors, peer_rank, tag);
  }

  std::unique_ptr<RecordingBackend> backend_owner_;
  RecordingBackend* backend_ = nullptr;
  int64_t max_wave_payload_bytes_;
};

TEST(ProcessGroupBatchTest, ReturnsWorkAfterPostingEveryOperation) {
  RecordingProcessGroup process_group(/*rank=*/0, /*world_size=*/2);
  std::vector<std::string> op_types = {"send", "recv", "send", "recv"};
  std::vector<torch::Tensor> tensors = {
      torch::zeros({2}),
      torch::zeros({2}),
      torch::zeros({2}),
      torch::zeros({2}),
  };
  std::vector<int64_t> remote_ranks(op_types.size(), 1);

  c10::intrusive_ptr<c10d::Work> work =
      process_group.batch_isend_irecv(op_types, tensors, remote_ranks);

  const RecordingBackend& backend = process_group.backend();
  ASSERT_NE(work, nullptr);
  ASSERT_EQ(backend.posted_operation_count(), op_types.size());
  EXPECT_TRUE(backend.wait_posted_counts().empty());

  EXPECT_TRUE(work->wait());

  ASSERT_EQ(backend.wait_posted_counts().size(), op_types.size());
  for (size_t posted_count_at_wait : backend.wait_posted_counts()) {
    EXPECT_EQ(posted_count_at_wait, op_types.size());
  }
}

TEST(ProcessGroupBatchTest, LargePayloadChunkingDoesNotDependOnStorageOffset) {
  constexpr int64_t kLargePayloadBytes = 64 * 1024 * 1024 + 1;

  RecordingProcessGroup zero_offset_group(/*rank=*/0, /*world_size=*/2);
  std::vector<std::string> zero_offset_types = {"send"};
  std::vector<torch::Tensor> zero_offset_tensors = {
      torch::empty({kLargePayloadBytes}, torch::kUInt8)};
  std::vector<int64_t> zero_offset_ranks = {1};
  c10::intrusive_ptr<c10d::Work> zero_offset_work =
      zero_offset_group.batch_isend_irecv(
          zero_offset_types, zero_offset_tensors, zero_offset_ranks);
  ASSERT_NE(zero_offset_work, nullptr);
  ASSERT_TRUE(zero_offset_work->wait());

  RecordingProcessGroup offset_group(/*rank=*/0, /*world_size=*/2);
  std::vector<std::string> offset_types = {"send"};
  torch::Tensor offset_storage =
      torch::empty({kLargePayloadBytes + 1}, torch::kUInt8);
  std::vector<torch::Tensor> offset_tensors = {
      offset_storage.narrow(/*dim=*/0, /*start=*/1, kLargePayloadBytes)};
  std::vector<int64_t> offset_ranks = {1};
  c10::intrusive_ptr<c10d::Work> offset_work = offset_group.batch_isend_irecv(
      offset_types, offset_tensors, offset_ranks);
  ASSERT_NE(offset_work, nullptr);
  ASSERT_TRUE(offset_work->wait());

  EXPECT_EQ(zero_offset_group.backend().posted_operation_count(), 2u);
  EXPECT_EQ(offset_group.backend().posted_operation_count(), 2u);
}

TEST(ProcessGroupBatchTest, ContiguousOffsetViewUsesOwnedStaging) {
  RecordingProcessGroup process_group(/*rank=*/0, /*world_size=*/2);
  std::vector<std::string> op_types = {"send"};
  torch::Tensor storage = torch::zeros({4}, torch::kFloat32);
  std::vector<torch::Tensor> tensors = {
      storage.narrow(/*dim=*/0, /*start=*/1, /*length=*/2)};
  std::vector<int64_t> remote_ranks = {1};
  const uintptr_t expected_data_ptr =
      reinterpret_cast<uintptr_t>(tensors.front().data_ptr());

  process_group.batch_isend_irecv(op_types, tensors, remote_ranks);

  ASSERT_EQ(process_group.backend().storage_offsets().size(), 1u);
  EXPECT_EQ(process_group.backend().storage_offsets().front(), 0);
  ASSERT_EQ(process_group.backend().data_ptrs().size(), 1u);
  EXPECT_NE(process_group.backend().data_ptrs().front(), expected_data_ptr);
}

TEST(ProcessGroupBatchTest, LimitsOutstandingPayloadToOneWave) {
  RecordingProcessGroup process_group(
      /*rank=*/0, /*world_size=*/2, /*max_wave_payload_bytes=*/8);
  std::vector<std::string> op_types = {"send", "recv", "send", "recv"};
  std::vector<torch::Tensor> tensors = {
      torch::zeros({2}, torch::kFloat32),
      torch::zeros({2}, torch::kFloat32),
      torch::zeros({2}, torch::kFloat32),
      torch::zeros({2}, torch::kFloat32),
  };
  std::vector<int64_t> remote_ranks(op_types.size(), 1);

  c10::intrusive_ptr<c10d::Work> work =
      process_group.batch_isend_irecv(op_types, tensors, remote_ranks);

  ASSERT_NE(work, nullptr);
  EXPECT_EQ(process_group.backend().posted_operation_count(), 1u);
  EXPECT_TRUE(work->wait());
  EXPECT_EQ(process_group.backend().posted_operation_count(), op_types.size());
  ASSERT_EQ(process_group.backend().wait_posted_counts().size(),
            op_types.size());
  EXPECT_EQ(process_group.backend().wait_posted_counts(),
            (std::vector<size_t>{1, 2, 3, 4}));
}

}  // namespace
}  // namespace xllm
