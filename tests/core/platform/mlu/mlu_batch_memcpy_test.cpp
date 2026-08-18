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

#include "core/platform/mlu/mlu_batch_memcpy.h"

#include <cn_api.h>
#include <cnrt.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include "core/framework/kv_cache/kv_cache_utils.h"
#include "core/platform/batch_memcpy.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"

namespace {

enum class BatchCopyFault : int8_t {
  NONE = 0,
  FAIL_SECOND_SUBMISSION = 1,
  FAIL_SUBMISSION_AND_SYNC = 2,
};

std::atomic<BatchCopyFault> g_batch_copy_fault{BatchCopyFault::NONE};
std::atomic<int32_t> g_batch_submit_calls{0};
std::atomic<int32_t> g_queue_sync_calls{0};

class ScopedBatchCopyFault final {
 public:
  explicit ScopedBatchCopyFault(BatchCopyFault fault) {
    g_batch_submit_calls.store(0, std::memory_order_relaxed);
    g_queue_sync_calls.store(0, std::memory_order_relaxed);
    g_batch_copy_fault.store(fault, std::memory_order_release);
  }

  ~ScopedBatchCopyFault() {
    g_batch_copy_fault.store(BatchCopyFault::NONE, std::memory_order_release);
  }

  ScopedBatchCopyFault(const ScopedBatchCopyFault&) = delete;
  ScopedBatchCopyFault& operator=(const ScopedBatchCopyFault&) = delete;
};

void wait_for_queue_gate(void* user_data) {
  std::atomic<bool>* gate_open = static_cast<std::atomic<bool>*>(user_data);
  while (!gate_open->load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

}  // namespace

extern "C" CNresult __real_cnMemcpyBatchAsync(
    CNaddr* dsts,
    CNaddr* srcs,
    size_t* bytes,
    size_t count,
    CNmemcpyBatchAsyncAttributes* attrs,
    size_t* attr_indexes,
    size_t num_attrs,
    CNqueue queue);

extern "C" CNresult __wrap_cnMemcpyBatchAsync(
    CNaddr* dsts,
    CNaddr* srcs,
    size_t* bytes,
    size_t count,
    CNmemcpyBatchAsyncAttributes* attrs,
    size_t* attr_indexes,
    size_t num_attrs,
    CNqueue queue) {
  const int32_t call =
      g_batch_submit_calls.fetch_add(1, std::memory_order_relaxed) + 1;
  const BatchCopyFault fault =
      g_batch_copy_fault.load(std::memory_order_acquire);
  if ((fault == BatchCopyFault::FAIL_SECOND_SUBMISSION && call == 2) ||
      (fault == BatchCopyFault::FAIL_SUBMISSION_AND_SYNC && call == 1)) {
    return CN_ERROR_INVALID_VALUE;
  }
  return __real_cnMemcpyBatchAsync(
      dsts, srcs, bytes, count, attrs, attr_indexes, num_attrs, queue);
}

extern "C" CNresult __real_cnQueueSync(CNqueue queue);

extern "C" CNresult __wrap_cnQueueSync(CNqueue queue) {
  g_queue_sync_calls.fetch_add(1, std::memory_order_relaxed);
  if (g_batch_copy_fault.load(std::memory_order_acquire) ==
      BatchCopyFault::FAIL_SUBMISSION_AND_SYNC) {
    return CN_ERROR_INVALID_VALUE;
  }
  return __real_cnQueueSync(queue);
}

namespace xllm::mlu {
namespace {

std::vector<torch::Tensor> rows(const torch::Tensor& tensor) {
  std::vector<torch::Tensor> result;
  result.reserve(static_cast<size_t>(tensor.size(0)));
  for (int64_t row = 0; row < tensor.size(0); ++row) {
    result.emplace_back(tensor[row]);
  }
  return result;
}

class MLUBatchMemcpyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() < 1) {
      GTEST_SKIP() << "MLU device is required for batch memcpy tests.";
    }
    device_ = std::make_unique<Device>(/*device_index=*/0);
    device_->set_device();
    stream_ = device_->get_stream_from_pool();
    batch_memcpy_ = create_batch_memcpy(*device_);
    ASSERT_NE(batch_memcpy_, nullptr);
  }

  void expect_round_trip(int64_t count, int64_t width) {
    torch::Tensor source;
    HostPageAlignedRegion source_region;
    create_host_page_aligned_tensor(
        {count, width}, torch::kUInt8, &source, &source_region);
    uint8_t* source_bytes = source.mutable_data_ptr<uint8_t>();
    for (int64_t index = 0; index < source.numel(); ++index) {
      source_bytes[index] = static_cast<uint8_t>((index * 17 + 3) % 251);
    }

    const torch::Tensor device_tensor = torch::zeros(
        {count, width},
        torch::TensorOptions().dtype(torch::kUInt8).device(device_->unwrap()));
    torch::Tensor restored;
    HostPageAlignedRegion restored_region;
    create_host_page_aligned_tensor(
        {count, width}, torch::kUInt8, &restored, &restored_region);

    ASSERT_TRUE(batch_memcpy_->submit_h2d(
        rows(source), rows(device_tensor), stream_.get()));
    ASSERT_TRUE(batch_memcpy_->copy_d2h(
        rows(device_tensor), rows(restored), stream_.get()));
    EXPECT_TRUE(torch::equal(source, restored));
  }

  std::unique_ptr<Device> device_;
  std::unique_ptr<Stream> stream_;
  std::unique_ptr<BatchMemcpy> batch_memcpy_;
};

TEST_F(MLUBatchMemcpyTest, RoundTripCompletesForOneDescriptor) {
  expect_round_trip(/*count=*/1, /*width=*/19);
}

TEST_F(MLUBatchMemcpyTest, RoundTripCompletesAtChunkLimit) {
  expect_round_trip(/*count=*/4096, /*width=*/3);
}

TEST_F(MLUBatchMemcpyTest, RoundTripCompletesAcrossChunkBoundary) {
  expect_round_trip(/*count=*/4097, /*width=*/3);
}

TEST_F(MLUBatchMemcpyTest, RoundTripSupportsDifferentTensorSizes) {
  const std::vector<int64_t> widths = {1, 7, 33};
  std::vector<HostPageAlignedRegion> source_regions(widths.size());
  std::vector<HostPageAlignedRegion> restored_regions(widths.size());
  std::vector<torch::Tensor> sources(widths.size());
  std::vector<torch::Tensor> restored(widths.size());
  std::vector<torch::Tensor> device_tensors;
  device_tensors.reserve(widths.size());

  for (size_t index = 0; index < widths.size(); ++index) {
    create_host_page_aligned_tensor({widths[index]},
                                    torch::kUInt8,
                                    &sources[index],
                                    &source_regions[index]);
    sources[index].fill_(static_cast<int64_t>(index + 1));
    create_host_page_aligned_tensor({widths[index]},
                                    torch::kUInt8,
                                    &restored[index],
                                    &restored_regions[index]);
    device_tensors.emplace_back(torch::zeros(
        {widths[index]},
        torch::TensorOptions().dtype(torch::kUInt8).device(device_->unwrap())));
  }

  ASSERT_TRUE(
      batch_memcpy_->submit_h2d(sources, device_tensors, stream_.get()));
  ASSERT_TRUE(batch_memcpy_->copy_d2h(device_tensors, restored, stream_.get()));
  for (size_t index = 0; index < widths.size(); ++index) {
    EXPECT_TRUE(torch::equal(sources[index], restored[index]));
  }
}

TEST_F(MLUBatchMemcpyTest, SubmitH2DReturnsBeforeCopyStreamCompletes) {
  torch::Tensor host;
  HostPageAlignedRegion host_region;
  create_host_page_aligned_tensor({16}, torch::kUInt8, &host, &host_region);
  host.fill_(23);
  const torch::Tensor device_tensor = torch::zeros(
      {16},
      torch::TensorOptions().dtype(torch::kUInt8).device(device_->unwrap()));
  ASSERT_EQ(device_->synchronize_default_stream(), 0);

  std::atomic<bool> gate_open{false};
  ASSERT_EQ(cnrtInvokeHostFunc(
                stream_->get_stream()->stream(),
                [](void* user_data) {
                  std::atomic<bool>* gate =
                      static_cast<std::atomic<bool>*>(user_data);
                  while (!gate->load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                  }
                },
                &gate_open),
            cnrtSuccess);

  std::future<bool> submit_result =
      std::async(std::launch::async, [this, &host, &device_tensor]() {
        device_->set_device();
        device_->init_device_context();
        return batch_memcpy_->submit_h2d(
            {host}, {device_tensor}, stream_.get());
      });
  const std::future_status submit_status =
      submit_result.wait_for(std::chrono::milliseconds(250));
  gate_open.store(true, std::memory_order_release);

  ASSERT_EQ(submit_result.wait_for(std::chrono::seconds(2)),
            std::future_status::ready);
  ASSERT_TRUE(submit_result.get());
  ASSERT_EQ(stream_->synchronize(), 0);
  EXPECT_EQ(submit_status, std::future_status::ready);
  EXPECT_TRUE(torch::all(device_tensor.cpu() == 23).item<bool>());
}

TEST_F(MLUBatchMemcpyTest, SubmissionFailureDrainsPreviouslySubmittedChunks) {
  constexpr int64_t kDescriptorCount = 4097;
  torch::Tensor host;
  HostPageAlignedRegion host_region;
  create_host_page_aligned_tensor(
      {kDescriptorCount, 1}, torch::kUInt8, &host, &host_region);
  host.fill_(31);
  const torch::Tensor device_tensor = torch::zeros(
      {kDescriptorCount, 1},
      torch::TensorOptions().dtype(torch::kUInt8).device(device_->unwrap()));
  ASSERT_EQ(device_->synchronize_default_stream(), 0);

  std::atomic<bool> gate_open{false};
  ASSERT_EQ(
      cnrtInvokeHostFunc(
          stream_->get_stream()->stream(), wait_for_queue_gate, &gate_open),
      cnrtSuccess);
  ScopedBatchCopyFault fault(BatchCopyFault::FAIL_SECOND_SUBMISSION);
  const std::vector<torch::Tensor> host_rows = rows(host);
  const std::vector<torch::Tensor> device_rows = rows(device_tensor);
  std::future<bool> submit_result =
      std::async(std::launch::async, [this, &host_rows, &device_rows]() {
        device_->set_device();
        device_->init_device_context();
        return batch_memcpy_->submit_h2d(host_rows, device_rows, stream_.get());
      });

  const std::future_status drain_status =
      submit_result.wait_for(std::chrono::milliseconds(100));
  gate_open.store(true, std::memory_order_release);

  ASSERT_EQ(submit_result.wait_for(std::chrono::seconds(2)),
            std::future_status::ready);
  EXPECT_FALSE(submit_result.get());
  EXPECT_EQ(drain_status, std::future_status::timeout);
  EXPECT_EQ(g_batch_submit_calls.load(std::memory_order_relaxed), 2);
  EXPECT_EQ(g_queue_sync_calls.load(std::memory_order_relaxed), 1);
  EXPECT_TRUE(
      torch::all(
          device_tensor.slice(/*dim=*/0, /*start=*/0, /*end=*/4096).cpu() == 31)
          .item<bool>());
  EXPECT_TRUE(torch::all(device_tensor[4096].cpu() == 0).item<bool>());
}

TEST_F(MLUBatchMemcpyTest, UndrainableSubmissionFailureTerminatesProcess) {
  torch::Tensor host;
  HostPageAlignedRegion host_region;
  create_host_page_aligned_tensor({1}, torch::kUInt8, &host, &host_region);
  const torch::Tensor device_tensor = torch::zeros(
      {1},
      torch::TensorOptions().dtype(torch::kUInt8).device(device_->unwrap()));

  EXPECT_DEATH(
      {
        ScopedBatchCopyFault fault(BatchCopyFault::FAIL_SUBMISSION_AND_SYNC);
        (void)batch_memcpy_->submit_h2d({host}, {device_tensor}, stream_.get());
      },
      "Failed to drain MLU batch memcpy queue after submission failure");
}

TEST_F(MLUBatchMemcpyTest, RejectsInvalidInputs) {
  torch::Tensor host;
  HostPageAlignedRegion host_region;
  create_host_page_aligned_tensor({2, 4}, torch::kUInt8, &host, &host_region);
  const torch::Tensor device_tensor = torch::zeros(
      {2, 4},
      torch::TensorOptions().dtype(torch::kUInt8).device(device_->unwrap()));

  EXPECT_FALSE(batch_memcpy_->submit_h2d(
      {host[0]}, {device_tensor[0], device_tensor[1]}, stream_.get()));
  EXPECT_FALSE(batch_memcpy_->submit_h2d(
      {host[0]}, {device_tensor.flatten()}, stream_.get()));
  EXPECT_FALSE(batch_memcpy_->submit_h2d(
      {host.transpose(0, 1)}, {device_tensor}, stream_.get()));
  EXPECT_FALSE(
      batch_memcpy_->submit_h2d({device_tensor[0]}, {host[0]}, stream_.get()));
  EXPECT_FALSE(
      batch_memcpy_->copy_d2h({host[0]}, {device_tensor[0]}, stream_.get()));
  EXPECT_FALSE(
      batch_memcpy_->submit_h2d({host[0]}, {device_tensor[0]}, nullptr));
  EXPECT_FALSE(batch_memcpy_->submit_h2d(
      {torch::Tensor()}, {device_tensor[0]}, stream_.get()));
}

TEST_F(MLUBatchMemcpyTest, RejectsStreamFromAnotherDevice) {
  if (Platform::device_count() < 2) {
    GTEST_SKIP() << "Two MLU devices are required for stream mismatch test.";
  }

  torch::Tensor host;
  HostPageAlignedRegion host_region;
  create_host_page_aligned_tensor({8}, torch::kUInt8, &host, &host_region);
  Device other_device(/*device_index=*/1);
  other_device.set_device();
  const torch::Tensor other_tensor =
      torch::zeros({8},
                   torch::TensorOptions()
                       .dtype(torch::kUInt8)
                       .device(other_device.unwrap()));
  device_->set_device();

  EXPECT_FALSE(
      batch_memcpy_->submit_h2d({host}, {other_tensor}, stream_.get()));
}

}  // namespace
}  // namespace xllm::mlu
