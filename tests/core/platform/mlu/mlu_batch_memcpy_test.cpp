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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/framework/kv_cache/kv_cache_utils.h"
#include "core/platform/batch_memcpy.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"

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

    ASSERT_TRUE(batch_memcpy_->copy_h2d(
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

  ASSERT_TRUE(batch_memcpy_->copy_h2d(sources, device_tensors, stream_.get()));
  ASSERT_TRUE(batch_memcpy_->copy_d2h(device_tensors, restored, stream_.get()));
  for (size_t index = 0; index < widths.size(); ++index) {
    EXPECT_TRUE(torch::equal(sources[index], restored[index]));
  }
}

TEST_F(MLUBatchMemcpyTest, RejectsInvalidInputs) {
  torch::Tensor host;
  HostPageAlignedRegion host_region;
  create_host_page_aligned_tensor({2, 4}, torch::kUInt8, &host, &host_region);
  const torch::Tensor device_tensor = torch::zeros(
      {2, 4},
      torch::TensorOptions().dtype(torch::kUInt8).device(device_->unwrap()));

  EXPECT_FALSE(batch_memcpy_->copy_h2d(
      {host[0]}, {device_tensor[0], device_tensor[1]}, stream_.get()));
  EXPECT_FALSE(batch_memcpy_->copy_h2d(
      {host[0]}, {device_tensor.flatten()}, stream_.get()));
  EXPECT_FALSE(batch_memcpy_->copy_h2d(
      {host.transpose(0, 1)}, {device_tensor}, stream_.get()));
  EXPECT_FALSE(
      batch_memcpy_->copy_h2d({device_tensor[0]}, {host[0]}, stream_.get()));
  EXPECT_FALSE(
      batch_memcpy_->copy_d2h({host[0]}, {device_tensor[0]}, stream_.get()));
  EXPECT_FALSE(batch_memcpy_->copy_h2d({host[0]}, {device_tensor[0]}, nullptr));
  EXPECT_FALSE(batch_memcpy_->copy_h2d(
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

  EXPECT_FALSE(batch_memcpy_->copy_h2d({host}, {other_tensor}, stream_.get()));
}

}  // namespace
}  // namespace xllm::mlu
