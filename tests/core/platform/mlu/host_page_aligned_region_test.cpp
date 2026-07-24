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

#include <cn_api.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <future>
#include <thread>
#include <utility>
#include <vector>

#include "core/framework/kv_cache/kv_cache_utils.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"

namespace xllm {
namespace {

class HostPageAlignedRegionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() < 1) {
      GTEST_SKIP() << "MLU device is required for pinned host memory tests.";
    }
    Device device(/*device_index=*/0);
    device.set_device();
    context_tensor_ =
        torch::zeros({1}, torch::TensorOptions().device(device.unwrap()));
  }

  torch::Tensor context_tensor_;
};

TEST_F(HostPageAlignedRegionTest, RoundsAllocationAndReturnsZeroedTensor) {
  const int64_t page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, 0);
  const std::vector<int64_t> dims = {page_size + 1};
  torch::Tensor tensor;
  HostPageAlignedRegion region;

  create_host_page_aligned_tensor(dims, torch::kUInt8, &tensor, &region);

  ASSERT_NE(region.base_ptr, nullptr);
  EXPECT_EQ(region.total_bytes, static_cast<size_t>(page_size * 2));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(region.base_ptr) %
                static_cast<uintptr_t>(page_size),
            static_cast<uintptr_t>(0));
  EXPECT_TRUE(tensor.device().is_cpu());
  EXPECT_TRUE(tensor.is_contiguous());
  unsigned int flags = 0;
  ASSERT_EQ(cnHostMemGetFlags(&flags, region.base_ptr), CN_SUCCESS);
  EXPECT_NE(flags & CN_MEMHOSTALLOC_PORTABLE, 0U);
  const uint8_t* bytes = tensor.const_data_ptr<uint8_t>();
  EXPECT_TRUE(std::all_of(
      bytes, bytes + tensor.numel(), [](uint8_t value) { return value == 0; }));
}

TEST_F(HostPageAlignedRegionTest, MoveTransfersSoleOwnership) {
  HostPageAlignedRegion original(/*bytes=*/17);
  void* ptr = original.base_ptr;
  const size_t bytes = original.total_bytes;

  HostPageAlignedRegion moved(std::move(original));

  EXPECT_EQ(original.base_ptr, nullptr);
  EXPECT_EQ(original.total_bytes, static_cast<size_t>(0));
  EXPECT_EQ(moved.base_ptr, ptr);
  EXPECT_EQ(moved.total_bytes, bytes);

  HostPageAlignedRegion assigned(/*bytes=*/33);
  assigned = std::move(moved);
  EXPECT_EQ(moved.base_ptr, nullptr);
  EXPECT_EQ(moved.total_bytes, static_cast<size_t>(0));
  EXPECT_EQ(assigned.base_ptr, ptr);
  EXPECT_EQ(assigned.total_bytes, bytes);
}

TEST_F(HostPageAlignedRegionTest, RepeatedAllocationAndReleaseSucceeds) {
  for (int32_t iteration = 0; iteration < 64; ++iteration) {
    HostPageAlignedRegion region(
        /*bytes=*/static_cast<size_t>(iteration + 1));
    ASSERT_NE(region.base_ptr, nullptr);
  }
}

TEST_F(HostPageAlignedRegionTest, ReleaseOnAnotherThreadRestoresItsContext) {
  HostPageAlignedRegion region(/*bytes=*/65);
  std::promise<CNresult> release_result;
  std::future<CNresult> result = release_result.get_future();

  std::thread release_thread(
      [region = std::move(region), &release_result]() mutable {
        Device device(/*device_index=*/0);
        device.set_device();
        CNcontext before = nullptr;
        CNresult ret = cnCtxGetCurrent(&before);
        if (ret != CN_SUCCESS) {
          release_result.set_value(ret);
          return;
        }
        {
          HostPageAlignedRegion owned(std::move(region));
        }
        CNcontext after = nullptr;
        ret = cnCtxGetCurrent(&after);
        release_result.set_value(ret == CN_SUCCESS && before == after
                                     ? CN_SUCCESS
                                     : CN_ERROR_INVALID_VALUE);
      });
  release_thread.join();

  EXPECT_EQ(result.get(), CN_SUCCESS);
}

}  // namespace
}  // namespace xllm
