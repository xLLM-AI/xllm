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

#include "core/framework/xtensor/xtensor.h"

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/xtensor/global_xtensor.h"
#include "core/framework/xtensor/phy_page_pool.h"
#include "core/platform/vmm_api.h"
#include "tests/core/framework/xtensor/xtensor_test_utils.h"
#include "tests/npu_test_environment.h"

namespace xllm {
namespace {

constexpr size_t kPoolPages = 4;

class XTensorEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    testing::init_npu_test_runtime();
    google::InitGoogleLogging("xtensor_test");
  }

  void TearDown() override {
    testing::finalize_npu_test_runtime();
    google::ShutdownGoogleLogging();
  }
};

::testing::Environment* const kEnvironment =
    ::testing::AddGlobalTestEnvironment(new XTensorEnvironment);

class XTensorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    uint32_t device_count = 0;
    ASSERT_EQ(aclrtGetDeviceCount(&device_count), ACL_SUCCESS);
    if (device_count == 0) {
      GTEST_SKIP() << "No NPU device available";
    }
    ASSERT_EQ(aclrtSetDevice(/*device_id=*/0), ACL_SUCCESS);
    auto& pool = PhyPagePool::get_instance();
    pool.init(device_, kPoolPages);
    resources_initialized_ = true;
    page_size_ = static_cast<size_t>(
        KVCacheConfig::get_instance().phy_page_granularity_size());
    ASSERT_GT(page_size_, 0);
    ASSERT_EQ(pool.num_available(), kPoolPages);
  }

  void TearDown() override {
    if (!resources_initialized_) {
      return;
    }
    EXPECT_EQ(PhyPagePool::get_instance().num_available(), kPoolPages);
    XTensorTestPeer::release_resources();
  }

  const torch::Device device_{"npu:0"};
  size_t page_size_ = 0;
  bool resources_initialized_ = false;
};

TEST_F(XTensorTest, ShortageAcrossTensorsLeavesBothUnmapped) {
  auto& pool = PhyPagePool::get_instance();
  XTensor blocker((kPoolPages - 1) * page_size_, torch::kUInt8, device_);
  ASSERT_TRUE(blocker.map_all());
  XTensor first(page_size_, torch::kUInt8, device_);
  XTensor second(page_size_, torch::kUInt8, device_);
  ASSERT_EQ(pool.num_available(), 1);

  EXPECT_FALSE(XTensor::map_pages({{&first, 0}, {&second, 0}}));
  EXPECT_EQ(first.get_phy_page_id(/*offset=*/0), -1);
  EXPECT_EQ(second.get_phy_page_id(/*offset=*/0), -1);
  EXPECT_EQ(pool.num_available(), 1);

  ASSERT_TRUE(blocker.unmap(/*offset=*/0));
  ASSERT_TRUE(XTensor::map_pages({{&first, 0}, {&second, 0}}));
  EXPECT_GE(first.get_phy_page_id(/*offset=*/0), 0);
  EXPECT_GE(second.get_phy_page_id(/*offset=*/0), 0);
  EXPECT_NE(first.get_phy_page_id(/*offset=*/0),
            second.get_phy_page_id(/*offset=*/0));
  EXPECT_EQ(pool.num_available(), 0);
}

TEST_F(XTensorTest, LaterInvalidTargetDoesNotMapEarlierTarget) {
  auto& pool = PhyPagePool::get_instance();
  XTensor first(page_size_, torch::kUInt8, device_);
  XTensor second(page_size_, torch::kUInt8, device_);
  const std::vector<offset_t> invalid_offsets = {
      -1, 1, static_cast<offset_t>(page_size_)};
  for (offset_t offset : invalid_offsets) {
    EXPECT_FALSE(XTensor::map_pages({{&first, 0}, {&second, offset}}));
    EXPECT_EQ(first.get_phy_page_id(/*offset=*/0), -1);
    EXPECT_EQ(second.get_phy_page_id(/*offset=*/0), -1);
    EXPECT_EQ(pool.num_available(), kPoolPages);
  }
  EXPECT_FALSE(XTensor::map_pages({{&first, 0}, {nullptr, 0}}));
  EXPECT_EQ(first.get_phy_page_id(/*offset=*/0), -1);
  EXPECT_EQ(pool.num_available(), kPoolPages);

  XTensor wrong_device(page_size_, torch::kUInt8, torch::Device(torch::kCPU));
  EXPECT_FALSE(XTensor::map_pages({{&first, 0}, {&wrong_device, 0}}));
  EXPECT_EQ(first.get_phy_page_id(/*offset=*/0), -1);
  EXPECT_EQ(pool.num_available(), kPoolPages);

  const auto weight_page_ids = pool.allocate_pages_from_right(/*count=*/1);
  ASSERT_EQ(weight_page_ids.size(), 1);
  XTensor weight_tensor(weight_page_ids, torch::kUInt8, device_);
  EXPECT_FALSE(XTensor::map_pages({{&first, 0}, {&weight_tensor, 0}}));
  EXPECT_EQ(first.get_phy_page_id(/*offset=*/0), -1);
  EXPECT_EQ(pool.num_available(), kPoolPages - 1);
}

TEST_F(XTensorTest, DuplicateAndExistingTargetsAreIdempotent) {
  auto& pool = PhyPagePool::get_instance();
  XTensor first(2 * page_size_, torch::kUInt8, device_);
  XTensor second(page_size_, torch::kUInt8, device_);
  ASSERT_TRUE(first.map(/*offset=*/0));
  const page_id_t existing_page = first.get_phy_page_id(/*offset=*/0);
  const offset_t next_offset = static_cast<offset_t>(page_size_);
  const std::vector<std::pair<XTensor*, offset_t>> targets = {
      {&first, 0},
      {&first, 0},
      {&first, next_offset},
      {&first, next_offset},
      {&second, 0},
      {&second, 0}};

  ASSERT_TRUE(XTensor::map_pages(targets));
  EXPECT_EQ(pool.num_available(), kPoolPages - 3);
  EXPECT_EQ(first.get_phy_page_id(/*offset=*/0), existing_page);
  EXPECT_GE(first.get_phy_page_id(next_offset), 0);
  EXPECT_GE(second.get_phy_page_id(/*offset=*/0), 0);
  EXPECT_NE(first.get_phy_page_id(next_offset), existing_page);
  EXPECT_NE(second.get_phy_page_id(/*offset=*/0), existing_page);
  EXPECT_NE(second.get_phy_page_id(/*offset=*/0),
            first.get_phy_page_id(next_offset));
  EXPECT_TRUE(XTensor::map_pages(targets));
  EXPECT_TRUE(XTensor::map_pages({}));
  EXPECT_EQ(pool.num_available(), kPoolPages - 3);
}

TEST_F(XTensorTest, MapAllExhaustionPreservesExistingMapping) {
  auto& pool = PhyPagePool::get_instance();
  XTensor tensor(3 * page_size_, torch::kUInt8, device_);
  ASSERT_TRUE(tensor.map(/*offset=*/0));
  const page_id_t existing_page = tensor.get_phy_page_id(/*offset=*/0);
  XTensor blocker(2 * page_size_, torch::kUInt8, device_);
  ASSERT_TRUE(blocker.map_all());
  ASSERT_EQ(pool.num_available(), 1);

  EXPECT_FALSE(tensor.map_all());
  EXPECT_EQ(tensor.get_phy_page_id(/*offset=*/0), existing_page);
  EXPECT_EQ(tensor.get_phy_page_id(static_cast<offset_t>(page_size_)), -1);
  EXPECT_EQ(tensor.get_phy_page_id(static_cast<offset_t>(2 * page_size_)), -1);
  EXPECT_EQ(pool.num_available(), 1);

  ASSERT_TRUE(blocker.unmap_all());
  ASSERT_TRUE(tensor.map_all());
  EXPECT_EQ(tensor.get_phy_page_id(/*offset=*/0), existing_page);
  EXPECT_EQ(pool.num_available(), kPoolPages - 3);
  EXPECT_TRUE(tensor.unmap_all());
  EXPECT_TRUE(tensor.unmap_all());
  EXPECT_EQ(pool.num_available(), kPoolPages);
}

TEST_F(XTensorTest, RealMapUnmapRemapPreservesContents) {
  auto& pool = PhyPagePool::get_instance();
  auto& global_tensor = GlobalXTensor::get_instance();
  global_tensor.init(device_);
  ASSERT_TRUE(global_tensor.is_initialized());
  XTensor tensor(page_size_, torch::kUInt8, device_);
  ASSERT_TRUE(tensor.map(/*offset=*/0));
  const page_id_t page_id = tensor.get_phy_page_id(/*offset=*/0);
  const std::vector<uint8_t> expected(256, 0x42);
  void* tensor_ptr = vir_ptr_to_void_ptr(tensor.vaddr());
  ASSERT_EQ(aclrtMemcpy(tensor_ptr,
                        expected.size(),
                        expected.data(),
                        expected.size(),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_SUCCESS);
  std::vector<uint8_t> observed(expected.size());
  ASSERT_EQ(aclrtMemcpy(observed.data(),
                        observed.size(),
                        global_tensor.get_vaddr_by_page_id(page_id),
                        observed.size(),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_SUCCESS);
  EXPECT_EQ(observed, expected);

  ASSERT_TRUE(tensor.unmap(/*offset=*/0));
  EXPECT_EQ(tensor.get_phy_page_id(/*offset=*/0), -1);
  EXPECT_EQ(pool.num_available(), kPoolPages);
  EXPECT_TRUE(tensor.unmap(/*offset=*/0));
  EXPECT_EQ(pool.num_available(), kPoolPages);
  ASSERT_TRUE(tensor.map(/*offset=*/0));
  EXPECT_EQ(tensor.get_phy_page_id(/*offset=*/0), page_id);
  ASSERT_EQ(aclrtMemcpy(observed.data(),
                        observed.size(),
                        tensor_ptr,
                        observed.size(),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_SUCCESS);
  EXPECT_EQ(observed, expected);
}

TEST_F(XTensorTest, AllocationOverflowPreservesCursorAndOutput) {
  XTensor tensor(page_size_, torch::kUInt8, device_);
  void* first_ptr = nullptr;
  ASSERT_TRUE(tensor.allocate(first_ptr, /*size=*/1));
  EXPECT_EQ(first_ptr, vir_ptr_to_void_ptr(tensor.vaddr()));
  void* rejected_ptr = nullptr;
  EXPECT_FALSE(
      tensor.allocate(rejected_ptr, std::numeric_limits<size_t>::max()));
  EXPECT_EQ(rejected_ptr, nullptr);
  EXPECT_EQ(tensor.alloc_offset(), 1);

  void* last_ptr = nullptr;
  ASSERT_TRUE(tensor.allocate(last_ptr, page_size_ - 1));
  EXPECT_EQ(vir_ptr_to_uintptr(tensor.vaddr()) + 1,
            reinterpret_cast<uintptr_t>(last_ptr));
  EXPECT_EQ(tensor.alloc_offset(), page_size_);
  EXPECT_FALSE(tensor.allocate(last_ptr, /*size=*/1));
  EXPECT_EQ(tensor.alloc_offset(), page_size_);
  EXPECT_EQ(vir_ptr_to_uintptr(tensor.vaddr()) + 1,
            reinterpret_cast<uintptr_t>(last_ptr));
}

TEST_F(XTensorTest, TensorViewsRespectByteBoundsAndAlignment) {
  XTensor tensor(page_size_, torch::kBFloat16, device_);
  ASSERT_TRUE(tensor.map_all());
  const torch::Tensor view = tensor.to_torch_tensor(/*offset=*/2, {2, 3});
  EXPECT_EQ(view.numel(), 6);
  EXPECT_EQ(view.scalar_type(), torch::kBFloat16);
  EXPECT_EQ(view.device(), device_);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(view.data_ptr()),
            vir_ptr_to_uintptr(tensor.vaddr()) + 2);
  EXPECT_EQ(tensor.to_torch_tensor(page_size_ - 2, {1}).numel(), 1);
  EXPECT_EQ(tensor.to_torch_tensor(page_size_, {0}).numel(), 0);

  EXPECT_DEATH(tensor.to_torch_tensor(page_size_ + 2, {0}),
               "Tensor byte offset is out of bounds");
  EXPECT_DEATH(tensor.to_torch_tensor(/*offset=*/1, {1}),
               "Tensor byte offset is not aligned");
  EXPECT_DEATH(tensor.to_torch_tensor(/*offset=*/0, {-1}),
               "Tensor dimensions must be nonnegative");
  EXPECT_DEATH(tensor.to_torch_tensor(page_size_ - 2, {2}),
               "Tensor view exceeds its virtual memory region");
  EXPECT_DEATH(tensor.to_torch_tensor(
                   /*offset=*/0, {std::numeric_limits<int64_t>::max(), 3}),
               "num_elems");
}

TEST_F(XTensorTest, VirtualSizeOverflowIsRejectedBeforeReservation) {
  EXPECT_DEATH(
      XTensor(std::numeric_limits<size_t>::max(), torch::kUInt8, device_),
      "size");
  EXPECT_DEATH(XTensor(/*size=*/0, torch::kUInt8, device_), "size");
}

}  // namespace
}  // namespace xllm
