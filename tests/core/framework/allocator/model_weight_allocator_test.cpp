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

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "core/framework/allocator/global_memory_region.h"
#include "core/framework/allocator/model_memory_manager.h"
#include "core/framework/allocator/virtual_memory/mapped_memory_region.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/framework/allocator/weight/weight_allocation.h"
#include "tests/core/framework/allocator/virtual_memory_test_utils.h"
#include "tests/npu_test_environment.h"

namespace xllm {

class ModelMemoryManagerTestPeer final {
 public:
  static void release_resources() {
    ModelMemoryManager::get_instance().destroy();
  }
};

namespace {

constexpr char kModelId[] = "virtual-memory-weight-regression";
constexpr size_t kPoolPages = 10;

class WeightAllocatorEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    testing::init_npu_test_runtime();
    google::InitGoogleLogging("model_weight_allocator_test");
  }

  void TearDown() override {
    testing::finalize_npu_test_runtime();
    google::ShutdownGoogleLogging();
  }
};

::testing::Environment* const kEnvironment =
    ::testing::AddGlobalTestEnvironment(new WeightAllocatorEnvironment);

class ModelWeightAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    uint32_t device_count = 0;
    ASSERT_EQ(aclrtGetDeviceCount(&device_count), ACL_SUCCESS);
    if (device_count == 0) {
      GTEST_SKIP() << "No NPU device available";
    }
    ASSERT_EQ(aclrtSetDevice(/*device_id=*/0), ACL_SUCCESS);

    const torch::Device device("npu:0");
    ModelMemoryManager::get_instance().init(device);
    auto& pool = PhysicalPagePool::get_instance();
    const size_t page_size = vmm::get_recommended_granularity(device.index());
    pool.init(device, kPoolPages, page_size);
    GlobalMemoryRegion::get_instance().init(device);
    resources_initialized_ = true;
    ASSERT_TRUE(GlobalMemoryRegion::get_instance().is_initialized());
    ASSERT_EQ(pool.num_available(), kPoolPages);
  }

  void TearDown() override {
    if (!resources_initialized_) {
      return;
    }
    ModelMemoryManager::get_instance().free_weight(kModelId);
    auto& pool = PhysicalPagePool::get_instance();
    pool.release_reserved_pages(blocking_page_ids_);
    EXPECT_EQ(pool.num_available(), kPoolPages);
    ModelMemoryManagerTestPeer::release_resources();
    VirtualMemoryTestPeer::release_resources();
  }

  bool resources_initialized_ = false;
  std::vector<page_id_t> blocking_page_ids_;
};

TEST_F(ModelWeightAllocatorTest, FragmentedTransferMatchesMappedWeights) {
  auto& pool = PhysicalPagePool::get_instance();
  blocking_page_ids_ = pool.allocate_pages_from_right(kPoolPages);
  ASSERT_EQ(blocking_page_ids_.size(), kPoolPages);
  const std::vector<page_id_t> weight_page_ids = {9, 7};
  pool.release_reserved_pages(weight_page_ids);
  std::erase_if(blocking_page_ids_,
                [](page_id_t page_id) { return page_id == 9 || page_id == 7; });

  auto& allocator = ModelMemoryManager::get_instance();
  ASSERT_TRUE(
      allocator.broadcast_alloc_weight_pages(kModelId, /*num_pages=*/2));
  ASSERT_EQ(pool.num_available(), 0);
  const size_t page_size = GlobalMemoryRegion::get_instance().page_size();
  void* weight_ptr = nullptr;
  ASSERT_TRUE(allocator.allocate_weight(kModelId, weight_ptr, 2 * page_size));

  std::vector<uint8_t> logical_weights(2 * page_size, 0x11);
  std::fill(logical_weights.begin() + page_size, logical_weights.end(), 0x22);
  ASSERT_EQ(aclrtMemcpy(weight_ptr,
                        logical_weights.size(),
                        logical_weights.data(),
                        logical_weights.size(),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_SUCCESS);

  const auto segments = allocator.get_model_weight_segments(kModelId);
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0].offset, 9 * page_size);
  EXPECT_EQ(segments[1].offset, 7 * page_size);
  std::vector<uint8_t> transferred_weights(logical_weights.size());
  const uintptr_t global_base = reinterpret_cast<uintptr_t>(
      GlobalMemoryRegion::get_instance().base_vaddr());
  size_t destination_offset = 0;
  for (const auto& segment : segments) {
    ASSERT_LE(segment.size, transferred_weights.size() - destination_offset);
    ASSERT_EQ(aclrtMemcpy(transferred_weights.data() + destination_offset,
                          segment.size,
                          reinterpret_cast<void*>(global_base + segment.offset),
                          segment.size,
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_SUCCESS);
    destination_offset += segment.size;
  }
  EXPECT_EQ(destination_offset, logical_weights.size());
  EXPECT_EQ(transferred_weights, logical_weights);
}

TEST_F(ModelWeightAllocatorTest, ContiguousAllocationRejectsOverflow) {
  auto& allocator = ModelMemoryManager::get_instance();
  ASSERT_TRUE(
      allocator.broadcast_alloc_weight_pages(kModelId, /*num_pages=*/1));
  const size_t page_size = GlobalMemoryRegion::get_instance().page_size();
  void* weight_ptr = nullptr;
  ASSERT_TRUE(allocator.allocate_weight(kModelId, weight_ptr, /*size=*/1));
  void* rejected_ptr = nullptr;
  EXPECT_FALSE(allocator.allocate_weight(
      kModelId, rejected_ptr, std::numeric_limits<size_t>::max()));
  EXPECT_EQ(rejected_ptr, nullptr);
  EXPECT_TRUE(allocator.allocate_weight(kModelId, rejected_ptr, page_size - 1));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(rejected_ptr),
            reinterpret_cast<uintptr_t>(weight_ptr) + 1);
  EXPECT_FALSE(allocator.allocate_weight(kModelId, rejected_ptr, /*size=*/1));
}

TEST_F(ModelWeightAllocatorTest,
       FragmentedAllocationOverflowPreservesCursorAndOutput) {
  auto& pool = PhysicalPagePool::get_instance();
  const size_t page_size = pool.page_size();
  auto page_ids = pool.allocate_pages_from_right(/*count=*/1);
  ASSERT_EQ(page_ids.size(), 1);
  auto region = std::make_unique<MappedMemoryRegion>(
      std::move(page_ids), torch::kUInt8, pool.device(), page_size);
  const uintptr_t base_address = vir_ptr_to_uintptr(region->vaddr());
  WeightAllocation allocation;
  allocation.set_fragmented(std::move(region), /*num_pages=*/1, {});

  void* first_ptr = nullptr;
  ASSERT_TRUE(allocation.allocate(first_ptr, /*size=*/1, page_size));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(first_ptr), base_address);
  void* rejected_ptr = nullptr;
  EXPECT_FALSE(allocation.allocate(
      rejected_ptr, std::numeric_limits<size_t>::max(), page_size));
  EXPECT_EQ(rejected_ptr, nullptr);
  EXPECT_EQ(allocation.current_offset(), 1);

  void* last_ptr = nullptr;
  ASSERT_TRUE(allocation.allocate(last_ptr, page_size - 1, page_size));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(last_ptr), base_address + 1);
  EXPECT_EQ(allocation.current_offset(), page_size);
  EXPECT_FALSE(allocation.allocate(last_ptr, /*size=*/1, page_size));
  EXPECT_EQ(allocation.current_offset(), page_size);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(last_ptr), base_address + 1);
}

TEST_F(ModelWeightAllocatorTest, ExhaustionDoesNotConsumePages) {
  auto& pool = PhysicalPagePool::get_instance();
  auto& allocator = ModelMemoryManager::get_instance();
  EXPECT_FALSE(
      allocator.broadcast_alloc_weight_pages(kModelId, kPoolPages + 1));
  EXPECT_EQ(pool.num_available(), kPoolPages);
  EXPECT_TRUE(allocator.get_model_weight_segments(kModelId).empty());
}

TEST_F(ModelWeightAllocatorTest, FailedFallbackPreservesExistingWeights) {
  auto& allocator = ModelMemoryManager::get_instance();
  ASSERT_TRUE(
      allocator.broadcast_alloc_weight_pages(kModelId, /*num_pages=*/1));
  const auto before = allocator.get_model_weight_segments(kModelId);
  ASSERT_EQ(before.size(), 1);
  EXPECT_FALSE(allocator.record_weight_fallback_allocation(kModelId, {}));
  const auto after = allocator.get_model_weight_segments(kModelId);
  ASSERT_EQ(after.size(), 1);
  EXPECT_EQ(after[0].offset, before[0].offset);
  EXPECT_EQ(after[0].size, before[0].size);
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), kPoolPages - 1);
  void* weight_ptr = nullptr;
  EXPECT_TRUE(allocator.allocate_weight(kModelId, weight_ptr, /*size=*/1));
}

TEST_F(ModelWeightAllocatorTest, KvShortageLeavesEveryLayerUnmapped) {
  auto& pool = PhysicalPagePool::get_instance();
  auto& allocator = ModelMemoryManager::get_instance();
  const size_t page_size = GlobalMemoryRegion::get_instance().page_size();
  const std::vector<int64_t> dims = {static_cast<int64_t>(page_size)};
  const auto k_tensors = allocator.create_k_tensors(
      kModelId, dims, torch::kUInt8, /*num_layers=*/2);
  const auto v_tensors = allocator.create_v_tensors(
      kModelId, dims, torch::kUInt8, /*num_layers=*/2);
  ASSERT_EQ(k_tensors.size(), 2);
  ASSERT_EQ(v_tensors.size(), 2);
  blocking_page_ids_ = pool.allocate_pages_from_right(kPoolPages - 3);
  ASSERT_EQ(pool.num_available(), 3);

  EXPECT_FALSE(allocator.map_to_kv_tensors(kModelId, {/*offset=*/0}));
  EXPECT_EQ(pool.num_available(), 3);
  for (int64_t layer_id = 0; layer_id < 2; ++layer_id) {
    const auto offsets = allocator.get_global_offsets_for_block(
        kModelId, layer_id, /*block_id=*/0, page_size);
    EXPECT_EQ(offsets.first, std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(offsets.second, std::numeric_limits<uint64_t>::max());
  }

  pool.release_reserved_pages(blocking_page_ids_);
  blocking_page_ids_.clear();
  ASSERT_TRUE(allocator.map_to_kv_tensors(kModelId, {/*offset=*/0}));
  EXPECT_EQ(pool.num_available(), kPoolPages - 4);
  ASSERT_TRUE(allocator.unmap_from_kv_tensors(kModelId, {/*offset=*/0}));
  EXPECT_EQ(pool.num_available(), kPoolPages);
}

}  // namespace
}  // namespace xllm
