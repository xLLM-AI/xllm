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

#include "core/distributed_runtime/model_memory_manager.h"

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/framework/allocator/global_memory_region.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
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

constexpr char kModelId[] = "virtual-memory-kv-regression";
constexpr size_t kPoolPages = 10;

class ModelMemoryEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    testing::init_npu_test_runtime();
    google::InitGoogleLogging("model_memory_manager_test");
  }

  void TearDown() override {
    testing::finalize_npu_test_runtime();
    google::ShutdownGoogleLogging();
  }
};

::testing::Environment* const kEnvironment =
    ::testing::AddGlobalTestEnvironment(new ModelMemoryEnvironment);

class ModelMemoryManagerTest : public ::testing::Test {
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
    auto& pool = PhysicalPagePool::get_instance();
    pool.release_reserved_pages(blocking_page_ids_);
    ModelMemoryManagerTestPeer::release_resources();
    VirtualMemoryTestPeer::release_resources();
  }

  bool resources_initialized_ = false;
  std::vector<page_id_t> blocking_page_ids_;
};

TEST_F(ModelMemoryManagerTest, KvShortageLeavesEveryLayerUnmapped) {
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
