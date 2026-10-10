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

#include "core/kv_cache/storage/kv_cache_memory_registry.h"

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/kv_cache/storage/paged_kv_cache_tensor_allocator.h"
#include "core/platform/vmm_api.h"
#include "tests/npu_test_environment.h"

namespace xllm {
namespace {

constexpr size_t kPoolPages = 4;

class KVCacheMemoryEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    testing::init_npu_test_runtime();
    google::InitGoogleLogging("kv_cache_memory_registry_test");
  }

  void TearDown() override {
    testing::finalize_npu_test_runtime();
    google::ShutdownGoogleLogging();
  }
};

::testing::Environment* const kEnvironment =
    ::testing::AddGlobalTestEnvironment(new KVCacheMemoryEnvironment);

class KVCacheMemoryRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    uint32_t device_count = 0;
    ASSERT_EQ(aclrtGetDeviceCount(&device_count), ACL_SUCCESS);
    if (device_count == 0) {
      GTEST_SKIP() << "No NPU device available";
    }
    ASSERT_EQ(aclrtSetDevice(/*device_id=*/0), ACL_SUCCESS);
    page_size_ = vmm::get_recommended_granularity(device_.index());
    ASSERT_GT(page_size_, 0);
    auto& pool = PhysicalPagePool::get_instance();
    pool.init(device_, kPoolPages, page_size_);
    resources_initialized_ = true;
    ASSERT_EQ(pool.num_available(), kPoolPages);
    registry_.init(device_, page_size_);
  }

  void TearDown() override {
    if (!resources_initialized_) {
      return;
    }
    registry_.clear();
    auto& pool = PhysicalPagePool::get_instance();
    EXPECT_EQ(pool.num_available(), kPoolPages);
    EXPECT_TRUE(pool.reset());
  }

  void create_model(const std::string& model_id, int64_t num_layers) {
    const std::vector<int64_t> dims{static_cast<int64_t>(page_size_)};
    registry_.create_k_tensors(model_id, dims, torch::kUInt8, num_layers);
    registry_.create_v_tensors(model_id, dims, torch::kUInt8, num_layers);
  }

  const torch::Device device_{"npu:0"};
  KVCacheMemoryRegistry registry_;
  size_t page_size_ = 0;
  bool resources_initialized_ = false;
};

TEST_F(KVCacheMemoryRegistryTest, ModelsKeepIndependentMappingsAndMetadata) {
  create_model("first", /*num_layers=*/1);
  create_model("second", /*num_layers=*/1);
  ASSERT_TRUE(registry_.map("first", {0}));
  ASSERT_TRUE(registry_.map("second", {0}));
  const auto first_offsets = registry_.get_global_offsets_for_block(
      "first", /*layer_id=*/0, /*block_id=*/0, page_size_);
  const auto second_offsets = registry_.get_global_offsets_for_block(
      "second", /*layer_id=*/0, /*block_id=*/0, page_size_);
  EXPECT_NE(first_offsets.first, UINT64_MAX);
  EXPECT_NE(first_offsets.second, UINT64_MAX);
  EXPECT_NE(first_offsets.first, second_offsets.first);
  EXPECT_NE(first_offsets.second, second_offsets.second);
  EXPECT_EQ(registry_.num_layers("first"), 1);
  EXPECT_FALSE(registry_.num_layers("missing").has_value());

  registry_.erase("first");
  EXPECT_FALSE(registry_.num_layers("first").has_value());
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), 2);
  EXPECT_EQ(registry_.get_global_offsets_for_block(
                "second", /*layer_id=*/0, /*block_id=*/0, page_size_),
            second_offsets);
  EXPECT_FALSE(registry_.map("first", {0}));
  EXPECT_EQ(registry_.get_global_offsets_for_block(
                "first", /*layer_id=*/0, /*block_id=*/0, page_size_),
            std::make_pair(UINT64_MAX, UINT64_MAX));
}

TEST_F(KVCacheMemoryRegistryTest, ShortageDoesNotPartiallyMapAnotherModel) {
  create_model("busy", /*num_layers=*/1);
  create_model("large", /*num_layers=*/2);
  ASSERT_TRUE(registry_.map("busy", {0}));
  const auto busy_offsets = registry_.get_global_offsets_for_block(
      "busy", /*layer_id=*/0, /*block_id=*/0, page_size_);
  EXPECT_FALSE(registry_.map("large", {0}));
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), 2);
  EXPECT_EQ(registry_.get_global_offsets_for_block(
                "busy", /*layer_id=*/0, /*block_id=*/0, page_size_),
            busy_offsets);
  for (int64_t layer_id = 0; layer_id < 2; ++layer_id) {
    EXPECT_EQ(registry_.get_global_offsets_for_block(
                  "large", layer_id, /*block_id=*/0, page_size_),
              std::make_pair(UINT64_MAX, UINT64_MAX));
  }

  ASSERT_TRUE(registry_.unmap("busy", {0}));
  ASSERT_TRUE(registry_.map("large", {0}));
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), 0);
  registry_.clear();
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), kPoolPages);
  registry_.init(device_, page_size_);
  create_model("fresh", /*num_layers=*/1);
  EXPECT_TRUE(registry_.map("fresh", {0}));
}

TEST_F(KVCacheMemoryRegistryTest, TensorAllocatorUsesItsInjectedRegistry) {
  auto allocator = create_paged_kv_cache_tensor_allocator(
      registry_, "model", /*num_layers=*/2);
  const std::vector<int64_t> shape{static_cast<int64_t>(page_size_ / 2)};
  const torch::Tensor first_key = allocator->allocate(
      KVCacheTensorRole::KEY, shape, torch::kBFloat16, device_);
  const torch::Tensor second_key = allocator->allocate(
      KVCacheTensorRole::KEY, shape, torch::kBFloat16, device_);
  const torch::Tensor first_value = allocator->allocate(
      KVCacheTensorRole::VALUE, shape, torch::kBFloat16, device_);
  const torch::Tensor second_value = allocator->allocate(
      KVCacheTensorRole::VALUE, shape, torch::kBFloat16, device_);
  EXPECT_NE(first_key.data_ptr(), second_key.data_ptr());
  EXPECT_NE(first_value.data_ptr(), second_value.data_ptr());
  EXPECT_EQ(first_key.sizes().vec(), shape);
  EXPECT_EQ(first_value.scalar_type(), torch::kBFloat16);
  EXPECT_EQ(first_value.device(), device_);
  EXPECT_EQ(registry_.num_layers("model"), 2);
  EXPECT_TRUE(registry_.map("model", {0}));
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), 0);
}

}  // namespace
}  // namespace xllm
