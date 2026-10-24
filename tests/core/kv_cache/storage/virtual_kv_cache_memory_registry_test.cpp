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

#include "core/kv_cache/storage/virtual_kv_cache_memory_registry.h"

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/framework/allocator/virtual_memory/shared_physical_page_mapping.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "core/kv_cache/storage/vmm_kv_tensor_allocator.h"
#include "core/platform/vmm_api.h"
#include "tests/npu_test_environment.h"

namespace xllm {
namespace {

constexpr size_t kPoolPages = 4;
constexpr int64_t kBlockSize = 128;
constexpr int64_t kLatentSize = 512;
constexpr int64_t kRopeSize = 64;
constexpr int64_t kIndexSize = 128;

void fill_block_pattern(const torch::Tensor& tensor,
                        int64_t block_id,
                        uint8_t pattern) {
  const size_t block_bytes =
      static_cast<size_t>(tensor.stride(/*dim=*/0)) * tensor.element_size();
  auto* block_ptr = static_cast<uint8_t*>(tensor.data_ptr()) +
                    static_cast<size_t>(block_id) * block_bytes;
  const std::vector<uint8_t> expected(block_bytes, pattern);
  ASSERT_EQ(aclrtMemcpy(block_ptr,
                        block_bytes,
                        expected.data(),
                        block_bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_SUCCESS);
}

void expect_block_pattern(const torch::Tensor& tensor,
                          int64_t block_id,
                          uint8_t pattern) {
  SCOPED_TRACE(block_id);
  const size_t block_bytes =
      static_cast<size_t>(tensor.stride(/*dim=*/0)) * tensor.element_size();
  const auto* block_ptr = static_cast<const uint8_t*>(tensor.data_ptr()) +
                          static_cast<size_t>(block_id) * block_bytes;
  std::vector<uint8_t> actual(block_bytes);
  ASSERT_EQ(aclrtMemcpy(actual.data(),
                        block_bytes,
                        block_ptr,
                        block_bytes,
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_SUCCESS);
  EXPECT_EQ(actual, std::vector<uint8_t>(block_bytes, pattern));
}

void expect_global_block_pattern(const SharedPhysicalPageMapping& mapping,
                                 uint64_t global_offset,
                                 const torch::Tensor& tensor,
                                 uint8_t pattern) {
  const size_t block_bytes =
      static_cast<size_t>(tensor.stride(/*dim=*/0)) * tensor.element_size();
  ASSERT_LE(global_offset, mapping.total_size());
  ASSERT_LE(block_bytes, mapping.total_size() - global_offset);
  const auto* block_ptr =
      static_cast<const uint8_t*>(mapping.base_vaddr()) + global_offset;
  std::vector<uint8_t> actual(block_bytes);
  ASSERT_EQ(aclrtMemcpy(actual.data(),
                        block_bytes,
                        block_ptr,
                        block_bytes,
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_SUCCESS);
  EXPECT_EQ(actual, std::vector<uint8_t>(block_bytes, pattern));
}

class KVCacheMemoryEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    testing::init_npu_test_runtime();
    google::InitGoogleLogging("virtual_kv_cache_memory_registry_test");
  }

  void TearDown() override {
    testing::finalize_npu_test_runtime();
    google::ShutdownGoogleLogging();
  }
};

::testing::Environment* const kEnvironment =
    ::testing::AddGlobalTestEnvironment(new KVCacheMemoryEnvironment);

class RecordingForwardingKVCacheTensorAllocator final
    : public KVCacheTensorAllocator {
 public:
  explicit RecordingForwardingKVCacheTensorAllocator(
      std::shared_ptr<KVCacheTensorAllocator> delegate)
      : delegate_(std::move(delegate)) {}

  torch::Tensor allocate(KVCacheTensorRole role,
                         const std::vector<int64_t>& shape,
                         torch::ScalarType dtype,
                         const torch::Device& device) override {
    roles_.emplace_back(role);
    tensors_.emplace_back(delegate_->allocate(role, shape, dtype, device));
    return tensors_.back();
  }

  const std::vector<KVCacheTensorRole>& roles() const { return roles_; }
  const std::vector<torch::Tensor>& tensors() const { return tensors_; }

 private:
  std::shared_ptr<KVCacheTensorAllocator> delegate_;
  std::vector<KVCacheTensorRole> roles_;
  std::vector<torch::Tensor> tensors_;
};

class VirtualKVCacheMemoryRegistryTest : public ::testing::Test {
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
    pool.init(device_, pool_pages(), page_size_);
    resources_initialized_ = true;
    ASSERT_EQ(pool.num_available(), pool_pages());
    registry_.init(device_, page_size_);
  }

  void TearDown() override {
    if (!resources_initialized_) {
      return;
    }
    registry_.clear();
    auto& pool = PhysicalPagePool::get_instance();
    EXPECT_EQ(pool.num_available(), pool_pages());
    EXPECT_TRUE(pool.reset());
  }

  void create_model(const std::string& model_id, int64_t num_layers) {
    const std::vector<int64_t> dims{static_cast<int64_t>(page_size_)};
    registry_.create_k_tensors(model_id, dims, torch::kUInt8, num_layers);
    registry_.create_v_tensors(model_id, dims, torch::kUInt8, num_layers);
  }

  virtual size_t pool_pages() const { return kPoolPages; }

  const torch::Device device_{"npu:0"};
  VirtualKVCacheMemoryRegistry registry_;
  size_t page_size_ = 0;
  bool resources_initialized_ = false;
};

class VirtualMLAKVCacheMemoryRegistryTest
    : public VirtualKVCacheMemoryRegistryTest {
 protected:
  size_t pool_pages() const override { return 18; }

  int64_t blocks_per_page() const {
    return static_cast<int64_t>(page_size_ /
                                (kBlockSize * kRopeSize * sizeof(uint16_t)));
  }

  std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
  create_mla_model(const std::string& model_id,
                   int64_t num_blocks,
                   int64_t num_layers) {
    return {registry_.create_k_tensors(model_id,
                                       {num_blocks, kBlockSize, 1, kLatentSize},
                                       torch::kBFloat16,
                                       num_layers),
            registry_.create_v_tensors(model_id,
                                       {num_blocks, kBlockSize, 1, kRopeSize},
                                       torch::kBFloat16,
                                       num_layers)};
  }
};

class VirtualIndexedKVCacheMemoryRegistryTest
    : public VirtualKVCacheMemoryRegistryTest {
 protected:
  size_t pool_pages() const override { return 62; }

  int64_t blocks_per_page() const {
    return static_cast<int64_t>(page_size_ /
                                (kBlockSize * kRopeSize * sizeof(uint16_t)));
  }

  std::tuple<std::vector<torch::Tensor>,
             std::vector<torch::Tensor>,
             std::vector<torch::Tensor>>
  create_indexed_model(const std::string& model_id,
                       int64_t num_blocks,
                       int64_t num_layers,
                       const std::vector<bool>& index_layer_owned) {
    auto keys =
        registry_.create_k_tensors(model_id,
                                   {num_blocks, kBlockSize, 1, kLatentSize},
                                   torch::kBFloat16,
                                   num_layers);
    auto values =
        registry_.create_v_tensors(model_id,
                                   {num_blocks, kBlockSize, 1, kRopeSize},
                                   torch::kBFloat16,
                                   num_layers);
    auto indices =
        registry_.create_index_tensors(model_id,
                                       {num_blocks, kBlockSize, 1, kIndexSize},
                                       torch::kBFloat16,
                                       num_layers,
                                       index_layer_owned);
    return {std::move(keys), std::move(values), std::move(indices)};
  }
};

TEST_F(VirtualKVCacheMemoryRegistryTest,
       PrepareRejectsDuplicateAndInvalidOffsetsWithoutAllocation) {
  create_model("model", /*num_layers=*/1);
  const size_t available = PhysicalPagePool::get_instance().num_available();
  EXPECT_TRUE(registry_.can_prepare_mapping("model", {0}));
  EXPECT_FALSE(registry_.can_prepare_mapping("missing", {0}));
  EXPECT_FALSE(registry_.can_prepare_mapping("model", {0, 0}));
  EXPECT_FALSE(registry_.can_prepare_mapping("model", {-1}));
  EXPECT_FALSE(registry_.can_prepare_mapping("model", {1}));
  EXPECT_FALSE(registry_.can_prepare_mapping(
      "model", {static_cast<offset_t>(page_size_)}));
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), available);
  EXPECT_EQ(registry_.get_global_offsets_for_block(
                "model", /*layer_id=*/0, /*block_id=*/0, page_size_),
            std::make_pair(UINT64_MAX, UINT64_MAX));
}

TEST_F(VirtualKVCacheMemoryRegistryTest,
       PrepareRejectsExistingMappingsWithoutChangingOwnership) {
  const std::vector<int64_t> dims{static_cast<int64_t>(2 * page_size_)};
  registry_.create_k_tensors("model", dims, torch::kUInt8, /*num_layers=*/1);
  registry_.create_v_tensors("model", dims, torch::kUInt8, /*num_layers=*/1);
  ASSERT_TRUE(registry_.map("model", {0}));
  const auto existing = registry_.get_global_offsets_for_block(
      "model", /*layer_id=*/0, /*block_id=*/0, page_size_);
  EXPECT_FALSE(registry_.can_prepare_mapping("model", {0}));
  EXPECT_FALSE(registry_.can_prepare_mapping(
      "model", {0, static_cast<offset_t>(page_size_)}));
  EXPECT_TRUE(registry_.can_prepare_mapping(
      "model", {static_cast<offset_t>(page_size_)}));
  EXPECT_EQ(registry_.get_global_offsets_for_block(
                "model", /*layer_id=*/0, /*block_id=*/0, page_size_),
            existing);
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), kPoolPages - 2);
  ASSERT_TRUE(registry_.unmap("model", {0}));
  EXPECT_TRUE(registry_.can_prepare_mapping("model", {0}));
}

TEST_F(VirtualKVCacheMemoryRegistryTest,
       ModelsKeepIndependentMappingsAndMetadata) {
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

TEST_F(VirtualKVCacheMemoryRegistryTest,
       ShortageDoesNotPartiallyMapAnotherModel) {
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

TEST_F(VirtualKVCacheMemoryRegistryTest,
       TensorAllocatorUsesItsInjectedRegistry) {
  auto allocator =
      create_vmm_kv_tensor_allocator(registry_, "model", /*num_layers=*/2);
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

TEST_F(VirtualMLAKVCacheMemoryRegistryTest,
       LatentAndRopeBoundariesKeepIndependentMappings) {
  ASSERT_EQ(page_size_ % (kBlockSize * kLatentSize * sizeof(uint16_t)), 0);
  const int64_t blocks_per_key_page = static_cast<int64_t>(
      page_size_ / (kBlockSize * kLatentSize * sizeof(uint16_t)));
  ASSERT_GT(blocks_per_key_page, 1);
  const int64_t blocks_per_group = blocks_per_page();
  auto [keys, values] =
      create_mla_model("mla", 2 * blocks_per_group, /*num_layers=*/1);
  auto& pool = PhysicalPagePool::get_instance();
  const offset_t next_offset = static_cast<offset_t>(page_size_);
  const offset_t invalid_offset = static_cast<offset_t>(2 * page_size_);
  EXPECT_FALSE(registry_.map("mla", {0, invalid_offset}));
  EXPECT_EQ(pool.num_available(), pool_pages());
  ASSERT_TRUE(registry_.map("mla", {0, next_offset}));
  EXPECT_EQ(pool.num_available(), 0);
  ASSERT_TRUE(registry_.map("mla", {0, next_offset}));
  EXPECT_EQ(pool.num_available(), 0);

  const std::vector<int64_t> blocks = {0,
                                       blocks_per_key_page - 1,
                                       blocks_per_key_page,
                                       blocks_per_group - 1,
                                       blocks_per_group,
                                       2 * blocks_per_group - 1};
  for (size_t i = 0; i < blocks.size(); ++i) {
    fill_block_pattern(keys.front(), blocks[i], static_cast<uint8_t>(20 + i));
    fill_block_pattern(
        values.front(), blocks[i], static_cast<uint8_t>(100 + i));
  }
  EXPECT_FALSE(registry_.unmap("mla", {0, invalid_offset}));
  EXPECT_EQ(pool.num_available(), 0);
  for (size_t i = 0; i < blocks.size(); ++i) {
    expect_block_pattern(keys.front(), blocks[i], static_cast<uint8_t>(20 + i));
    expect_block_pattern(
        values.front(), blocks[i], static_cast<uint8_t>(100 + i));
  }
  EXPECT_EQ(registry_.get_global_offsets_for_block(
                "mla", /*layer_id=*/0, /*block_id=*/0, page_size_),
            std::make_pair(UINT64_MAX, UINT64_MAX));

  ASSERT_TRUE(registry_.unmap("mla", {0}));
  EXPECT_EQ(pool.num_available(), 9);
  expect_block_pattern(keys.front(), blocks_per_group, /*pattern=*/24);
  expect_block_pattern(values.front(), blocks_per_group, /*pattern=*/104);
  expect_block_pattern(keys.front(), 2 * blocks_per_group - 1, /*pattern=*/25);
  expect_block_pattern(
      values.front(), 2 * blocks_per_group - 1, /*pattern=*/105);
  ASSERT_TRUE(registry_.map("mla", {0}));
  EXPECT_EQ(pool.num_available(), 0);
  fill_block_pattern(keys.front(), /*block_id=*/0, /*pattern=*/50);
  fill_block_pattern(values.front(), /*block_id=*/0, /*pattern=*/150);
  expect_block_pattern(keys.front(), blocks_per_group, /*pattern=*/24);
  expect_block_pattern(values.front(), blocks_per_group, /*pattern=*/104);
}

TEST_F(VirtualMLAKVCacheMemoryRegistryTest,
       TransferOffsetsUseIndependentRoleBlockSizes) {
  const int64_t blocks_per_group = blocks_per_page();
  auto [keys, values] =
      create_mla_model("mla", 2 * blocks_per_group, /*num_layers=*/1);
  EXPECT_FALSE(registry_.get_global_offsets_for_block("mla", 0, 0).has_value());
  ASSERT_TRUE(registry_.map("mla", {0, static_cast<offset_t>(page_size_)}));
  auto& pool = PhysicalPagePool::get_instance();
  SharedPhysicalPageMapping mapping;
  mapping.init(device_, pool.get_all_pages(), page_size_);
  ASSERT_TRUE(mapping.is_initialized());
  const std::vector<int64_t> blocks{
      0, 1, blocks_per_group / 8, blocks_per_group - 1, blocks_per_group};
  for (int64_t block : blocks) {
    const uint8_t key_pattern = static_cast<uint8_t>(10 + block);
    const uint8_t value_pattern = static_cast<uint8_t>(30 + block);
    fill_block_pattern(keys.front(), block, key_pattern);
    fill_block_pattern(values.front(), block, value_pattern);
    const auto offsets =
        registry_.get_global_offsets_for_block("mla", 0, block);
    ASSERT_TRUE(offsets.has_value());
    EXPECT_FALSE(offsets->index_offset.has_value());
    EXPECT_EQ(offsets->k_offset % page_size_,
              block * kBlockSize * kLatentSize * sizeof(uint16_t) % page_size_);
    EXPECT_EQ(offsets->v_offset % page_size_,
              block * kBlockSize * kRopeSize * sizeof(uint16_t) % page_size_);
    expect_global_block_pattern(
        mapping, offsets->k_offset, keys.front(), key_pattern);
    expect_global_block_pattern(
        mapping, offsets->v_offset, values.front(), value_pattern);
  }
  EXPECT_FALSE(registry_.get_global_offsets_for_block("missing", 0, 0));
  EXPECT_FALSE(registry_.get_global_offsets_for_block("mla", -1, 0));
  EXPECT_FALSE(registry_.get_global_offsets_for_block("mla", 1, 0));
  EXPECT_FALSE(registry_.get_global_offsets_for_block("mla", 0, -1));
  EXPECT_FALSE(
      registry_.get_global_offsets_for_block("mla", 0, 2 * blocks_per_group));
  ASSERT_TRUE(registry_.unmap("mla", {0}));
  EXPECT_FALSE(registry_.get_global_offsets_for_block("mla", 0, 0));
  EXPECT_TRUE(
      registry_.get_global_offsets_for_block("mla", 0, blocks_per_group));
  ASSERT_TRUE(registry_.map("mla", {0}));
  fill_block_pattern(keys.front(), /*block_id=*/1, /*pattern=*/201);
  fill_block_pattern(values.front(), /*block_id=*/1, /*pattern=*/202);
  const auto remapped = registry_.get_global_offsets_for_block("mla", 0, 1);
  ASSERT_TRUE(remapped.has_value());
  expect_global_block_pattern(mapping, remapped->k_offset, keys.front(), 201);
  expect_global_block_pattern(mapping, remapped->v_offset, values.front(), 202);
}

TEST_F(VirtualKVCacheMemoryRegistryTest,
       TransferRejectsBlocksThatDoNotDivideOnePage) {
  registry_.create_k_tensors("large_block",
                             {1, static_cast<int64_t>(2 * page_size_)},
                             torch::kUInt8,
                             /*num_layers=*/1);
  registry_.create_v_tensors("large_block",
                             {1, static_cast<int64_t>(page_size_)},
                             torch::kUInt8,
                             /*num_layers=*/1);
  ASSERT_TRUE(registry_.map("large_block", {0}));
  EXPECT_FALSE(registry_.get_global_offsets_for_block("large_block", 0, 0));
  registry_.erase("large_block");

  ASSERT_NE(page_size_ % 3, 0U);
  registry_.create_k_tensors("misaligned",
                             {static_cast<int64_t>(page_size_), 3},
                             torch::kUInt8,
                             /*num_layers=*/1);
  registry_.create_v_tensors("misaligned",
                             {static_cast<int64_t>(page_size_), 1},
                             torch::kUInt8,
                             /*num_layers=*/1);
  ASSERT_TRUE(registry_.map("misaligned", {0}));
  EXPECT_FALSE(registry_.get_global_offsets_for_block("misaligned", 0, 0));
}

TEST_F(VirtualMLAKVCacheMemoryRegistryTest,
       ShortageDoesNotPartiallyMapLatentRopeOrAnotherLayer) {
  create_model("busy", /*num_layers=*/1);
  auto [keys, values] =
      create_mla_model("mla", blocks_per_page(), /*num_layers=*/2);
  ASSERT_TRUE(registry_.map("busy", {0}));
  auto& pool = PhysicalPagePool::get_instance();
  ASSERT_EQ(pool.num_available(), 16);
  const auto busy_offsets = registry_.get_global_offsets_for_block(
      "busy", /*layer_id=*/0, /*block_id=*/0, page_size_);

  EXPECT_FALSE(registry_.map("mla", {0}));
  EXPECT_EQ(pool.num_available(), 16);
  EXPECT_EQ(registry_.get_global_offsets_for_block(
                "busy", /*layer_id=*/0, /*block_id=*/0, page_size_),
            busy_offsets);
  ASSERT_TRUE(registry_.unmap("mla", {0}));
  EXPECT_EQ(pool.num_available(), 16);
  ASSERT_TRUE(registry_.unmap("busy", {0}));
  ASSERT_TRUE(registry_.map("mla", {0}));
  EXPECT_EQ(pool.num_available(), 0);
  for (size_t layer = 0; layer < keys.size(); ++layer) {
    fill_block_pattern(
        keys[layer], blocks_per_page() - 1, static_cast<uint8_t>(30 + layer));
    fill_block_pattern(values[layer],
                       blocks_per_page() - 1,
                       static_cast<uint8_t>(130 + layer));
  }
  for (size_t layer = 0; layer < keys.size(); ++layer) {
    expect_block_pattern(
        keys[layer], blocks_per_page() - 1, static_cast<uint8_t>(30 + layer));
    expect_block_pattern(values[layer],
                         blocks_per_page() - 1,
                         static_cast<uint8_t>(130 + layer));
  }
}

TEST_F(VirtualMLAKVCacheMemoryRegistryTest,
       IncompleteLogicalPagesDoNotChangeExistingMappings) {
  ASSERT_GT(blocks_per_page(), 1);
  auto [keys, values] =
      create_mla_model("mla", 2 * blocks_per_page() - 1, /*num_layers=*/1);
  auto& pool = PhysicalPagePool::get_instance();
  const offset_t partial_offset = static_cast<offset_t>(page_size_);
  EXPECT_FALSE(registry_.map("mla", {0, partial_offset}));
  EXPECT_EQ(pool.num_available(), pool_pages());
  ASSERT_TRUE(registry_.map("mla", {0}));
  fill_block_pattern(keys.front(), /*block_id=*/0, /*pattern=*/51);
  fill_block_pattern(values.front(), /*block_id=*/0, /*pattern=*/151);
  EXPECT_FALSE(registry_.map("mla", {partial_offset}));
  EXPECT_FALSE(registry_.unmap("mla", {0, partial_offset}));
  EXPECT_EQ(pool.num_available(), 9);
  expect_block_pattern(keys.front(), /*block_id=*/0, /*pattern=*/51);
  expect_block_pattern(values.front(), /*block_id=*/0, /*pattern=*/151);
}

TEST_F(VirtualIndexedKVCacheMemoryRegistryTest,
       SparseIndexRegionsKeepLayerIdsAndGroupBoundaries) {
  const int64_t blocks_per_group = blocks_per_page();
  const int64_t blocks_per_index_page = static_cast<int64_t>(
      page_size_ / (kBlockSize * kIndexSize * sizeof(uint16_t)));
  ASSERT_EQ(blocks_per_group, 2 * blocks_per_index_page);
  auto [keys, values, indices] = create_indexed_model(
      "indexed", 2 * blocks_per_group, /*num_layers=*/3, {true, false, true});
  ASSERT_EQ(indices.size(), 3);
  ASSERT_TRUE(indices[0].defined());
  EXPECT_FALSE(indices[1].defined());
  ASSERT_TRUE(indices[2].defined());
  EXPECT_NE(indices[0].data_ptr(), indices[2].data_ptr());
  const offset_t next_offset = static_cast<offset_t>(page_size_);
  const offset_t invalid_offset = static_cast<offset_t>(2 * page_size_);
  auto& pool = PhysicalPagePool::get_instance();
  EXPECT_FALSE(registry_.map("indexed", {0, invalid_offset}));
  EXPECT_EQ(pool.num_available(), pool_pages());
  ASSERT_TRUE(registry_.map("indexed", {0, next_offset}));
  EXPECT_EQ(pool.num_available(), 0);
  EXPECT_TRUE(registry_.map("indexed", {0, next_offset}));
  EXPECT_EQ(pool.num_available(), 0);

  const std::vector<int64_t> blocks{0,
                                    blocks_per_index_page - 1,
                                    blocks_per_index_page,
                                    blocks_per_group - 1,
                                    blocks_per_group,
                                    2 * blocks_per_group - 1};
  for (size_t layer = 0; layer < keys.size(); ++layer) {
    for (size_t block = 0; block < blocks.size(); ++block) {
      const uint8_t pattern = static_cast<uint8_t>(10 + 20 * layer + block);
      fill_block_pattern(keys[layer], blocks[block], pattern);
      fill_block_pattern(values[layer], blocks[block], pattern + 70);
      if (indices[layer].defined()) {
        fill_block_pattern(indices[layer], blocks[block], pattern + 140);
      }
    }
  }
  EXPECT_FALSE(registry_.unmap("indexed", {0, invalid_offset}));
  EXPECT_EQ(pool.num_available(), 0);
  for (size_t layer = 0; layer < keys.size(); ++layer) {
    for (size_t block = 0; block < blocks.size(); ++block) {
      const uint8_t pattern = static_cast<uint8_t>(10 + 20 * layer + block);
      expect_block_pattern(keys[layer], blocks[block], pattern);
      expect_block_pattern(values[layer], blocks[block], pattern + 70);
      if (indices[layer].defined()) {
        expect_block_pattern(indices[layer], blocks[block], pattern + 140);
      }
    }
  }

  ASSERT_TRUE(registry_.unmap("indexed", {0}));
  EXPECT_EQ(pool.num_available(), 31);
  ASSERT_TRUE(registry_.map("indexed", {0}));
  EXPECT_EQ(pool.num_available(), 0);
  for (size_t layer = 0; layer < keys.size(); ++layer) {
    fill_block_pattern(keys[layer], /*block_id=*/0, /*pattern=*/200);
    fill_block_pattern(values[layer], /*block_id=*/0, /*pattern=*/201);
    if (indices[layer].defined()) {
      fill_block_pattern(indices[layer], /*block_id=*/0, /*pattern=*/202);
    }
    const uint8_t last_pattern = static_cast<uint8_t>(15 + 20 * layer);
    expect_block_pattern(keys[layer], 2 * blocks_per_group - 1, last_pattern);
    expect_block_pattern(
        values[layer], 2 * blocks_per_group - 1, last_pattern + 70);
    if (indices[layer].defined()) {
      expect_block_pattern(
          indices[layer], 2 * blocks_per_group - 1, last_pattern + 140);
    }
  }
  registry_.erase("indexed");
  EXPECT_EQ(pool.num_available(), pool_pages());
}

TEST_F(VirtualIndexedKVCacheMemoryRegistryTest,
       TransferOffsetsPreserveSparseIndexOwnership) {
  const int64_t blocks_per_group = blocks_per_page();
  auto [keys, values, indices] = create_indexed_model(
      "indexed", 2 * blocks_per_group, /*num_layers=*/3, {true, false, true});
  ASSERT_TRUE(registry_.map("indexed", {0, static_cast<offset_t>(page_size_)}));
  auto& pool = PhysicalPagePool::get_instance();
  SharedPhysicalPageMapping mapping;
  mapping.init(device_, pool.get_all_pages(), page_size_);
  ASSERT_TRUE(mapping.is_initialized());
  const std::vector<int64_t> blocks{
      1, blocks_per_group / 2 - 1, blocks_per_group / 2, blocks_per_group};
  for (size_t layer = 0; layer < keys.size(); ++layer) {
    for (int64_t block : blocks) {
      fill_block_pattern(keys[layer], block, /*pattern=*/31);
      fill_block_pattern(values[layer], block, /*pattern=*/41);
      const auto offsets = registry_.get_global_offsets_for_block(
          "indexed", static_cast<int64_t>(layer), block);
      ASSERT_TRUE(offsets.has_value());
      expect_global_block_pattern(mapping, offsets->k_offset, keys[layer], 31);
      expect_global_block_pattern(
          mapping, offsets->v_offset, values[layer], 41);
      ASSERT_EQ(offsets->index_offset.has_value(), indices[layer].defined());
      if (!indices[layer].defined()) {
        continue;
      }
      fill_block_pattern(indices[layer], block, /*pattern=*/51);
      EXPECT_EQ(
          offsets->index_offset.value() % page_size_,
          block * kBlockSize * kIndexSize * sizeof(uint16_t) % page_size_);
      expect_global_block_pattern(
          mapping, offsets->index_offset.value(), indices[layer], 51);
    }
  }
  ASSERT_TRUE(registry_.unmap("indexed", {0}));
  for (int64_t layer = 0; layer < 3; ++layer) {
    EXPECT_FALSE(registry_.get_global_offsets_for_block("indexed", layer, 1));
    const auto offsets = registry_.get_global_offsets_for_block(
        "indexed", layer, blocks_per_group);
    ASSERT_TRUE(offsets.has_value());
    EXPECT_EQ(offsets->index_offset.has_value(), layer != 1);
  }
}

TEST_F(VirtualIndexedKVCacheMemoryRegistryTest,
       ShortageKeepsKeyValueAndIndexMappingsAtomic) {
  create_model("busy", /*num_layers=*/1);
  create_indexed_model(
      "indexed", 2 * blocks_per_page(), /*num_layers=*/3, {true, false, true});
  ASSERT_TRUE(registry_.map("busy", {0}));
  const auto busy_offsets = registry_.get_global_offsets_for_block(
      "busy", /*layer_id=*/0, /*block_id=*/0, page_size_);
  auto& pool = PhysicalPagePool::get_instance();
  EXPECT_EQ(pool.num_available(), 60);
  const offset_t next_offset = static_cast<offset_t>(page_size_);
  EXPECT_FALSE(registry_.map("indexed", {0, next_offset}));
  EXPECT_EQ(pool.num_available(), 60);
  EXPECT_EQ(registry_.get_global_offsets_for_block(
                "busy", /*layer_id=*/0, /*block_id=*/0, page_size_),
            busy_offsets);
  ASSERT_TRUE(registry_.unmap("indexed", {0, next_offset}));
  EXPECT_EQ(pool.num_available(), 60);
  ASSERT_TRUE(registry_.unmap("busy", {0}));
  ASSERT_TRUE(registry_.map("indexed", {0, next_offset}));
  EXPECT_EQ(pool.num_available(), 0);
  registry_.erase("indexed");
  EXPECT_EQ(pool.num_available(), pool_pages());
}

TEST_F(VirtualIndexedKVCacheMemoryRegistryTest,
       TensorAllocatorSkipsSharedIndexLayers) {
  const std::vector<bool> index_layer_owned{false, true, false, true};
  auto allocator = create_vmm_kv_tensor_allocator(
      registry_, "indexed", /*num_layers=*/4, index_layer_owned);
  const std::vector<int64_t> key_shape{
      blocks_per_page(), kBlockSize, 1, kLatentSize};
  const std::vector<int64_t> value_shape{
      blocks_per_page(), kBlockSize, 1, kRopeSize};
  const std::vector<int64_t> index_shape{
      blocks_per_page(), kBlockSize, 1, kIndexSize};
  std::vector<torch::Tensor> indices(4);
  for (size_t layer = 0; layer < index_layer_owned.size(); ++layer) {
    allocator->allocate(
        KVCacheTensorRole::KEY, key_shape, torch::kBFloat16, device_);
    allocator->allocate(
        KVCacheTensorRole::VALUE, value_shape, torch::kBFloat16, device_);
    if (index_layer_owned[layer]) {
      indices[layer] = allocator->allocate(
          KVCacheTensorRole::INDEX, index_shape, torch::kBFloat16, device_);
    }
  }
  ASSERT_TRUE(indices[1].defined());
  ASSERT_TRUE(indices[3].defined());
  EXPECT_NE(indices[1].data_ptr(), indices[3].data_ptr());
  ASSERT_TRUE(registry_.map("indexed", {0}));
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), 22);
  fill_block_pattern(indices[1], /*block_id=*/0, /*pattern=*/40);
  fill_block_pattern(indices[3], /*block_id=*/0, /*pattern=*/140);
  expect_block_pattern(indices[1], /*block_id=*/0, /*pattern=*/40);
  expect_block_pattern(indices[3], /*block_id=*/0, /*pattern=*/140);
  ASSERT_TRUE(registry_.unmap("indexed", {0}));
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), pool_pages());
}

TEST_F(VirtualIndexedKVCacheMemoryRegistryTest,
       DefaultIndexOwnershipMapsAllLayersAndRejectsTransfer) {
  auto allocator =
      create_vmm_kv_tensor_allocator(registry_, "indexed", /*num_layers=*/2);
  const std::vector<int64_t> shape{blocks_per_page(), kBlockSize, 1, kRopeSize};
  std::vector<torch::Tensor> indices;
  for (size_t layer = 0; layer < 2; ++layer) {
    allocator->allocate(
        KVCacheTensorRole::KEY, shape, torch::kBFloat16, device_);
    allocator->allocate(
        KVCacheTensorRole::VALUE, shape, torch::kBFloat16, device_);
    indices.emplace_back(allocator->allocate(
        KVCacheTensorRole::INDEX, shape, torch::kBFloat16, device_));
  }
  ASSERT_TRUE(registry_.map("indexed", {0}));
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), 56);
  fill_block_pattern(indices[0], /*block_id=*/0, /*pattern=*/41);
  fill_block_pattern(indices[1], /*block_id=*/0, /*pattern=*/141);
  expect_block_pattern(indices[0], /*block_id=*/0, /*pattern=*/41);
  expect_block_pattern(indices[1], /*block_id=*/0, /*pattern=*/141);
  EXPECT_EQ(registry_.get_global_offsets_for_block(
                "indexed",
                /*layer_id=*/0,
                /*block_id=*/0,
                kBlockSize * kRopeSize * sizeof(uint16_t)),
            std::make_pair(UINT64_MAX, UINT64_MAX));
  registry_.clear();
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), pool_pages());
}

TEST_F(VirtualIndexedKVCacheMemoryRegistryTest,
       UnownedIndexWidthDoesNotChangeKVPageGeometry) {
  const int64_t blocks_per_group = blocks_per_page();
  const std::vector<int64_t> key_shape{
      blocks_per_group, kBlockSize, 1, kLatentSize};
  const std::vector<int64_t> value_shape{
      blocks_per_group, kBlockSize, 1, kRopeSize};
  const std::vector<int64_t> index_shape{blocks_per_group, kBlockSize, 1, 96};
  registry_.create_k_tensors("indexed", key_shape, torch::kBFloat16, 3);
  registry_.create_v_tensors("indexed", value_shape, torch::kBFloat16, 3);
  const auto indices = registry_.create_index_tensors(
      "indexed", index_shape, torch::kBFloat16, 3, {false, false, false});
  ASSERT_EQ(indices.size(), 3U);
  for (const torch::Tensor& index : indices) {
    EXPECT_FALSE(index.defined());
  }

  ASSERT_TRUE(registry_.map("indexed", {0}));
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), 35);
  for (int64_t layer = 0; layer < 3; ++layer) {
    const auto offsets =
        registry_.get_global_offsets_for_block("indexed", layer, 1);
    ASSERT_TRUE(offsets.has_value());
    EXPECT_FALSE(offsets->index_offset.has_value());
  }
  registry_.erase("indexed");
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), pool_pages());

  registry_.create_k_tensors(
      "symmetric", value_shape, torch::kBFloat16, /*num_layers=*/3);
  registry_.create_v_tensors(
      "symmetric", value_shape, torch::kBFloat16, /*num_layers=*/3);
  registry_.create_index_tensors("symmetric",
                                 index_shape,
                                 torch::kBFloat16,
                                 /*num_layers=*/3,
                                 {false, false, false});
  ASSERT_TRUE(registry_.map("symmetric", {0}));
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), pool_pages() - 6);
  const size_t block_bytes = kBlockSize * kRopeSize * sizeof(uint16_t);
  for (int64_t layer = 0; layer < 3; ++layer) {
    const auto offsets =
        registry_.get_global_offsets_for_block("symmetric", layer, 1);
    ASSERT_TRUE(offsets.has_value());
    EXPECT_FALSE(offsets->index_offset.has_value());
    EXPECT_EQ(registry_.get_global_offsets_for_block(
                  "symmetric", layer, /*block_id=*/1, block_bytes),
              std::make_pair(offsets->k_offset, offsets->v_offset));
  }
  registry_.erase("symmetric");
  EXPECT_EQ(PhysicalPagePool::get_instance().num_available(), pool_pages());
}

TEST_F(VirtualIndexedKVCacheMemoryRegistryTest,
       AllocateKVCacheUsesSparseVirtualIndexRegions) {
  ASSERT_EQ(page_size_ % (kBlockSize * kRopeSize * sizeof(uint16_t)), 0);
  const int64_t blocks_per_group = blocks_per_page();
  const std::vector<bool> index_layer_owned{true, false, true};

  KVCacheCapacity capacity;
  capacity.n_blocks(2 * blocks_per_group).block_size(kBlockSize);
  ModelArgs model_args;
  model_args.model_type("glm_moe_dsa")
      .enable_mla(true)
      .n_heads(8)
      .n_kv_heads(2)
      .head_dim(64)
      .kv_lora_rank(kLatentSize)
      .qk_rope_head_dim(kRopeSize)
      .index_n_heads(1)
      .index_head_dim(kIndexSize);
  const KVCacheShape shape(capacity, model_args, /*world_size=*/1);
  ASSERT_EQ(
      shape.key_cache_shape(),
      (std::vector<int64_t>{2 * blocks_per_group, kBlockSize, 1, kLatentSize}));
  ASSERT_EQ(
      shape.value_cache_shape(),
      (std::vector<int64_t>{2 * blocks_per_group, kBlockSize, 1, kRopeSize}));
  ASSERT_EQ(
      shape.index_cache_shape(),
      (std::vector<int64_t>{2 * blocks_per_group, kBlockSize, 1, kIndexSize}));

  auto virtual_allocator = create_vmm_kv_tensor_allocator(
      registry_, "indexed", /*num_layers=*/3, index_layer_owned);
  auto allocator = std::make_shared<RecordingForwardingKVCacheTensorAllocator>(
      std::move(virtual_allocator));
  KVCacheCreateOptions options;
  options.device(device_)
      .dtype(torch::kBFloat16)
      .num_layers(3)
      .model_id("indexed")
      .model_type("glm_moe_dsa")
      .enable_virtual_memory(true)
      .enable_lighting_indexer(true)
      .indexer_cache_enabled_layers(index_layer_owned)
      .tensor_allocator(allocator);

  std::vector<KVCache> caches;
  allocate_kv_caches(caches, shape, options);
  ASSERT_EQ(caches.size(), 3U);
  ASSERT_TRUE(caches[0].get_index_cache().defined());
  EXPECT_FALSE(caches[1].get_index_cache().defined());
  ASSERT_TRUE(caches[2].get_index_cache().defined());
  ASSERT_EQ(allocator->roles(),
            (std::vector<KVCacheTensorRole>{KVCacheTensorRole::KEY,
                                            KVCacheTensorRole::VALUE,
                                            KVCacheTensorRole::INDEX,
                                            KVCacheTensorRole::KEY,
                                            KVCacheTensorRole::VALUE,
                                            KVCacheTensorRole::KEY,
                                            KVCacheTensorRole::VALUE,
                                            KVCacheTensorRole::INDEX}));
  EXPECT_EQ(caches[0].get_k_cache().data_ptr(),
            allocator->tensors()[0].data_ptr());
  EXPECT_EQ(caches[0].get_v_cache().data_ptr(),
            allocator->tensors()[1].data_ptr());
  EXPECT_EQ(caches[0].get_index_cache().data_ptr(),
            allocator->tensors()[2].data_ptr());
  EXPECT_EQ(caches[1].get_k_cache().data_ptr(),
            allocator->tensors()[3].data_ptr());
  EXPECT_EQ(caches[1].get_v_cache().data_ptr(),
            allocator->tensors()[4].data_ptr());
  EXPECT_EQ(caches[2].get_k_cache().data_ptr(),
            allocator->tensors()[5].data_ptr());
  EXPECT_EQ(caches[2].get_v_cache().data_ptr(),
            allocator->tensors()[6].data_ptr());
  EXPECT_EQ(caches[2].get_index_cache().data_ptr(),
            allocator->tensors()[7].data_ptr());
  EXPECT_EQ(caches[0].get_k_cache().scalar_type(), torch::kBFloat16);
  EXPECT_EQ(caches[0].get_v_cache().scalar_type(), torch::kBFloat16);
  EXPECT_EQ(caches[0].get_index_cache().scalar_type(), torch::kBFloat16);
  auto& pool = PhysicalPagePool::get_instance();
  EXPECT_EQ(pool.num_available(), pool_pages());

  ASSERT_TRUE(registry_.map("indexed", {0}));
  EXPECT_EQ(pool.num_available(), 31);
  for (size_t layer = 0; layer < caches.size(); ++layer) {
    const uint8_t pattern = static_cast<uint8_t>(10 + layer * 20);
    fill_block_pattern(
        caches[layer].get_k_cache(), blocks_per_group - 1, pattern);
    fill_block_pattern(
        caches[layer].get_v_cache(), blocks_per_group - 1, pattern + 70);
    if (index_layer_owned[layer]) {
      fill_block_pattern(
          caches[layer].get_index_cache(), blocks_per_group - 1, pattern + 140);
    }
  }
  for (size_t layer = 0; layer < caches.size(); ++layer) {
    const uint8_t pattern = static_cast<uint8_t>(10 + layer * 20);
    expect_block_pattern(
        caches[layer].get_k_cache(), blocks_per_group - 1, pattern);
    expect_block_pattern(
        caches[layer].get_v_cache(), blocks_per_group - 1, pattern + 70);
    if (index_layer_owned[layer]) {
      expect_block_pattern(
          caches[layer].get_index_cache(), blocks_per_group - 1, pattern + 140);
    }
  }
  ASSERT_TRUE(registry_.unmap("indexed", {0}));
  EXPECT_EQ(pool.num_available(), pool_pages());
  registry_.erase("indexed");
}

}  // namespace
}  // namespace xllm
