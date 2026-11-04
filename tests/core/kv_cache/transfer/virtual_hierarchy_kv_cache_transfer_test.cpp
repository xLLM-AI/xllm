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

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/kv_cache/layout/kv_cache_capacity.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/layout/vmm_kv_cache_layout.h"
#include "core/kv_cache/storage/virtual_kv_cache_memory_registry.h"
#include "core/kv_cache/storage/vmm_kv_tensor_allocator.h"
#include "core/kv_cache/transfer/hierarchy_kv_cache_transfer.h"
#include "core/platform/vmm_api.h"
#include "framework/model/model_args.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace {

class VirtualHierarchyKVCacheTransferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() < 1) {
      GTEST_SKIP() << "An NPU device is required for virtual Host KV transfer.";
    }
    device_.set_device();
    device_.init_device_context();
    page_size_ = vmm::get_recommended_granularity(device_.index());
    ASSERT_GT(page_size_, 0U);
    registry_.init(device_.unwrap(), page_size_);
  }

  void TearDown() override {
    registry_.clear();
    if (pool_initialized_) {
      auto& pool = PhysicalPagePool::get_instance();
      EXPECT_EQ(pool.num_available(), pool_pages_);
      EXPECT_TRUE(pool.reset());
    }
  }

  void run_round_trip(const ModelArgs& model_args) {
    constexpr int64_t kBlockSize = 128;
    constexpr int64_t kLayerCount = 2;
    constexpr int64_t kHostBlockId = 2;
    constexpr uint64_t kBatchId = 17;

    KVCacheCapacity capacity;
    capacity.n_blocks(1).block_size(kBlockSize).n_layers(kLayerCount);
    const VmmKVCacheLayout layout(
        capacity, model_args, /*world_size=*/1, torch::kBFloat16);
    const int64_t blocks_per_page =
        static_cast<int64_t>(layout.page_layout(page_size_).blocks_per_page());
    ASSERT_GT(blocks_per_page, 2);
    pool_pages_ = layout.physical_pages(page_size_);
    auto& pool = PhysicalPagePool::get_instance();
    pool.init(device_.unwrap(), pool_pages_, page_size_);
    pool_initialized_ = true;

    capacity.n_blocks(2 * blocks_per_page);
    const KVCacheShape cache_shape(capacity, model_args, /*world_size=*/1);
    KVCacheCreateOptions create_options;
    create_options.device(device_.unwrap())
        .dtype(torch::kBFloat16)
        .num_layers(kLayerCount)
        .model_id("host_round_trip")
        .model_type(model_args.model_type())
        .enable_vmm_kvcache(true)
        .enable_lighting_indexer(model_args.index_n_heads() > 0)
        .tensor_allocator(create_vmm_kv_tensor_allocator(
            registry_, "host_round_trip", kLayerCount));
    std::vector<KVCache> caches;
    allocate_kv_caches(caches, cache_shape, create_options);
    ASSERT_EQ(caches.size(), static_cast<size_t>(kLayerCount));
    ASSERT_EQ(pool.num_available(), pool_pages_);
    ASSERT_TRUE(registry_.map("host_round_trip", {0}));
    ASSERT_EQ(pool.num_available(), 0U);

    const int64_t source_block_id = blocks_per_page - 1;
    const int64_t destination_block_id = blocks_per_page;
    std::vector<BlockTypeTensorMap> expected(caches.size());
    for (size_t layer = 0; layer < caches.size(); ++layer) {
      const BlockTypeTensorMap tensors =
          caches[layer].get_block_type_tensors(BlockType::KV);
      ASSERT_EQ(tensors.size(), model_args.index_n_heads() > 0 ? 3U : 2U);
      for (const auto& [role, tensor] : tensors) {
        const torch::Tensor block = tensor[source_block_id];
        const int64_t offset =
            static_cast<int64_t>(layer * 10) + static_cast<int32_t>(role);
        const torch::Tensor values =
            (torch::arange(block.numel(), torch::kInt64) + offset)
                .remainder(127)
                .reshape(block.sizes())
                .to(block.options());
        block.copy_(values);
        expected[layer].emplace(role, tensor[source_block_id].cpu());
      }
    }
    ASSERT_EQ(device_.synchronize_default_stream(), 0);

    HierarchyKVCacheTransfer::Options transfer_options;
    transfer_options.layers(kLayerCount)
        .host_blocks_factor(2.0)
        .layers_wise_copy_batchs(2);
    std::unique_ptr<Stream> compute_stream = device_.current_stream();
    HierarchyKVCacheTransfer transfer(transfer_options,
                                      device_.unwrap(),
                                      compute_stream.get(),
                                      &caches,
                                      cache_shape,
                                      create_options);
    BlockTransferInfo offload_info(source_block_id, kHostBlockId);
    offload_info.block_type = BlockType::KV;
    offload_info.transfer_type = TransferType::D2H2G;
    ASSERT_EQ(transfer.transfer_kv_blocks(kBatchId, {offload_info}), 1U);

    // Only one logical page group fits: reload must reuse the released backing
    // at another virtual address, after the original source is unmapped.
    ASSERT_TRUE(registry_.unmap("host_round_trip", {0}));
    ASSERT_EQ(pool.num_available(), pool_pages_);
    EXPECT_FALSE(registry_.get_global_offsets_for_block(
        "host_round_trip", /*layer_id=*/0, source_block_id));
    ASSERT_TRUE(
        registry_.map("host_round_trip", {static_cast<offset_t>(page_size_)}));
    ASSERT_EQ(pool.num_available(), 0U);
    for (const KVCache& cache : caches) {
      for (const auto& [role, tensor] :
           cache.get_block_type_tensors(BlockType::KV)) {
        tensor[destination_block_id].zero_();
      }
    }
    ASSERT_EQ(device_.synchronize_default_stream(), 0);

    BlockTransferInfo load_info(kHostBlockId, destination_block_id);
    load_info.block_type = BlockType::KV;
    load_info.transfer_type = TransferType::H2D;
    ASSERT_EQ(transfer.transfer_kv_blocks(kBatchId, {load_info}), 1U);
    ModelParamsView input_params = ModelInputSnapshot(LlmModelParams()).view();
    input_params.meta.batch_id = kBatchId;
    transfer.set_layer_synchronizer(input_params);
    ASSERT_NE(input_params.parallel.layer_wise_load_synchronizer, nullptr);
    for (int64_t layer = 0; layer < kLayerCount; ++layer) {
      ASSERT_TRUE(input_params.synchronize_layer(layer));
    }
    ASSERT_TRUE(transfer.wait_for_load(kBatchId));
    ASSERT_EQ(device_.synchronize_default_stream(), 0);
    for (size_t layer = 0; layer < caches.size(); ++layer) {
      for (const auto& [role, tensor] :
           caches[layer].get_block_type_tensors(BlockType::KV)) {
        SCOPED_TRACE(layer);
        SCOPED_TRACE(static_cast<int32_t>(role));
        EXPECT_TRUE(torch::equal(tensor[destination_block_id].cpu(),
                                 expected[layer].at(role)));
      }
    }
    transfer.shutdown();
    ASSERT_TRUE(registry_.unmap("host_round_trip",
                                {static_cast<offset_t>(page_size_)}));
    EXPECT_EQ(pool.num_available(), pool_pages_);
  }

  Device device_{/*device_index=*/0};
  VirtualKVCacheMemoryRegistry registry_;
  size_t page_size_ = 0;
  size_t pool_pages_ = 0;
  bool pool_initialized_ = false;
};

TEST_F(VirtualHierarchyKVCacheTransferTest,
       RestoresVirtualKVAfterSourcePagesAreUnmapped) {
  ModelArgs model_args;
  model_args.model_type("test_model")
      .n_layers(2)
      .n_heads(2)
      .n_kv_heads(1)
      .head_dim(64);
  run_round_trip(model_args);
}

TEST_F(VirtualHierarchyKVCacheTransferTest,
       RestoresVirtualMLAAndIndexAfterSourcePagesAreUnmapped) {
  ModelArgs model_args;
  model_args.model_type("glm_moe_dsa")
      .n_layers(2)
      .enable_mla(true)
      .n_heads(8)
      .n_kv_heads(2)
      .head_dim(64)
      .kv_lora_rank(512)
      .qk_rope_head_dim(64)
      .index_n_heads(1)
      .index_head_dim(128);
  run_round_trip(model_args);
}

}  // namespace
}  // namespace xllm
