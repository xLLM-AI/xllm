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

#include "core/kv_cache/transfer/hierarchy_kv_cache_transfer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/kv_cache/layout/kv_cache_capacity.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "framework/model/model_args.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "tests/core/kv_cache/dcp_indexer_test_fixture.h"

namespace xllm {
namespace {

TEST(HierarchyKVCacheTransferTest,
     RoundTripPublishesStrategyLayerReadyGranularity) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP()
        << "An accelerator device is required for hierarchy KV transfer.";
  }

  constexpr int64_t kBlockCount = 2;
  constexpr int64_t kBlockSize = 4;
  constexpr int64_t kLayerCount = 4;
  constexpr int64_t kSourceBlockId = 0;
  constexpr int64_t kDestinationBlockId = 1;
  constexpr uint64_t kBatchId = 7;

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();

  KVCacheCapacity capacity;
  capacity.n_blocks(kBlockCount).block_size(kBlockSize);
  ModelArgs model_args;
  model_args.model_type("test_model")
      .n_layers(kLayerCount)
      .n_heads(2)
      .n_kv_heads(1)
      .head_dim(8);
  const KVCacheShape cache_shape(capacity, model_args, /*world_size=*/1);

  KVCacheCreateOptions create_options;
  create_options.device(device.unwrap())
      .dtype(torch::kFloat32)
      .num_layers(kLayerCount)
      .model_type("test_model");
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, cache_shape, create_options);
  ASSERT_EQ(caches.size(), static_cast<size_t>(kLayerCount));

  for (size_t layer_index = 0; layer_index < caches.size(); ++layer_index) {
    const double layer_value = static_cast<double>(layer_index);
    caches[layer_index].get_k_cache()[kSourceBlockId].fill_(3.0 + layer_value);
    caches[layer_index].get_v_cache()[kSourceBlockId].fill_(7.0 + layer_value);
    caches[layer_index].get_k_cache()[kDestinationBlockId].zero_();
    caches[layer_index].get_v_cache()[kDestinationBlockId].zero_();
  }
  ASSERT_EQ(device.synchronize_default_stream(), 0);

  HierarchyKVCacheTransfer::Options transfer_options;
  transfer_options.layers(kLayerCount)
      .host_blocks_factor(2.0)
      .layers_wise_copy_batchs(2);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  HierarchyKVCacheTransfer transfer(transfer_options,
                                    device.unwrap(),
                                    compute_stream.get(),
                                    &caches,
                                    cache_shape,
                                    create_options);

  GTEST_FLAG_SET(death_test_style, "threadsafe");
  LlmModelParams missing_storage;
  ModelParamsView missing(missing_storage);
  missing.meta.batch_id = kBatchId;
  missing.meta.requires_host_restore = true;
  EXPECT_DEATH(transfer.set_layer_synchronizer(missing),
               "Missing Host restore handle");

  BlockTransferInfo offload_info(kSourceBlockId, /*dst_block_id=*/0);
  offload_info.block_type = BlockType::KV;
  offload_info.transfer_type = TransferType::D2H2G;
  EXPECT_EQ(transfer.transfer_kv_blocks(kBatchId, {offload_info}), 1U);

  BlockTransferInfo load_info(/*src_block_id=*/0, kDestinationBlockId);
  load_info.block_type = BlockType::KV;
  load_info.transfer_type = TransferType::H2D;
  EXPECT_EQ(transfer.transfer_kv_blocks(kBatchId, {load_info}), 1U);

  LlmModelParams input_storage;
  ModelParamsView input_params(input_storage);
  input_params.meta.batch_id = kBatchId;
  input_params.meta.requires_host_restore = true;
  transfer.set_layer_synchronizer(input_params);
  ASSERT_NE(input_params.parallel.layer_wise_load_synchronizer, nullptr);
  EXPECT_FALSE(input_params.meta.requires_host_restore);
  const auto synchronizer = input_params.parallel.layer_wise_load_synchronizer;
  transfer.set_layer_synchronizer(input_params);
  EXPECT_EQ(input_params.parallel.layer_wise_load_synchronizer, synchronizer);
  EXPECT_FALSE(input_params.meta.requires_host_restore);
  missing.meta.batch_id = kBatchId;
  EXPECT_DEATH(transfer.set_layer_synchronizer(missing),
               "Missing Host restore handle");
  EXPECT_EQ(input_params.parallel.layer_wise_load_synchronizer->size(), 2U);
  EXPECT_EQ(input_params.parallel.layers_per_event, 2U);
  for (uint32_t layer_index = 0; layer_index < kLayerCount; ++layer_index) {
    ASSERT_TRUE(input_params.synchronize_layer(layer_index));
  }

  for (KVCache& cache : caches) {
    EXPECT_TRUE(torch::equal(cache.get_k_cache()[kSourceBlockId],
                             cache.get_k_cache()[kDestinationBlockId]));
    EXPECT_TRUE(torch::equal(cache.get_v_cache()[kSourceBlockId],
                             cache.get_v_cache()[kDestinationBlockId]));
  }
}

#if defined(USE_NPU)
class HierarchyIndexerTransferTest : public test::DcpIndexerConfigTest {};

TEST_P(HierarchyIndexerTransferTest,
       RoundTripRestoresEveryPhysicalIndexerPageAndScale) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP()
        << "An accelerator device is required for hierarchy KV transfer.";
  }

  constexpr int64_t kBlockCount = 3;
  constexpr int64_t kBlockSize = 16;
  constexpr int64_t kSourceBlockId = 1;
  constexpr int64_t kDestinationBlockId = 2;
  constexpr int64_t kUntouchedBlockId = 0;
  constexpr uint64_t kBatchId = 11;

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();

  ModelArgs model_args;
  model_args.n_layers(1)
      .n_heads(2)
      .n_kv_heads(2)
      .head_dim(16)
      .index_n_heads(1)
      .index_head_dim(16);
  KVCacheCapacity capacity;
  capacity.n_blocks(kBlockCount)
      .block_size(kBlockSize)
      .enable_indexer_cache_quant(true);
  const KVCacheShape cache_shape(capacity, model_args, /*world_size=*/1);
  ASSERT_TRUE(cache_shape.has_index_cache_shape());
  ASSERT_TRUE(cache_shape.has_index_cache_scale_shape());
  ASSERT_EQ(cache_shape.index_cache_shape()[0],
            kBlockCount * expected_factor());

  KVCacheCreateOptions create_options;
  create_options.device(device.unwrap())
      .dtype(torch::kBFloat16)
      .num_layers(1)
      .model_type("test_model")
      .enable_lighting_indexer(true)
      .enable_indexer_cache_quant(true);
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, cache_shape, create_options);
  ASSERT_EQ(caches.size(), 1U);

  KVCache& cache = caches.front();
  torch::Tensor index_cache = cache.get_index_cache();
  std::optional<torch::Tensor> index_scale = cache.get_indexer_cache_scale();
  ASSERT_TRUE(index_scale.has_value());
  const int64_t pages_per_block = expected_factor();
  ASSERT_GT(pages_per_block, 0);
  ASSERT_EQ(index_cache.size(0), kBlockCount * pages_per_block);
  ASSERT_EQ(index_scale->size(0), kBlockCount * pages_per_block);

  const torch::indexing::Slice source_pages(
      kSourceBlockId * pages_per_block, (kSourceBlockId + 1) * pages_per_block);
  const torch::indexing::Slice destination_pages(
      kDestinationBlockId * pages_per_block,
      (kDestinationBlockId + 1) * pages_per_block);
  const torch::indexing::Slice untouched_pages(
      kUntouchedBlockId * pages_per_block,
      (kUntouchedBlockId + 1) * pages_per_block);
  index_cache.zero_();
  index_scale->zero_();
  index_cache.index({untouched_pages}).fill_(9);
  index_scale->index({untouched_pages}).fill_(9.0);
  ASSERT_EQ(device.synchronize_default_stream(), 0);

  HierarchyKVCacheTransfer::Options transfer_options;
  transfer_options.layers(1)
      .host_blocks_factor(1.5)
      .layers_wise_copy_batchs(1)
      .kv_split_size(GetParam())
      .kv_split_rank(0);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  HierarchyKVCacheTransfer transfer(transfer_options,
                                    device.unwrap(),
                                    compute_stream.get(),
                                    &caches,
                                    cache_shape,
                                    create_options);

  // Updates after registration must remain visible through the transfer view.
  for (int64_t page = kSourceBlockId * pages_per_block;
       page < (kSourceBlockId + 1) * pages_per_block;
       ++page) {
    index_cache[page].fill_(page + 2);
    (*index_scale)[page].fill_((page + 1) / 8.0);
  }
  ASSERT_EQ(device.synchronize_default_stream(), 0);
  const torch::Tensor expected_index =
      index_cache.index({source_pages}).clone();
  const torch::Tensor expected_scale =
      index_scale->index({source_pages}).clone();

  BlockTransferInfo offload_info(kSourceBlockId, /*dst_block_id=*/0);
  offload_info.block_type = BlockType::KV;
  offload_info.transfer_type = TransferType::D2H2G;
  ASSERT_EQ(transfer.transfer_kv_blocks(kBatchId, {offload_info}), 1U);

  index_cache.index({source_pages}).fill_(-1);
  index_scale->index({source_pages}).fill_(-1.0);
  ASSERT_EQ(device.synchronize_default_stream(), 0);

  BlockTransferInfo load_info(/*src_block_id=*/0, kDestinationBlockId);
  load_info.block_type = BlockType::KV;
  load_info.transfer_type = TransferType::H2D;
  ASSERT_EQ(transfer.transfer_kv_blocks(kBatchId, {load_info}), 1U);
  LlmModelParams input_storage;
  ModelParamsView input_params(input_storage);
  input_params.meta.batch_id = kBatchId;
  input_params.meta.requires_host_restore = true;
  transfer.set_layer_synchronizer(input_params);
  ASSERT_TRUE(input_params.synchronize_layer(0));
  ASSERT_EQ(device.synchronize_default_stream(), 0);

  EXPECT_TRUE(
      torch::equal(index_cache.index({destination_pages}), expected_index));
  EXPECT_TRUE(
      torch::equal(index_scale->index({destination_pages}), expected_scale));
  EXPECT_TRUE(
      torch::equal(index_cache.index({untouched_pages}),
                   torch::full_like(index_cache.index({untouched_pages}), 9)));
  EXPECT_TRUE(torch::equal(
      index_scale->index({untouched_pages}),
      torch::full_like(index_scale->index({untouched_pages}), 9.0)));
}

INSTANTIATE_TEST_SUITE_P(KvSplitSizes,
                         HierarchyIndexerTransferTest,
                         ::testing::Values(2, 4));
#endif

TEST(HierarchyKVCacheTransferTest, RejectsMixedOffloadBatchBeforeSubmission) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP()
        << "An accelerator device is required for hierarchy KV transfer.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  KVCacheCapacity capacity;
  capacity.n_blocks(2).block_size(4);
  ModelArgs model_args;
  model_args.model_type("test_model")
      .n_layers(1)
      .n_heads(2)
      .n_kv_heads(1)
      .head_dim(8);
  const KVCacheShape cache_shape(capacity, model_args, /*world_size=*/1);
  KVCacheCreateOptions create_options;
  create_options.device(device.unwrap())
      .dtype(torch::kFloat32)
      .num_layers(1)
      .model_type("test_model");
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, cache_shape, create_options);

  HierarchyKVCacheTransfer::Options transfer_options;
  transfer_options.layers(1).host_blocks_factor(2.0).layers_wise_copy_batchs(1);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  HierarchyKVCacheTransfer transfer(transfer_options,
                                    device.unwrap(),
                                    compute_stream.get(),
                                    &caches,
                                    cache_shape,
                                    create_options);

  BlockTransferInfo d2h_info(/*src_block_id=*/0, /*dst_block_id=*/0);
  d2h_info.block_type = BlockType::KV;
  d2h_info.transfer_type = TransferType::D2H2G;
  BlockTransferInfo h2d_info(/*src_block_id=*/0, /*dst_block_id=*/1);
  h2d_info.block_type = BlockType::KV;
  h2d_info.transfer_type = TransferType::H2D;

  EXPECT_DEATH(
      {
        (void)transfer.transfer_kv_blocks(/*batch_id=*/7, {d2h_info, h2d_info});
      },
      "mixed transfer types");
}

}  // namespace
}  // namespace xllm
