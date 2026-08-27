/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/kv_cache_transfer/hierarchy_kv_cache_transfer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "framework/kv_cache/kv_cache_capacity.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/model/model_args.h"
#include "platform/device.h"
#include "platform/platform.h"

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
  transfer_options.tp_rank(0)
      .tp_size(1)
      .layers(kLayerCount)
      .host_blocks_factor(2.0)
      .layers_wise_copy_batchs(2);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  HierarchyKVCacheTransfer transfer(transfer_options,
                                    device.unwrap(),
                                    compute_stream.get(),
                                    &caches,
                                    cache_shape,
                                    create_options);

  BlockTransferInfo offload_info(kSourceBlockId, /*dst_block_id=*/0);
  offload_info.block_type = BlockType::KV;
  offload_info.transfer_type = TransferType::D2H2G;
  EXPECT_EQ(transfer.transfer_kv_blocks(kBatchId, {offload_info}), 1U);

  BlockTransferInfo load_info(/*src_block_id=*/0, kDestinationBlockId);
  load_info.block_type = BlockType::KV;
  load_info.transfer_type = TransferType::H2D;
  EXPECT_EQ(transfer.transfer_kv_blocks(kBatchId, {load_info}), 1U);

  ModelInputParams input_params;
  input_params.meta.batch_id = kBatchId;
  transfer.set_layer_synchronizer(input_params);
  ASSERT_NE(input_params.parallel.layer_wise_load_synchronizer, nullptr);
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
  transfer_options.tp_rank(0)
      .tp_size(1)
      .layers(1)
      .host_blocks_factor(2.0)
      .layers_wise_copy_batchs(1);
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
