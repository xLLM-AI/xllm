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

#include "framework/kv_cache_transfer/hierarchy_kv_cache_transfer.h"

#include <cnrt.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "framework/kv_cache/kv_cache_capacity.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/model/model_args.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {

class HierarchyKVCacheTransferTestPeer final {
 public:
  static void replace_batch_memcpy(HierarchyKVCacheTransfer* transfer,
                                   std::unique_ptr<BatchMemcpy> batch_memcpy) {
    transfer->batch_memcpy_ = std::move(batch_memcpy);
  }

  static void set_layer_batch_ranges(
      HierarchyKVCacheTransfer* transfer,
      std::vector<HierarchyKVCacheTransfer::LayerBatchRange> ranges) {
    transfer->layer_batch_ranges_ = std::move(ranges);
  }

  static bool load_from_host(
      HierarchyKVCacheTransfer* transfer,
      std::shared_ptr<LayerSynchronizer> synchronizer,
      const std::vector<BlockTransferInfo>& block_transfer_info) {
    return transfer->load_from_host(std::move(synchronizer),
                                    block_transfer_info);
  }
};

namespace {

struct LoadFailureState {
  std::atomic<bool> gate_open{false};
  std::atomic<bool> copy_finished{false};
  std::atomic<bool> record_failed{false};
  std::atomic<bool> abort_called{false};
  std::atomic<bool> abort_saw_completed_copy{false};
};

void finish_pending_copy(void* user_data) {
  LoadFailureState* state = static_cast<LoadFailureState*>(user_data);
  while (!state->gate_open.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  state->copy_finished.store(true, std::memory_order_release);
}

bool wait_for_flag(const std::atomic<bool>& flag) {
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!flag.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  return flag.load(std::memory_order_acquire);
}

class PendingBatchMemcpy final : public BatchMemcpy {
 public:
  explicit PendingBatchMemcpy(LoadFailureState* state) : state_(state) {}

  void init(int32_t /*device_id*/) override {}

  bool submit_h2d(const std::vector<torch::Tensor>& /*src_tensors*/,
                  const std::vector<torch::Tensor>& /*dst_tensors*/,
                  Stream* stream) override {
    return cnrtInvokeHostFunc(stream->get_stream()->stream(),
                              finish_pending_copy,
                              state_) == cnrtSuccess;
  }

  bool copy_d2h(const std::vector<torch::Tensor>& /*src_tensors*/,
                const std::vector<torch::Tensor>& /*dst_tensors*/,
                Stream* /*stream*/) override {
    ADD_FAILURE() << "Unexpected D2H copy.";
    return false;
  }

 private:
  LoadFailureState* state_ = nullptr;
};

class SecondRecordFailsSynchronizer final : public LayerSynchronizer {
 public:
  explicit SecondRecordFailsSynchronizer(LoadFailureState* state)
      : state_(state) {}

  bool synchronize_layer(int64_t /*layer_index*/) override {
    ADD_FAILURE() << "Unexpected layer synchronization.";
    return false;
  }

  bool record_stream(int64_t /*layer_index*/, Stream* /*stream*/) override {
    ++record_count_;
    if (record_count_ == 1) {
      return true;
    }
    state_->record_failed.store(true, std::memory_order_release);
    return false;
  }

  void abort() override {
    state_->abort_saw_completed_copy.store(
        state_->copy_finished.load(std::memory_order_acquire),
        std::memory_order_release);
    state_->abort_called.store(true, std::memory_order_release);
  }

  uint32_t size() const override { return 2; }

 private:
  LoadFailureState* state_ = nullptr;
  int32_t record_count_ = 0;
};

TEST(HierarchyKVCacheTransferTest,
     RestoreRegistersReadyEventsForConfiguredLayerGroups) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for hierarchy KV cache transfer.";
  }

  constexpr int64_t kBlockCount = 2;
  constexpr int64_t kBlockSize = 4;
  constexpr int64_t kLayerCount = 3;
  constexpr int64_t kSourceBlockId = 0;
  constexpr int64_t kDestinationBlockId = 1;
  constexpr uint64_t kBatchId = 7;
  constexpr double kHostBlocksFactor = 2.0;

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

  for (size_t layer_idx = 0; layer_idx < caches.size(); ++layer_idx) {
    const double layer_value = static_cast<double>(layer_idx);
    caches[layer_idx].get_k_cache()[kSourceBlockId].fill_(3.0 + layer_value);
    caches[layer_idx].get_v_cache()[kSourceBlockId].fill_(7.0 + layer_value);
    caches[layer_idx].get_k_cache()[kDestinationBlockId].zero_();
    caches[layer_idx].get_v_cache()[kDestinationBlockId].zero_();
  }
  ASSERT_EQ(device.synchronize_default_stream(), 0);

  HierarchyKVCacheTransfer::Options transfer_options;
  transfer_options.tp_rank(0)
      .tp_size(1)
      .layers(kLayerCount)
      .host_blocks_factor(kHostBlocksFactor)
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
  EXPECT_EQ(input_params.parallel.layer_wise_load_synchronizer->size(), 3U);
  EXPECT_EQ(input_params.parallel.layers_per_bacth_copy, 1U);
  for (uint32_t layer_idx = 0; layer_idx < kLayerCount; ++layer_idx) {
    ASSERT_TRUE(input_params.synchronize_layer(layer_idx));
  }

  for (KVCache& cache : caches) {
    EXPECT_TRUE(torch::equal(cache.get_k_cache()[kSourceBlockId],
                             cache.get_k_cache()[kDestinationBlockId]));
    EXPECT_TRUE(torch::equal(cache.get_v_cache()[kSourceBlockId],
                             cache.get_v_cache()[kDestinationBlockId]));
  }
}

TEST(HierarchyKVCacheTransferTest,
     QuantizedIndexerRoundTripRestoresIndexAndScale) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for hierarchy KV cache transfer.";
  }

  constexpr int64_t kBlockCount = 2;
  constexpr int64_t kBlockSize = 4;
  constexpr int64_t kSourceBlockId = 0;
  constexpr int64_t kDestinationBlockId = 1;
  constexpr uint64_t kBatchId = 7;
  constexpr double kHostBlocksFactor = 2.0;

  HostCacheValidationOptions validation_options;
  validation_options.host_blocks_factor = kHostBlocksFactor;
  validation_options.device_block_count = kBlockCount;
  validation_options.supports_host_kv_offload = true;
  validation_options.indexer_cache_dtype = "int8";
  validation_options.model_type = "deepseek_v32";
  EXPECT_FALSE(validate_host_cache_options(validation_options).has_value());

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();

  KVCacheCapacity capacity;
  capacity.n_blocks(kBlockCount)
      .block_size(kBlockSize)
      .enable_indexer_cache_quant(true);

  ModelArgs model_args;
  model_args.model_type("deepseek_v32")
      .enable_mla(true)
      .n_heads(8)
      .n_kv_heads(2)
      .head_dim(8)
      .kv_lora_rank(8)
      .qk_rope_head_dim(4)
      .index_n_heads(1)
      .index_head_dim(4);
  const KVCacheShape cache_shape(capacity, model_args, /*world_size=*/1);

  KVCacheCreateOptions create_options;
  create_options.device(device.unwrap())
      .dtype(torch::kBFloat16)
      .num_layers(1)
      .model_type("deepseek_v32")
      .enable_lighting_indexer(true)
      .enable_indexer_cache_quant(true);
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, cache_shape, create_options);
  ASSERT_EQ(caches.size(), 1U);

  torch::Tensor source_index = caches[0].get_index_cache()[kSourceBlockId];
  const std::optional<torch::Tensor> index_scale =
      caches[0].get_indexer_cache_scale();
  ASSERT_TRUE(index_scale.has_value());
  torch::Tensor source_scale = index_scale.value()[kSourceBlockId];
  source_index.fill_(37);
  source_scale.fill_(0.625F);
  caches[0].get_index_cache()[kDestinationBlockId].zero_();
  index_scale.value()[kDestinationBlockId].zero_();

  HierarchyKVCacheTransfer::Options transfer_options;
  transfer_options.tp_rank(0)
      .tp_size(1)
      .layers(1)
      .host_blocks_factor(kHostBlocksFactor)
      .layers_wise_copy_batchs(1);
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

  ModelInputParams params;
  params.meta.batch_id = kBatchId;
  transfer.set_layer_synchronizer(params);
  ASSERT_NE(params.parallel.layer_wise_load_synchronizer, nullptr);
  ASSERT_TRUE(params.parallel.layer_wise_load_synchronizer->synchronize_layer(
      /*layer_index=*/0));

  EXPECT_TRUE(
      torch::equal(source_index.cpu(),
                   caches[0].get_index_cache()[kDestinationBlockId].cpu()));
  EXPECT_TRUE(torch::equal(source_scale.cpu(),
                           index_scale.value()[kDestinationBlockId].cpu()));
}

TEST(HierarchyKVCacheTransferTest, EventFailureDrainsEarlierCopyBeforeAbort) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for hierarchy KV cache transfer.";
  }

  constexpr int64_t kLayerCount = 1;
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();

  KVCacheCapacity capacity;
  capacity.n_blocks(2).block_size(4);
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

  HierarchyKVCacheTransfer::Options transfer_options;
  transfer_options.tp_rank(0)
      .tp_size(1)
      .layers(kLayerCount)
      .host_blocks_factor(2.0)
      .layers_wise_copy_batchs(1);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  HierarchyKVCacheTransfer transfer(transfer_options,
                                    device.unwrap(),
                                    compute_stream.get(),
                                    &caches,
                                    cache_shape,
                                    create_options);

  LoadFailureState state;
  HierarchyKVCacheTransferTestPeer::replace_batch_memcpy(
      &transfer, std::make_unique<PendingBatchMemcpy>(&state));
  HierarchyKVCacheTransferTestPeer::set_layer_batch_ranges(
      &transfer,
      {{/*begin_layer=*/0, /*end_layer=*/1},
       {/*begin_layer=*/1, /*end_layer=*/1}});
  std::shared_ptr<LayerSynchronizer> synchronizer =
      std::make_shared<SecondRecordFailsSynchronizer>(&state);
  BlockTransferInfo load_info(/*src_block_id=*/0, /*dst_block_id=*/1);
  load_info.block_type = BlockType::KV;
  load_info.transfer_type = TransferType::H2D;

  std::future<bool> load_result = std::async(
      std::launch::async, [&device, &transfer, &synchronizer, &load_info]() {
        device.set_device();
        device.init_device_context();
        return HierarchyKVCacheTransferTestPeer::load_from_host(
            &transfer, synchronizer, {load_info});
      });

  const bool record_failure_observed = wait_for_flag(state.record_failed);
  if (!record_failure_observed) {
    state.gate_open.store(true, std::memory_order_release);
  }
  ASSERT_TRUE(record_failure_observed);
  const std::future_status drain_status =
      load_result.wait_for(std::chrono::milliseconds(100));
  EXPECT_FALSE(state.abort_called.load(std::memory_order_acquire));

  state.gate_open.store(true, std::memory_order_release);

  ASSERT_EQ(load_result.wait_for(std::chrono::seconds(2)),
            std::future_status::ready);
  EXPECT_FALSE(load_result.get());
  EXPECT_EQ(drain_status, std::future_status::timeout);
  EXPECT_TRUE(state.copy_finished.load(std::memory_order_acquire));
  EXPECT_TRUE(state.abort_called.load(std::memory_order_acquire));
  EXPECT_TRUE(state.abort_saw_completed_copy.load(std::memory_order_acquire));
}

}  // namespace
}  // namespace xllm
