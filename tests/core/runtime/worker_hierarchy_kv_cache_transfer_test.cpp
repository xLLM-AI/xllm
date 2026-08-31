/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/framework/kv_cache/kv_cache_capacity.h"
#include "core/framework/kv_cache/kv_cache_shape.h"
#include "core/framework/model/model_args.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/runtime/llm_worker_impl.h"
#include "core/runtime/options.h"
#include "core/util/slice.h"

namespace xllm {
namespace {

constexpr int64_t kBlockCount = 2;
constexpr int64_t kBlockSize = 4;
constexpr int64_t kLayerCount = 1;

ModelArgs make_model_args() {
  ModelArgs model_args;
  model_args.model_type("test_model")
      .n_layers(kLayerCount)
      .n_heads(2)
      .n_kv_heads(1)
      .head_dim(8);
  return model_args;
}

KVCacheShape make_cache_shape(const ModelArgs& model_args) {
  KVCacheCapacity capacity;
  capacity.n_blocks(kBlockCount).block_size(kBlockSize);
  return KVCacheShape(capacity, model_args, /*world_size=*/1);
}

KVCacheCreateOptions make_create_options(const torch::Device& device) {
  KVCacheCreateOptions create_options;
  create_options.device(device)
      .dtype(torch::kFloat32)
      .num_layers(kLayerCount)
      .model_type("test_model");
  return create_options;
}

runtime::Options make_runtime_options(double host_blocks_factor) {
  runtime::Options options;
  options.model_id("test_model")
      .host_blocks_factor(host_blocks_factor)
      .world_size(1)
      .dp_size(1)
      .cp_size(1);
  return options;
}

class TestHierarchyWorker final : public LLMWorkerImpl {
 public:
  TestHierarchyWorker(const ParallelArgs& parallel_args,
                      const torch::Device& device,
                      const runtime::Options& options,
                      const ModelArgs& model_args)
      : LLMWorkerImpl(parallel_args, device, options) {
    dtype_ = torch::kFloat32;
    context_ =
        ModelContext(parallel_args,
                     model_args,
                     QuantArgs(),
                     torch::TensorOptions().dtype(dtype_).device(device));
  }

  void initialize_hierarchy_cache(const KVCacheShape& cache_shape,
                                  const KVCacheCreateOptions& create_options) {
    allocate_kv_caches(kv_caches_, cache_shape, create_options);
    init_hierarchy_kv_cache_transfer(cache_shape, create_options);
  }

  std::shared_ptr<HierarchyKVCacheTransfer> recreate_hierarchy_transfer() {
    std::shared_ptr<HierarchyKVCacheTransfer> transfer =
        create_hierarchy_kv_cache_transfer();
    register_hierarchy_kv_cache(*transfer,
                                HierarchyKVCacheTransfer::CacheRole::TARGET,
                                compute_stream_.get());
    EXPECT_TRUE(transfer->finalize_registration());
    return transfer;
  }
};

class WorkerHierarchyKVCacheTransferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() < 1) {
      GTEST_SKIP()
          << "An accelerator device is required for hierarchy KV transfer.";
    }

    device_ = std::make_unique<Device>(/*device_index=*/0);
    device_->set_device();
    device_->init_device_context();
  }

  std::unique_ptr<Device> device_;
};

TEST_F(WorkerHierarchyKVCacheTransferTest,
       DisabledHostCacheKeepsTransferUnbound) {
  const ModelArgs model_args = make_model_args();
  const KVCacheShape cache_shape = make_cache_shape(model_args);
  const KVCacheCreateOptions create_options =
      make_create_options(device_->unwrap());
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  TestHierarchyWorker worker(parallel_args,
                             device_->unwrap(),
                             make_runtime_options(/*host_blocks_factor=*/0.0),
                             model_args);
  worker.initialize_hierarchy_cache(cache_shape, create_options);

  EXPECT_EQ(worker.get_hierarchy_kv_cache_transfer(), nullptr);

  std::vector<BlockTransferInfo> transfer_info = {
      BlockTransferInfo(/*src_block_id=*/0, /*dst_block_id=*/1)};
  EXPECT_EQ(worker.transfer_kv_blocks(/*batch_id=*/1, transfer_info), 0U);
  Slice<BlockTransferInfo> transfer_slice(transfer_info);
  EXPECT_EQ(worker.prefetch_kv_blocks(transfer_slice),
            std::vector<uint8_t>({0}));
}

TEST_F(WorkerHierarchyKVCacheTransferTest,
       SharedTransferCanBeClearedAndRecreatedFromSavedContext) {
  const ModelArgs model_args = make_model_args();
  const KVCacheShape cache_shape = make_cache_shape(model_args);
  const KVCacheCreateOptions create_options =
      make_create_options(device_->unwrap());
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  TestHierarchyWorker worker(parallel_args,
                             device_->unwrap(),
                             make_runtime_options(/*host_blocks_factor=*/2.0),
                             model_args);
  worker.initialize_hierarchy_cache(cache_shape, create_options);

  std::shared_ptr<HierarchyKVCacheTransfer> transfer =
      worker.get_hierarchy_kv_cache_transfer();
  ASSERT_NE(transfer, nullptr);
  EXPECT_EQ(transfer.use_count(), 2);
  std::weak_ptr<HierarchyKVCacheTransfer> original_transfer = transfer;
  transfer.reset();
  EXPECT_EQ(original_transfer.use_count(), 1);

  worker.clear_hierarchy_kv_cache_transfer();
  EXPECT_TRUE(original_transfer.expired());
  worker.clear_hierarchy_kv_cache_transfer();

  std::shared_ptr<HierarchyKVCacheTransfer> recreated_transfer =
      worker.recreate_hierarchy_transfer();
  ASSERT_NE(recreated_transfer, nullptr);
  EXPECT_EQ(worker.get_hierarchy_kv_cache_transfer(), nullptr);
  HierarchyKVCacheTransfer* recreated_transfer_ptr = recreated_transfer.get();
  worker.set_hierarchy_kv_cache_transfer(recreated_transfer);
  EXPECT_EQ(worker.get_hierarchy_kv_cache_transfer().get(),
            recreated_transfer_ptr);
  recreated_transfer.reset();
  EXPECT_EQ(worker.get_hierarchy_kv_cache_transfer().use_count(), 2);
  worker.clear_hierarchy_kv_cache_transfer();
}

TEST_F(WorkerHierarchyKVCacheTransferTest,
       WorkerDestructionReleasesOwnedTransfer) {
  const ModelArgs model_args = make_model_args();
  const KVCacheShape cache_shape = make_cache_shape(model_args);
  const KVCacheCreateOptions create_options =
      make_create_options(device_->unwrap());
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  std::weak_ptr<HierarchyKVCacheTransfer> owned_transfer;

  {
    auto worker = std::make_unique<TestHierarchyWorker>(
        parallel_args,
        device_->unwrap(),
        make_runtime_options(/*host_blocks_factor=*/2.0),
        model_args);
    worker->initialize_hierarchy_cache(cache_shape, create_options);
    std::shared_ptr<HierarchyKVCacheTransfer> transfer =
        worker->get_hierarchy_kv_cache_transfer();
    ASSERT_NE(transfer, nullptr);
    owned_transfer = transfer;
    transfer.reset();
    EXPECT_EQ(owned_transfer.use_count(), 1);
  }

  EXPECT_TRUE(owned_transfer.expired());
}

}  // namespace
}  // namespace xllm
