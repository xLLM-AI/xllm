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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/kv_cache/kv_cache_capacity.h"
#include "core/framework/kv_cache/kv_cache_shape.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/runtime/llm_worker_impl.h"
#include "core/runtime/mtp_worker_impl.h"
#include "core/runtime/options.h"
#include "core/util/slice.h"

namespace xllm {
namespace {

constexpr int64_t kBlockCount = 2;
constexpr int64_t kBlockSize = 4;

ModelArgs make_model_args(const std::string& model_type,
                          int64_t layer_count,
                          int64_t head_dim) {
  ModelArgs model_args;
  model_args.model_type(model_type)
      .n_layers(layer_count)
      .n_heads(2)
      .n_kv_heads(1)
      .head_dim(head_dim);
  return model_args;
}

KVCacheShape make_cache_shape(const ModelArgs& model_args) {
  KVCacheCapacity capacity;
  capacity.n_blocks(kBlockCount).block_size(kBlockSize);
  return KVCacheShape(capacity, model_args, /*world_size=*/1);
}

KVCacheShape make_dsv4_cache_shape(const ModelArgs& model_args,
                                   int64_t c4_count,
                                   int64_t c128_count) {
  KVCacheCapacity capacity;
  capacity.n_blocks(kBlockCount)
      .block_size(kBlockSize)
      .swa_count(kBlockCount)
      .c4_count(c4_count)
      .c128_count(c128_count);
  return KVCacheShape(capacity, model_args, /*world_size=*/1);
}

KVCacheCreateOptions make_create_options(const torch::Device& device,
                                         const ModelArgs& model_args) {
  KVCacheCreateOptions create_options;
  create_options.device(device)
      .dtype(torch::kFloat32)
      .num_layers(model_args.n_layers())
      .model_type(model_args.model_type())
      .block_size(kBlockSize)
      .head_dim(model_args.head_dim())
      .index_head_dim(std::max<int64_t>(model_args.index_head_dim(), 1))
      .window_size(std::max<int64_t>(model_args.window_size(), 1))
      .compress_ratios(model_args.compress_ratios());
  return create_options;
}

runtime::Options make_runtime_options(double host_blocks_factor) {
  runtime::Options options;
  options.model_id("test_target")
      .block_size(kBlockSize)
      .num_speculative_tokens(1)
      .host_blocks_factor(host_blocks_factor)
      .layers_wise_copy_batchs(1)
      .world_size(1)
      .dp_size(1)
      .cp_size(1);
  return options;
}

class RecordingTransferWorker final : public LLMWorkerImpl {
 public:
  RecordingTransferWorker(const ParallelArgs& parallel_args,
                          const torch::Device& device,
                          const runtime::Options& options,
                          uint32_t transfer_result)
      : LLMWorkerImpl(parallel_args, device, options),
        transfer_result_(transfer_result) {}

  uint32_t transfer_kv_blocks(
      uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info) override {
    last_batch_id_ = batch_id;
    last_transfer_size_ = block_transfer_info.size();
    ++vector_transfer_count_;
    return transfer_result_;
  }

  uint32_t transfer_kv_blocks(
      uint64_t batch_id,
      Slice<BlockTransferInfo>& block_transfer_info) override {
    last_batch_id_ = batch_id;
    last_transfer_size_ = block_transfer_info.size();
    ++slice_transfer_count_;
    return transfer_result_;
  }

  std::vector<uint8_t> prefetch_kv_blocks(
      Slice<BlockTransferInfo>& block_transfer_info) override {
    ++prefetch_count_;
    return std::vector<uint8_t>(block_transfer_info.size(), /*value=*/1);
  }

  uint32_t vector_transfer_count() const { return vector_transfer_count_; }
  uint32_t slice_transfer_count() const { return slice_transfer_count_; }
  uint32_t prefetch_count() const { return prefetch_count_; }
  uint64_t last_batch_id() const { return last_batch_id_; }
  size_t last_transfer_size() const { return last_transfer_size_; }

 private:
  uint32_t transfer_result_ = 0;
  uint32_t vector_transfer_count_ = 0;
  uint32_t slice_transfer_count_ = 0;
  uint32_t prefetch_count_ = 0;
  uint64_t last_batch_id_ = 0;
  size_t last_transfer_size_ = 0;
};

class HierarchyTransferTestWorker final : public LLMWorkerImpl {
 public:
  HierarchyTransferTestWorker(const ParallelArgs& parallel_args,
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

  void initialize_hierarchy_cache(const KVCacheShape& cache_shape) {
    const ModelArgs& model_args = context_.get_model_args();
    const KVCacheCreateOptions create_options =
        make_create_options(device_.unwrap(), model_args);
    allocate_kv_caches(kv_caches_, cache_shape, create_options);
    init_hierarchy_kv_cache_transfer(cache_shape, create_options);
  }

  void fill_block(int64_t block_id, double value) {
    for (size_t layer_index = 0; layer_index < kv_caches_.size();
         ++layer_index) {
      const double layer_value = value + static_cast<double>(layer_index);
      kv_caches_[layer_index].get_k_cache()[block_id].fill_(layer_value);
      kv_caches_[layer_index].get_v_cache()[block_id].fill_(layer_value + 0.5);
    }
  }

  void zero_block(int64_t block_id) {
    for (KVCache& cache : kv_caches_) {
      cache.get_k_cache()[block_id].zero_();
      cache.get_v_cache()[block_id].zero_();
    }
  }

  bool blocks_equal(int64_t source_block_id,
                    int64_t destination_block_id) const {
    for (const KVCache& cache : kv_caches_) {
      if (!torch::equal(cache.get_k_cache()[source_block_id],
                        cache.get_k_cache()[destination_block_id]) ||
          !torch::equal(cache.get_v_cache()[source_block_id],
                        cache.get_v_cache()[destination_block_id])) {
        return false;
      }
    }
    return true;
  }

  void fill_block(BlockType block_type, int64_t block_id, double value) {
    for (KVCache& cache : kv_caches_) {
      BlockTypeTensorMap tensors = cache.get_block_type_tensors(block_type);
      for (auto& [role, tensor] : tensors) {
        tensor[block_id].fill_(value + static_cast<double>(role));
      }
    }
  }

  void zero_block(BlockType block_type, int64_t block_id) {
    for (KVCache& cache : kv_caches_) {
      BlockTypeTensorMap tensors = cache.get_block_type_tensors(block_type);
      for (auto& [role, tensor] : tensors) {
        tensor[block_id].zero_();
      }
    }
  }

  bool blocks_equal(BlockType block_type,
                    int64_t source_block_id,
                    int64_t destination_block_id) const {
    for (const KVCache& cache : kv_caches_) {
      const BlockTypeTensorMap tensors =
          cache.get_block_type_tensors(block_type);
      for (const auto& [role, tensor] : tensors) {
        if (!torch::equal(tensor[source_block_id],
                          tensor[destination_block_id])) {
          return false;
        }
      }
    }
    return true;
  }
};

class TestMTPWorker final : public MTPWorkerImpl {
 public:
  TestMTPWorker(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options)
      : MTPWorkerImpl(parallel_args, device, options, WorkerType::LLM) {}

  void replace_transfer_workers(std::unique_ptr<LLMWorkerImpl> target,
                                std::unique_ptr<LLMWorkerImpl> draft) {
    impl_ = std::move(target);
    draft_impl_ = std::move(draft);
  }

  void prepare_hierarchy_transfers() { prepare_hierarchy_kv_cache_transfers(); }

  void finalize_hierarchy_transfers() {
    finalize_hierarchy_kv_cache_transfers();
  }

  std::shared_ptr<HierarchyKVCacheTransfer> target_transfer_owner() const {
    return get_hierarchy_kv_cache_transfer();
  }

  std::shared_ptr<HierarchyKVCacheTransfer> target_worker_transfer() const {
    return impl_->get_hierarchy_kv_cache_transfer();
  }

  std::shared_ptr<HierarchyKVCacheTransfer> draft_transfer_owner() const {
    return draft_transfer_owner_;
  }

  std::shared_ptr<HierarchyKVCacheTransfer> draft_worker_transfer() const {
    return draft_impl_->get_hierarchy_kv_cache_transfer();
  }
};

class MTPHostOffloadTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() < 1) {
      GTEST_SKIP() << "An accelerator is required for MTP host offload tests.";
    }
  }
};

TEST_F(MTPHostOffloadTest, TransfersEveryBlockToTargetAndDraft) {
  constexpr uint64_t kBatchId = 42;
  const torch::Device device(Platform::type_torch(), /*index=*/0);
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  const runtime::Options options = make_runtime_options(0.0);
  TestMTPWorker worker(parallel_args, device, options);

  const std::vector<BlockTransferInfo> transfer_info = {
      BlockTransferInfo(/*src_block_id=*/1, /*dst_block_id=*/2),
      BlockTransferInfo(/*src_block_id=*/3, /*dst_block_id=*/4)};
  auto target = std::make_unique<RecordingTransferWorker>(
      parallel_args,
      device,
      options,
      static_cast<uint32_t>(transfer_info.size()));
  auto draft = std::make_unique<RecordingTransferWorker>(
      parallel_args,
      device,
      options,
      static_cast<uint32_t>(transfer_info.size()));
  RecordingTransferWorker* target_ptr = target.get();
  RecordingTransferWorker* draft_ptr = draft.get();
  worker.replace_transfer_workers(std::move(target), std::move(draft));

  const uint32_t transferred =
      worker.transfer_kv_blocks(kBatchId, transfer_info);

  EXPECT_EQ(transferred, transfer_info.size());
  EXPECT_EQ(target_ptr->vector_transfer_count(), 1);
  EXPECT_EQ(draft_ptr->vector_transfer_count(), 1);
  EXPECT_EQ(target_ptr->last_batch_id(), kBatchId);
  EXPECT_EQ(draft_ptr->last_batch_id(), kBatchId);
  EXPECT_EQ(target_ptr->last_transfer_size(), transfer_info.size());
  EXPECT_EQ(draft_ptr->last_transfer_size(), transfer_info.size());
}

TEST_F(MTPHostOffloadTest, RejectsMismatchedTargetAndDraftTransferCounts) {
  constexpr uint64_t kBatchId = 73;
  const torch::Device device(Platform::type_torch(), /*index=*/0);
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  const runtime::Options options = make_runtime_options(0.0);
  TestMTPWorker worker(parallel_args, device, options);

  const std::vector<BlockTransferInfo> transfer_info = {
      BlockTransferInfo(/*src_block_id=*/5, /*dst_block_id=*/6)};
  auto target = std::make_unique<RecordingTransferWorker>(
      parallel_args, device, options, /*transfer_result=*/1);
  auto draft = std::make_unique<RecordingTransferWorker>(
      parallel_args, device, options, /*transfer_result=*/0);
  RecordingTransferWorker* target_ptr = target.get();
  RecordingTransferWorker* draft_ptr = draft.get();
  worker.replace_transfer_workers(std::move(target), std::move(draft));
  Slice<BlockTransferInfo> transfer_slice(transfer_info);

  const uint32_t transferred =
      worker.transfer_kv_blocks(kBatchId, transfer_slice);

  EXPECT_EQ(transferred, 0);
  EXPECT_EQ(target_ptr->slice_transfer_count(), 1);
  EXPECT_EQ(draft_ptr->slice_transfer_count(), 1);
}

TEST_F(MTPHostOffloadTest, StorePrefetchRemainsDisabledInDualTransferStage) {
  const torch::Device device(Platform::type_torch(), /*index=*/0);
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  const runtime::Options options = make_runtime_options(0.0);
  TestMTPWorker worker(parallel_args, device, options);

  const std::vector<BlockTransferInfo> transfer_info = {
      BlockTransferInfo(/*src_block_id=*/5, /*dst_block_id=*/6)};
  auto target = std::make_unique<RecordingTransferWorker>(
      parallel_args, device, options, /*transfer_result=*/1);
  auto draft = std::make_unique<RecordingTransferWorker>(
      parallel_args, device, options, /*transfer_result=*/1);
  RecordingTransferWorker* target_ptr = target.get();
  RecordingTransferWorker* draft_ptr = draft.get();
  worker.replace_transfer_workers(std::move(target), std::move(draft));
  Slice<BlockTransferInfo> transfer_slice(transfer_info);

  const std::vector<uint8_t> hits = worker.prefetch_kv_blocks(transfer_slice);

  EXPECT_EQ(hits, std::vector<uint8_t>({0}));
  EXPECT_EQ(target_ptr->prefetch_count(), 0U);
  EXPECT_EQ(draft_ptr->prefetch_count(), 0U);
}

TEST_F(MTPHostOffloadTest, BindsAndReleasesTargetAndDraftTransferOwners) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  const runtime::Options options = make_runtime_options(2.0);
  const ModelArgs target_model_args =
      make_model_args("test_target", /*layer_count=*/2, /*head_dim=*/8);
  const ModelArgs draft_model_args =
      make_model_args("test_draft", /*layer_count=*/1, /*head_dim=*/4);
  const KVCacheShape target_shape = make_cache_shape(target_model_args);
  const KVCacheShape draft_shape = make_cache_shape(draft_model_args);
  std::weak_ptr<HierarchyKVCacheTransfer> target_lifetime;
  std::weak_ptr<HierarchyKVCacheTransfer> draft_lifetime;

  {
    auto worker = std::make_unique<TestMTPWorker>(
        parallel_args, device.unwrap(), options);
    auto target = std::make_unique<HierarchyTransferTestWorker>(
        parallel_args, device.unwrap(), options, target_model_args);
    auto draft = std::make_unique<HierarchyTransferTestWorker>(
        parallel_args, device.unwrap(), options, draft_model_args);
    HierarchyTransferTestWorker* target_ptr = target.get();
    HierarchyTransferTestWorker* draft_ptr = draft.get();

    worker->replace_transfer_workers(std::move(target), std::move(draft));
    worker->prepare_hierarchy_transfers();
    worker->prepare_hierarchy_transfers();

    ASSERT_NE(target_ptr->get_hierarchy_kv_cache_transfer(), nullptr);
    ASSERT_NE(draft_ptr->get_hierarchy_kv_cache_transfer(), nullptr);
    EXPECT_FALSE(target_ptr->get_hierarchy_kv_cache_transfer()
                     ->registration_finalized());
    EXPECT_FALSE(
        draft_ptr->get_hierarchy_kv_cache_transfer()->registration_finalized());

    target_ptr->initialize_hierarchy_cache(target_shape);
    draft_ptr->initialize_hierarchy_cache(draft_shape);
    EXPECT_FALSE(target_ptr->get_hierarchy_kv_cache_transfer()
                     ->registration_finalized());
    EXPECT_FALSE(
        draft_ptr->get_hierarchy_kv_cache_transfer()->registration_finalized());
    worker->finalize_hierarchy_transfers();
    worker->finalize_hierarchy_transfers();

    std::shared_ptr<HierarchyKVCacheTransfer> target_owner =
        worker->target_transfer_owner();
    std::shared_ptr<HierarchyKVCacheTransfer> target_worker_transfer =
        worker->target_worker_transfer();
    std::shared_ptr<HierarchyKVCacheTransfer> draft_owner =
        worker->draft_transfer_owner();
    std::shared_ptr<HierarchyKVCacheTransfer> draft_worker_transfer =
        worker->draft_worker_transfer();
    ASSERT_NE(target_owner, nullptr);
    ASSERT_NE(draft_owner, nullptr);
    EXPECT_EQ(target_owner.get(), target_worker_transfer.get());
    EXPECT_EQ(draft_owner.get(), draft_worker_transfer.get());
    EXPECT_NE(target_owner.get(), draft_owner.get());
    EXPECT_TRUE(target_owner->registration_finalized());
    EXPECT_TRUE(draft_owner->registration_finalized());
    target_lifetime = target_owner;
    draft_lifetime = draft_owner;
  }

  EXPECT_TRUE(target_lifetime.expired());
  EXPECT_TRUE(draft_lifetime.expired());
}

TEST_F(MTPHostOffloadTest, DualTransferRoundTripUsesIndependentSynchronizers) {
  constexpr uint64_t kBatchId = 91;
  constexpr int64_t kSourceBlockId = 0;
  constexpr int64_t kDestinationBlockId = 1;
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  const runtime::Options options = make_runtime_options(2.0);
  const ModelArgs target_model_args =
      make_model_args("test_target", /*layer_count=*/2, /*head_dim=*/8);
  const ModelArgs draft_model_args =
      make_model_args("test_draft", /*layer_count=*/1, /*head_dim=*/4);
  const KVCacheShape target_shape = make_cache_shape(target_model_args);
  const KVCacheShape draft_shape = make_cache_shape(draft_model_args);
  TestMTPWorker worker(parallel_args, device.unwrap(), options);
  auto target = std::make_unique<HierarchyTransferTestWorker>(
      parallel_args, device.unwrap(), options, target_model_args);
  auto draft = std::make_unique<HierarchyTransferTestWorker>(
      parallel_args, device.unwrap(), options, draft_model_args);
  HierarchyTransferTestWorker* target_ptr = target.get();
  HierarchyTransferTestWorker* draft_ptr = draft.get();
  worker.replace_transfer_workers(std::move(target), std::move(draft));
  worker.prepare_hierarchy_transfers();
  target_ptr->initialize_hierarchy_cache(target_shape);
  draft_ptr->initialize_hierarchy_cache(draft_shape);
  worker.finalize_hierarchy_transfers();

  target_ptr->fill_block(kSourceBlockId, /*value=*/3.0);
  draft_ptr->fill_block(kSourceBlockId, /*value=*/13.0);
  target_ptr->zero_block(kDestinationBlockId);
  draft_ptr->zero_block(kDestinationBlockId);
  ASSERT_EQ(device.synchronize_default_stream(), 0);

  BlockTransferInfo offload_info(kSourceBlockId, /*dst_block_id=*/0);
  offload_info.block_type = BlockType::KV;
  offload_info.transfer_type = TransferType::D2H2G;
  EXPECT_EQ(worker.transfer_kv_blocks(kBatchId, {offload_info}), 1U);

  BlockTransferInfo load_info(/*src_block_id=*/0, kDestinationBlockId);
  load_info.block_type = BlockType::KV;
  load_info.transfer_type = TransferType::H2D;
  EXPECT_EQ(worker.transfer_kv_blocks(kBatchId, {load_info}), 1U);

  ModelInputParams target_input_params;
  target_input_params.meta.batch_id = kBatchId;
  worker.set_hierarchy_layer_synchronizer(target_input_params);
  ASSERT_NE(target_input_params.parallel.layer_wise_load_synchronizer, nullptr);

  ModelInputParams draft_input_params = target_input_params;
  draft_ptr->set_hierarchy_layer_synchronizer(draft_input_params);
  ASSERT_NE(draft_input_params.parallel.layer_wise_load_synchronizer, nullptr);
  EXPECT_NE(target_input_params.parallel.layer_wise_load_synchronizer.get(),
            draft_input_params.parallel.layer_wise_load_synchronizer.get());
  EXPECT_EQ(target_input_params.parallel.layers_per_event, 2U);
  EXPECT_EQ(draft_input_params.parallel.layers_per_event, 1U);

  for (uint32_t layer_index = 0; layer_index < 2; ++layer_index) {
    ASSERT_TRUE(target_input_params.synchronize_layer(layer_index));
  }
  ASSERT_TRUE(draft_input_params.synchronize_layer(/*layer_idx=*/0));
  EXPECT_TRUE(target_ptr->blocks_equal(kSourceBlockId, kDestinationBlockId));
  EXPECT_TRUE(draft_ptr->blocks_equal(kSourceBlockId, kDestinationBlockId));
}

TEST_F(MTPHostOffloadTest, Dsv4DraftSkipsUnsupportedCompressedBlockTypes) {
  constexpr uint64_t kBatchId = 92;
  constexpr int64_t kSourceBlockId = 0;
  constexpr int64_t kDestinationBlockId = 1;
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  const runtime::Options options = make_runtime_options(2.0);

  ModelArgs target_model_args =
      make_model_args("deepseek_v4", /*layer_count=*/3, /*head_dim=*/8);
  target_model_args.index_head_dim(4)
      .window_size(kBlockSize)
      .compress_ratios({1, 4, 128});
  ModelArgs draft_model_args =
      make_model_args("deepseek_v4_mtp", /*layer_count=*/1, /*head_dim=*/8);
  draft_model_args.index_head_dim(4)
      .window_size(kBlockSize)
      .compress_ratios({1});
  const KVCacheShape target_shape = make_dsv4_cache_shape(
      target_model_args, /*c4_count=*/kBlockCount, /*c128_count=*/kBlockCount);
  const KVCacheShape draft_shape =
      make_dsv4_cache_shape(draft_model_args, /*c4_count=*/0, /*c128_count=*/0);

  TestMTPWorker worker(parallel_args, device.unwrap(), options);
  auto target = std::make_unique<HierarchyTransferTestWorker>(
      parallel_args, device.unwrap(), options, target_model_args);
  auto draft = std::make_unique<HierarchyTransferTestWorker>(
      parallel_args, device.unwrap(), options, draft_model_args);
  HierarchyTransferTestWorker* target_ptr = target.get();
  HierarchyTransferTestWorker* draft_ptr = draft.get();
  worker.replace_transfer_workers(std::move(target), std::move(draft));
  worker.prepare_hierarchy_transfers();
  target_ptr->initialize_hierarchy_cache(target_shape);
  draft_ptr->initialize_hierarchy_cache(draft_shape);
  worker.finalize_hierarchy_transfers();

  ASSERT_TRUE(
      worker.draft_transfer_owner()->supports_block_type(BlockType::SWA));
  ASSERT_FALSE(
      worker.draft_transfer_owner()->supports_block_type(BlockType::C4));
  ASSERT_FALSE(
      worker.draft_transfer_owner()->supports_block_type(BlockType::C128));

  const std::vector<BlockType> block_types = {
      BlockType::SWA, BlockType::C4, BlockType::C128};
  std::vector<BlockTransferInfo> offload_info;
  std::vector<BlockTransferInfo> load_info;
  offload_info.reserve(block_types.size());
  load_info.reserve(block_types.size());
  for (BlockType block_type : block_types) {
    target_ptr->fill_block(block_type, kSourceBlockId, /*value=*/3.0);
    target_ptr->zero_block(block_type, kDestinationBlockId);
    if (block_type == BlockType::SWA) {
      draft_ptr->fill_block(block_type, kSourceBlockId, /*value=*/13.0);
      draft_ptr->zero_block(block_type, kDestinationBlockId);
    }

    offload_info.emplace_back(kSourceBlockId, /*dst_block_id=*/0);
    offload_info.back().block_type = block_type;
    offload_info.back().transfer_type = TransferType::D2H2G;
    load_info.emplace_back(/*src_block_id=*/0, kDestinationBlockId);
    load_info.back().block_type = block_type;
    load_info.back().transfer_type = TransferType::H2D;
  }
  ASSERT_EQ(device.synchronize_default_stream(), 0);

  EXPECT_EQ(worker.transfer_kv_blocks(kBatchId, offload_info),
            block_types.size());
  EXPECT_EQ(worker.transfer_kv_blocks(kBatchId, load_info), block_types.size());

  ModelInputParams target_input_params;
  target_input_params.meta.batch_id = kBatchId;
  worker.set_hierarchy_layer_synchronizer(target_input_params);
  for (uint32_t layer_index = 0; layer_index < 3; ++layer_index) {
    ASSERT_TRUE(target_input_params.synchronize_layer(layer_index));
  }
  ModelInputParams draft_input_params;
  draft_input_params.meta.batch_id = kBatchId;
  draft_ptr->set_hierarchy_layer_synchronizer(draft_input_params);
  ASSERT_TRUE(draft_input_params.synchronize_layer(/*layer_idx=*/0));

  for (BlockType block_type : block_types) {
    EXPECT_TRUE(target_ptr->blocks_equal(
        block_type, kSourceBlockId, kDestinationBlockId));
  }
  EXPECT_TRUE(draft_ptr->blocks_equal(
      BlockType::SWA, kSourceBlockId, kDestinationBlockId));
}

}  // namespace
}  // namespace xllm
