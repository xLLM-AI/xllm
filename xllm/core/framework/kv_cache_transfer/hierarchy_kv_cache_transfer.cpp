/* Copyright 2025-2026 The xLLM Authors.

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

#include <algorithm>
#include <memory>
#include <vector>

namespace xllm {
namespace {

constexpr uint32_t TIMEOUT_MS = 60000;

// Streams reserved for concurrent D2H offload callers. D2H runs synchronously
// on the RemoteWorker copy threadpool (4 threads, see remote_worker.h); reserve
// one stream per such thread so concurrent offloads never block on the stream
// queue.
constexpr size_t kOffloadStreamCount = 4;

std::vector<HierarchyKVCacheTransfer::LayerBatchRange> build_layer_batch_ranges(
    int64_t num_layers,
    uint32_t requested_batches) {
  std::vector<HierarchyKVCacheTransfer::LayerBatchRange> ranges;
  if (num_layers <= 0) {
    return ranges;
  }

  uint32_t layers_per_batch =
      requested_batches == 0
          ? static_cast<uint32_t>(num_layers)
          : static_cast<uint32_t>(num_layers) / requested_batches;
  layers_per_batch = std::max<uint32_t>(layers_per_batch, 1);

  for (int64_t begin = 0; begin < num_layers; begin += layers_per_batch) {
    ranges.push_back(
        {begin, std::min<int64_t>(begin + layers_per_batch, num_layers)});
  }
  return ranges;
}

bool has_tensor(const torch::Tensor& tensor) {
  return tensor.defined() && tensor.numel() > 0;
}

BlockTypeTensorMap build_block_type_tensor_map(const KVCache& kv_cache,
                                               BlockType type) {
  BlockTypeTensorMap map;

  const torch::Tensor key_cache = kv_cache.get_k_cache();
  const torch::Tensor value_cache = kv_cache.get_v_cache();
  const torch::Tensor index_cache = kv_cache.get_index_cache();
  const torch::Tensor conv_cache = kv_cache.get_conv_cache();
  const torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  const torch::Tensor swa_cache = kv_cache.get_swa_cache();
  const std::optional<torch::Tensor> index_cache_scale =
      kv_cache.get_indexer_cache_scale();

  switch (type) {
    case BlockType::KV:
      if (has_tensor(conv_cache) || has_tensor(ssm_cache) ||
          has_tensor(swa_cache)) {
        return {};
      }
      if (has_tensor(key_cache)) {
        map.emplace(KVCacheTensorRole::KEY, key_cache);
      }
      if (has_tensor(value_cache)) {
        map.emplace(KVCacheTensorRole::VALUE, value_cache);
      }
      if (has_tensor(index_cache)) {
        map.emplace(KVCacheTensorRole::INDEX, index_cache);
      }
      // INT8 indexer cache carries a per-token fp32 scale that must travel with
      // the int8 index values during offload/reload.
      if (index_cache_scale.has_value() &&
          has_tensor(index_cache_scale.value())) {
        map.emplace(KVCacheTensorRole::INDEX_SCALE, index_cache_scale.value());
      }
      return map;
    case BlockType::SINGLE:
      if (has_tensor(conv_cache)) {
        map.emplace(KVCacheTensorRole::CONV, conv_cache);
      }
      if (has_tensor(ssm_cache)) {
        map.emplace(KVCacheTensorRole::SSM, ssm_cache);
      }
      return map;
    case BlockType::SWA:
      if (!has_tensor(swa_cache) || has_tensor(key_cache) ||
          has_tensor(value_cache) || has_tensor(index_cache)) {
        return {};
      }
      map.emplace(KVCacheTensorRole::SWA, swa_cache);
      return map;
    case BlockType::C4:
      // DSV4 compress-ratio-4 layer: has swa + key + index (no value).
      if (!has_tensor(swa_cache) || has_tensor(value_cache) ||
          !has_tensor(key_cache) || !has_tensor(index_cache)) {
        return {};
      }
      map.emplace(KVCacheTensorRole::KEY, key_cache);
      map.emplace(KVCacheTensorRole::INDEX, index_cache);
      return map;
    case BlockType::C128:
      // DSV4 compress-ratio-128 layer: has swa + key, but no index/value.
      if (!has_tensor(swa_cache) || has_tensor(value_cache) ||
          !has_tensor(key_cache) || has_tensor(index_cache)) {
        return {};
      }
      map.emplace(KVCacheTensorRole::KEY, key_cache);
      return map;
    default:
      return {};
  }
}

}  // namespace

HierarchyKVCacheTransfer::HierarchyKVCacheTransfer(
    const Options& options,
    const torch::Device& device,
    std::vector<xllm::KVCache>* kv_caches_ptr,
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options)
    : options_(options),
      device_(device),
      kv_caches_ptr_(kv_caches_ptr),
      kv_cache_shape_(kv_cache_shape),
      create_options_(create_options) {
  CHECK(kv_caches_ptr_ != nullptr) << "kv_caches_ptr must not be null.";

  device_.set_device();
  device_.init_device_context();
  load_threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/2,
      /*init_func=*/[this]() mutable { device_.set_device(); },
      /*cpu_binding=*/false,
      /*pool_name=*/"HierarchyKVCacheTransfer.load");
  // D2H offload runs synchronously on the caller (RemoteWorker copy thread) so
  // its copied-block count can be returned to the scheduler; it is not posted
  // to a local pool. Size the shared stream pool to cover the H2D load threads
  // plus the concurrent D2H callers.
  const size_t num_streams = load_threadpool_->size() + kOffloadStreamCount;
  for (size_t i = 0; i < num_streams; ++i) {
    copy_stream_.enqueue(device_.get_stream_from_pool(TIMEOUT_MS));
  }

  build_device_block_type_map();
  layer_batch_ranges_ = build_layer_batch_ranges(
      options_.layers(), options_.layers_wise_copy_batchs());

  if (options_.host_blocks_factor() > 1.0) {
    batch_memcpy_ = create_batch_memcpy(device_);
    create_host_cache();
  }
}

void HierarchyKVCacheTransfer::build_device_block_type_map() {
  device_kv_caches_.clear();
  device_block_type_layer_ids_.clear();

  const std::vector<BlockType> kBlockTypes = {BlockType::KV,
                                              BlockType::SINGLE,
                                              BlockType::SWA,
                                              BlockType::C4,
                                              BlockType::C128};

  for (int64_t layer_id = 0;
       layer_id < static_cast<int64_t>(kv_caches_ptr_->size());
       ++layer_id) {
    KVCache& kv_cache = kv_caches_ptr_->at(static_cast<size_t>(layer_id));
    for (BlockType type : kBlockTypes) {
      BlockTypeTensorMap tensor_map =
          build_block_type_tensor_map(kv_cache, type);
      if (!tensor_map.empty()) {
        device_kv_caches_[type].push_back(&kv_cache);
        device_block_type_layer_ids_[type].push_back(layer_id);
      }
    }
  }
}

void HierarchyKVCacheTransfer::create_host_cache() {
  CHECK(!device_kv_caches_.empty())
      << "device block type caches must not be empty.";

  for (const auto& [block_type, group_caches] : device_kv_caches_) {
    if (group_caches.empty()) {
      continue;
    }

    const int64_t layer_count = static_cast<int64_t>(group_caches.size());

    KVCacheCreateOptions host_opts = create_options_;
    host_opts.device(torch::Device(torch::kCPU))
        .enable_xtensor(false)
        .enable_raw_device_allocator(false)
        .host_blocks_factor(options_.host_blocks_factor());
#if defined(USE_NPU)
    host_opts.enable_kv_cache_huge_page_allocator(false);
#endif

    host_kv_caches_[block_type] = std::make_unique<KVCache>(
        kv_cache_shape_, host_opts, block_type, layer_count);
  }

  LOG(INFO) << "HierarchyKVCacheTransfer: created host cache for "
            << host_kv_caches_.size() << " block type groups.";
}

HierarchyKVCacheTransfer::CopyPlan HierarchyKVCacheTransfer::build_copy_plan(
    const std::vector<BlockTransferInfo>& block_transfer_info,
    const LayerBatchRange& layer_batch_range) const {
  CopyPlan plan;
  if (block_transfer_info.empty()) {
    return plan;
  }

  const TransferType transfer_type = block_transfer_info.front().transfer_type;

  for (const auto& info : block_transfer_info) {
    BlockType type = info.block_type;
    auto device_it = device_kv_caches_.find(type);
    auto layer_ids_it = device_block_type_layer_ids_.find(type);
    auto host_it = host_kv_caches_.find(type);
    if (device_it == device_kv_caches_.end() ||
        layer_ids_it == device_block_type_layer_ids_.end() ||
        host_it == host_kv_caches_.end()) {
      continue;
    }

    const auto& group_caches = device_it->second;
    const auto& layer_ids = layer_ids_it->second;
    const KVCache* host_cache = host_it->second.get();
    CHECK(host_cache != nullptr) << "host cache instance must not be null.";
    const BlockTypeTensorMap host_tensors =
        host_cache->get_block_type_tensors(type);

    int32_t host_block_id = -1;
    int32_t device_block_id = -1;
    switch (transfer_type) {
      case TransferType::H2D:
        host_block_id = info.src_block_id;
        device_block_id = info.dst_block_id;
        break;
      case TransferType::D2H2G:
        host_block_id = info.dst_block_id;
        device_block_id = info.src_block_id;
        break;
      default:
        LOG(FATAL) << "Unsupported transfer type for copy plan: "
                   << static_cast<uint32_t>(transfer_type);
    }

    CHECK_GE(host_block_id, 0) << "host block id must be non-negative.";

    for (size_t layer_slot = 0; layer_slot < group_caches.size();
         ++layer_slot) {
      const int64_t absolute_layer_id = layer_ids[layer_slot];
      if (absolute_layer_id < layer_batch_range.begin_layer ||
          absolute_layer_id >= layer_batch_range.end_layer) {
        continue;
      }

      BlockTypeTensorMap device_tensors =
          build_block_type_tensor_map(*group_caches[layer_slot], type);
      for (const auto& [role, device_tensor] : device_tensors) {
        auto host_tensor_it = host_tensors.find(role);
        if (host_tensor_it == host_tensors.end()) {
          continue;
        }

        // device_tensor shape: [num_blocks, ...per_block_dims]
        // host_tensor shape: [num_host_blocks, num_layers, ...per_block_dims]
        const torch::Tensor& host_tensor = host_tensor_it->second;
        CHECK_LT(host_block_id, host_tensor.size(0))
            << "host block id out of range.";
        torch::Tensor device_block = device_tensor[device_block_id];
        torch::Tensor host_block_layer =
            host_tensor[host_block_id][static_cast<int64_t>(layer_slot)];

        if (transfer_type == TransferType::H2D) {
          plan.src_tensors.emplace_back(host_block_layer);
          plan.dst_tensors.emplace_back(device_block);
        } else {
          plan.src_tensors.emplace_back(device_block);
          plan.dst_tensors.emplace_back(host_block_layer);
        }
      }
    }
  }

  return plan;
}

uint32_t HierarchyKVCacheTransfer::transfer_kv_blocks(
    uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  CHECK(!block_transfer_info.empty());

  // This runs synchronously on the caller's thread (a brpc RPC worker thread
  // for remote workers), which has no ACL context of its own. Both branches
  // below touch device resources on this thread — D2H offload issues the copy
  // inline, and H2D creates the layer synchronizer's events
  // (aclrtCreateEventWithFlag). Establish the context here so those calls do
  // not fail with ACL_ERROR_RT_CONTEXT_NULL (107002). Idempotent; the async H2D
  // copy posted to load_threadpool_ already has context via that pool's
  // init_func.
  device_.set_device();

  switch (block_transfer_info[0].transfer_type) {
    case TransferType::D2H2G:
      return offload(block_transfer_info);
    case TransferType::H2D: {
      // Create and register the synchronizer synchronously, before scheduling
      // the async copy. The scheduler issues transfer_kv_blocks before step, so
      // by the time this returns the entry is in the map and the forward path's
      // set_layer_synchronizer is guaranteed to find it. Registering inside the
      // async load_from_host would race the forward lookup (it could miss the
      // entry, skip the wait, and read KV before the H2D copy completed).
      auto synchronizer = create_layer_synchronizer(
          static_cast<int64_t>(layer_batch_ranges_.size()));
      CHECK(synchronizer != nullptr)
          << "Failed to create layer synchronizer for H2D batch_id=" << batch_id
          << "; H2D copy cannot be backed and the pool has already advanced "
             "kv_cache_tokens_num_.";
      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto existing = layer_wise_load_synchronizer_.find(batch_id);
        if (existing != layer_wise_load_synchronizer_.end()) {
          LOG(ERROR)
              << "layer_wise_load_synchronizer collision at batch_id="
              << batch_id
              << ", previous entry was never consumed (batch cancelled or "
                 "batch_id reused). Overwriting; stale entry's events will "
                 "release when its refcount drops.";
        }
        layer_wise_load_synchronizer_[batch_id] = synchronizer;
      }
      load_threadpool_->schedule(
          [this,
           synchronizer,
           block_transfer_info = std::move(block_transfer_info)]() mutable {
            load_from_host(synchronizer, block_transfer_info);
          });
      return 0;
    }
    default:
      LOG(ERROR) << "Unsupported transfer type: "
                 << static_cast<uint32_t>(block_transfer_info[0].transfer_type);
      return 0;
  }
}

uint32_t HierarchyKVCacheTransfer::transfer_kv_blocks(
    uint64_t /*batch_id*/,
    Slice<BlockTransferInfo>& block_transfer_info) {
  CHECK(!block_transfer_info.empty());
  LOG(ERROR) << "Slice-based transfer not supported in Phase 1 (no store).";
  return 0;
}

uint32_t HierarchyKVCacheTransfer::offload(
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  if (block_transfer_info.empty()) {
    return 0;
  }

  if (batch_memcpy_ == nullptr) {
    return block_transfer_info.size();
  }

  Slice<BlockTransferInfo> slice(block_transfer_info);
  if (!offload_to_host(slice)) {
    LOG(ERROR) << "Offload to host failed.";
    return 0;
  }
  return block_transfer_info.size();
}

bool HierarchyKVCacheTransfer::offload_to_host(
    Slice<BlockTransferInfo>& block_transfer_info) {
  if (block_transfer_info.empty()) {
    return true;
  }

  CHECK(batch_memcpy_ != nullptr) << "batch memcpy must be initialized.";
  std::unique_ptr<Stream> stream;
  copy_stream_.wait_dequeue(stream);
  bool success = true;
  for (const auto& range : layer_batch_ranges_) {
    CopyPlan plan = build_copy_plan(
        static_cast<std::vector<BlockTransferInfo>>(block_transfer_info),
        range);
    if (plan.src_tensors.empty()) {
      continue;
    }
    if (!batch_memcpy_->copy_d2h(
            plan.src_tensors, plan.dst_tensors, stream.get())) {
      success = false;
      break;
    }
  }
  copy_stream_.enqueue(std::move(stream));
  return success;
}

bool HierarchyKVCacheTransfer::load_from_host(
    std::shared_ptr<LayerSynchronizer> synchronizer,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  if (block_transfer_info.empty()) {
    return true;
  }

  CHECK(synchronizer != nullptr) << "layer synchronizer must not be null.";
  CHECK(batch_memcpy_ != nullptr) << "batch memcpy must be initialized.";

  std::unique_ptr<Stream> stream;
  copy_stream_.wait_dequeue(stream);
  bool success = true;
  for (size_t range_idx = 0; range_idx < layer_batch_ranges_.size();
       ++range_idx) {
    CopyPlan plan =
        build_copy_plan(block_transfer_info, layer_batch_ranges_[range_idx]);
    if (plan.src_tensors.empty()) {
      if (!synchronizer->record_stream(static_cast<int64_t>(range_idx),
                                       stream.get())) {
        success = false;
        break;
      }
      continue;
    }
    if (!batch_memcpy_->copy_h2d(
            plan.src_tensors, plan.dst_tensors, stream.get())) {
      success = false;
      break;
    }
    if (!synchronizer->record_stream(static_cast<int64_t>(range_idx),
                                     stream.get())) {
      success = false;
      break;
    }
  }

  copy_stream_.enqueue(std::move(stream));
  // On failure some ranges were never recorded; abort the synchronizer so a
  // forward thread spinning on those layers unblocks and reports failure
  // (aborting the forward) instead of hanging or reading uncopied KV cache.
  if (!success) {
    synchronizer->abort();
  }
  return success;
}

void HierarchyKVCacheTransfer::set_layer_synchronizer(
    ModelInputParams& params) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = layer_wise_load_synchronizer_.find(params.meta.batch_id);
  if (it != layer_wise_load_synchronizer_.end()) {
    params.parallel.layer_wise_load_synchronizer = it->second;
    params.parallel.layers_per_bacth_copy =
        layer_batch_ranges_.empty()
            ? options_.layers()
            : static_cast<uint32_t>(layer_batch_ranges_[0].end_layer -
                                    layer_batch_ranges_[0].begin_layer);
    layer_wise_load_synchronizer_.erase(it);
  }
}

}  // namespace xllm
