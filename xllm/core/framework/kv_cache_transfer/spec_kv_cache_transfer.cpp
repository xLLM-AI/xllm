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

#include "framework/kv_cache_transfer/spec_kv_cache_transfer.h"

#include <glog/logging.h>

#include "common/macros.h"

namespace xllm {
SpecKVCacheTransfer::SpecKVCacheTransfer(const uint16_t listen_port,
                                         const InstanceRole& instance_role,
                                         bool enable_lighting_indexer)
    : LlmDataDistTransfer(listen_port, instance_role, enable_lighting_indexer) {
}

void SpecKVCacheTransfer::register_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  UNUSED_PARAMETER(kv_cache_shape);
  UNUSED_PARAMETER(dtype);
  register_kv_cache_internal(kv_caches, layer_registered_caches_);
}

void SpecKVCacheTransfer::register_kv_cache_spec(
    std::vector<xllm::KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  UNUSED_PARAMETER(kv_cache_shape);
  UNUSED_PARAMETER(dtype);
  register_kv_cache_internal(kv_caches, spec_layer_registered_caches_);
}

void SpecKVCacheTransfer::register_kv_cache_internal(
    std::vector<xllm::KVCache>& kv_caches,
    LayerRegisteredCaches& layer_registered_caches) {
  register_layer_registered_caches(kv_caches, layer_registered_caches);
}

void SpecKVCacheTransfer::free_kv_cache() {
  layer_registered_caches_.clear();
  spec_layer_registered_caches_.clear();
  has_grouped_cache_layout_ = false;
}

bool SpecKVCacheTransfer::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<uint64_t>& src_linear_state_ids,
    const std::vector<uint64_t>& dst_linear_state_ids) {
  if (has_grouped_cache_layout_) {
    return LlmDataDistTransfer::pull_kv_blocks(src_cluster_id,
                                               src_addr,
                                               src_blocks,
                                               dst_blocks,
                                               src_linear_state_ids,
                                               dst_linear_state_ids);
  }
  const bool base_success =
      LlmDataDistTransfer::pull_kv_blocks(src_cluster_id,
                                          src_addr,
                                          src_blocks,
                                          dst_blocks,
                                          src_linear_state_ids,
                                          dst_linear_state_ids);
  bool spec_success = true;
  for (int64_t layer_id = 0;
       layer_id < static_cast<int64_t>(spec_layer_registered_caches_.size());
       ++layer_id) {
    const auto& registered_caches = spec_layer_registered_caches_[layer_id];
    for (const RegisteredCache& registered_cache : registered_caches) {
      const bool sequence_scoped = registered_cache.sequence_scoped;
      const std::vector<uint64_t>& src_ids =
          sequence_scoped ? src_linear_state_ids : src_blocks;
      const std::vector<uint64_t>& dst_ids =
          sequence_scoped ? dst_linear_state_ids : dst_blocks;
      if (src_ids.empty() || dst_ids.empty()) {
        continue;
      }
      CacheIndex cache_index{src_cluster_id, registered_cache.cache.cache_id};
      KvCacheExtParam ext_param{};
      ext_param.src_layer_range = {0, 0};
      ext_param.dst_layer_range = {0, 0};
      ext_param.tensor_num_per_layer = 1;
      auto ret = llm_data_dist_->PullKvBlocks(
          cache_index, registered_cache.cache, src_ids, dst_ids, ext_param);
      if (ret != LLM_SUCCESS) {
        LOG(ERROR) << "Pull spec KvBlocks failed, layer = " << layer_id
                   << ", ret = " << std::hex << ret;
        spec_success = false;
      }
    }
  }
  return base_success && spec_success;
}

bool SpecKVCacheTransfer::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft,
    int32_t kv_split_rank,
    int32_t kv_split_size) {
  if (is_spec_draft) {
    return push_kv_blocks_spec(
        merged_kv_infos, layer_synchronizer, kv_split_rank, kv_split_size);
  } else {
    return push_kv_blocks_internal(merged_kv_infos,
                                   layer_synchronizer,
                                   layer_registered_caches_,
                                   kv_split_rank,
                                   kv_split_size);
  }
}

bool SpecKVCacheTransfer::push_kv_blocks_spec(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    int32_t kv_split_rank,
    int32_t kv_split_size) {
  return push_kv_blocks_internal(merged_kv_infos,
                                 layer_synchronizer,
                                 spec_layer_registered_caches_,
                                 kv_split_rank,
                                 kv_split_size);
}

bool SpecKVCacheTransfer::push_kv_blocks_internal(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    const LayerRegisteredCaches& layer_registered_caches,
    int32_t kv_split_rank,
    int32_t kv_split_size) {
  return push_layer_registered_caches(layer_registered_caches,
                                      merged_kv_infos,
                                      layer_synchronizer,
                                      kv_split_rank,
                                      kv_split_size);
}

folly::SemiFuture<bool> SpecKVCacheTransfer::push_kv_blocks_async(
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    const ParallelArgs& parallel_args,
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer,
    bool is_spec_draft) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        transfer_kv_infos,
                        &parallel_args,
                        layer_synchronizer,
                        is_spec_draft,
                        promise = std::move(promise)]() mutable {
    std::unordered_map<std::string, KVCacheInfo> merged_kv_infos;
    std::vector<TransferKVInfo> filtered_kv_infos;
    const std::vector<TransferKVInfo>* kv_infos = &transfer_kv_infos;
    // When the KV cache is actually sharded across ranks
    // (kv_split_size_effective > 1), filter remote_blocks_ids down to this
    // rank's slice. When kv_split_size==1 each rank holds the full replica and
    // we keep the legacy 1:1 remote_blocks_ids mapping.
    const int32_t kv_split_size = parallel_args.kv_split_size_effective();
    if (kv_split_size > 1) {
      filtered_kv_infos = filter_kv_split_infos(
          parallel_args.kv_split_rank(), kv_split_size, *kv_infos);
      kv_infos = &filtered_kv_infos;
      if (kv_infos->empty()) {
        promise.setValue(true);
        return;
      }
    }
    merge_kv_blocks(merged_kv_infos, *kv_infos, parallel_args);
    bool success = true;
    if (!merged_kv_infos.empty()) {
      success = this->push_kv_blocks(merged_kv_infos,
                                     layer_synchronizer,
                                     is_spec_draft,
                                     parallel_args.kv_split_rank(),
                                     parallel_args.kv_split_size_effective());
    }
    promise.setValue(success);
  });
  return future;
}
}  // namespace xllm
