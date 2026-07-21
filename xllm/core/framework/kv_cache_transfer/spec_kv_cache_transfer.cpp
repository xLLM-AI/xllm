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

#include <optional>

#include "common/macros.h"
#include "util/timer.h"

namespace xllm {

namespace {

std::optional<int32_t> get_remote_tp_size(
    const std::vector<TransferKVInfo>& transfer_kv_infos) {
  for (const TransferKVInfo& info : transfer_kv_infos) {
    const int32_t remote_dp_size = info.remote_instance_info.dp_size;
    const size_t remote_world_size =
        info.remote_instance_info.cluster_ids.size();
    if (remote_dp_size <= 0 || remote_world_size == 0 ||
        remote_world_size % static_cast<size_t>(remote_dp_size) != 0) {
      continue;
    }
    return static_cast<int32_t>(remote_world_size /
                                static_cast<size_t>(remote_dp_size));
  }
  return std::nullopt;
}

void merge_heterogeneous_kv_blocks(
    std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo>& merged,
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    int32_t source_shard_rank) {
  for (const TransferKVInfo& info : transfer_kv_infos) {
    const int32_t dst_dp_size = info.remote_instance_info.dp_size;
    const size_t dst_world_size = info.remote_instance_info.cluster_ids.size();
    CHECK_GT(dst_dp_size, 0);
    CHECK_EQ(dst_world_size % static_cast<size_t>(dst_dp_size), 0);
    const int32_t dst_tp_size =
        static_cast<int32_t>(dst_world_size / dst_dp_size);
    CHECK_GT(dst_tp_size, 0);
    CHECK_GE(info.dp_rank, 0);
    CHECK_LT(info.dp_rank, dst_dp_size);

    // Every Prefill head shard must reach a Decode worker. The generic
    // merge_kv_blocks() route is one-to-one in TP rank and drops source ranks
    // whose rank is >= dst_tp_size (for TP2 -> TP1, rank 1 is dropped). Map
    // source shards modulo the destination TP group instead; staging row
    // ranges keep multiple source shards disjoint on the selected worker.
    const int32_t dst_rank =
        info.dp_rank * dst_tp_size + source_shard_rank % dst_tp_size;
    CHECK_LT(static_cast<size_t>(dst_rank), dst_world_size);
    const uint64_t dst_cluster_id =
        info.remote_instance_info.cluster_ids[dst_rank];
    const std::string& dst_addr = info.remote_instance_info.addrs[dst_rank];
    const std::string key = std::to_string(dst_cluster_id) + "_" + dst_addr;
    auto& kv_info = merged[key];
    kv_info.dst_cluster_id = dst_cluster_id;
    kv_info.dst_addr = dst_addr;
    kv_info.src_blocks.insert(kv_info.src_blocks.end(),
                              info.local_blocks_ids.begin(),
                              info.local_blocks_ids.end());
    kv_info.dst_blocks.insert(kv_info.dst_blocks.end(),
                              info.remote_blocks_ids.begin(),
                              info.remote_blocks_ids.end());
    kv_info.src_linear_state_ids.insert(kv_info.src_linear_state_ids.end(),
                                        info.local_linear_state_ids.begin(),
                                        info.local_linear_state_ids.end());
    kv_info.dst_linear_state_ids.insert(kv_info.dst_linear_state_ids.end(),
                                        info.remote_linear_state_ids.begin(),
                                        info.remote_linear_state_ids.end());
  }
}

}  // namespace

SpecKVCacheTransfer::SpecKVCacheTransfer(const uint16_t listen_port,
                                         const InstanceRole& instance_role,
                                         bool enable_lighting_indexer,
                                         bool enable_mla,
                                         bool draft_body_uses_tp1)
    : LlmDataDistTransfer(listen_port, instance_role, enable_lighting_indexer) {
  enable_mla_ = enable_mla;
  draft_body_uses_tp1_ = draft_body_uses_tp1;
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
  // Register matching staging cache IDs on both Prefill and Decode before the
  // first DataDist link. Prefill pushes each local TP shard into a disjoint row
  // range; Decode merges those already-local rows into its TP1 cache.
  const bool source_is_sharded = role_ == LlmRole::kPrompt;
  register_hetero_staging_caches(layer_registered_caches_,
                                 hetero_staging_registered_caches_,
                                 /*source_shard_count=*/2,
                                 source_is_sharded);
  register_hetero_staging_caches(spec_layer_registered_caches_,
                                 spec_hetero_staging_registered_caches_,
                                 /*source_shard_count=*/2,
                                 source_is_sharded && !draft_body_uses_tp1_);
}

bool SpecKVCacheTransfer::pull_replicated_spec_kv_blocks(
    uint64_t src_cluster_id,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
  CHECK_EQ(src_blocks.size(), dst_blocks.size());
  bool success = true;
  for (size_t layer_id = 0; layer_id < spec_layer_registered_caches_.size();
       ++layer_id) {
    for (const RegisteredCache& cache :
         spec_layer_registered_caches_[layer_id]) {
      CacheIndex source{src_cluster_id, cache.cache.cache_id};
      KvCacheExtParam ext_param{};
      ext_param.src_layer_range = {0, 0};
      ext_param.dst_layer_range = {0, 0};
      ext_param.tensor_num_per_layer = 1;
      const auto ret = llm_data_dist_->PullKvBlocks(
          source, cache.cache, src_blocks, dst_blocks, ext_param);
      if (ret != LLM_SUCCESS) {
        LOG(ERROR) << "Pull replicated TP1 draft KV failed, layer=" << layer_id
                   << ", role=" << cache.role.to_string()
                   << ", ret=" << std::hex << ret;
        success = false;
      }
    }
  }
  return success;
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
  hetero_staging_registered_caches_.clear();
  spec_hetero_staging_registered_caches_.clear();
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

bool SpecKVCacheTransfer::pull_hetero_kv_blocks(
    const std::vector<uint64_t>& src_cluster_ids,
    const std::vector<std::string>& src_addrs,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<uint64_t>& src_linear_state_ids,
    const std::vector<uint64_t>& dst_linear_state_ids) {
  (void)src_addrs;
  Timer phase_timer;
  // DataDist PUSH does not expose a Decode-side completion primitive for the
  // large recurrent state tensors. Keep CONV/SSM on the established
  // synchronous PULL path while KEY/VALUE are pre-pushed into staging.
  const bool linear_success =
      pull_and_merge_sharded_caches(layer_registered_caches_,
                                    hetero_staging_registered_caches_,
                                    src_cluster_ids,
                                    /*src_blocks=*/{},
                                    /*dst_blocks=*/{},
                                    src_linear_state_ids,
                                    dst_linear_state_ids);
  if (!linear_success) {
    return false;
  }
  const double linear_seconds = phase_timer.elapsed_seconds();
  phase_timer.reset();
  const bool target_success =
      merge_pre_pushed_sharded_caches(layer_registered_caches_,
                                      hetero_staging_registered_caches_,
                                      dst_blocks,
                                      /*dst_linear_state_ids=*/{},
                                      /*source_shard_count=*/2);
  if (!target_success) {
    return false;
  }
  const double target_merge_seconds = phase_timer.elapsed_seconds();
  phase_timer.reset();
  // Keep the one-layer MTP draft cache on the established synchronous pull
  // path while validating the new layer-overlapped target-cache push.  The
  // draft pull is small (~1 ms), and this isolates whether its newly-added
  // layer event observes the cache before the MTP prefill write is complete.
  const bool draft_success =
      draft_body_uses_tp1_
          ? pull_replicated_spec_kv_blocks(
                src_cluster_ids.front(), src_blocks, dst_blocks)
          : pull_and_merge_sharded_caches(
                spec_layer_registered_caches_,
                spec_hetero_staging_registered_caches_,
                src_cluster_ids,
                src_blocks,
                dst_blocks,
                /*src_linear_state_ids=*/{},
                /*dst_linear_state_ids=*/{});
  if (draft_success) {
    const double draft_seconds = phase_timer.elapsed_seconds();
    LOG(INFO) << "Merged heterogeneous TP KV cache (target KV pre-pushed, "
                 "linear state and draft pulled): source_shards="
              << 2 << ", blocks=" << dst_blocks.size()
              << ", linear_states=" << dst_linear_state_ids.size()
              << ", linear_ms=" << linear_seconds * 1000.0
              << ", target_merge_ms=" << target_merge_seconds * 1000.0
              << ", draft_ms=" << draft_seconds * 1000.0 << ", total_ms="
              << (linear_seconds + target_merge_seconds + draft_seconds) *
                     1000.0;
  }
  return draft_success;
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

bool SpecKVCacheTransfer::push_kv_blocks_to_hetero_staging(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft,
    int64_t source_shard_rank,
    int64_t source_shard_count) {
  const LayerRegisteredCaches& source_caches =
      is_spec_draft ? spec_layer_registered_caches_ : layer_registered_caches_;
  const LayerRegisteredCaches& staging_caches =
      is_spec_draft ? spec_hetero_staging_registered_caches_
                    : hetero_staging_registered_caches_;
  return push_layer_registered_caches_to_staging(source_caches,
                                                 staging_caches,
                                                 merged_kv_infos,
                                                 layer_synchronizer,
                                                 source_shard_rank,
                                                 source_shard_count);
}

folly::SemiFuture<bool> SpecKVCacheTransfer::push_kv_blocks_async(
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    const ParallelArgs& parallel_args,
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer,
    bool is_spec_draft) {
  const int32_t local_dp_size = parallel_args.dp_size();
  const int32_t kv_split_size = parallel_args.kv_split_size_effective();
  const std::optional<int32_t> remote_tp_size =
      get_remote_tp_size(transfer_kv_infos);
  bool heterogeneous_non_mla = false;
  int32_t local_tp_size = 1;
  if (!enable_mla_ && local_dp_size > 0 && kv_split_size > 0 &&
      remote_tp_size.has_value()) {
    local_tp_size = parallel_args.world_size() / local_dp_size / kv_split_size;
    if (local_tp_size != remote_tp_size.value()) {
      heterogeneous_non_mla = true;
      LOG(INFO) << "Push non-MLA heterogeneous KV shards to decode staging: "
                << "prefill_tp_size=" << local_tp_size
                << ", decode_tp_size=" << remote_tp_size.value()
                << ", is_spec_draft=" << is_spec_draft
                << "; decode will only perform a local merge.";
    }
  }
  const int64_t source_shard_rank =
      heterogeneous_non_mla ? parallel_args.rank() % local_tp_size : 0;

  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  // In heterogeneous non-MLA mode Decode intentionally restores the draft
  // cache from the source shards with a synchronous PULL.  Pushing the same
  // one-layer draft cache into staging is therefore redundant: no Decode
  // path consumes spec_hetero_staging_registered_caches_.  The source cache
  // remains alive until the synchronous FirstGeneration RPC returns, so
  // skipping this PUSH does not shorten its lifetime for the later PULL.
  if (heterogeneous_non_mla && is_spec_draft) {
    VLOG(5) << "Skip redundant heterogeneous MTP draft staging PUSH; "
               "Decode restores draft KV from source shards.";
    promise.setValue(true);
    return future;
  }
  threadpool_.schedule([this,
                        transfer_kv_infos,
                        &parallel_args,
                        layer_synchronizer,
                        is_spec_draft,
                        heterogeneous_non_mla,
                        local_tp_size,
                        source_shard_rank,
                        promise = std::move(promise)]() mutable {
    std::unordered_map<std::string, KVCacheInfo> merged_kv_infos;
    std::vector<TransferKVInfo> filtered_kv_infos;
    const std::vector<TransferKVInfo>* kv_infos = &transfer_kv_infos;
    // When the KV cache is actually sharded across ranks
    // (kv_split_size_effective > 1), filter remote_blocks_ids down to this
    // rank's slice. When kv_split_size==1 each rank holds the full replica and
    // we keep the legacy 1:1 remote_blocks_ids mapping.
    const int32_t effective_kv_split_size =
        parallel_args.kv_split_size_effective();
    if (effective_kv_split_size > 1) {
      filtered_kv_infos = filter_kv_split_infos(
          parallel_args.kv_split_rank(), effective_kv_split_size, *kv_infos);
      kv_infos = &filtered_kv_infos;
      if (kv_infos->empty()) {
        promise.setValue(true);
        return;
      }
    }
    if (heterogeneous_non_mla) {
      merge_heterogeneous_kv_blocks(
          merged_kv_infos, *kv_infos, source_shard_rank);
    } else {
      merge_kv_blocks(merged_kv_infos, *kv_infos, parallel_args);
    }
    bool success = true;
    if (!merged_kv_infos.empty()) {
      if (heterogeneous_non_mla) {
        success = this->push_kv_blocks_to_hetero_staging(merged_kv_infos,
                                                         layer_synchronizer,
                                                         is_spec_draft,
                                                         source_shard_rank,
                                                         local_tp_size);
      } else {
        success = this->push_kv_blocks(merged_kv_infos,
                                       layer_synchronizer,
                                       is_spec_draft,
                                       parallel_args.kv_split_rank(),
                                       parallel_args.kv_split_size_effective());
      }
    }
    promise.setValue(success);
  });
  return future;
}
}  // namespace xllm
