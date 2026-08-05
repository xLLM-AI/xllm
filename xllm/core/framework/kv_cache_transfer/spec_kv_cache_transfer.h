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

#pragma once

#include <memory>
#include <mutex>

#include "framework/kv_cache_transfer/llm_data_dist_transfer.h"
#include "framework/parallel_state/parallel_args.h"

namespace xllm {

using namespace llm_datadist;

class SpecKVCacheTransfer : public LlmDataDistTransfer {
 public:
  SpecKVCacheTransfer(const uint16_t listen_port,
                      const InstanceRole& instance_role,
                      bool enable_lighting_indexer = false,
                      bool enable_mla = false,
                      bool draft_body_uses_tp1 = false);

  virtual ~SpecKVCacheTransfer() = default;

  void register_kv_cache(std::vector<xllm::KVCache>& kv_caches,
                         const KVCacheShape& kv_cache_shape,
                         const torch::ScalarType dtype) override;

  void register_kv_cache_spec(std::vector<xllm::KVCache>& kv_caches,
                              const KVCacheShape& kv_cache_shape,
                              const torch::ScalarType dtype) override;

  void register_kv_cache_internal(
      std::vector<xllm::KVCache>& kv_caches,
      LayerRegisteredCaches& layer_registered_caches);

  void free_kv_cache() override;

  bool pull_kv_blocks(const uint64_t src_cluster_id,
                      const std::string& src_addr,
                      const std::vector<KVTransferMapping>& mappings) override;

  bool pull_hetero_kv_blocks(
      const std::vector<uint64_t>& src_cluster_ids,
      const std::vector<std::string>& src_addrs,
      const std::vector<KVTransferMapping>& mappings) override;

  folly::SemiFuture<bool> push_kv_blocks_async(
      const std::vector<TransferKVInfo>& transfer_kv_infos,
      const ParallelArgs& parallel_args,
      std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer,
      bool is_spec_draft) override;

  bool push_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
      bool is_spec_draft,
      int32_t kv_split_rank,
      int32_t kv_split_size) override;

  bool push_kv_blocks_spec(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
      int32_t kv_split_rank = 0,
      int32_t kv_split_size = 1);

  bool push_kv_blocks_internal(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
      const LayerRegisteredCaches& layer_registered_caches,
      int32_t kv_split_rank = 0,
      int32_t kv_split_size = 1);

  bool push_kv_blocks_to_hetero_staging(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
      bool is_spec_draft,
      int64_t source_shard_rank,
      int64_t source_shard_count);

 private:
  bool pull_and_merge_sharded_caches(
      const LayerRegisteredCaches& layer_registered_caches,
      const LayerRegisteredCaches& staging_registered_caches,
      const std::vector<uint64_t>& src_cluster_ids,
      const std::vector<KVTransferMapping>& mappings,
      bool sequence_scoped);

  bool merge_pre_pushed_sharded_caches(
      const LayerRegisteredCaches& layer_registered_caches,
      const LayerRegisteredCaches& staging_registered_caches,
      const std::vector<KVTransferMapping>& mappings,
      int64_t source_shard_count,
      bool sequence_scoped);

  bool push_layer_registered_caches_to_staging(
      const LayerRegisteredCaches& layer_registered_caches,
      const LayerRegisteredCaches& staging_registered_caches,
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
      int64_t source_shard_rank,
      int64_t source_shard_count);

  void register_hetero_staging_caches(
      const LayerRegisteredCaches& source_registered_caches,
      LayerRegisteredCaches& staging_registered_caches,
      int64_t source_shard_count,
      bool source_is_sharded);

  bool pull_replicated_spec_kv_blocks(
      uint64_t src_cluster_id,
      const std::vector<KVTransferMapping>& mappings);

  bool draft_body_uses_tp1_ = false;
  bool heterogeneous_pd_enabled_ = false;
  LayerRegisteredCaches hetero_staging_registered_caches_;
  LayerRegisteredCaches spec_hetero_staging_registered_caches_;
  LayerRegisteredCaches spec_layer_registered_caches_;
  bool parallel_shard_pull_ = true;
  // Staging tensors are shared by all heterogeneous requests. Keep the full
  // restore transaction serialized until request-scoped staging slots exist.
  std::mutex hetero_restore_mutex_;
  // Created only for the opt-in heterogeneous path. Homogeneous PD should not
  // reserve a worker thread for a code path it cannot enter.
  std::unique_ptr<ThreadPool> shard_pull_threadpool_;
};

}  // namespace xllm
