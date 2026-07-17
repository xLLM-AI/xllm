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

#include "framework/kv_cache_transfer/llm_data_dist_transfer.h"
#include "framework/parallel_state/parallel_args.h"

namespace xllm {

using namespace llm_datadist;

class SpecKVCacheTransfer : public LlmDataDistTransfer {
 public:
  SpecKVCacheTransfer(const uint16_t listen_port,
                      const InstanceRole& instance_role,
                      bool enable_lighting_indexer = false);

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

  bool pull_kv_blocks(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<uint64_t>& src_blocks,
      const std::vector<uint64_t>& dst_blocks,
      const std::vector<uint64_t>& src_linear_state_ids,
      const std::vector<uint64_t>& dst_linear_state_ids) override;

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

 private:
  LayerRegisteredCaches spec_layer_registered_caches_;
};

}  // namespace xllm
