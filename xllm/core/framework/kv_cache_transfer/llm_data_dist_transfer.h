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

#include <llm_datadist/llm_datadist.h>

#include "framework/kv_cache_transfer/kv_cache_transfer.h"

namespace xllm {

using namespace llm_datadist;

struct RegisteredCache {
  KVCacheTensorRole role;
  int32_t group_id;
  bool sequence_scoped;
  Cache cache;
};

using LayerRegisteredCaches = std::vector<std::vector<RegisteredCache>>;

class LlmDataDistTransfer : public KVCacheTransfer {
 public:
  LlmDataDistTransfer(const uint16_t listen_port,
                      const InstanceRole& instance_role,
                      bool enable_lighting_indexer = false);
  virtual ~LlmDataDistTransfer() = default;

  virtual void initialize(int32_t device_id) override;

  virtual void finalize() override;

  virtual void register_kv_cache(std::vector<xllm::KVCache>& kv_caches,
                                 const KVCacheShape& kv_cache_shape,
                                 const torch::ScalarType dtype) override;

  virtual void free_kv_cache() override;

  virtual void get_cache_info(uint64_t& cluster_id, std::string& addr) override;

  virtual bool link_cluster(const uint64_t cluster_id,
                            const std::string& remote_addr,
                            const uint16_t port) override;

  virtual bool unlink_cluster(const uint64_t& cluster_id,
                              const std::string& remote_addr,
                              const uint16_t port,
                              bool force_flag = true) override;

  virtual bool pull_kv_blocks(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<uint64_t>& src_blocks,
      const std::vector<uint64_t>& dst_blocks,
      const std::vector<uint64_t>& src_linear_state_ids,
      const std::vector<uint64_t>& dst_linear_state_ids) override;

  virtual bool push_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
      bool is_spec_draft,
      int32_t kv_split_rank,
      int32_t kv_split_size) override;

  ClusterInfo create_cluster_info(const uint64_t& cluster_id,
                                  const std::string& remote_ip,
                                  const uint16_t& remote_port);

 protected:
  RegisteredCache register_cache_tensor(int64_t layer_id,
                                        const KVCacheTensor& cache_tensor);

  void register_layer_registered_caches(
      std::vector<xllm::KVCache>& kv_caches,
      LayerRegisteredCaches& layer_registered_caches);

  bool push_layer_registered_caches(
      const LayerRegisteredCaches& layer_registered_caches,
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
      int32_t kv_split_rank = 0,
      int32_t kv_split_size = 1);

 protected:
  uint64_t cluster_id_;
  std::string host_ip_;
  uint16_t listen_port_;
  bool enable_mla_ = false;
  bool enable_lighting_indexer_ = false;
  bool has_grouped_cache_layout_ = false;
  LlmRole role_ = LlmRole::kMix;
  std::unordered_set<uint64_t> linked_cluster_ids;

  std::shared_ptr<LlmDataDist> llm_data_dist_;
  LayerRegisteredCaches layer_registered_caches_;
};

}  // namespace xllm
