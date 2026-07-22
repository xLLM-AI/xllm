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

#include <folly/futures/Future.h>

#include "common/types.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache/kv_cache_tensor_allocator.h"
#if defined(USE_NPU)
#include "platform/npu/npu_layer_synchronizer.h"
#endif
#if defined(USE_MLU)
#include "platform/mlu/mlu_layer_synchronizer.h"
#endif
#if defined(USE_DCU)
#include "platform/dcu/dcu_layer_synchronizer.h"
#endif
#include "framework/parallel_state/parallel_args.h"
#include "platform/device.h"
#include "util/threadpool.h"

namespace xllm {

#if defined(USE_NPU)
using KVPushSynchronizerImpl = NPULayerSynchronizerImpl;
#elif defined(USE_MLU)
using KVPushSynchronizerImpl = MLULayerSynchronizerImpl;
#elif defined(USE_DCU)
using KVPushSynchronizerImpl = DCULayerSynchronizerImpl;
#endif

// In KV-split mode, filters and remaps remote_blocks_ids so that each KV-split
// rank only sees the remote blocks assigned to it. When `kv_split_size == 1`
// the caller should skip this entirely (every rank holds the full KV replica
// and `remote_blocks_ids` is 1:1 with `local_blocks_ids`).
//
// Note: prior to the KV-split / CP decoupling refactor this was named
// filter_cp_kv_infos and gated on cp_size>1. The behavior is identical when
// kv_split_size == cp_size (the legacy default), so callers that pass cp_rank
// / cp_size keep working byte-for-byte.
std::vector<TransferKVInfo> filter_kv_split_infos(
    int32_t kv_split_rank,
    int32_t kv_split_size,
    const std::vector<TransferKVInfo>& kv_infos);

class KVCacheTransfer {
 public:
  struct KVCacheInfo {
    uint64_t dst_cluster_id;
    std::string dst_addr;
    std::vector<uint64_t> src_blocks;
    std::vector<uint64_t> dst_blocks;
    std::vector<uint64_t> src_linear_state_ids;
    std::vector<uint64_t> dst_linear_state_ids;

    // XTensor mode: destination offsets from D-node (per-layer)
    // dst_xtensor_layer_offsets[layer_id] = {k_offsets, v_offsets}
    std::vector<XTensorLayerOffsets> dst_xtensor_layer_offsets;

    // Group-aware block mappings keyed by cache-layout group id.
    std::vector<KVBlockTransferGroup> block_transfer_groups;
  };

  static std::vector<std::string> rotate_dst_rank(
      const std::vector<std::string>& keys,
      int32_t kv_split_rank);

  KVCacheTransfer() = default;
  virtual ~KVCacheTransfer() = default;

  virtual void initialize(int32_t device_id) {};

  virtual void finalize() {};

  virtual void allocate_kv_cache(std::vector<xllm::KVCache>& kv_caches,
                                 const int64_t num_layers,
                                 const KVCacheShape& kv_cache_shape,
                                 const torch::ScalarType dtype) {};

  virtual void allocate_kv_cache_spec(std::vector<xllm::KVCache>& kv_caches,
                                      const int64_t num_layers,
                                      const KVCacheShape& kv_cache_shape,
                                      torch::ScalarType dtype) {
    NOT_IMPLEMENTED();
  };

  virtual void free_kv_cache() {};

  virtual void register_kv_cache(std::vector<xllm::KVCache>& kv_caches,
                                 const KVCacheShape& kv_cache_shape,
                                 const torch::ScalarType dtype) {};

  virtual void register_kv_cache_spec(std::vector<xllm::KVCache>& kv_caches,
                                      const KVCacheShape& kv_cache_shape,
                                      const torch::ScalarType dtype) {
    NOT_IMPLEMENTED();
  };

  virtual void get_cache_info(uint64_t& cluster_id, std::string& addr) = 0;

  virtual bool link_cluster(const uint64_t cluster_id,
                            const std::string& remote_addr,
                            const uint16_t port) = 0;

  virtual bool unlink_cluster(const uint64_t& cluster_id,
                              const std::string& remote_addr,
                              const uint16_t port,
                              bool force_flag = true) = 0;

  virtual bool pull_kv_blocks(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<uint64_t>& src_blocks,
      const std::vector<uint64_t>& dst_blocks,
      const std::vector<uint64_t>& src_linear_state_ids,
      const std::vector<uint64_t>& dst_linear_state_ids) = 0;

  virtual folly::SemiFuture<bool> pull_kv_blocks_async(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<uint64_t>& src_blocks,
      const std::vector<uint64_t>& dst_blocks,
      const std::vector<uint64_t>& src_linear_state_ids,
      const std::vector<uint64_t>& dst_linear_state_ids);

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
  virtual folly::SemiFuture<bool> push_kv_blocks_async(
      const std::vector<TransferKVInfo>& transfer_kv_infos,
      const ParallelArgs& parallel_args,
      std::shared_ptr<KVPushSynchronizerImpl> layer_synchronizer,
      bool is_spec_draft);
#endif

  virtual void merge_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      const std::vector<TransferKVInfo>& transfer_kv_infos,
      const ParallelArgs& parallel_args);

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
  virtual bool push_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
      bool is_spec_draft,
      int32_t kv_split_rank,
      int32_t kv_split_size) = 0;
#endif

#if defined(USE_NPU)
  virtual std::vector<torch::Tensor> convert_to_torch_tensor(
      const std::vector<int64_t>& dims,
      const torch::ScalarType dtype,
      const std::vector<uintptr_t>& addresses,
      const aclFormat format = ACL_FORMAT_ND);
#endif

 protected:
  // working thread
  ThreadPool threadpool_{/*num_threads=*/1,
                         /*cpu_binding=*/false,
                         /*pool_name=*/"KVCacheTransfer.async"};
};

class KVCacheTransferFactory {
 public:
  using AllocateKVCacheFunc = std::function<bool(
      const KVCacheShape&,
      bool use_huge_page_allocator,
      std::shared_ptr<KVCacheTensorAllocator> tensor_allocator)>;

  static std::shared_ptr<KVCacheTransfer> create(
      const std::string& transfer_type,
      uint16_t transfer_listen_port,
      InstanceRole instance_role,
      const Device& device,
      const KVCacheShape& kv_cache_shape,
      torch::ScalarType dtype,
      std::vector<xllm::KVCache>& kv_caches,
      int64_t num_layers,
      AllocateKVCacheFunc allocate_kv_cache_func,
      bool enable_lighting_indexer,
      const std::string& model_type = "",
      const std::string& model_id = "");
};

}  // namespace xllm
