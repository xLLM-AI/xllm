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

#include <torch/torch.h>

#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/macros.h"
#include "common/types.h"
#include "framework/block/block.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/model/model_input_params.h"
#include "platform/batch_memcpy.h"
#include "platform/device.h"
#include "platform/layer_synchronizer.h"
#include "util/blockingconcurrentqueue.h"
#include "util/threadpool.h"

namespace xllm {
class HierarchyKVCacheTransfer {
 public:
  struct LayerBatchRange {
    int64_t begin_layer = 0;
    int64_t end_layer = 0;
  };

  struct CopyPlan {
    std::vector<torch::Tensor> src_tensors;
    std::vector<torch::Tensor> dst_tensors;
  };

  using GroupedCaches = std::map<BlockType, std::vector<KVCache*>>;

  // Host prefix caches: one real KVCache per block type, allocated over
  // page-aligned + mlock'd + NPU-registered host memory. Shape per tensor is
  // [host_blocks, layer_count, ...per_block_dims].
  using HostGroupedCaches = std::map<BlockType, std::unique_ptr<KVCache>>;

  struct Options {
    PROPERTY(uint32_t, tp_rank);
    PROPERTY(uint32_t, tp_size);
    PROPERTY(uint32_t, layers);
    PROPERTY(double, host_blocks_factor) = 0.0;
    PROPERTY(uint32_t, layers_wise_copy_batchs) = 1;
    PROPERTY(bool, enable_mla) = false;
    PROPERTY(bool, enable_kvcache_store) = false;
    PROPERTY(std::string, store_protocol) = "rdma";
    PROPERTY(std::string, store_master_server_address) = "";
    PROPERTY(std::string, store_metadata_server) = "";
    PROPERTY(std::string, store_local_hostname) = "";
  };

  HierarchyKVCacheTransfer(const Options& options,
                           const torch::Device& device,
                           std::vector<xllm::KVCache>* kv_caches_ptr,
                           const KVCacheShape& kv_cache_shape,
                           const KVCacheCreateOptions& create_options);
  ~HierarchyKVCacheTransfer() = default;

  uint32_t transfer_kv_blocks(
      const uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info);

  uint32_t transfer_kv_blocks(const uint64_t batch_id,
                              Slice<BlockTransferInfo>& block_transfer_info);

  void set_layer_synchronizer(ModelInputParams& params);

 private:
  void build_device_block_type_map();
  void create_host_cache();
  CopyPlan build_copy_plan(
      const std::vector<BlockTransferInfo>& block_transfer_info,
      const LayerBatchRange& layer_batch_range) const;

  uint32_t offload(const std::vector<BlockTransferInfo>& block_transfer_info);
  bool offload_to_host(Slice<BlockTransferInfo>& block_transfer_info);
  bool load_from_host(
      std::shared_ptr<LayerSynchronizer> synchronizer,
      const std::vector<BlockTransferInfo>& block_transfer_info);

 private:
  Options options_;
  Device device_;

  std::unique_ptr<ThreadPool> load_threadpool_;
  moodycamel::BlockingConcurrentQueue<std::unique_ptr<Stream>> copy_stream_;

  std::vector<xllm::KVCache>* kv_caches_ptr_ = nullptr;
  KVCacheShape kv_cache_shape_;
  KVCacheCreateOptions create_options_;
  GroupedCaches device_kv_caches_;
  std::map<BlockType, std::vector<int64_t>> device_block_type_layer_ids_;
  HostGroupedCaches host_kv_caches_;
  std::vector<LayerBatchRange> layer_batch_ranges_;

  std::unique_ptr<BatchMemcpy> batch_memcpy_;

  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<LayerSynchronizer>>
      layer_wise_load_synchronizer_;
};

}  // namespace xllm
