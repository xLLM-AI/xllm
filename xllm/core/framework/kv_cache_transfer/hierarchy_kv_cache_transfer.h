/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/macros.h"
#include "common/types.h"
#include "framework/block/block.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/kv_cache_transfer/host_transfer/transfer.h"
#include "framework/model/model_input_params.h"
#include "platform/device.h"
#include "util/threadpool.h"

namespace xllm {

class KVCacheStore;

class HierarchyKVCacheTransfer final {
 public:
  using GroupedCaches = std::map<BlockType, std::vector<KVCache*>>;
  using HostGroupedCaches = std::map<BlockType, std::unique_ptr<KVCache>>;
  using CacheHandle = uint32_t;

  enum class CacheRole : uint8_t {
    TARGET = 0,
    DRAFT,
  };

  struct CacheRegistration {
    CacheRole role = CacheRole::TARGET;
    std::vector<KVCache>* device_kv_caches = nullptr;
    KVCacheShape kv_cache_shape;
    KVCacheCreateOptions create_options;
    const Stream* producer_stream = nullptr;
    std::string store_key_component;
  };

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
    PROPERTY(std::string, store_namespace) = "";
    PROPERTY(uint32_t, store_worker_id) = 0;
  };

  HierarchyKVCacheTransfer(const Options& options, const torch::Device& device);
  HierarchyKVCacheTransfer(const Options& options,
                           const torch::Device& device,
                           const Stream* compute_stream,
                           std::vector<xllm::KVCache>* kv_caches_ptr,
                           const KVCacheShape& kv_cache_shape,
                           const KVCacheCreateOptions& create_options);
  ~HierarchyKVCacheTransfer();

  CacheHandle register_cache(CacheRegistration registration);
  bool finalize_registration();
  void shutdown();
  [[nodiscard]] bool registration_finalized() const {
    return registration_finalized_;
  }

  uint32_t transfer_kv_blocks(
      uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info);
  uint32_t transfer_kv_blocks(uint64_t batch_id,
                              Slice<BlockTransferInfo>& block_transfer_info);
  std::vector<uint8_t> prefetch_kv_blocks(
      Slice<BlockTransferInfo>& block_transfer_info);
  void set_layer_synchronizer(ModelInputParams& params);
  [[nodiscard]] bool supports_block_type(BlockType block_type) const;
  [[nodiscard]] bool supports_block_type(CacheRole role,
                                         BlockType block_type) const;

 private:
  struct CacheDomain {
    CacheHandle handle = 0;
    CacheRole role = CacheRole::TARGET;
    std::vector<KVCache>* device_kv_caches = nullptr;
    KVCacheShape kv_cache_shape;
    KVCacheCreateOptions create_options;
    GroupedCaches device_caches_by_type;
    std::map<BlockType, std::vector<int64_t>> layer_ids_by_type;
    HostGroupedCaches host_caches_by_type;
    std::unique_ptr<HostKVLayout> host_layout;
    const Stream* producer_stream = nullptr;
    std::string store_key_component;
  };

  static int32_t domain_group_id(CacheHandle handle, BlockType block_type);
  GroupedCaches build_device_groups(CacheDomain* domain) const;
  void create_host_cache(CacheDomain* domain);
  HostKVLayout create_host_kv_layout(const CacheDomain& domain) const;
  HostKVRequest make_request(
      const std::vector<BlockTransferInfo>& block_transfer_info,
      TransferType transfer_type) const;

  uint32_t offload(const std::vector<BlockTransferInfo>& block_transfer_info);
  bool offload_to_host(const HostKVRequest& request);
  bool load_from_host(const HostKVRequest& request,
                      const HostKVLoadHandle& handle);

  Options options_;
  Device device_;
  std::unique_ptr<ThreadPool> load_threadpool_;

  std::vector<CacheDomain> cache_domains_;
  bool registration_finalized_ = false;
  bool shutdown_ = false;

  std::unique_ptr<HostKVTransfer> host_kv_transfer_;
  std::unique_ptr<KVCacheStore> kv_cache_store_;

  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, HostKVLoadHandle> load_handles_;
};

}  // namespace xllm
