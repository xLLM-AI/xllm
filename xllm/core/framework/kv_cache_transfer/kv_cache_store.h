/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <optional>
#include <string>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache_transfer/mooncake_store_backend.h"
#include "framework/model/model_input_params.h"
#include "util/slice.h"

namespace xllm {

struct HostCacheStoreEntry {
  uint32_t cache_handle = 0;
  std::string key_component;
  KVCache* cache = nullptr;
};

using HostCacheStoreIndex =
    std::map<BlockType, std::vector<HostCacheStoreEntry>>;

struct KVCacheStoreInitConfig {
  std::string localhost_name = "127.0.0.1";
  std::string protocol = "tcp";
  std::string rdma_devices;
  std::string metadata_server;
  std::string master_server_address;
  std::string model_id;
  int32_t replica_num = 1;
  uint32_t tp_rank = 0;
  uint32_t tp_size = 1;
  bool enable_mla = false;
};

class KVCacheStore final {
 public:
  KVCacheStore() = default;
  ~KVCacheStore();

  bool init(const KVCacheStoreInitConfig& config,
            HostCacheStoreIndex store_index);

  uint32_t batch_put(
      const std::vector<BlockTransferInfo>& block_transfer_info) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    return batch_put(slice);
  }

  uint32_t batch_get(
      const std::vector<BlockTransferInfo>& block_transfer_info) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    return batch_get(slice);
  }

  uint32_t batch_put(Slice<BlockTransferInfo>& block_transfer_info);
  uint32_t batch_get(Slice<BlockTransferInfo>& block_transfer_info);
  std::vector<uint8_t> batch_get_with_status(
      Slice<BlockTransferInfo>& block_transfer_info);

 private:
  friend class KVCacheStoreTestPeer;

  struct StoreEntry {
    uint32_t cache_handle = 0;
    std::string key_component;
    std::string schema_hash;
    KVCache* cache = nullptr;
    BlockType block_type = BlockType::KV;
    std::string key_prefix;
    std::vector<torch::Tensor> block_tensors;
  };

  struct PhysicalRequest {
    size_t logical_index = 0;
    const StoreEntry* entry = nullptr;
    std::string key;
  };

  struct RequestGroup {
    std::string key;
    std::vector<size_t> request_indices;
  };

  struct GroupedRequests {
    std::vector<PhysicalRequest> requests;
    std::vector<RequestGroup> groups;
  };

  KVCacheStore(const KVCacheStore&) = delete;
  KVCacheStore& operator=(const KVCacheStore&) = delete;

  void initialize_store_index(HostCacheStoreIndex store_index);
  std::string build_schema_hash(BlockType block_type,
                                const KVCache& cache) const;
  std::string build_key_prefix(const std::string& key_component,
                               BlockType block_type,
                               const std::string& schema_hash) const;
  std::string build_key(const StoreEntry& entry,
                        const BlockTransferInfo& block_info) const;
  std::vector<PhysicalRequest> build_requests(
      Slice<BlockTransferInfo>& block_transfer_info) const;
  static std::vector<RequestGroup> group_requests(
      const std::vector<PhysicalRequest>& requests);
  GroupedRequests build_grouped_requests(
      Slice<BlockTransferInfo>& block_transfer_info) const;
  static std::vector<uint8_t> aggregate_results(
      size_t logical_count,
      const std::vector<PhysicalRequest>& requests,
      const std::vector<uint8_t>& physical_results);
  static std::optional<std::string> get_store_device_names(
      const KVCacheStoreInitConfig& config);
  std::optional<MooncakeMultiBuffer> build_multi_buffer(const StoreEntry& entry,
                                                        int32_t block_id) const;
  std::optional<std::vector<MooncakeRegisteredRange>> collect_ranges() const;

 private:
  bool is_initialized_ = false;
  KVCacheStoreInitConfig config_;
  std::map<BlockType, std::vector<StoreEntry>> store_index_;
  size_t max_entries_per_type_ = 0;
  std::unique_ptr<MooncakeStoreBackend> backend_;
};

}  // namespace xllm
