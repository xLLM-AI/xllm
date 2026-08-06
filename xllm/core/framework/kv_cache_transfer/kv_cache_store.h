/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <Mooncake/mooncake-store/include/client_service.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "util/slice.h"

namespace xllm {

using HostGroupedCaches = std::map<BlockType, std::unique_ptr<KVCache>>;

struct KVCacheStoreInitConfig {
  std::string localhost_name = "127.0.0.1";
  std::string protocol = "tcp";
  std::string metadata_server;
  std::string master_server_address;
  std::string model_id;
  int32_t replica_num = 1;
  uint32_t tp_rank = 0;
  uint32_t tp_size = 1;
};

class KVCacheStore final {
 public:
  KVCacheStore() = default;
  ~KVCacheStore();

  bool init(const KVCacheStoreInitConfig& config,
            HostGroupedCaches* host_kv_caches);

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

  uint32_t batch_exist(std::vector<std::string>&& keys);

 private:
  KVCacheStore(const KVCacheStore&) = delete;
  KVCacheStore& operator=(const KVCacheStore&) = delete;

  std::string build_key(const BlockTransferInfo& block_info) const;
  std::vector<mooncake::Slice> generate_mooncake_slices(BlockType type,
                                                        int32_t block_id) const;

 private:
  bool is_initialized_ = false;
  KVCacheStoreInitConfig config_;
  std::string cache_schema_hash_;
  mooncake::ReplicateConfig rep_config_;
  HostGroupedCaches* host_kv_caches_ = nullptr;
  std::vector<void*> registered_addresses_;
  std::shared_ptr<mooncake::Client> client_ptr_;
};

}  // namespace xllm
