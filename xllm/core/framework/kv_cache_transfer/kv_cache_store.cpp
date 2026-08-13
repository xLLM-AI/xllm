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

#include "framework/kv_cache_transfer/kv_cache_store.h"

#include <Mooncake/mooncake-store/include/utils.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <unordered_map>
#include <utility>

#include "util/hash_util.h"

namespace xllm {

bool KVCacheStore::init(const KVCacheStoreInitConfig& config,
                        HostGroupedCaches* host_kv_caches) {
  CHECK(!is_initialized_) << "KVCacheStore is already initialized.";
  CHECK(host_kv_caches != nullptr && !host_kv_caches->empty())
      << "KVCacheStore requires typed Host caches.";
  config_ = config;
  host_kv_caches_ = host_kv_caches;

  std::optional<std::string> device_names = std::nullopt;
  if (config_.protocol == "rdma") {
    const char* configured_devices = std::getenv("DEVICE_NAMES");
    if (configured_devices != nullptr) {
      device_names = configured_devices;
      LOG(INFO) << "Mooncake RDMA device_names: " << device_names.value();
    } else {
      LOG(WARNING) << "DEVICE_NAMES is not set; falling back to TCP.";
      config_.protocol = "tcp";
    }
  }

  auto client = mooncake::Client::Create(config_.localhost_name,
                                         config_.metadata_server,
                                         config_.protocol,
                                         device_names,
                                         config_.master_server_address);
  if (!client.has_value()) {
    LOG(ERROR) << "Failed to create Mooncake Store client for "
               << config_.localhost_name;
    return false;
  }
  client_ptr_ = client.value();
  rep_config_.replica_num = config_.replica_num;

  std::string cache_schema = "tp=" + std::to_string(config_.tp_size);
  for (const auto& [type, cache] : *host_kv_caches_) {
    CHECK(cache != nullptr);
    const BlockTypeTensorMap tensors = cache->get_block_type_tensors(type);
    CHECK(!tensors.empty()) << "Host cache has no tensors for BlockType "
                            << static_cast<int32_t>(type);

    size_t slot_bytes = 0;
    int64_t host_blocks = -1;
    cache_schema.append("|type=");
    cache_schema.append(std::to_string(static_cast<int32_t>(type)));
    for (const auto& [role, tensor] : tensors) {
      CHECK(tensor.defined() && tensor.dim() > 0 && tensor.is_contiguous());
      if (host_blocks < 0) {
        host_blocks = tensor.size(0);
      } else {
        CHECK_EQ(host_blocks, tensor.size(0));
      }
      slot_bytes += static_cast<size_t>(tensor[0].numel()) *
                    static_cast<size_t>(tensor.element_size());
      cache_schema.append(",role=");
      cache_schema.append(std::to_string(static_cast<int32_t>(role)));
      cache_schema.append(",dtype=");
      cache_schema.append(
          std::to_string(static_cast<int32_t>(tensor.scalar_type())));
      cache_schema.append(",shape=");
      for (int64_t dim = 1; dim < tensor.dim(); ++dim) {
        cache_schema.append(std::to_string(tensor.size(dim)));
        cache_schema.push_back('x');
      }

      if (config_.protocol == "rdma") {
        void* address = tensor.data_ptr();
        const size_t bytes = static_cast<size_t>(tensor.numel()) *
                             static_cast<size_t>(tensor.element_size());
        auto result =
            client_ptr_->RegisterLocalMemory(address,
                                             bytes,
                                             /*location=*/"cpu:0",
                                             /*remote_accessible=*/false,
                                             /*update_metadata=*/false);
        if (!result.has_value()) {
          LOG(ERROR) << "Failed to register Mooncake Host tensor: "
                     << toString(result.error());
          return false;
        }
        registered_addresses_.emplace_back(address);
      }
    }
    LOG(INFO) << "KVCacheStore init OK: type=" << static_cast<int32_t>(type)
              << ", host_blocks=" << host_blocks
              << ", slot_bytes=" << slot_bytes
              << ", protocol=" << config_.protocol;
  }
  const XXH3Key schema_hash = hash_string(cache_schema);
  cache_schema_hash_.assign(reinterpret_cast<const char*>(schema_hash.data),
                            sizeof(schema_hash.data));

  is_initialized_ = true;
  return true;
}

KVCacheStore::~KVCacheStore() {
  if (client_ptr_ != nullptr) {
    for (void* address : registered_addresses_) {
      auto result = client_ptr_->unregisterLocalMemory(
          address, /*update_metadata=*/false);
      if (!result.has_value()) {
        LOG(WARNING) << "Failed to unregister Mooncake Host tensor: "
                     << toString(result.error());
      }
    }
    client_ptr_.reset();
  }
}

std::string KVCacheStore::build_key(const BlockTransferInfo& block_info) const {
  std::string key = "xllm-kv-v2:";
  key.append(std::to_string(config_.model_id.size()));
  key.push_back(':');
  key.append(config_.model_id);
  key.push_back(':');
  key.append(std::to_string(config_.tp_size));
  key.push_back(':');
  key.append(std::to_string(static_cast<int32_t>(block_info.block_type)));
  key.push_back(':');
  key.append(std::to_string(config_.tp_rank));
  key.push_back(':');
  key.append(cache_schema_hash_);
  key.append(reinterpret_cast<const char*>(block_info.hash_key),
             XXH3_128BITS_HASH_VALUE_LEN);
  return key;
}

uint32_t KVCacheStore::batch_put(
    Slice<BlockTransferInfo>& block_transfer_info) {
  if (!is_initialized_ || block_transfer_info.empty()) {
    return 0;
  }

  std::vector<std::string> all_keys;
  all_keys.reserve(block_transfer_info.size());
  for (const BlockTransferInfo& block_info : block_transfer_info) {
    all_keys.emplace_back(build_key(block_info));
  }
  const auto exists = client_ptr_->BatchIsExist(all_keys);

  std::vector<std::string> put_keys;
  std::vector<std::vector<mooncake::Slice>> put_slices;
  put_keys.reserve(block_transfer_info.size());
  put_slices.reserve(block_transfer_info.size());
  uint32_t success_count = 0;
  for (size_t i = 0; i < block_transfer_info.size(); ++i) {
    const bool already_exists =
        i < exists.size() && exists[i].has_value() && exists[i].value();
    if (already_exists) {
      ++success_count;
      continue;
    }
    put_keys.emplace_back(all_keys[i]);
    put_slices.emplace_back(
        generate_mooncake_slices(block_transfer_info[i].block_type,
                                 block_transfer_info[i].dst_block_id));
  }

  if (put_keys.empty()) {
    return success_count;
  }
  const auto results = client_ptr_->BatchPut(put_keys, put_slices, rep_config_);
  for (size_t i = 0; i < put_keys.size() && i < results.size(); ++i) {
    if (results[i].has_value()) {
      ++success_count;
    }
  }
  return success_count;
}

uint32_t KVCacheStore::batch_get(
    Slice<BlockTransferInfo>& block_transfer_info) {
  const std::vector<uint8_t> statuses =
      batch_get_with_status(block_transfer_info);
  return static_cast<uint32_t>(
      std::count(statuses.begin(), statuses.end(), static_cast<uint8_t>(1)));
}

std::vector<uint8_t> KVCacheStore::batch_get_with_status(
    Slice<BlockTransferInfo>& block_transfer_info) {
  std::vector<uint8_t> statuses(block_transfer_info.size(), /*value=*/0);
  if (!is_initialized_ || block_transfer_info.empty()) {
    return statuses;
  }

  std::vector<std::string> all_keys;
  all_keys.reserve(block_transfer_info.size());
  for (const BlockTransferInfo& block_info : block_transfer_info) {
    all_keys.emplace_back(build_key(block_info));
  }
  const auto exists = client_ptr_->BatchIsExist(all_keys);

  std::vector<std::string> get_keys;
  std::unordered_map<std::string, std::vector<mooncake::Slice>> get_slices;
  std::vector<size_t> get_positions;
  get_keys.reserve(block_transfer_info.size());
  get_positions.reserve(block_transfer_info.size());
  get_slices.reserve(block_transfer_info.size());
  for (size_t i = 0; i < block_transfer_info.size(); ++i) {
    const bool exists_in_store =
        i < exists.size() && exists[i].has_value() && exists[i].value();
    if (!exists_in_store) {
      continue;
    }
    get_positions.emplace_back(i);
    get_keys.emplace_back(all_keys[i]);
    get_slices.emplace(
        all_keys[i],
        generate_mooncake_slices(block_transfer_info[i].block_type,
                                 block_transfer_info[i].dst_block_id));
  }

  if (get_keys.empty()) {
    return statuses;
  }
  const auto results = client_ptr_->BatchGet(get_keys, get_slices);
  for (size_t i = 0; i < get_keys.size() && i < results.size(); ++i) {
    if (results[i].has_value()) {
      statuses[get_positions[i]] = 1;
    }
  }
  return statuses;
}

uint32_t KVCacheStore::batch_exist(std::vector<std::string>&& keys) {
  if (!is_initialized_) {
    return 0;
  }
  const auto exists = client_ptr_->BatchIsExist(keys);
  return static_cast<uint32_t>(
      std::count_if(exists.begin(), exists.end(), [](const auto& result) {
        return result.has_value() && result.value();
      }));
}

std::vector<mooncake::Slice> KVCacheStore::generate_mooncake_slices(
    BlockType type,
    int32_t block_id) const {
  CHECK(host_kv_caches_ != nullptr);
  const auto cache_it = host_kv_caches_->find(type);
  CHECK(cache_it != host_kv_caches_->end() && cache_it->second != nullptr)
      << "Missing Host cache for BlockType " << static_cast<int32_t>(type);
  const BlockTypeTensorMap tensors =
      cache_it->second->get_block_type_tensors(type);

  std::vector<mooncake::Slice> slices;
  slices.reserve(tensors.size());
  for (const auto& tensor_entry : tensors) {
    const torch::Tensor& tensor = tensor_entry.second;
    CHECK_GE(block_id, 0);
    CHECK_LT(block_id, tensor.size(0));
    torch::Tensor block = tensor[block_id];
    CHECK(block.is_contiguous());
    slices.emplace_back(
        mooncake::Slice{block.data_ptr(),
                        static_cast<size_t>(block.numel()) *
                            static_cast<size_t>(block.element_size())});
  }
  return slices;
}

}  // namespace xllm
