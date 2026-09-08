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

#include "framework/kv_cache_transfer/kv_cache_store.h"

#include <Mooncake/mooncake-store/include/utils.h>
#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "util/hash_util.h"

namespace xllm {
namespace {

void append_key_field(std::string& key, const std::string& value) {
  key.append(std::to_string(value.size()));
  key.push_back(':');
  key.append(value);
  key.push_back(':');
}

}  // namespace

bool KVCacheStore::init(const KVCacheStoreInitConfig& config,
                        HostCacheStoreIndex store_index) {
  CHECK(!is_initialized_) << "KVCacheStore is already initialized.";
  CHECK(!config.model_id.empty())
      << "KVCacheStore requires a target model identity.";
  CHECK_GT(config.tp_size, 0U);
  CHECK_LT(config.tp_rank, config.tp_size);
  config_ = config;
  initialize_store_index(std::move(store_index));

  const std::optional<std::string> device_names =
      get_store_device_names(config_);
  if (config_.protocol == "rdma") {
    LOG(INFO) << "Mooncake RDMA device_names: "
              << device_names.value_or("auto-discover");
  } else if (!config_.rdma_devices.empty()) {
    LOG(WARNING) << "Ignoring store_rdma_devices for Store protocol "
                 << config_.protocol << ".";
  }

  const std::optional<std::vector<MooncakeRegisteredRange>> ranges =
      collect_ranges();
  if (!ranges.has_value()) {
    LOG(ERROR) << "Failed to collect Mooncake Host tensor ranges.";
    return false;
  }
  MooncakeStoreBackendConfig backend_config;
  backend_config.localhost_name = config_.localhost_name;
  backend_config.protocol = config_.protocol;
  backend_config.rdma_devices = device_names.value_or("");
  backend_config.metadata_server = config_.metadata_server;
  backend_config.master_server_address = config_.master_server_address;
  backend_config.replica_num = config_.replica_num;
  backend_ = std::make_unique<MooncakeStoreBackend>();
  if (!backend_->init(backend_config, *ranges)) {
    backend_.reset();
    return false;
  }

  is_initialized_ = true;
  return true;
}

std::optional<std::string> KVCacheStore::get_store_device_names(
    const KVCacheStoreInitConfig& config) {
  if (config.protocol != "rdma" || config.rdma_devices.empty()) {
    return std::nullopt;
  }
  return config.rdma_devices;
}

KVCacheStore::~KVCacheStore() { backend_.reset(); }

void KVCacheStore::initialize_store_index(HostCacheStoreIndex store_index) {
  CHECK(store_index_.empty()) << "KVCacheStore index is already initialized.";
  CHECK(!store_index.empty()) << "KVCacheStore requires Host caches.";
  for (auto& [block_type, entries] : store_index) {
    CHECK(!entries.empty()) << "KVCacheStore block type has no entries.";
    std::unordered_set<uint32_t> cache_handles;
    std::unordered_set<std::string> key_components;
    std::vector<StoreEntry>& store_entries = store_index_[block_type];
    store_entries.reserve(entries.size());
    max_entries_per_type_ = std::max(max_entries_per_type_, entries.size());
    for (HostCacheStoreEntry& entry : entries) {
      CHECK(entry.cache != nullptr) << "KVCacheStore cache must not be null.";
      CHECK(!entry.key_component.empty())
          << "KVCacheStore key component must not be empty.";
      CHECK(cache_handles.emplace(entry.cache_handle).second)
          << "Duplicate KVCacheStore cache handle for BlockType "
          << static_cast<int32_t>(block_type) << ".";
      CHECK(key_components.emplace(entry.key_component).second)
          << "Duplicate KVCacheStore key component for BlockType "
          << static_cast<int32_t>(block_type) << ": " << entry.key_component;
      std::string schema_hash = build_schema_hash(block_type, *entry.cache);
      std::string key_prefix =
          build_key_prefix(entry.key_component, block_type, schema_hash);
      const BlockTypeTensorMap tensors =
          entry.cache->get_block_type_tensors(block_type);
      std::vector<torch::Tensor> block_tensors;
      block_tensors.reserve(tensors.size());
      for (const auto& tensor_entry : tensors) {
        block_tensors.emplace_back(tensor_entry.second);
      }
      store_entries.emplace_back(StoreEntry{entry.cache_handle,
                                            std::move(entry.key_component),
                                            std::move(schema_hash),
                                            entry.cache,
                                            block_type,
                                            std::move(key_prefix),
                                            std::move(block_tensors)});
    }
  }
}

std::string KVCacheStore::build_schema_hash(BlockType block_type,
                                            const KVCache& cache) const {
  const BlockTypeTensorMap tensors = cache.get_block_type_tensors(block_type);
  CHECK(!tensors.empty()) << "Host cache has no tensors for BlockType "
                          << static_cast<int32_t>(block_type);

  std::string cache_schema = config_.enable_mla
                                 ? "parallel=mla"
                                 : "tp=" + std::to_string(config_.tp_size);
  cache_schema.append("|type=");
  cache_schema.append(std::to_string(static_cast<int32_t>(block_type)));
  int64_t host_blocks = -1;
  for (const auto& [role, tensor] : tensors) {
    CHECK(tensor.defined() && tensor.dim() > 0 && tensor.is_contiguous());
    if (host_blocks < 0) {
      host_blocks = tensor.size(0);
    } else {
      CHECK_EQ(host_blocks, tensor.size(0));
    }
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
  }
  const XXH3Key schema_hash = hash_string(cache_schema);
  return std::string(reinterpret_cast<const char*>(schema_hash.data),
                     sizeof(schema_hash.data));
}

std::string KVCacheStore::build_key_prefix(
    const std::string& key_component,
    BlockType block_type,
    const std::string& schema_hash) const {
  std::string prefix = "xllm-kv-v3:";
  append_key_field(prefix, config_.model_id);
  append_key_field(prefix, key_component);
  if (config_.enable_mla) {
    prefix.append("mla:");
  } else {
    prefix.append(std::to_string(config_.tp_size));
    prefix.push_back(':');
  }
  prefix.append(std::to_string(static_cast<int32_t>(block_type)));
  prefix.push_back(':');
  if (!config_.enable_mla) {
    prefix.append(std::to_string(config_.tp_rank));
    prefix.push_back(':');
  }
  prefix.append(schema_hash);
  return prefix;
}

std::string KVCacheStore::build_key(const StoreEntry& entry,
                                    const BlockTransferInfo& block_info) const {
  CHECK(entry.block_type == block_info.block_type);
  std::string key = entry.key_prefix;
  key.append(reinterpret_cast<const char*>(block_info.hash_key),
             XXH3_128BITS_HASH_VALUE_LEN);
  return key;
}

std::vector<KVCacheStore::PhysicalRequest> KVCacheStore::build_requests(
    Slice<BlockTransferInfo>& block_transfer_info) const {
  std::vector<PhysicalRequest> requests;
  requests.reserve(block_transfer_info.size() * max_entries_per_type_);
  for (size_t logical_index = 0; logical_index < block_transfer_info.size();
       ++logical_index) {
    const BlockTransferInfo& block_info = block_transfer_info[logical_index];
    const auto entries_it = store_index_.find(block_info.block_type);
    if (entries_it == store_index_.end()) {
      LOG(ERROR) << "KVCacheStore has no entry for BlockType "
                 << static_cast<int32_t>(block_info.block_type) << ".";
      continue;
    }
    for (const StoreEntry& entry : entries_it->second) {
      requests.emplace_back(
          PhysicalRequest{logical_index, &entry, build_key(entry, block_info)});
    }
  }
  return requests;
}

std::vector<KVCacheStore::RequestGroup> KVCacheStore::group_requests(
    const std::vector<PhysicalRequest>& requests) {
  std::vector<RequestGroup> groups;
  groups.reserve(requests.size());
  std::unordered_map<std::string, size_t> group_indices;
  group_indices.reserve(requests.size());
  for (size_t request_index = 0; request_index < requests.size();
       ++request_index) {
    const PhysicalRequest& request = requests[request_index];
    const auto [group_it, inserted] =
        group_indices.emplace(request.key, groups.size());
    if (inserted) {
      groups.emplace_back(RequestGroup{request.key, {}});
    }
    groups[group_it->second].request_indices.emplace_back(request_index);
  }
  return groups;
}

KVCacheStore::GroupedRequests KVCacheStore::build_grouped_requests(
    Slice<BlockTransferInfo>& block_transfer_info) const {
  GroupedRequests grouped;
  grouped.requests = build_requests(block_transfer_info);
  grouped.groups = group_requests(grouped.requests);
  return grouped;
}

std::vector<uint8_t> KVCacheStore::aggregate_results(
    size_t logical_count,
    const std::vector<PhysicalRequest>& requests,
    const std::vector<uint8_t>& physical_results) {
  std::vector<uint32_t> required_counts(logical_count, 0);
  std::vector<uint32_t> success_counts(logical_count, 0);
  for (size_t request_index = 0; request_index < requests.size();
       ++request_index) {
    const size_t logical_index = requests[request_index].logical_index;
    CHECK_LT(logical_index, logical_count);
    ++required_counts[logical_index];
    if (request_index < physical_results.size() &&
        physical_results[request_index] != 0) {
      ++success_counts[logical_index];
    }
  }

  std::vector<uint8_t> logical_results(logical_count, /*value=*/0);
  for (size_t logical_index = 0; logical_index < logical_count;
       ++logical_index) {
    if (required_counts[logical_index] > 0 &&
        required_counts[logical_index] == success_counts[logical_index]) {
      logical_results[logical_index] = 1;
    }
  }
  return logical_results;
}

uint32_t KVCacheStore::batch_put(
    Slice<BlockTransferInfo>& block_transfer_info) {
  if (!is_initialized_ || block_transfer_info.empty()) {
    return 0;
  }
  if (config_.enable_mla && config_.tp_rank != 0U) {
    VLOG(1) << "KVCacheStore skips MLA remote put on non-zero rank: tp_rank="
            << config_.tp_rank;
    return static_cast<uint32_t>(block_transfer_info.size());
  }

  const GroupedRequests grouped = build_grouped_requests(block_transfer_info);
  const std::vector<PhysicalRequest>& requests = grouped.requests;
  const std::vector<RequestGroup>& groups = grouped.groups;
  if (groups.empty()) {
    return 0;
  }

  std::vector<uint8_t> physical_results(requests.size(), /*value=*/0);
  std::vector<std::string> put_keys;
  std::vector<MooncakeMultiBuffer> put_buffers;
  std::vector<size_t> put_group_indices;
  put_keys.reserve(groups.size());
  put_buffers.reserve(groups.size());
  put_group_indices.reserve(groups.size());
  for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
    const RequestGroup& group = groups[group_index];
    const PhysicalRequest& request = requests[group.request_indices.front()];
    const std::optional<MooncakeMultiBuffer> buffer = build_multi_buffer(
        *request.entry,
        block_transfer_info[request.logical_index].dst_block_id);
    if (!buffer.has_value()) {
      continue;
    }
    put_keys.emplace_back(group.key);
    put_buffers.emplace_back(*buffer);
    put_group_indices.emplace_back(group_index);
  }

  if (!put_keys.empty() && backend_ != nullptr) {
    const std::vector<uint8_t> results =
        backend_->batch_put(put_keys, put_buffers);
    for (size_t result_index = 0; result_index < put_group_indices.size() &&
                                  result_index < results.size();
         ++result_index) {
      if (results[result_index] == 0) {
        continue;
      }
      for (size_t request_index :
           groups[put_group_indices[result_index]].request_indices) {
        physical_results[request_index] = 1;
      }
    }
  }

  const std::vector<uint8_t> logical_results =
      aggregate_results(block_transfer_info.size(), requests, physical_results);
  return static_cast<uint32_t>(std::count(
      logical_results.begin(), logical_results.end(), static_cast<uint8_t>(1)));
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

  const std::vector<PhysicalRequest> requests =
      build_requests(block_transfer_info);
  if (requests.empty()) {
    return statuses;
  }

  std::unordered_set<std::string> unique_keys;
  unique_keys.reserve(requests.size());
  for (const PhysicalRequest& request : requests) {
    if (!unique_keys.emplace(request.key).second) {
      LOG(ERROR) << "Duplicate KVCacheStore BatchGet key in one request.";
      return statuses;
    }
  }

  std::vector<uint8_t> physical_results(requests.size(), /*value=*/0);
  for (size_t request_index = 0; request_index < requests.size();
       ++request_index) {
    const PhysicalRequest& request = requests[request_index];
    const std::optional<MooncakeMultiBuffer> buffer = build_multi_buffer(
        *request.entry,
        block_transfer_info[request.logical_index].dst_block_id);
    if (!buffer.has_value() || backend_ == nullptr) {
      continue;
    }
    physical_results[request_index] =
        backend_->get(request.key, *buffer) ? 1 : 0;
  }
  return aggregate_results(
      block_transfer_info.size(), requests, physical_results);
}

std::optional<MooncakeMultiBuffer> KVCacheStore::build_multi_buffer(
    const StoreEntry& entry,
    int32_t block_id) const {
  if (entry.block_tensors.empty()) {
    LOG(ERROR) << "Missing Host cache for BlockType "
               << static_cast<int32_t>(entry.block_type);
    return std::nullopt;
  }

  MooncakeMultiBuffer buffer;
  buffer.addresses.reserve(entry.block_tensors.size());
  buffer.sizes.reserve(entry.block_tensors.size());
  for (const torch::Tensor& tensor : entry.block_tensors) {
    if (block_id < 0 || block_id >= tensor.size(0)) {
      LOG(ERROR) << "Invalid Host cache block id=" << block_id;
      return std::nullopt;
    }
    torch::Tensor block = tensor[block_id];
    if (!block.is_contiguous() || block.numel() <= 0 ||
        static_cast<uint64_t>(block.numel()) >
            std::numeric_limits<size_t>::max() /
                static_cast<size_t>(block.element_size())) {
      LOG(ERROR) << "Invalid Host cache tensor for BlockType "
                 << static_cast<int32_t>(entry.block_type);
      return std::nullopt;
    }
    buffer.addresses.emplace_back(block.data_ptr());
    buffer.sizes.emplace_back(static_cast<size_t>(block.numel()) *
                              static_cast<size_t>(block.element_size()));
  }
  return buffer;
}

std::optional<std::vector<MooncakeRegisteredRange>>
KVCacheStore::collect_ranges() const {
  std::vector<MooncakeRegisteredRange> ranges;
  for (const auto& [block_type, entries] : store_index_) {
    for (const StoreEntry& entry : entries) {
      const BlockTypeTensorMap tensors =
          entry.cache->get_block_type_tensors(block_type);
      for (const auto& tensor_entry : tensors) {
        const torch::Tensor& tensor = tensor_entry.second;
        if (!tensor.defined() || tensor.numel() <= 0 ||
            static_cast<uint64_t>(tensor.numel()) >
                std::numeric_limits<size_t>::max() /
                    static_cast<size_t>(tensor.element_size())) {
          LOG(ERROR) << "Invalid Host tensor registration range.";
          return std::nullopt;
        }
        ranges.emplace_back(MooncakeRegisteredRange{
            tensor.data_ptr(),
            static_cast<size_t>(tensor.numel()) *
                static_cast<size_t>(tensor.element_size())});
      }
    }
  }
  return ranges;
}

}  // namespace xllm
