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

#include <Mooncake/mooncake-store/include/client_service.h>
#include <Mooncake/mooncake-store/include/utils.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "util/hash_util.h"

namespace xllm {
namespace {

std::string participant_name(CacheParticipant participant) {
  switch (participant) {
    case CacheParticipant::TARGET:
      return "TARGET";
    case CacheParticipant::DRAFT:
      return "DRAFT";
  }
  LOG(FATAL) << "Unsupported cache participant: "
             << static_cast<int32_t>(participant);
  return "UNKNOWN";
}

void append_key_field(std::string& key, const std::string& value) {
  key.append(std::to_string(value.size()));
  key.push_back(':');
  key.append(value);
  key.push_back(':');
}

struct ComponentRequest final {
  size_t logical_index = 0;
  const HostCacheComponentSchema* component = nullptr;
  std::string key;
};

std::vector<mooncake::Slice> generate_mooncake_slices(
    const HostCacheSliceProvider* slice_provider,
    const HostCacheComponentSchema& component,
    int32_t block_id) {
  CHECK(slice_provider != nullptr);
  const std::vector<torch::Tensor> tensors = slice_provider->component_tensors(
      component.participant, component.block_type, block_id);
  CHECK(!tensors.empty()) << "Missing Host cache slices for participant="
                          << participant_name(component.participant)
                          << ", type="
                          << static_cast<int32_t>(component.block_type);

  std::vector<mooncake::Slice> slices;
  slices.reserve(tensors.size());
  for (const torch::Tensor& tensor : tensors) {
    CHECK(tensor.defined() && tensor.is_contiguous());
    slices.emplace_back(
        mooncake::Slice{tensor.data_ptr(),
                        static_cast<size_t>(tensor.numel()) *
                            static_cast<size_t>(tensor.element_size())});
  }
  return slices;
}

bool copy_slices(const std::vector<mooncake::Slice>& source,
                 const std::vector<mooncake::Slice>& destination) {
  if (source.size() != destination.size()) {
    return false;
  }
  for (size_t index = 0; index < source.size(); ++index) {
    if (source[index].size != destination[index].size ||
        source[index].ptr == nullptr || destination[index].ptr == nullptr) {
      return false;
    }
    if (source[index].ptr != destination[index].ptr) {
      std::memcpy(
          destination[index].ptr, source[index].ptr, source[index].size);
    }
  }
  return true;
}

}  // namespace

struct KVCacheStore::Impl final {
  mooncake::ReplicateConfig rep_config;
  std::shared_ptr<mooncake::Client> client;
};

KVCacheStore::KVCacheStore() : impl_(std::make_unique<Impl>()) {}

bool KVCacheStore::init(const KVCacheStoreInitConfig& config,
                        const HostCacheSliceProvider* slice_provider) {
  CHECK(!is_initialized_) << "KVCacheStore is already initialized.";
  CHECK(slice_provider != nullptr)
      << "KVCacheStore requires a Host cache slice provider.";
  config_ = config;
  slice_provider_ = slice_provider;
  components_ = slice_provider_->store_components();
  CHECK(!components_.empty())
      << "KVCacheStore requires at least one Host cache component.";
  std::sort(components_.begin(),
            components_.end(),
            [](const HostCacheComponentSchema& lhs,
               const HostCacheComponentSchema& rhs) {
              if (lhs.participant != rhs.participant) {
                return static_cast<int32_t>(lhs.participant) <
                       static_cast<int32_t>(rhs.participant);
              }
              return static_cast<int32_t>(lhs.block_type) <
                     static_cast<int32_t>(rhs.block_type);
            });

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
  impl_->client = client.value();
  impl_->rep_config.replica_num = config_.replica_num;

  std::unordered_set<std::string> component_ids;
  for (const HostCacheComponentSchema& component : components_) {
    CHECK_GT(component.tp_size, 0u);
    CHECK_LT(component.tp_rank, component.tp_size);
    CHECK(!component.model_identity.empty());
    CHECK(!component.schema_fingerprint.empty());
    const std::string component_id =
        participant_name(component.participant) + ":" +
        std::to_string(static_cast<int32_t>(component.block_type));
    CHECK(component_ids.emplace(component_id).second)
        << "Duplicate Host cache Store component: " << component_id;

    const std::vector<torch::Tensor> tensors =
        slice_provider_->component_storage_tensors(component.participant,
                                                   component.block_type);
    CHECK(!tensors.empty())
        << "Host cache component has no tensors: " << component_id;
    int64_t host_blocks = -1;
    size_t slot_bytes = 0;
    for (const torch::Tensor& tensor : tensors) {
      CHECK(tensor.defined() && tensor.dim() > 0 && tensor.is_contiguous());
      if (host_blocks < 0) {
        host_blocks = tensor.size(0);
      } else {
        CHECK_EQ(host_blocks, tensor.size(0));
      }
      slot_bytes += static_cast<size_t>(tensor[0].numel()) *
                    static_cast<size_t>(tensor.element_size());
      if (config_.protocol != "rdma") {
        continue;
      }
      void* address = tensor.data_ptr();
      const size_t bytes = static_cast<size_t>(tensor.numel()) *
                           static_cast<size_t>(tensor.element_size());
      auto result =
          impl_->client->RegisterLocalMemory(address,
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
    LOG(INFO) << "KVCacheStore init OK: type="
              << static_cast<int32_t>(component.block_type)
              << ", participant=" << participant_name(component.participant)
              << ", host_blocks=" << host_blocks
              << ", slot_bytes=" << slot_bytes
              << ", protocol=" << config_.protocol;
  }

  is_initialized_ = true;
  return true;
}

KVCacheStore::~KVCacheStore() {
  if (impl_->client != nullptr) {
    for (void* address : registered_addresses_) {
      auto result = impl_->client->unregisterLocalMemory(
          address, /*update_metadata=*/false);
      if (!result.has_value()) {
        LOG(WARNING) << "Failed to unregister Mooncake Host tensor: "
                     << toString(result.error());
      }
    }
    impl_->client.reset();
  }
}

std::string KVCacheStore::build_component_key(
    const HostCacheComponentSchema& component,
    const BlockTransferInfo& block_info) const {
  std::string key = "xllm-kv-v3:";
  append_key_field(key, config_.model_id);
  append_key_field(key, participant_name(component.participant));
  append_key_field(key, component.model_identity);
  append_key_field(key, "composite-host-v1");
  key.append(std::to_string(component.tp_size));
  key.push_back(':');
  key.append(std::to_string(component.tp_rank));
  key.push_back(':');
  key.append(std::to_string(static_cast<int32_t>(component.block_type)));
  key.push_back(':');
  append_key_field(key, component.schema_fingerprint);
  key.append(reinterpret_cast<const char*>(block_info.hash_key),
             XXH3_128BITS_HASH_VALUE_LEN);
  return key;
}

std::vector<const HostCacheComponentSchema*> KVCacheStore::required_components(
    BlockType block_type) const {
  std::vector<const HostCacheComponentSchema*> components;
  for (const HostCacheComponentSchema& component : components_) {
    if (component.block_type == block_type) {
      components.emplace_back(&component);
    }
  }
  return components;
}

uint32_t KVCacheStore::batch_put(
    Slice<BlockTransferInfo>& block_transfer_info) {
  if (!is_initialized_ || block_transfer_info.empty()) {
    return 0;
  }

  std::vector<ComponentRequest> requests;
  for (size_t logical_index = 0; logical_index < block_transfer_info.size();
       ++logical_index) {
    const BlockTransferInfo& block_info = block_transfer_info[logical_index];
    const std::vector<const HostCacheComponentSchema*> components =
        required_components(block_info.block_type);
    for (const HostCacheComponentSchema* component : components) {
      requests.push_back({logical_index,
                          component,
                          build_component_key(*component, block_info)});
    }
  }
  if (requests.empty()) {
    return 0;
  }

  std::vector<std::string> all_keys;
  all_keys.reserve(requests.size());
  for (const ComponentRequest& request : requests) {
    all_keys.emplace_back(request.key);
  }
  const auto exists = impl_->client->BatchIsExist(all_keys);

  std::vector<std::string> put_keys;
  std::vector<std::vector<mooncake::Slice>> put_slices;
  std::vector<std::vector<size_t>> put_request_groups;
  std::unordered_map<std::string, size_t> put_key_indices;
  std::vector<uint32_t> completed_components(block_transfer_info.size(), 0);
  std::vector<uint32_t> required_component_counts(block_transfer_info.size(),
                                                  0);
  for (size_t request_index = 0; request_index < requests.size();
       ++request_index) {
    const ComponentRequest& request = requests[request_index];
    ++required_component_counts[request.logical_index];
    const bool already_exists = request_index < exists.size() &&
                                exists[request_index].has_value() &&
                                exists[request_index].value();
    if (already_exists) {
      ++completed_components[request.logical_index];
      continue;
    }
    const auto [put_it, inserted] =
        put_key_indices.emplace(request.key, put_keys.size());
    if (inserted) {
      put_keys.emplace_back(request.key);
      put_slices.emplace_back(generate_mooncake_slices(
          slice_provider_,
          *request.component,
          block_transfer_info[request.logical_index].dst_block_id));
      put_request_groups.emplace_back();
    }
    put_request_groups[put_it->second].emplace_back(request_index);
  }

  if (!put_keys.empty()) {
    const auto results =
        impl_->client->BatchPut(put_keys, put_slices, impl_->rep_config);
    for (size_t i = 0; i < put_request_groups.size() && i < results.size();
         ++i) {
      if (!results[i].has_value()) {
        continue;
      }
      for (size_t request_index : put_request_groups[i]) {
        ++completed_components[requests[request_index].logical_index];
      }
    }
  }

  uint32_t success_count = 0;
  for (size_t logical_index = 0; logical_index < block_transfer_info.size();
       ++logical_index) {
    if (required_component_counts[logical_index] > 0 &&
        completed_components[logical_index] ==
            required_component_counts[logical_index]) {
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

  std::vector<ComponentRequest> requests;
  std::vector<uint32_t> required_component_counts(block_transfer_info.size(),
                                                  0);
  for (size_t logical_index = 0; logical_index < block_transfer_info.size();
       ++logical_index) {
    const BlockTransferInfo& block_info = block_transfer_info[logical_index];
    const std::vector<const HostCacheComponentSchema*> components =
        required_components(block_info.block_type);
    for (const HostCacheComponentSchema* component : components) {
      requests.push_back({logical_index,
                          component,
                          build_component_key(*component, block_info)});
      ++required_component_counts[logical_index];
    }
  }
  if (requests.empty()) {
    return statuses;
  }

  std::vector<std::string> all_keys;
  all_keys.reserve(requests.size());
  for (const ComponentRequest& request : requests) {
    all_keys.emplace_back(request.key);
  }
  const auto exists = impl_->client->BatchIsExist(all_keys);

  std::vector<uint32_t> existing_component_counts(block_transfer_info.size(),
                                                  0);
  for (size_t request_index = 0; request_index < requests.size();
       ++request_index) {
    if (request_index < exists.size() && exists[request_index].has_value() &&
        exists[request_index].value()) {
      ++existing_component_counts[requests[request_index].logical_index];
    }
  }

  std::vector<std::string> get_keys;
  std::unordered_map<std::string, std::vector<mooncake::Slice>> get_slices;
  std::vector<std::vector<size_t>> get_request_groups;
  std::unordered_map<std::string, size_t> get_key_indices;
  for (size_t request_index = 0; request_index < requests.size();
       ++request_index) {
    const ComponentRequest& request = requests[request_index];
    const size_t logical_index = request.logical_index;
    if (required_component_counts[logical_index] == 0 ||
        existing_component_counts[logical_index] !=
            required_component_counts[logical_index]) {
      continue;
    }
    const auto [get_it, inserted] =
        get_key_indices.emplace(request.key, get_keys.size());
    if (inserted) {
      get_keys.emplace_back(request.key);
      get_slices.emplace(request.key,
                         generate_mooncake_slices(
                             slice_provider_,
                             *request.component,
                             block_transfer_info[logical_index].dst_block_id));
      get_request_groups.emplace_back();
    }
    get_request_groups[get_it->second].emplace_back(request_index);
  }
  if (get_keys.empty()) {
    return statuses;
  }

  const auto results = impl_->client->BatchGet(get_keys, get_slices);
  std::vector<uint32_t> fetched_component_counts(block_transfer_info.size(), 0);
  for (size_t i = 0; i < get_request_groups.size() && i < results.size(); ++i) {
    if (!results[i].has_value()) {
      continue;
    }
    const std::vector<mooncake::Slice>& source_slices =
        get_slices.at(get_keys[i]);
    for (size_t request_index : get_request_groups[i]) {
      const ComponentRequest& request = requests[request_index];
      const std::vector<mooncake::Slice> destination_slices =
          generate_mooncake_slices(
              slice_provider_,
              *request.component,
              block_transfer_info[request.logical_index].dst_block_id);
      if (copy_slices(source_slices, destination_slices)) {
        ++fetched_component_counts[request.logical_index];
      }
    }
  }
  for (size_t logical_index = 0; logical_index < block_transfer_info.size();
       ++logical_index) {
    if (required_component_counts[logical_index] > 0 &&
        fetched_component_counts[logical_index] ==
            required_component_counts[logical_index]) {
      statuses[logical_index] = 1;
    }
  }
  return statuses;
}

uint32_t KVCacheStore::batch_exist(std::vector<std::string>&& keys) {
  if (!is_initialized_) {
    return 0;
  }
  const auto exists = impl_->client->BatchIsExist(keys);
  return static_cast<uint32_t>(
      std::count_if(exists.begin(), exists.end(), [](const auto& result) {
        return result.has_value() && result.value();
      }));
}

}  // namespace xllm
