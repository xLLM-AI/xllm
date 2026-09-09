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

#include "framework/kv_cache_transfer/mooncake_store_backend.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace xllm {

MooncakeStoreBackend::~MooncakeStoreBackend() { unregister_ranges(); }

bool MooncakeStoreBackend::init(
    const MooncakeStoreBackendConfig& config,
    const std::vector<MooncakeRegisteredRange>& ranges) {
  const std::optional<std::vector<MooncakeRegisteredRange>> unique =
      unique_ranges(ranges);
  if (!unique.has_value()) {
    LOG(ERROR) << "Invalid Mooncake Host tensor registration ranges.";
    return false;
  }

  try {
    client_ptr_ = mooncake::RealClient::create();
    const int setup_result =
        client_ptr_->setup_real(config.localhost_name,
                                config.metadata_server,
                                /*global_segment_size=*/0,
                                /*local_buffer_size=*/0,
                                config.protocol,
                                config.rdma_devices,
                                config.master_server_address,
                                /*transfer_engine=*/nullptr,
                                /*ipc_socket_path=*/"",
                                /*enable_ssd_offload=*/false,
                                /*ssd_offload_path=*/"",
                                /*tenant_id=*/"default");
    if (setup_result != 0) {
      LOG(ERROR) << "Failed to initialize Mooncake RealClient: result="
                 << setup_result;
      client_ptr_.reset();
      return false;
    }
    rep_config_.replica_num = config.replica_num;
    registered_ranges_.reserve(unique->size());
    for (const MooncakeRegisteredRange& range : *unique) {
      const int register_result =
          client_ptr_->register_buffer(range.address, range.bytes);
      if (register_result != 0) {
        LOG(ERROR) << "Failed to register Mooncake Host tensor: result="
                   << register_result;
        unregister_ranges();
        client_ptr_.reset();
        return false;
      }
      registered_ranges_.emplace_back(range);
    }
  } catch (const std::exception& error) {
    LOG(ERROR) << "Mooncake RealClient initialization failed: " << error.what();
    unregister_ranges();
    client_ptr_.reset();
    return false;
  } catch (...) {
    LOG(ERROR) << "Mooncake RealClient initialization failed unexpectedly.";
    unregister_ranges();
    client_ptr_.reset();
    return false;
  }
  return true;
}

std::vector<uint8_t> MooncakeStoreBackend::batch_put(
    const std::vector<std::string>& keys,
    const std::vector<MooncakeMultiBuffer>& buffers) {
  std::vector<uint8_t> statuses(keys.size(), /*value=*/0);
  if (client_ptr_ == nullptr || keys.size() != buffers.size()) {
    LOG(ERROR) << "Invalid Mooncake multi-buffer Put request.";
    return statuses;
  }

  std::vector<std::vector<void*>> all_buffers;
  std::vector<std::vector<size_t>> all_sizes;
  all_buffers.reserve(buffers.size());
  all_sizes.reserve(buffers.size());
  for (const MooncakeMultiBuffer& buffer : buffers) {
    all_buffers.emplace_back(buffer.addresses);
    all_sizes.emplace_back(buffer.sizes);
  }

  try {
    const std::vector<int> results = client_ptr_->batch_put_from_multi_buffers(
        keys, all_buffers, all_sizes, rep_config_);
    if (results.size() > keys.size()) {
      LOG(ERROR) << "Mooncake Put returned extra result items.";
    }
    const size_t result_count = std::min(keys.size(), results.size());
    for (size_t result_index = 0; result_index < result_count; ++result_index) {
      statuses[result_index] = put_succeeded(results[result_index]) ? 1 : 0;
    }
  } catch (const std::exception& error) {
    LOG(ERROR) << "Mooncake Put failed: " << error.what();
  } catch (...) {
    LOG(ERROR) << "Mooncake Put failed unexpectedly.";
  }
  return statuses;
}

bool MooncakeStoreBackend::get(const std::string& key,
                               const MooncakeMultiBuffer& buffer) {
  if (client_ptr_ == nullptr ||
      buffer.addresses.size() != buffer.sizes.size()) {
    LOG(ERROR) << "Invalid Mooncake multi-buffer Get request for key=" << key;
    return false;
  }

  int64_t expected_bytes = 0;
  for (const size_t size : buffer.sizes) {
    if (size > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        expected_bytes >
            std::numeric_limits<int64_t>::max() - static_cast<int64_t>(size)) {
      LOG(ERROR) << "Mooncake Get size overflows for key=" << key;
      return false;
    }
    expected_bytes += static_cast<int64_t>(size);
  }
  if (expected_bytes > std::numeric_limits<int32_t>::max()) {
    LOG(ERROR) << "Mooncake Get exceeds int32 result range for key=" << key;
    return false;
  }

  try {
    const std::vector<std::string> keys = {key};
    const std::vector<std::vector<void*>> all_buffers = {buffer.addresses};
    const std::vector<std::vector<size_t>> all_sizes = {buffer.sizes};
    const std::vector<int> results =
        client_ptr_->batch_get_into_multi_buffers(keys,
                                                  all_buffers,
                                                  all_sizes,
                                                  /*prefer_same_node=*/false);
    if (results.size() > 1) {
      LOG(ERROR) << "Mooncake Get returned extra result items for key=" << key;
    }
    return get_succeeded(expected_bytes, results);
  } catch (const std::exception& error) {
    LOG(ERROR) << "Mooncake Get failed for key=" << key << ": " << error.what();
  } catch (...) {
    LOG(ERROR) << "Mooncake Get failed unexpectedly for key=" << key;
  }
  return false;
}

std::optional<std::vector<MooncakeRegisteredRange>>
MooncakeStoreBackend::unique_ranges(
    const std::vector<MooncakeRegisteredRange>& ranges) {
  std::vector<MooncakeRegisteredRange> unique;
  unique.reserve(ranges.size());
  for (const MooncakeRegisteredRange& range : ranges) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(range.address);
    if (range.address == nullptr || range.bytes == 0 ||
        range.bytes > std::numeric_limits<uintptr_t>::max() - address) {
      return std::nullopt;
    }
    const auto same_address = std::find_if(
        unique.begin(),
        unique.end(),
        [address](const MooncakeRegisteredRange& item) {
          return reinterpret_cast<uintptr_t>(item.address) == address;
        });
    if (same_address == unique.end()) {
      unique.emplace_back(range);
    } else if (same_address->bytes != range.bytes) {
      return std::nullopt;
    }
  }

  std::vector<MooncakeRegisteredRange> ordered = unique;
  std::sort(ordered.begin(),
            ordered.end(),
            [](const MooncakeRegisteredRange& lhs,
               const MooncakeRegisteredRange& rhs) {
              return reinterpret_cast<uintptr_t>(lhs.address) <
                     reinterpret_cast<uintptr_t>(rhs.address);
            });
  for (size_t range_index = 1; range_index < ordered.size(); ++range_index) {
    const MooncakeRegisteredRange& previous = ordered[range_index - 1];
    const uintptr_t previous_end =
        reinterpret_cast<uintptr_t>(previous.address) + previous.bytes;
    if (reinterpret_cast<uintptr_t>(ordered[range_index].address) <
        previous_end) {
      return std::nullopt;
    }
  }
  return unique;
}

bool MooncakeStoreBackend::get_succeeded(int64_t expected_bytes,
                                         const std::vector<int>& results) {
  if (expected_bytes < 0 ||
      expected_bytes > std::numeric_limits<int32_t>::max() || results.empty()) {
    return false;
  }
  const int32_t result = static_cast<int32_t>(results.front());
  return result >= 0 && result == static_cast<int32_t>(expected_bytes);
}

bool MooncakeStoreBackend::put_succeeded(int result) {
  return result == 0 ||
         result == static_cast<int>(mooncake::ErrorCode::OBJECT_ALREADY_EXISTS);
}

void MooncakeStoreBackend::unregister_ranges() {
  if (client_ptr_ == nullptr) {
    return;
  }
  for (auto range_it = registered_ranges_.rbegin();
       range_it != registered_ranges_.rend();
       ++range_it) {
    try {
      const int unregister_result =
          client_ptr_->unregister_buffer(range_it->address);
      if (unregister_result != 0) {
        LOG(WARNING) << "Failed to unregister Mooncake Host tensor: result="
                     << unregister_result;
      }
    } catch (const std::exception& error) {
      LOG(WARNING) << "Failed to unregister Mooncake Host tensor: "
                   << error.what();
    } catch (...) {
      LOG(WARNING) << "Failed to unregister Mooncake Host tensor unexpectedly.";
    }
  }
  registered_ranges_.clear();
}

}  // namespace xllm
