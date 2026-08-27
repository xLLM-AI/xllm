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

#include "framework/kv_cache_transfer/host_transfer/transfer.h"

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_set>
#include <utility>

#include "framework/kv_cache_transfer/host_transfer/basic_transfer.h"
#include "platform/layer_synchronizer.h"

namespace xllm {
namespace {

using BlockKey = std::pair<int32_t, int64_t>;

class BlockKeyHash final {
 public:
  size_t operator()(const BlockKey& key) const {
    const size_t group_hash = std::hash<int32_t>{}(key.first);
    const size_t block_hash = std::hash<int64_t>{}(key.second);
    return group_hash ^
           (block_hash + 0x9e3779b9U + (group_hash << 6U) + (group_hash >> 2U));
  }
};

}  // namespace

HostKVTransfer::HostKVTransfer(HostKVLayout layout)
    : layout_(std::move(layout)) {}

bool HostKVTransfer::load(const HostKVRequest& request,
                          const HostKVLoadHandle& handle) {
  if (!valid_request(request, /*is_load=*/true) || !valid_handle(handle)) {
    if (handle.synchronizer != nullptr) {
      handle.synchronizer->abort();
    }
    return false;
  }
  const bool success = load_impl(request, handle);
  if (!success) {
    handle.synchronizer->abort();
  }
  return success;
}

bool HostKVTransfer::offload(const HostKVRequest& request) {
  if (!valid_request(request, /*is_load=*/false)) {
    return false;
  }
  return offload_impl(request);
}

bool HostKVTransfer::valid_request(const HostKVRequest& request,
                                   bool is_load) const {
  if (request.mappings.empty()) {
    LOG(ERROR) << "Host KV request must not be empty.";
    return false;
  }

  std::unordered_set<BlockKey, BlockKeyHash> destinations;
  destinations.reserve(request.mappings.size());
  for (const HostKVMapping& mapping : request.mappings) {
    const HostKVGroupLayout* group = layout_.find_group(mapping.group_id);
    if (group == nullptr || mapping.host_block_id < 0 ||
        mapping.host_block_id >= layout_.host_block_count(mapping.group_id) ||
        mapping.device_block_id < 0 ||
        mapping.device_block_id >=
            layout_.device_block_count(mapping.group_id)) {
      LOG(ERROR) << "Host KV request contains an unknown group or block.";
      return false;
    }

    const int64_t destination =
        is_load ? mapping.device_block_id : mapping.host_block_id;
    if (!destinations.emplace(mapping.group_id, destination).second) {
      LOG(ERROR) << "Host KV request contains a duplicate destination.";
      return false;
    }
  }
  return true;
}

bool HostKVTransfer::valid_handle(const HostKVLoadHandle& handle) const {
  if (handle.synchronizer == nullptr) {
    LOG(ERROR) << "Host KV load requires a synchronizer.";
    return false;
  }
  if (handle.synchronizer->size() != load_event_count() ||
      handle.layers_per_event != layers_per_event()) {
    LOG(ERROR) << "Host KV load handle does not match the transfer strategy.";
    return false;
  }
  return true;
}

std::unique_ptr<HostKVTransfer> create_host_kv_transfer(
    HostKVLayout layout,
    const Device& device,
    const Stream& compute_stream,
    const HostKVTransferConfig& config) {
  return std::make_unique<BasicHostKVTransfer>(
      std::move(layout), device, compute_stream, config.layer_copy_batches);
}

}  // namespace xllm
