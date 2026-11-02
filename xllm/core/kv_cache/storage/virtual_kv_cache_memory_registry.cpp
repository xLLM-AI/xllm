/* Copyright 2026 The xLLM Authors.

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

#include "core/kv_cache/storage/virtual_kv_cache_memory_registry.h"

#include <glog/logging.h>

#include <algorithm>
#include <numeric>

namespace xllm {

void VirtualKVCacheMemoryRegistry::init(const torch::Device& device,
                                        size_t page_size) {
  CHECK_GT(page_size, 0);
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) {
    CHECK_EQ(device_, device) << "KV cache registry device mismatch";
    CHECK_EQ(page_size_, page_size) << "KV cache registry page size mismatch";
    return;
  }
  device_ = device;
  page_size_ = page_size;
  initialized_ = true;
}

std::vector<torch::Tensor> VirtualKVCacheMemoryRegistry::create_k_tensors(
    const std::string& model_id,
    const std::vector<int64_t>& dims,
    torch::ScalarType dtype,
    int64_t num_layers,
    const std::vector<bool>& layer_owned) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(initialized_) << "KV cache registry is not initialized";
  CHECK(!model_id.empty());
  return regions_[model_id].create_k_tensors(
      dims, dtype, num_layers, device_, page_size_, layer_owned);
}

std::vector<torch::Tensor> VirtualKVCacheMemoryRegistry::create_v_tensors(
    const std::string& model_id,
    const std::vector<int64_t>& dims,
    torch::ScalarType dtype,
    int64_t num_layers,
    const std::vector<bool>& layer_owned) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(initialized_) << "KV cache registry is not initialized";
  CHECK(!model_id.empty());
  return regions_[model_id].create_v_tensors(
      dims, dtype, num_layers, device_, page_size_, layer_owned);
}

std::vector<torch::Tensor> VirtualKVCacheMemoryRegistry::create_index_tensors(
    const std::string& model_id,
    const std::vector<int64_t>& dims,
    torch::ScalarType dtype,
    int64_t num_layers,
    const std::vector<bool>& layer_owned) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(initialized_) << "KV cache registry is not initialized";
  CHECK(!model_id.empty());
  return regions_[model_id].create_index_tensors(
      dims, dtype, num_layers, device_, page_size_, layer_owned);
}

bool VirtualKVCacheMemoryRegistry::bind_mapping_group(
    const std::string& group_id,
    std::vector<std::string> member_ids) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (member_ids.empty() ||
      std::find(member_ids.begin(), member_ids.end(), group_id) ==
          member_ids.end()) {
    LOG(ERROR) << "KV mapping group must contain its group ID";
    return false;
  }
  std::sort(member_ids.begin(), member_ids.end());
  if (std::adjacent_find(member_ids.begin(), member_ids.end()) !=
      member_ids.end()) {
    LOG(ERROR) << "KV mapping group contains duplicate members";
    return false;
  }
  const auto group = mapping_groups_.find(group_id);
  if (group != mapping_groups_.end()) {
    return group->second == member_ids;
  }
  for (const std::string& member_id : member_ids) {
    const auto member = regions_.find(member_id);
    if (member == regions_.end() || member->second.mapping_block_bytes() == 0 ||
        mapping_started_.count(member_id) != 0 ||
        member_groups_.count(member_id) != 0) {
      LOG(ERROR) << "KV mapping group member is unavailable: " << member_id;
      return false;
    }
  }
  for (const std::string& member_id : member_ids) {
    member_groups_.emplace(member_id, group_id);
  }
  mapping_groups_.emplace(group_id, std::move(member_ids));
  return true;
}

bool VirtualKVCacheMemoryRegistry::collect_page_targets(
    const std::string& model_id,
    const std::vector<offset_t>& offsets,
    std::vector<std::pair<MappedVirtualMemoryRegion*, offset_t>>& targets)
    const {
  const auto model = regions_.find(model_id);
  if (model == regions_.end()) {
    LOG(ERROR) << "Model " << model_id << " has no KV cache regions";
    return false;
  }
  const auto membership = member_groups_.find(model_id);
  if (membership == member_groups_.end()) {
    return model->second.collect_page_targets(
        offsets, model->second.mapping_block_bytes(), targets);
  }
  const auto& members = mapping_groups_.at(membership->second);
  size_t common_block_bytes = 0;
  for (const std::string& member_id : members) {
    common_block_bytes = std::gcd(common_block_bytes,
                                  regions_.at(member_id).mapping_block_bytes());
  }
  for (const std::string& member_id : members) {
    if (!regions_.at(member_id).collect_page_targets(
            offsets, common_block_bytes, targets)) {
      return false;
    }
  }
  return true;
}

bool VirtualKVCacheMemoryRegistry::can_prepare_mapping(
    const std::string& model_id,
    const std::vector<offset_t>& offsets) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::unordered_set<offset_t> unique_offsets(offsets.begin(),
                                                    offsets.end());
  if (unique_offsets.size() != offsets.size()) {
    return false;
  }
  std::vector<std::pair<MappedVirtualMemoryRegion*, offset_t>> targets;
  if (!collect_page_targets(model_id, offsets, targets)) {
    return false;
  }
  return std::all_of(targets.begin(), targets.end(), [](const auto& target) {
    return target.first->get_phy_page_id(target.second) < 0;
  });
}

void VirtualKVCacheMemoryRegistry::mark_mapping_started(
    const std::string& model_id) {
  if (mapping_started_.count(model_id) != 0) {
    return;
  }
  const auto membership = member_groups_.find(model_id);
  if (membership == member_groups_.end()) {
    mapping_started_.emplace(model_id);
    return;
  }
  const auto& members = mapping_groups_.at(membership->second);
  mapping_started_.insert(members.begin(), members.end());
}

bool VirtualKVCacheMemoryRegistry::map(const std::string& model_id,
                                       const std::vector<offset_t>& offsets) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::pair<MappedVirtualMemoryRegion*, offset_t>> targets;
  if (!collect_page_targets(model_id, offsets, targets) ||
      !MappedVirtualMemoryRegion::map_pages(targets)) {
    return false;
  }
  if (!offsets.empty()) {
    mark_mapping_started(model_id);
  }
  return true;
}

bool VirtualKVCacheMemoryRegistry::unmap(const std::string& model_id,
                                         const std::vector<offset_t>& offsets) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::pair<MappedVirtualMemoryRegion*, offset_t>> targets;
  if (!collect_page_targets(model_id, offsets, targets)) {
    return false;
  }
  for (const auto& [region, offset] : targets) {
    CHECK(region->unmap(offset));
  }
  return true;
}

std::optional<int64_t> VirtualKVCacheMemoryRegistry::num_layers(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = regions_.find(model_id);
  if (it == regions_.end()) {
    return std::nullopt;
  }
  return it->second.num_layers();
}

std::optional<VirtualKVCacheBlockGlobalOffsets>
VirtualKVCacheMemoryRegistry::get_global_offsets_for_block(
    const std::string& model_id,
    int64_t layer_id,
    int64_t block_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = regions_.find(model_id);
  if (it == regions_.end()) {
    LOG(ERROR) << "Model " << model_id << " has no KV cache regions";
    return std::nullopt;
  }
  return it->second.get_global_offsets_for_block(layer_id, block_id);
}

std::pair<uint64_t, uint64_t>
VirtualKVCacheMemoryRegistry::get_global_offsets_for_block(
    const std::string& model_id,
    int64_t layer_id,
    int64_t block_id,
    size_t block_size) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = regions_.find(model_id);
  if (it == regions_.end()) {
    LOG(ERROR) << "Model " << model_id << " has no KV cache regions";
    return {UINT64_MAX, UINT64_MAX};
  }
  return it->second.get_global_offsets_for_block(
      layer_id, block_id, block_size);
}

void VirtualKVCacheMemoryRegistry::erase(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto membership = member_groups_.find(model_id);
  if (membership == member_groups_.end()) {
    mapping_started_.erase(model_id);
    regions_.erase(model_id);
    return;
  }
  const auto group = mapping_groups_.find(membership->second);
  for (const std::string& member_id : group->second) {
    member_groups_.erase(member_id);
    mapping_started_.erase(member_id);
    regions_.erase(member_id);
  }
  mapping_groups_.erase(group);
}

void VirtualKVCacheMemoryRegistry::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  mapping_groups_.clear();
  member_groups_.clear();
  mapping_started_.clear();
  regions_.clear();
  initialized_ = false;
  page_size_ = 0;
}

}  // namespace xllm
