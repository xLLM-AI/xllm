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

#include "core/kv_cache/storage/kv_cache_memory_regions.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <limits>
#include <memory>
#include <utility>

namespace xllm {

std::vector<torch::Tensor> KVCacheMemoryRegions::create_k_tensors(
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers,
    const torch::Device& device,
    size_t page_size) {
  return create_tensors(
      dims, dtype, num_layers, device, page_size, /*name=*/"K", k_regions_);
}

std::vector<torch::Tensor> KVCacheMemoryRegions::create_v_tensors(
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers,
    const torch::Device& device,
    size_t page_size) {
  return create_tensors(
      dims, dtype, num_layers, device, page_size, /*name=*/"V", v_regions_);
}

std::vector<torch::Tensor> KVCacheMemoryRegions::create_tensors(
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers,
    const torch::Device& device,
    size_t page_size,
    const char* name,
    std::vector<std::unique_ptr<MappedMemoryRegion>>& regions) {
  CHECK(num_layers_ == 0 || num_layers_ == num_layers)
      << "Number of KV cache layers mismatch";
  CHECK(regions.empty()) << name << " tensors already created";
  CHECK(!dims.empty()) << name << " tensor dims cannot be empty";
  CHECK_GT(num_layers, 0);

  size_t size = torch::scalarTypeToTypeMeta(dtype).itemsize();
  for (int64_t dim : dims) {
    CHECK_GT(dim, 0);
    CHECK_LE(size,
             std::numeric_limits<size_t>::max() / static_cast<size_t>(dim));
    size *= static_cast<size_t>(dim);
  }

  CHECK_GT(page_size, 0);
  CHECK_LE(size, std::numeric_limits<size_t>::max() - (page_size - 1));
  if (size % page_size != 0) {
    size_t aligned_size = ((size + page_size - 1) / page_size) * page_size;
    LOG(WARNING) << name << " tensor size " << size
                 << " is not aligned to page size " << page_size
                 << ", aligning to " << aligned_size;
    size = aligned_size;
  }

  num_layers_ = num_layers;
  std::vector<torch::Tensor> tensors;
  tensors.reserve(num_layers);
  regions.reserve(num_layers);
  for (int64_t i = 0; i < num_layers; ++i) {
    auto region =
        std::make_unique<MappedMemoryRegion>(size, dtype, device, page_size);
    tensors.emplace_back(region->to_torch_tensor(/*offset=*/0, dims));
    regions.emplace_back(std::move(region));
  }
  return tensors;
}

bool KVCacheMemoryRegions::map(const std::vector<offset_t>& offsets) {
  if (k_regions_.empty() || v_regions_.empty()) {
    LOG(ERROR) << "KV tensors not created";
    return false;
  }

  CHECK_GT(num_layers_, 0);
  CHECK_EQ(k_regions_.size(), num_layers_);
  CHECK_EQ(v_regions_.size(), num_layers_);
  CHECK_LE(offsets.size(),
           std::numeric_limits<size_t>::max() / k_regions_.size() / 2);
  std::vector<std::pair<MappedMemoryRegion*, offset_t>> targets;
  targets.reserve(offsets.size() * k_regions_.size() * 2);
  for (size_t i = 0; i < k_regions_.size(); ++i) {
    for (offset_t offset : offsets) {
      targets.emplace_back(k_regions_[i].get(), offset);
      targets.emplace_back(v_regions_[i].get(), offset);
    }
  }
  return MappedMemoryRegion::map_pages(targets);
}

bool KVCacheMemoryRegions::unmap(const std::vector<offset_t>& offsets) {
  if (k_regions_.empty() || v_regions_.empty()) {
    LOG(ERROR) << "Cannot unmap before KV tensors are created";
    return false;
  }

  CHECK_GT(num_layers_, 0);
  CHECK_EQ(k_regions_.size(), num_layers_);
  CHECK_EQ(v_regions_.size(), num_layers_);
  for (size_t i = 0; i < k_regions_.size(); ++i) {
    const auto& k_region = k_regions_[i];
    const auto& v_region = v_regions_[i];
    for (offset_t offset : offsets) {
      if (offset < 0 || static_cast<size_t>(offset) >= k_region->size() ||
          static_cast<size_t>(offset) >= v_region->size() ||
          static_cast<size_t>(offset) % k_region->page_size() != 0 ||
          static_cast<size_t>(offset) % v_region->page_size() != 0) {
        LOG(ERROR) << "Invalid KV unmapping offset " << offset;
        return false;
      }
    }
  }

  for (size_t i = 0; i < k_regions_.size(); ++i) {
    for (offset_t offset : offsets) {
      CHECK(k_regions_[i]->unmap(offset));
      CHECK(v_regions_[i]->unmap(offset));
    }
  }
  return true;
}

std::pair<uint64_t, uint64_t>
KVCacheMemoryRegions::get_global_offsets_for_block(int64_t layer_id,
                                                   int64_t block_id,
                                                   size_t block_size) const {
  constexpr uint64_t kInvalidOffset = UINT64_MAX;
  if (layer_id < 0 || layer_id >= num_layers_) {
    LOG(ERROR) << "Invalid layer_id " << layer_id
               << " (num_layers=" << num_layers_ << ")";
    return {kInvalidOffset, kInvalidOffset};
  }
  if (k_regions_.empty() || v_regions_.empty()) {
    LOG(ERROR) << "KV tensors not created";
    return {kInvalidOffset, kInvalidOffset};
  }

  if (block_id < 0 || block_size == 0 ||
      static_cast<uint64_t>(block_id) >
          std::numeric_limits<size_t>::max() / block_size) {
    LOG(ERROR) << "Invalid KV block geometry: block_id=" << block_id
               << ", block_size=" << block_size;
    return {kInvalidOffset, kInvalidOffset};
  }

  const auto& k_region = k_regions_[layer_id];
  const auto& v_region = v_regions_[layer_id];
  const size_t block_offset = static_cast<size_t>(block_id) * block_size;
  if (block_size > k_region->size() || block_size > v_region->size() ||
      block_offset > k_region->size() - block_size ||
      block_offset > v_region->size() - block_size) {
    LOG(ERROR) << "KV block " << block_id << " is outside layer " << layer_id
               << " regions";
    return {kInvalidOffset, kInvalidOffset};
  }

  const size_t page_size = k_region->page_size();
  if (page_size == 0 || v_region->page_size() != page_size) {
    LOG(ERROR) << "KV regions have incompatible page sizes";
    return {kInvalidOffset, kInvalidOffset};
  }
  const size_t offset_within_page = block_offset % page_size;
  if (block_size > page_size - offset_within_page) {
    LOG(ERROR) << "KV block " << block_id << " spans multiple physical pages";
    return {kInvalidOffset, kInvalidOffset};
  }

  const size_t aligned_offset = block_offset - offset_within_page;
  if (aligned_offset >
      static_cast<size_t>(std::numeric_limits<offset_t>::max())) {
    LOG(ERROR) << "KV mapping offset exceeds supported range";
    return {kInvalidOffset, kInvalidOffset};
  }
  const offset_t local_offset = static_cast<offset_t>(aligned_offset);
  const page_id_t k_page_id = k_region->get_phy_page_id(local_offset);
  if (k_page_id < 0) {
    LOG(ERROR) << "K cache block " << block_id << " at layer " << layer_id
               << " is not mapped (local_offset=" << local_offset << ")";
    return {kInvalidOffset, kInvalidOffset};
  }

  const page_id_t v_page_id = v_region->get_phy_page_id(local_offset);
  if (v_page_id < 0) {
    LOG(ERROR) << "V cache block " << block_id << " at layer " << layer_id
               << " is not mapped (local_offset=" << local_offset << ")";
    return {kInvalidOffset, kInvalidOffset};
  }

  const uint64_t max_page_id =
      (kInvalidOffset - 1 - offset_within_page) / page_size;
  if (static_cast<uint64_t>(k_page_id) > max_page_id ||
      static_cast<uint64_t>(v_page_id) > max_page_id) {
    LOG(ERROR) << "KV physical page offset exceeds supported range";
    return {kInvalidOffset, kInvalidOffset};
  }
  return {static_cast<uint64_t>(k_page_id) * page_size + offset_within_page,
          static_cast<uint64_t>(v_page_id) * page_size + offset_within_page};
}

}  // namespace xllm
