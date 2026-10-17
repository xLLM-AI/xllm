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

#include "core/kv_cache/storage/virtual_kv_cache_memory_regions.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>

#include "core/framework/allocator/torch/mapped_virtual_memory_tensor_view.h"
#include "core/kv_cache/layout/virtual_kv_cache_page_layout.h"

namespace xllm {
namespace {

std::optional<uint64_t> get_role_block_global_offset(
    const MappedVirtualMemoryRegion& region,
    size_t block_id,
    size_t block_bytes) {
  const size_t page_size = region.page_size();
  if (block_bytes == 0 || block_bytes > page_size ||
      page_size % block_bytes != 0) {
    LOG(ERROR) << "Virtual memory transfer requires each block to divide one "
                  "physical page: block_bytes="
               << block_bytes << ", page_size=" << page_size;
    return std::nullopt;
  }
  if (block_id > std::numeric_limits<size_t>::max() / block_bytes) {
    LOG(ERROR) << "KV transfer block byte offset overflows";
    return std::nullopt;
  }
  const size_t block_offset = block_id * block_bytes;
  if (block_bytes > region.size() ||
      block_offset > region.size() - block_bytes) {
    LOG(ERROR) << "KV transfer block is outside its virtual region";
    return std::nullopt;
  }
  const size_t offset_within_page = block_offset % page_size;
  const size_t aligned_offset = block_offset - offset_within_page;
  if (aligned_offset >
      static_cast<size_t>(std::numeric_limits<offset_t>::max())) {
    LOG(ERROR) << "KV transfer mapping offset exceeds supported range";
    return std::nullopt;
  }
  const page_id_t page_id =
      region.get_phy_page_id(static_cast<offset_t>(aligned_offset));
  if (page_id < 0) {
    LOG(ERROR) << "KV transfer block is not mapped: block_id=" << block_id;
    return std::nullopt;
  }
  const uint64_t max_page_id =
      (UINT64_MAX - offset_within_page - block_bytes) / page_size;
  if (static_cast<uint64_t>(page_id) > max_page_id) {
    LOG(ERROR) << "KV transfer physical byte offset exceeds supported range";
    return std::nullopt;
  }
  return static_cast<uint64_t>(page_id) * page_size + offset_within_page;
}

}  // namespace

std::vector<torch::Tensor> VirtualKVCacheMemoryRegions::create_k_tensors(
    const std::vector<int64_t>& dims,
    torch::ScalarType dtype,
    int64_t num_layers,
    const torch::Device& device,
    size_t page_size) {
  k_block_bytes_ = VirtualKVCachePageLayout::tensor_block_bytes(
      dims, torch::scalarTypeToTypeMeta(dtype).itemsize());
  return create_tensors(
      dims, dtype, num_layers, device, page_size, /*name=*/"K", k_regions_);
}

std::vector<torch::Tensor> VirtualKVCacheMemoryRegions::create_v_tensors(
    const std::vector<int64_t>& dims,
    torch::ScalarType dtype,
    int64_t num_layers,
    const torch::Device& device,
    size_t page_size) {
  v_block_bytes_ = VirtualKVCachePageLayout::tensor_block_bytes(
      dims, torch::scalarTypeToTypeMeta(dtype).itemsize());
  return create_tensors(
      dims, dtype, num_layers, device, page_size, /*name=*/"V", v_regions_);
}

std::vector<torch::Tensor> VirtualKVCacheMemoryRegions::create_index_tensors(
    const std::vector<int64_t>& dims,
    torch::ScalarType dtype,
    int64_t num_layers,
    const torch::Device& device,
    size_t page_size,
    const std::vector<bool>& layer_owned) {
  CHECK_EQ(dtype, torch::kBFloat16)
      << "Virtual memory INDEX cache supports only BF16 tensors";
  index_block_bytes_ = VirtualKVCachePageLayout::tensor_block_bytes(
      dims, torch::scalarTypeToTypeMeta(dtype).itemsize());
  return create_tensors(dims,
                        dtype,
                        num_layers,
                        device,
                        page_size,
                        /*name=*/"INDEX",
                        index_regions_,
                        layer_owned);
}

std::vector<torch::Tensor> VirtualKVCacheMemoryRegions::create_tensors(
    const std::vector<int64_t>& dims,
    torch::ScalarType dtype,
    int64_t num_layers,
    const torch::Device& device,
    size_t page_size,
    const char* name,
    std::vector<std::unique_ptr<MappedVirtualMemoryRegion>>& regions,
    const std::vector<bool>& layer_owned) {
  CHECK(num_layers_ == 0 || num_layers_ == num_layers)
      << "Number of KV cache layers mismatch";
  CHECK(regions.empty()) << name << " tensors already created";
  CHECK(!dims.empty()) << name << " tensor dims cannot be empty";
  CHECK_GT(num_layers, 0);
  CHECK(layer_owned.empty() ||
        layer_owned.size() == static_cast<size_t>(num_layers))
      << "INDEX cache ownership mask must match num_layers";
  CHECK_GT(dims.front(), 0);
  CHECK(num_blocks_ == 0 || num_blocks_ == static_cast<size_t>(dims.front()))
      << "Number of KV cache blocks mismatch";

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
  num_blocks_ = static_cast<size_t>(dims.front());
  std::vector<torch::Tensor> tensors(num_layers);
  regions.resize(num_layers);
  for (int64_t i = 0; i < num_layers; ++i) {
    if (!layer_owned.empty() && !layer_owned[i]) {
      continue;
    }
    auto region =
        std::make_unique<MappedVirtualMemoryRegion>(size, device, page_size);
    tensors[i] = create_mapped_virtual_memory_tensor_view(
        *region, dtype, /*offset=*/0, dims);
    regions[i] = std::move(region);
  }
  return tensors;
}

bool VirtualKVCacheMemoryRegions::collect_page_targets(
    const std::vector<offset_t>& offsets,
    std::vector<std::pair<MappedVirtualMemoryRegion*, offset_t>>& targets)
    const {
  if (k_regions_.empty() || v_regions_.empty()) {
    LOG(ERROR) << "KV tensors not created";
    return false;
  }

  CHECK_GT(num_layers_, 0);
  CHECK_EQ(k_regions_.size(), num_layers_);
  CHECK_EQ(v_regions_.size(), num_layers_);
  const size_t page_size = k_regions_.front()->page_size();
  size_t index_layers = 0;
  if (!index_regions_.empty()) {
    CHECK_EQ(index_regions_.size(), num_layers_);
    index_layers = static_cast<size_t>(std::count_if(
        index_regions_.begin(), index_regions_.end(), [](const auto& region) {
          return region != nullptr;
        }));
  }
  std::vector<size_t> tensor_block_bytes{k_block_bytes_, v_block_bytes_};
  if (index_layers > 0) {
    tensor_block_bytes.emplace_back(index_block_bytes_);
  }
  const VirtualKVCachePageLayout layout(std::move(tensor_block_bytes),
                                        page_size);
  const size_t k_pages = layout.pages_per_tensor(/*role_index=*/0);
  const size_t v_pages = layout.pages_per_tensor(/*role_index=*/1);
  size_t index_pages = 0;
  std::vector<size_t> tensor_counts{k_regions_.size(), v_regions_.size()};
  if (index_layers > 0) {
    index_pages = layout.pages_per_tensor(/*role_index=*/2);
    tensor_counts.emplace_back(index_layers);
  }
  const size_t pages_per_offset = layout.physical_pages(tensor_counts);
  CHECK_LE(offsets.size(),
           std::numeric_limits<size_t>::max() / pages_per_offset);
  targets.reserve(offsets.size() * pages_per_offset);

  const size_t complete_pages = num_blocks_ / layout.blocks_per_page();
  for (offset_t offset : offsets) {
    if (offset < 0 || static_cast<size_t>(offset) % page_size != 0 ||
        static_cast<size_t>(offset) / page_size >= complete_pages) {
      LOG(ERROR) << "Invalid KV logical page offset " << offset;
      return false;
    }
  }

  const auto append_targets = [&offsets, &targets, page_size](
                                  MappedVirtualMemoryRegion& region,
                                  size_t pages_per_tensor) {
    CHECK_EQ(region.page_size(), page_size);
    CHECK_LE(pages_per_tensor, std::numeric_limits<size_t>::max() / page_size);
    const size_t page_span = pages_per_tensor * page_size;
    for (offset_t offset : offsets) {
      if (static_cast<size_t>(offset) >
          static_cast<size_t>(std::numeric_limits<offset_t>::max()) /
              pages_per_tensor) {
        LOG(ERROR) << "KV mapping offset exceeds supported range";
        return false;
      }
      const size_t role_offset = static_cast<size_t>(offset) * pages_per_tensor;
      if (page_span > region.size() ||
          role_offset > region.size() - page_span) {
        LOG(ERROR) << "KV logical page exceeds its tensor region";
        return false;
      }
      for (size_t page = 0; page < pages_per_tensor; ++page) {
        targets.emplace_back(
            &region, static_cast<offset_t>(role_offset + page * page_size));
      }
    }
    return true;
  };
  for (size_t i = 0; i < k_regions_.size(); ++i) {
    if (!append_targets(*k_regions_[i], k_pages) ||
        !append_targets(*v_regions_[i], v_pages) ||
        (index_pages > 0 && index_regions_[i] != nullptr &&
         !append_targets(*index_regions_[i], index_pages))) {
      return false;
    }
  }
  return true;
}

bool VirtualKVCacheMemoryRegions::map(const std::vector<offset_t>& offsets) {
  std::vector<std::pair<MappedVirtualMemoryRegion*, offset_t>> targets;
  return collect_page_targets(offsets, targets) &&
         MappedVirtualMemoryRegion::map_pages(targets);
}

bool VirtualKVCacheMemoryRegions::unmap(const std::vector<offset_t>& offsets) {
  std::vector<std::pair<MappedVirtualMemoryRegion*, offset_t>> targets;
  if (!collect_page_targets(offsets, targets)) {
    return false;
  }
  for (const auto& [region, offset] : targets) {
    CHECK(region->unmap(offset));
  }
  return true;
}

std::optional<VirtualKVCacheBlockGlobalOffsets>
VirtualKVCacheMemoryRegions::get_global_offsets_for_block(
    int64_t layer_id,
    int64_t block_id) const {
  if (layer_id < 0 || layer_id >= num_layers_ || block_id < 0 ||
      static_cast<uint64_t>(block_id) >= num_blocks_) {
    LOG(ERROR) << "Invalid virtual KV transfer block: layer_id=" << layer_id
               << ", block_id=" << block_id;
    return std::nullopt;
  }
  if (k_regions_.empty() || v_regions_.empty()) {
    LOG(ERROR) << "KV tensors not created";
    return std::nullopt;
  }
  const auto k_offset = get_role_block_global_offset(
      *k_regions_[layer_id], static_cast<size_t>(block_id), k_block_bytes_);
  const auto v_offset = get_role_block_global_offset(
      *v_regions_[layer_id], static_cast<size_t>(block_id), v_block_bytes_);
  if (!k_offset.has_value() || !v_offset.has_value()) {
    return std::nullopt;
  }
  VirtualKVCacheBlockGlobalOffsets offsets{
      k_offset.value(), v_offset.value(), std::nullopt};
  if (!index_regions_.empty() && index_regions_[layer_id] != nullptr) {
    offsets.index_offset =
        get_role_block_global_offset(*index_regions_[layer_id],
                                     static_cast<size_t>(block_id),
                                     index_block_bytes_);
    if (!offsets.index_offset.has_value()) {
      return std::nullopt;
    }
  }
  return offsets;
}

std::pair<uint64_t, uint64_t>
VirtualKVCacheMemoryRegions::get_global_offsets_for_block(
    int64_t layer_id,
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
  if (k_block_bytes_ != v_block_bytes_ || !index_regions_.empty()) {
    LOG(ERROR) << "Virtual memory KV transfer requires equal key and value "
                  "block sizes without INDEX cache";
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
