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

#include "core/virtual_memory/mapped_memory_region.h"

#include <glog/logging.h>

#include <limits>
#include <unordered_set>

#include "core/platform/vmm_api.h"
#include "core/virtual_memory/physical_page_pool.h"

namespace xllm {
namespace {

// Align size up to page_size granularity
size_t align_up(size_t size, size_t page_size) {
  CHECK_GT(page_size, 0);
  CHECK_LE(size, std::numeric_limits<size_t>::max() - (page_size - 1));
  return ((size + page_size - 1) / page_size) * page_size;
}

VirPtr alloc_virtual_mem(size_t size, size_t page_size) {
  CHECK(size % page_size == 0)
      << "alloc size not aligned: " << size;  // Ensure alignment.

  VirPtr vaddr;
  vmm::create_vir_ptr(vaddr, size);
  return vaddr;
}

void release_virtual_mem(VirPtr vaddr, size_t size) {
  if (is_null_vir_ptr(vaddr)) {
    return;
  }
  vmm::release_vir_ptr(vaddr, size);
}

void return_pages_to_pool(
    std::unordered_map<page_id_t, std::unique_ptr<PhysicalPage>>& mapping) {
  std::vector<std::unique_ptr<PhysicalPage>> pages_to_return;
  pages_to_return.reserve(mapping.size());
  for (auto& entry : mapping) {
    pages_to_return.push_back(std::move(entry.second));
  }
  mapping.clear();

  if (!pages_to_return.empty()) {
    PhysicalPagePool::get_instance().batch_put(pages_to_return);
  }
}

void unmap_pages(
    VirPtr vaddr,
    size_t page_size,
    const std::unordered_map<page_id_t, std::unique_ptr<PhysicalPage>>&
        mapping) {
  if (is_null_vir_ptr(vaddr)) {
    return;
  }

  for (const auto& entry : mapping) {
    VirPtr addr =
        add_vir_ptr_offset(vaddr, static_cast<size_t>(entry.first) * page_size);
    vmm::unmap(addr, page_size, page_size);
  }
}

void cleanup_pages_and_vmem(
    VirPtr vaddr,
    size_t size,
    size_t page_size,
    std::unordered_map<page_id_t, std::unique_ptr<PhysicalPage>>& mapping) {
  unmap_pages(vaddr, page_size, mapping);
  return_pages_to_pool(mapping);
  release_virtual_mem(vaddr, size);
}

void release_preallocated_pages(const std::vector<page_id_t>& page_ids) {
  if (page_ids.empty()) {
    return;
  }

  PhysicalPagePool::get_instance().release_reserved_pages(page_ids);
  LOG(INFO) << "MappedMemoryRegion: freed " << page_ids.size()
            << " reserved pages";
}

}  // namespace

MappedMemoryRegion::MappedMemoryRegion(size_t size,
                                       torch::Dtype dtype,
                                       torch::Device dev,
                                       size_t page_size)
    : vaddr_(0), size_(0), page_size_(page_size), dtype_(dtype), dev_(dev) {
  CHECK_GT(size, 0);
  size_ = align_up(size, page_size_);
  CHECK_LE(size_, static_cast<size_t>(std::numeric_limits<offset_t>::max()));
  vaddr_ = alloc_virtual_mem(size_, page_size_);
}

MappedMemoryRegion::MappedMemoryRegion(std::vector<page_id_t> page_ids,
                                       torch::Dtype dtype,
                                       torch::Device dev,
                                       size_t page_size)
    : vaddr_(0),
      size_(0),
      page_size_(page_size),
      dtype_(dtype),
      dev_(dev),
      use_preallocated_pages_(true),
      preallocated_page_ids_(std::move(page_ids)) {
  if (preallocated_page_ids_.empty()) {
    LOG(ERROR) << "MappedMemoryRegion: empty page_ids for preallocated mode";
    return;
  }

  CHECK_GT(page_size_, 0);
  CHECK_LE(
      preallocated_page_ids_.size(),
      static_cast<size_t>(std::numeric_limits<offset_t>::max()) / page_size_);
  size_ = preallocated_page_ids_.size() * page_size_;
  vaddr_ = alloc_virtual_mem(size_, page_size_);

  if (!map_with_page_ids(preallocated_page_ids_)) {
    LOG(ERROR) << "MappedMemoryRegion: failed to map preallocated pages";
    for (size_t i = 0; i < mapped_preallocated_pages_; ++i) {
      VirPtr address = add_vir_ptr_offset(vaddr_, i * page_size_);
      vmm::unmap(address, page_size_, page_size_);
    }
    mapped_preallocated_pages_ = 0;
    vmm::release_vir_ptr(vaddr_, size_);
    vaddr_ = {};
    size_ = 0;
  }
}

MappedMemoryRegion::~MappedMemoryRegion() {
  if (use_preallocated_pages_) {
    for (size_t i = 0; i < mapped_preallocated_pages_; ++i) {
      VirPtr addr = add_vir_ptr_offset(vaddr_, i * page_size_);
      vmm::unmap(addr, page_size_, page_size_);
    }
    release_virtual_mem(vaddr_, size_);
    release_preallocated_pages(preallocated_page_ids_);
    return;
  }

  cleanup_pages_and_vmem(vaddr_, size_, page_size_, mapping_);
}

bool MappedMemoryRegion::valid_offset_(offset_t offset) const {
  return offset >= 0 && page_size_ > 0 &&
         static_cast<size_t>(offset) % page_size_ == 0 &&
         static_cast<size_t>(offset) < size_ && !is_null_vir_ptr(vaddr_);
}

bool MappedMemoryRegion::map(offset_t offset) {
  return map_pages({{this, offset}});
}

bool MappedMemoryRegion::map_pages(
    const std::vector<std::pair<MappedMemoryRegion*, offset_t>>& targets) {
  auto& pool = PhysicalPagePool::get_instance();
  std::vector<std::pair<MappedMemoryRegion*, offset_t>> missing;
  missing.reserve(targets.size());
  std::unordered_map<MappedMemoryRegion*, std::unordered_set<offset_t>> seen;
  seen.reserve(targets.size());
  for (const auto& [tensor, offset] : targets) {
    if (tensor == nullptr || tensor->use_preallocated_pages_ ||
        !tensor->valid_offset_(offset) || tensor->dev_ != pool.device() ||
        tensor->page_size_ != pool.page_size()) {
      LOG(ERROR) << "Invalid MappedMemoryRegion mapping target at offset "
                 << offset;
      return false;
    }
    if (!seen[tensor].emplace(offset).second ||
        tensor->mapping_.find(offset / tensor->page_size_) !=
            tensor->mapping_.end()) {
      continue;
    }
    missing.emplace_back(tensor, offset);
  }

  if (missing.empty()) {
    return true;
  }
  for (const auto& [tensor, offsets] : seen) {
    tensor->mapping_.reserve(tensor->mapping_.size() + offsets.size());
  }

  auto pages = pool.batch_get(missing.size());
  if (pages.empty()) {
    LOG(WARNING) << "Not enough physical pages for " << missing.size()
                 << " MappedMemoryRegion mappings";
    return false;
  }

  for (size_t i = 0; i < missing.size(); ++i) {
    auto [tensor, offset] = missing[i];
    VirPtr addr = add_vir_ptr_offset(tensor->vaddr_, offset);
    PhyMemHandle handle = pages[i]->get_phy_handle();
    vmm::map(addr, handle, tensor->page_size_, tensor->dev_.index());
    tensor->mapping_.emplace(offset / tensor->page_size_, std::move(pages[i]));
  }
  return true;
}

bool MappedMemoryRegion::unmap(offset_t offset) {
  if (use_preallocated_pages_ || !valid_offset_(offset)) {
    LOG(ERROR) << "Invalid MappedMemoryRegion unmapping offset " << offset;
    return false;
  }

  page_id_t page_id = offset / page_size_;

  auto it = mapping_.find(page_id);
  if (it == mapping_.end()) {
    // Already unmapped (idempotent: return true)
    return true;
  }

  VirPtr vaddr = add_vir_ptr_offset(vaddr_, offset);
  vmm::unmap(vaddr, page_size_, page_size_);

  // Return the physical page to pool
  std::vector<std::unique_ptr<PhysicalPage>> pages_to_return;
  pages_to_return.reserve(1);
  pages_to_return.emplace_back(std::move(it->second));
  mapping_.erase(it);
  PhysicalPagePool::get_instance().batch_put(pages_to_return);

  return true;
}

bool MappedMemoryRegion::map_all() {
  std::vector<std::pair<MappedMemoryRegion*, offset_t>> targets;
  targets.reserve(size_ / page_size_);
  for (size_t offset = 0; offset < size_; offset += page_size_) {
    targets.emplace_back(this, static_cast<offset_t>(offset));
  }
  return map_pages(targets);
}

bool MappedMemoryRegion::unmap_all() {
  for (size_t offset = 0; offset < size_; offset += page_size_) {
    page_id_t page_id = offset / page_size_;
    // Only unmap if the page is mapped
    if (mapping_.find(page_id) != mapping_.end()) {
      if (!unmap(offset)) {
        LOG(ERROR) << "Failed to unmap page at offset " << offset;
        return false;
      }
    }
  }
  return true;
}

bool MappedMemoryRegion::map_with_page_ids(
    const std::vector<page_id_t>& page_ids) {
  if (!use_preallocated_pages_ || mapped_preallocated_pages_ != 0 ||
      page_ids.empty() || page_ids.size() != size_ / page_size_) {
    LOG(ERROR)
        << "Invalid preallocated MappedMemoryRegion mapping size or state";
    return false;
  }
  auto& pool = PhysicalPagePool::get_instance();
  if (dev_ != pool.device() || page_size_ != pool.page_size()) {
    LOG(ERROR) << "Preallocated MappedMemoryRegion geometry differs from "
                  "physical page pool";
    return false;
  }
  const auto& all_pages = pool.get_all_pages();
  std::unordered_set<page_id_t> seen;
  seen.reserve(page_ids.size());
  for (page_id_t page_id : page_ids) {
    if (page_id < 0 || static_cast<size_t>(page_id) >= all_pages.size() ||
        all_pages[page_id] == nullptr || !seen.emplace(page_id).second) {
      LOG(ERROR) << "Invalid or duplicate preallocated page_id " << page_id;
      return false;
    }
  }

  for (size_t i = 0; i < page_ids.size(); ++i) {
    VirPtr addr = add_vir_ptr_offset(vaddr_, i * page_size_);
    PhyMemHandle handle = all_pages[page_ids[i]]->get_phy_handle();
    vmm::map(addr, handle, page_size_, dev_.index());
    ++mapped_preallocated_pages_;
  }
  return true;
}

page_id_t MappedMemoryRegion::get_phy_page_id(offset_t offset) const {
  if (!valid_offset_(offset)) {
    return -1;
  }

  page_id_t local_page_id = offset / page_size_;
  auto it = mapping_.find(local_page_id);
  if (it == mapping_.end()) {
    // Not mapped, return -1
    return -1;
  }
  return it->second->page_id();
}

}  // namespace xllm
