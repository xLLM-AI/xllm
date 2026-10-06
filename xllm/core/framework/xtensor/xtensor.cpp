/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/framework/xtensor/xtensor.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <unordered_set>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/xtensor/phy_page_pool.h"
#include "core/platform/vmm_api.h"

namespace xllm {
namespace {

// Align size up to page_size granularity
size_t align_up(size_t size, size_t page_size) {
  CHECK_GT(page_size, 0);
  CHECK_LE(size, std::numeric_limits<size_t>::max() - (page_size - 1));
  return ((size + page_size - 1) / page_size) * page_size;
}

VirPtr alloc_virtual_mem(size_t size) {
  size_t page_size =
      ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size();
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
    std::unordered_map<page_id_t, std::unique_ptr<PhyPage>>& mapping) {
  std::vector<std::unique_ptr<PhyPage>> pages_to_return;
  pages_to_return.reserve(mapping.size());
  for (auto& entry : mapping) {
    pages_to_return.push_back(std::move(entry.second));
  }
  mapping.clear();

  if (!pages_to_return.empty()) {
    PhyPagePool::get_instance().batch_put(pages_to_return);
  }
}

void unmap_pages(
    VirPtr vaddr,
    size_t page_size,
    const std::unordered_map<page_id_t, std::unique_ptr<PhyPage>>& mapping) {
  if (is_null_vir_ptr(vaddr)) {
    return;
  }

  for (const auto& entry : mapping) {
    VirPtr addr =
        add_vir_ptr_offset(vaddr, static_cast<size_t>(entry.first) * page_size);
    vmm::unmap(addr, page_size);
  }
}

void cleanup_pages_and_vmem(
    VirPtr vaddr,
    size_t size,
    size_t page_size,
    std::unordered_map<page_id_t, std::unique_ptr<PhyPage>>& mapping) {
  unmap_pages(vaddr, page_size, mapping);
  return_pages_to_pool(mapping);
  release_virtual_mem(vaddr, size);
}

void free_preallocated_weight_pages(const std::vector<page_id_t>& page_ids) {
  if (page_ids.empty()) {
    return;
  }

  PhyPagePool::get_instance().free_weight_pages(page_ids);
  LOG(INFO) << "XTensor: freed " << page_ids.size()
            << " preallocated weight pages";
}

}  // namespace

XTensor::XTensor(size_t size, torch::Dtype dtype, torch::Device dev)
    : vaddr_(0),
      size_(0),
      page_size_(
          ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size()),
      dtype_(dtype),
      dev_(dev) {
  CHECK_GT(size, 0);
  size_ = align_up(size, page_size_);
  CHECK_LE(size_, static_cast<size_t>(std::numeric_limits<offset_t>::max()));
  vaddr_ = alloc_virtual_mem(size_);
}

XTensor::XTensor(const std::vector<page_id_t>& page_ids,
                 torch::Dtype dtype,
                 torch::Device dev)
    : vaddr_(0),
      size_(0),
      page_size_(
          ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size()),
      dtype_(dtype),
      dev_(dev),
      use_preallocated_pages_(true),
      preallocated_page_ids_(page_ids) {
  if (page_ids.empty()) {
    LOG(ERROR) << "XTensor: empty page_ids for preallocated mode";
    return;
  }

  CHECK_GT(page_size_, 0);
  CHECK_LE(
      page_ids.size(),
      static_cast<size_t>(std::numeric_limits<offset_t>::max()) / page_size_);
  size_ = page_ids.size() * page_size_;
  vaddr_ = alloc_virtual_mem(size_);

  if (!map_with_page_ids(page_ids)) {
    LOG(ERROR) << "XTensor: failed to map preallocated pages";
    vmm::release_vir_ptr(vaddr_, size_);
    vaddr_ = {};
    size_ = 0;
  }
}

XTensor::~XTensor() {
  if (use_preallocated_pages_) {
    for (size_t i = 0; i < mapped_preallocated_pages_; ++i) {
      VirPtr addr = add_vir_ptr_offset(vaddr_, i * page_size_);
      vmm::unmap(addr, page_size_);
    }
    release_virtual_mem(vaddr_, size_);
    free_preallocated_weight_pages(preallocated_page_ids_);
    return;
  }

  cleanup_pages_and_vmem(vaddr_, size_, page_size_, mapping_);
}

bool XTensor::valid_offset_(offset_t offset) const {
  return offset >= 0 && page_size_ > 0 &&
         static_cast<size_t>(offset) % page_size_ == 0 &&
         static_cast<size_t>(offset) < size_ && !is_null_vir_ptr(vaddr_);
}

bool XTensor::map(offset_t offset) { return map_pages({{this, offset}}); }

bool XTensor::map_pages(
    const std::vector<std::pair<XTensor*, offset_t>>& targets) {
  auto& pool = PhyPagePool::get_instance();
  std::vector<std::pair<XTensor*, offset_t>> missing;
  missing.reserve(targets.size());
  std::unordered_map<XTensor*, std::unordered_set<offset_t>> seen;
  seen.reserve(targets.size());
  for (const auto& [tensor, offset] : targets) {
    if (tensor == nullptr || tensor->use_preallocated_pages_ ||
        !tensor->valid_offset_(offset) || tensor->dev_ != pool.device()) {
      LOG(ERROR) << "Invalid XTensor mapping target at offset " << offset;
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
                 << " XTensor mappings";
    return false;
  }

  for (size_t i = 0; i < missing.size(); ++i) {
    auto [tensor, offset] = missing[i];
    VirPtr addr = add_vir_ptr_offset(tensor->vaddr_, offset);
    PhyMemHandle handle = pages[i]->get_phy_handle();
    vmm::map(addr, handle, tensor->dev_.index());
    tensor->mapping_.emplace(offset / tensor->page_size_, std::move(pages[i]));
  }
  return true;
}

bool XTensor::unmap(offset_t offset) {
  if (use_preallocated_pages_ || !valid_offset_(offset)) {
    LOG(ERROR) << "Invalid XTensor unmapping offset " << offset;
    return false;
  }

  page_id_t page_id = offset / page_size_;

  auto it = mapping_.find(page_id);
  if (it == mapping_.end()) {
    // Already unmapped (idempotent: return true)
    return true;
  }

  VirPtr vaddr = add_vir_ptr_offset(vaddr_, offset);
  vmm::unmap(vaddr, page_size_);

  // Return the physical page to pool
  std::vector<std::unique_ptr<PhyPage>> pages_to_return;
  pages_to_return.reserve(1);
  pages_to_return.emplace_back(std::move(it->second));
  mapping_.erase(it);
  PhyPagePool::get_instance().batch_put(pages_to_return);

  return true;
}

bool XTensor::map_all() {
  std::vector<std::pair<XTensor*, offset_t>> targets;
  targets.reserve(size_ / page_size_);
  for (size_t offset = 0; offset < size_; offset += page_size_) {
    targets.emplace_back(this, static_cast<offset_t>(offset));
  }
  return map_pages(targets);
}

bool XTensor::unmap_all() {
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

bool XTensor::map_with_page_ids(const std::vector<page_id_t>& page_ids) {
  if (!use_preallocated_pages_ || mapped_preallocated_pages_ != 0 ||
      page_ids.empty() || page_ids.size() != size_ / page_size_) {
    LOG(ERROR) << "Invalid preallocated XTensor mapping size or state";
    return false;
  }
  auto& pool = PhyPagePool::get_instance();
  if (dev_ != pool.device()) {
    LOG(ERROR) << "Preallocated XTensor device differs from physical page pool";
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
    vmm::map(addr, handle, dev_.index());
    ++mapped_preallocated_pages_;
  }
  return true;
}

bool XTensor::allocate(void*& ptr, size_t size) {
  // Check if there's enough space
  if (size > size_ - alloc_offset_) {
    LOG(ERROR) << "XTensor::allocate failed: requested " << size
               << " bytes at offset " << alloc_offset_ << ", but only "
               << (size_ - alloc_offset_) << " bytes available"
               << " (total size: " << size_ << ")";
    return false;
  }

  ptr = vir_ptr_to_void_ptr(add_vir_ptr_offset(vaddr_, alloc_offset_));
  // Update allocation offset
  alloc_offset_ += size;

  VLOG(2) << "XTensor::allocate: size=" << size
          << ", new_alloc_offset=" << alloc_offset_;

  return true;
}

torch::Tensor XTensor::to_torch_tensor() const {
  int64_t num_elems = static_cast<int64_t>(size_ / torch::elementSize(dtype_));
  return to_torch_tensor(/*offset=*/0, {num_elems});
}

page_id_t XTensor::get_phy_page_id(offset_t offset) const {
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

torch::Tensor XTensor::to_torch_tensor(size_t offset,
                                       const std::vector<int64_t>& dims) const {
  CHECK_LE(offset, size_) << "Tensor byte offset is out of bounds";
  size_t element_size = torch::elementSize(dtype_);
  CHECK_EQ(offset % element_size, 0) << "Tensor byte offset is not aligned";
  for (int64_t dim : dims) {
    CHECK_GE(dim, 0) << "Tensor dimensions must be nonnegative";
  }
  size_t num_elems =
      std::find(dims.begin(), dims.end(), 0) == dims.end() ? 1 : 0;
  for (int64_t dim : dims) {
    if (dim == 0 || num_elems == 0) {
      continue;
    }
    CHECK_LE(num_elems,
             std::numeric_limits<size_t>::max() / static_cast<size_t>(dim));
    num_elems *= static_cast<size_t>(dim);
  }
  CHECK_LE(num_elems, (size_ - offset) / element_size)
      << "Tensor view exceeds its virtual memory region";
  uintptr_t addr = vir_ptr_to_uintptr(vaddr_) + offset;

#if defined(USE_NPU)
  c10::DeviceType device_type = c10::DeviceType::PrivateUse1;
  torch::TensorOptions option =
      torch::TensorOptions().dtype(dtype_).device(dev_);

  auto tensor = torch::empty({0}, option);
  auto address = reinterpret_cast<void*>(addr);
  torch::DataPtr c10_data_ptr(address, address, [](void*) {}, tensor.device());

  size_t tensor_nbytes = at::detail::computeStorageNbytesContiguous(
      dims, tensor.dtype().itemsize());
  torch::Storage storage;
  // get npu storage constructor from register and construct storage
  auto fptr = c10::GetStorageImplCreate(device_type);
  auto allocator = c10::GetAllocator(device_type);

  // PyTorch 2.7+: StorageImpl now takes DataPtr instead of raw allocator
  storage = fptr(c10::StorageImpl::use_byte_size_t(),
                 c10::SymInt(tensor_nbytes),
                 std::move(c10_data_ptr),
                 allocator,
                 true);

  tensor.set_(storage, 0, dims);

  return tensor;
#else
  // For non-NPU devices, use torch::from_blob
  auto options =
      torch::TensorOptions().dtype(dtype_).device(dev_).requires_grad(false);
  return torch::from_blob(reinterpret_cast<void*>(addr), dims, options);
#endif
}

}  // namespace xllm
