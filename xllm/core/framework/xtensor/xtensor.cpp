/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "xtensor.h"

#include <glog/logging.h>

#include "common/global_flags.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/util/tensor_helper.h"
#include "phy_page_pool.h"
#include "platform/vmm_api.h"

namespace xllm {

// Align size up to page_size granularity
static inline size_t align_up(size_t size, size_t page_size) {
  return ((size + page_size - 1) / page_size) * page_size;
}

static inline VirPtr alloc_virtual_mem(size_t size) {
  size_t page_size =
      ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size();
  CHECK(size % page_size == 0)
      << "alloc size not aligned: " << size;  // Ensure alignment.

  VirPtr vaddr;
  vmm::create_vir_ptr(vaddr, size);
  return vaddr;
}

static inline void unmap_and_release_virtual_mem(VirPtr vaddr,
                                                 size_t size,
                                                 size_t page_size) {
  if (is_null_vir_ptr(vaddr)) {
    return;
  }

  for (size_t offset = 0; offset < size; offset += page_size) {
    VirPtr addr = add_vir_ptr_offset(vaddr, offset);
    vmm::unmap(addr, page_size);
  }
  vmm::release_vir_ptr(vaddr, size);
}

static inline void release_virtual_mem(VirPtr vaddr, size_t size) {
  if (is_null_vir_ptr(vaddr)) {
    return;
  }
  vmm::release_vir_ptr(vaddr, size);
}

static inline void return_pages_to_pool(
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

static inline void unmap_pages(
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

static inline void cleanup_pages_and_vmem(
    VirPtr vaddr,
    size_t size,
    size_t page_size,
    std::unordered_map<page_id_t, std::unique_ptr<PhyPage>>& mapping) {
  unmap_pages(vaddr, page_size, mapping);
  return_pages_to_pool(mapping);
  release_virtual_mem(vaddr, size);
}

static inline void free_preallocated_weight_pages(
    const std::vector<page_id_t>& page_ids) {
  if (page_ids.empty()) {
    return;
  }

  PhyPagePool::get_instance().free_weight_pages(page_ids);
  LOG(INFO) << "XTensor: freed " << page_ids.size()
            << " preallocated weight pages";
}

XTensor::XTensor(size_t size, torch::Dtype dtype, torch::Device dev)
    : vaddr_(0),
      size_(0),
      page_size_(
          ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size()),
      dtype_(dtype),
      dev_(dev) {
  // Align size to page_size_
  size_ = align_up(size, page_size_);
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
    unmap_and_release_virtual_mem(vaddr_, size_, page_size_);
    free_preallocated_weight_pages(preallocated_page_ids_);
    return;
  }

  cleanup_pages_and_vmem(vaddr_, size_, page_size_, mapping_);
}

bool XTensor::map(offset_t offset) {
  CHECK(offset % page_size_ == 0)
      << "Offset not aligned to page size: " << offset;

  page_id_t page_id = offset / page_size_;

  // Check if already mapped (idempotent: return true if already mapped)
  if (mapping_.find(page_id) != mapping_.end()) {
    return true;
  }

  // Get a physical page from pool
  auto phy_pages = PhyPagePool::get_instance().batch_get(1);
  if (phy_pages.empty()) {
    LOG(ERROR) << "Failed to get physical page from pool";
    return false;
  }

  // Map the physical page
  VirPtr vaddr = add_vir_ptr_offset(vaddr_, offset);
  PhyMemHandle phy_handle = phy_pages[0]->get_phy_handle();
  vmm::map(vaddr, phy_handle);

  mapping_[page_id] = std::move(phy_pages[0]);
  return true;
}

bool XTensor::unmap(offset_t offset) {
  CHECK(offset % page_size_ == 0)
      << "Offset not aligned to page size: " << offset;

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
  pages_to_return.push_back(std::move(it->second));
  mapping_.erase(it);
  PhyPagePool::get_instance().batch_put(pages_to_return);

  return true;
}

bool XTensor::map_all() {
  for (size_t offset = 0; offset < size_; offset += page_size_) {
    if (!map(offset)) {
      LOG(ERROR) << "Failed to map page at offset " << offset;
      return false;
    }
  }
  return true;
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
  auto& pool = PhyPagePool::get_instance();
  const auto& all_pages = pool.get_all_pages();

  for (size_t i = 0; i < page_ids.size(); ++i) {
    page_id_t phy_page_id = page_ids[i];

    if (phy_page_id < 0 ||
        static_cast<size_t>(phy_page_id) >= all_pages.size()) {
      LOG(ERROR) << "XTensor::map_with_page_ids: invalid page_id "
                 << phy_page_id;
      return false;
    }

    PhyPage* page = all_pages[phy_page_id];
    if (page == nullptr) {
      LOG(ERROR) << "XTensor::map_with_page_ids: null page at page_id "
                 << phy_page_id;
      return false;
    }

    // Map the physical page to the i-th position in virtual space
    size_t offset = i * page_size_;
    VirPtr vaddr = add_vir_ptr_offset(vaddr_, offset);

    PhyMemHandle phy_handle = page->get_phy_handle();
    vmm::map(vaddr, phy_handle);

    // Note: we don't store in mapping_ since we don't own these pages
    // They will be freed via free_weight_pages() in PhyPagePool
  }

  LOG(INFO) << "XTensor::map_with_page_ids: mapped " << page_ids.size()
            << " preallocated pages";
  return true;
}

bool XTensor::map_phy_page_(PhyPage* page, offset_t offset) {
  CHECK(offset % page_size_ == 0)
      << "Offset not aligned to page size: " << offset;
  CHECK(page) << "Page is null";

  VirPtr vaddr = add_vir_ptr_offset(vaddr_, offset);
  PhyMemHandle phy_handle = page->get_phy_handle();
  vmm::map(vaddr, phy_handle);
  return true;
}

bool XTensor::allocate(void*& ptr, size_t size) {
  // Check if there's enough space
  if (alloc_offset_ + size > size_) {
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
  auto num_elems = static_cast<int64_t>(size_ / torch::elementSize(dtype_));
  return to_torch_tensor(0, {num_elems});
}

page_id_t XTensor::get_phy_page_id(offset_t offset) const {
  CHECK(offset % page_size_ == 0)
      << "Offset not aligned to page size: " << offset;

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
  uintptr_t addr = vir_ptr_to_uintptr(vaddr_) + offset;
  auto dtype = dtype_;

#if defined(USE_NPU)
  c10::DeviceType device_type = c10::DeviceType::PrivateUse1;
  torch::TensorOptions option =
      torch::TensorOptions().dtype(dtype).device(device_type);

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
      torch::TensorOptions().dtype(dtype).device(dev_).requires_grad(false);
  return torch::from_blob(reinterpret_cast<void*>(addr), dims, options);
#endif
}

}  // namespace xllm
