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

#include "sleepable_allocator.h"

#include <glog/logging.h>

#include <algorithm>

namespace xllm {
namespace {

inline size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

const char* to_string(MemTag tag) {
  switch (tag) {
    case MemTag::WEIGHTS:
      return "weights";
    case MemTag::KV_CACHE:
      return "kv_cache";
    default:
      return "unknown";
  }
}

// ============================== SleepableRegion ==============================

SleepableRegion::SleepableRegion(MemTag tag,
                                 const torch::Device& device,
                                 size_t bytes,
                                 size_t chunk_bytes)
    : tag_(tag),
      device_(device),
      device_id_(device.index() < 0 ? 0 : device.index()) {
  CHECK_GT(bytes, 0u) << "SleepableRegion size must be > 0";

  const size_t granularity = vmm::get_recommended_granularity(device_id_);
  CHECK_GT(granularity, 0u) << "Invalid allocation granularity";

  // chunk_bytes == 0 => map the whole region as one chunk (minimal waste for
  // per-layer weight buffers). Otherwise honor the requested chunk size.
  size_t chunk = (chunk_bytes == 0) ? bytes : chunk_bytes;
  chunk_bytes_ = align_up(chunk, granularity);
  aligned_size_ = align_up(bytes, chunk_bytes_);

  vmm::create_vir_ptr(vaddr_, aligned_size_);

  VLOG(1) << "SleepableRegion[" << to_string(tag_)
          << "] reserved vaddr=" << base() << ", requested=" << bytes
          << ", aligned_size=" << aligned_size_
          << ", chunk_bytes=" << chunk_bytes_
          << ", num_chunks=" << (aligned_size_ / chunk_bytes_)
          << ", device_id=" << device_id_;
}

SleepableRegion::~SleepableRegion() {
  if (mapped_) {
    unmap();
  }
  vmm::release_vir_ptr(vaddr_, aligned_size_);
}

void SleepableRegion::map() {
  if (mapped_) {
    return;
  }
  const size_t num_chunks = aligned_size_ / chunk_bytes_;
  handles_.reserve(num_chunks);
  for (size_t i = 0; i < num_chunks; ++i) {
    PhyMemHandle handle;
    vmm::create_phy_mem_handle(handle, device_id_, chunk_bytes_);
    VirPtr addr = add_vir_ptr_offset(vaddr_, i * chunk_bytes_);
    vmm::map(addr, handle, chunk_bytes_, device_id_);
    handles_.emplace_back(handle);
  }
  mapped_ = true;
}

void SleepableRegion::unmap() {
  if (!mapped_) {
    return;
  }
  const size_t num_chunks = aligned_size_ / chunk_bytes_;
  for (size_t i = 0; i < num_chunks; ++i) {
    VirPtr addr = add_vir_ptr_offset(vaddr_, i * chunk_bytes_);
    vmm::unmap_chunk(addr, chunk_bytes_);
    vmm::release_phy_mem_handle(handles_[i]);
  }
  handles_.clear();
  mapped_ = false;
}

// ============================ SleepableAllocator ============================

SleepableAllocator& SleepableAllocator::get_instance() {
  static SleepableAllocator instance;
  return instance;
}

void* SleepableAllocator::reserve_and_map(MemTag tag,
                                          const torch::Device& device,
                                          size_t bytes,
                                          size_t chunk_bytes) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto region =
      std::make_unique<SleepableRegion>(tag, device, bytes, chunk_bytes);
  region->map();
  void* base = region->base();
  regions_.emplace_back(std::move(region));
  return base;
}

void SleepableAllocator::sleep() {
  std::lock_guard<std::mutex> lock(mtx_);
  if (sleeping_) {
    LOG(WARNING) << "SleepableAllocator already sleeping";
    return;
  }
  size_t released = 0;
  size_t weight_regions = 0;
  size_t weight_bytes = 0;
  size_t kv_bytes = 0;
  for (auto& region : regions_) {
    if (region->is_mapped()) {
      released += region->size();
      if (region->tag() == MemTag::WEIGHTS) {
        ++weight_regions;
        weight_bytes += region->size();
      } else {
        kv_bytes += region->size();
      }
    }
    region->unmap();
  }
  sleeping_ = true;
  LOG(INFO) << "SleepableAllocator: deep sleep released " << regions_.size()
            << " region(s), " << released << " bytes (weights=" << weight_bytes
            << " B in " << weight_regions << " layer(s), kv=" << kv_bytes
            << " B)";
}

void SleepableAllocator::wake_up(const std::vector<MemTag>& tags) {
  std::lock_guard<std::mutex> lock(mtx_);
  for (auto& region : regions_) {
    const bool selected =
        tags.empty() ||
        std::find(tags.begin(), tags.end(), region->tag()) != tags.end();
    if (selected) {
      region->map();
    }
  }
  sleeping_ = false;
  LOG(INFO) << "SleepableAllocator: wake_up (tags=" << tags.size() << ")";
}

bool SleepableAllocator::is_sleeping() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return sleeping_;
}

size_t SleepableAllocator::mapped_bytes() const {
  std::lock_guard<std::mutex> lock(mtx_);
  size_t total = 0;
  for (auto& region : regions_) {
    if (region->is_mapped()) {
      total += region->size();
    }
  }
  return total;
}

}  // namespace xllm
