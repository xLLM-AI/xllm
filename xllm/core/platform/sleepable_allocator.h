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

#pragma once

#include <torch/torch.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "vmm_api.h"

namespace xllm {

// Memory category managed by the sleep allocator (mirrors vllm-ascend's
// "weights" / "kv_cache" tags).
enum class MemTag : int32_t {
  WEIGHTS = 0,
  KV_CACHE = 1,
};

const char* to_string(MemTag tag);

// A VMM-backed contiguous device region whose physical HBM can be released
// (sleep) and re-acquired (wakeup) while the virtual address stays stable.
//
// Deep-sleep only: on sleep the physical memory is discarded (never offloaded);
// on wakeup fresh physical memory is mapped back to the SAME virtual address so
// tensors built over it (via from_blob) stay valid. Contents are garbage after
// wakeup (weights re-loaded, KV re-prefilled).
//
// Physical memory is mapped in large chunks (chunk_bytes, GiB-scale) instead of
// fixed 2MB pages, so map/unmap issue only a handful of driver calls. Unlike
// xtensor's PhyPagePool, sleep actually returns the physical memory to the
// driver (aclrtFreePhysical).
class SleepableRegion final {
 public:
  SleepableRegion(MemTag tag,
                  const torch::Device& device,
                  size_t bytes,
                  size_t chunk_bytes);
  ~SleepableRegion();

  SleepableRegion(const SleepableRegion&) = delete;
  SleepableRegion& operator=(const SleepableRegion&) = delete;

  void* base() const { return vir_ptr_to_void_ptr(vaddr_); }
  size_t size() const { return aligned_size_; }
  MemTag tag() const { return tag_; }
  bool is_mapped() const { return mapped_; }

  void map();    // wakeup: aclrtMallocPhysical + aclrtMapMem
  void unmap();  // sleep:  aclrtUnmapMem + aclrtFreePhysical (vaddr kept)

 private:
  MemTag tag_;
  torch::Device device_;
  int32_t device_id_;
  size_t aligned_size_;
  size_t chunk_bytes_;
  VirPtr vaddr_;
  bool mapped_ = false;
  std::vector<PhyMemHandle> handles_;
};

// Per-worker singleton owning the tagged VMM regions. Both KV cache and (in
// manual-loader mode) each decoder layer's contiguous weight buffer are
// allocated as regions here. Tensors are built directly over a region's base
// address (from_blob / convert_to_torch_tensor), so no device-allocator routing
// is required -- which is what makes this work on torch_npu, whose pluggable
// allocator is only reachable from Python.
class SleepableAllocator final {
 public:
  static SleepableAllocator& get_instance();

  // Reserve a region of exactly `bytes` and map it immediately. Returns the
  // stable base pointer the caller builds tensors on. chunk_bytes = 0 maps the
  // whole region as a single physical chunk (good for per-layer weight
  // buffers); pass an explicit chunk_bytes (e.g. 1 GiB) for very large regions
  // like KV.
  void* reserve_and_map(MemTag tag,
                        const torch::Device& device,
                        size_t bytes,
                        size_t chunk_bytes = 0);

  // Enable/query routing of manual-loader weight buffers into WEIGHTS regions.
  // Set by the worker before model load when RL sleep mode is on.
  void set_weights_enabled(bool enabled) { weights_enabled_ = enabled; }
  bool weights_enabled() const { return weights_enabled_; }

  // sleep (deep): unmap all regions, returning physical HBM to the driver.
  void sleep();
  // wakeup: re-map the given tags (empty = all).
  void wake_up(const std::vector<MemTag>& tags = {});

  bool is_sleeping() const;
  size_t mapped_bytes() const;

 private:
  SleepableAllocator() = default;

  mutable std::mutex mtx_;
  std::vector<std::unique_ptr<SleepableRegion>> regions_;
  std::atomic<bool> weights_enabled_{false};
  bool sleeping_ = false;
};

}  // namespace xllm
