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

#pragma once

#include <c10/core/Device.h>

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include "common/macros.h"

namespace xllm {
namespace cuda {

// Device-level read-write lock manager for protecting CUDA Graph capture
// operations. Uses shared_mutex to allow multiple execute/replay operations to
// run concurrently while ensuring capture operations have exclusive access.
// Prevents prepare streams and capture streams from executing conflicting GPU
// work concurrently on the same device during cudaStreamCaptureModeGlobal
// capture windows.
class DeviceCaptureLock {
 public:
  // Get singleton instance.
  static DeviceCaptureLock& get_instance() {
    static DeviceCaptureLock instance;
    return instance;
  }

  // Get write lock (exclusive) for a specific device.
  // Use this for capture operations that require exclusive access.
  // Creates a new shared_mutex if one doesn't exist for the device.
  std::shared_mutex& get_write_lock(c10::DeviceIndex device_index) {
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    auto it = locks_.find(device_index);
    if (it == locks_.end()) {
      locks_[device_index] = std::make_unique<std::shared_mutex>();
      return *locks_[device_index];
    }
    return *it->second;
  }

  // Get read lock (shared) for a specific device.
  // Use this for replay/execute operations that can run concurrently.
  // Creates a new shared_mutex if one doesn't exist for the device.
  // Note: Returns the same shared_mutex as get_write_lock, but semantically
  // indicates shared access.
  std::shared_mutex& get_read_lock(c10::DeviceIndex device_index) {
    // Same implementation as get_write_lock - returns the same mutex
    // The difference is in how callers use it (unique_lock vs shared_lock)
    return get_write_lock(device_index);
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(DeviceCaptureLock);
  DeviceCaptureLock() = default;
  ~DeviceCaptureLock() = default;

  // Map from device index to shared_mutex.
  std::unordered_map<c10::DeviceIndex, std::unique_ptr<std::shared_mutex>>
      locks_;
  // Mutex to protect the map itself.
  std::mutex map_mutex_;
};

}  // namespace cuda
}  // namespace xllm
