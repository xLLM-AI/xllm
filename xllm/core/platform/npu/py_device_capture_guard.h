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

#pragma once

#include <c10/core/Device.h>
#include <glog/logging.h>

#include <mutex>
#include <optional>

#include "platform/npu/device_capture_lock.h"

namespace xllm {
namespace npu {

// Holds a device's capture lock across a capture driven from Python.
//
// A Python graph runner captures while the worker may be copying the next
// batch to the device. ACL graph capture breaks under that H2D even when the
// streams differ and the capture mode is thread-local -- see the comment in
// WorkerImpl::prepare_work_before_execute_on_stream_impl -- so such a capture
// has to hold the same per-device lock the native capture path holds.
//
// Acquisition is separate from construction because Python enters and leaves
// the lock as a context manager, and the acquiring call releases the GIL.
class PyDeviceCaptureGuard final {
 public:
  explicit PyDeviceCaptureGuard(int32_t device_index)
      : device_index_(static_cast<c10::DeviceIndex>(device_index)) {}

  void acquire() {
    CHECK(!lock_.has_value()) << "device capture lock is already held";
    lock_.emplace(DeviceCaptureLock::get_instance().get_lock(device_index_));
  }

  void release() { lock_.reset(); }

 private:
  c10::DeviceIndex device_index_;
  std::optional<std::unique_lock<std::mutex>> lock_;
};

}  // namespace npu
}  // namespace xllm
