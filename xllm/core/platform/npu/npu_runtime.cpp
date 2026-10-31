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

#include "core/platform/npu/npu_runtime.h"

#include <glog/logging.h>
#include <pybind11/embed.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>

#include <limits>

namespace xllm::npu {

void init_npu(int32_t device_index) {
  CHECK_GE(device_index, 0);
  CHECK_LE(device_index, std::numeric_limits<c10::DeviceIndex>::max());
  CHECK(Py_IsInitialized())
      << "Initialize the process Python runtime before initializing an NPU";

  const c10::DeviceIndex index = static_cast<c10::DeviceIndex>(device_index);
  // Preserve the caller's device selection during Python lazy initialization.
  c10_npu::set_device(index);
  {
    pybind11::gil_scoped_acquire gil;
    pybind11::module_::import("torch_npu.npu").attr("init")();
  }
  c10_npu::set_device(index);
}

}  // namespace xllm::npu
