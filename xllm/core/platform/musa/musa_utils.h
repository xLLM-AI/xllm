/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <glog/logging.h>
#include <musa_runtime.h>

#include <cstdint>

namespace xllm::musa {

inline int32_t get_device_multiprocessor_count(int32_t device_id) {
  musaDeviceProp properties{};
  musaError_t error = musaGetDeviceProperties(&properties, device_id);
  if (error != musaSuccess) {
    LOG(FATAL) << "Failed to get properties for MUSA device " << device_id
               << ": " << musaGetErrorString(error);
  }
  return properties.multiProcessorCount;
}

}  // namespace xllm::musa
