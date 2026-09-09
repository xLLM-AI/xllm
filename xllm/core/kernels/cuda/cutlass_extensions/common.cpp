/* Copyright 2025-2026 The xLLM Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ===========================================================================*/

// ref to:
// https://github.com/vllm-project/vllm/blob/main/csrc/cutlass_extensions/common.cpp

#include "cutlass_extensions/common.hpp"

#include <glog/logging.h>

#include "core/platform/cuda/cuda_utils.h"

namespace xllm {
namespace kernel {
namespace cuda {

int32_t get_sm_version_num() {
  int32_t major_capability, minor_capability;
  cudaDeviceGetAttribute(
      &major_capability, cudaDevAttrComputeCapabilityMajor, 0);
  cudaDeviceGetAttribute(
      &minor_capability, cudaDevAttrComputeCapabilityMinor, 0);
  int32_t version_num = major_capability * 10 + minor_capability;
  return version_num;
}

int32_t get_device_sm_count(int device) {
  return xllm::cuda::get_device_sm_count(device);
}

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm
