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

#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <nvtx3/nvToolsExt.h>
#include <torch/torch.h>

#include <utility>

namespace xllm::cuda {

class NvtxRange {
 public:
  NvtxRange(const std::string& name) { nvtxRangePush(name.c_str()); }

  ~NvtxRange() { nvtxRangePop(); }
};

inline int32_t get_device_sm_count(int32_t device_id) {
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
  if (err != cudaSuccess) {
    LOG(FATAL) << "Failed to get device properties for device " << device_id;
  }
  return prop.multiProcessorCount;
}

inline std::pair<int32_t, int32_t> get_compute_capability(int32_t device_id) {
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
  if (err != cudaSuccess) {
    LOG(FATAL) << "Failed to get compute capability for device " << device_id;
  }
  return std::make_pair(prop.major, prop.minor);
}

inline int32_t get_cuda_version() {
  int32_t version;
  cudaError_t err = cudaRuntimeGetVersion(&version);
  if (err != cudaSuccess) {
    LOG(FATAL) << "Failed to get CUDA version!";
  }
  return version;
}

inline bool cuda_version_at_least(int32_t major, int32_t minor) {
  int32_t version = get_cuda_version();
  return version >= major * 1000 + minor * 10;
}

// Whether to enable Programmatic Dependent Launch (PDL). See
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
// Only supported for >= sm90, and currently only for FA2, CUDA core, and
// trtllm-gen decode.
inline bool support_pdl(int32_t device_id) {
  auto [major, minor] = get_compute_capability(device_id);
  return major >= 9;
}

inline bool support_sm90a(int32_t device_id) {
  auto [major, minor] = get_compute_capability(device_id);
  return (major == 9) && cuda_version_at_least(12, 3);
}

inline bool support_sm100a(int32_t device_id) {
  auto [major, minor] = get_compute_capability(device_id);
  return (major == 10) && cuda_version_at_least(12, 8);
}

inline bool support_sm100f(int32_t device_id) {
  auto [major, minor] = get_compute_capability(device_id);
  return (major == 10) && cuda_version_at_least(12, 9);
}

inline bool support_sm120a(int32_t device_id) {
  auto [major, minor] = get_compute_capability(device_id);
  return (major == 12) && (minor == 0) && cuda_version_at_least(12, 8);
}

}  // namespace xllm::cuda
