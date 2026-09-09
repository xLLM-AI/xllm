/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.
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
// https://github.com/vllm-project/vllm/blob/main/csrc/cutlass_extensions/common.hpp

#pragma once
// clang-format off
#include "cutlass/cutlass.h"
#include <climits>
#include "cuda_runtime.h"
#include <iostream>
// clang-format on

// Helper macro for checking CUTLASS errors
#define CUTLASS_CHECK(status)                       \
  {                                                 \
    cutlass::Status error = status;                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, \
                cutlassGetStatusString(error));     \
  }

namespace xllm {
namespace kernel {
namespace cuda {

inline int get_cuda_max_shared_memory_per_block_opt_in(int const device) {
  int max_shared_mem_per_block_opt_in = 0;
  cudaDeviceGetAttribute(&max_shared_mem_per_block_opt_in,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         device);
  return max_shared_mem_per_block_opt_in;
}

int32_t get_sm_version_num();

// Get the number of SMs (Streaming Multiprocessors) on a CUDA device.
int32_t get_device_sm_count(int device);

// Kernel wrapper to guard against compilation on unsupported architectures.
// Reduces binary size by excluding code paths that won't be executed.
template <typename Kernel>
struct enable_sm90_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

template <typename Kernel>
struct enable_sm90_only : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ == 900
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

template <typename Kernel>
struct enable_sm100_only : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ == 1000
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

template <typename Kernel>
struct enable_sm120_only : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ == 1200
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm
