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

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(USE_NPU)
#include "acl/acl.h"
#elif defined(USE_MLU)
#include <cn_api.h>
#elif defined(USE_CUDA) || defined(USE_ILU)
#include <cuda.h>
#elif defined(USE_MUSA)
#include <musa.h>
#elif defined(USE_DCU)
#include <hip/hip_runtime_api.h>
#endif

namespace xllm {

#if defined(USE_NPU)
using VirPtr = void*;
using PhyMemHandle = aclrtDrvMemHandle;
#elif defined(USE_MLU)
using VirPtr = CNaddr;
using PhyMemHandle = CNmemGenericAllocationHandle;
#elif defined(USE_CUDA) || defined(USE_ILU)
using VirPtr = CUdeviceptr;
using PhyMemHandle = CUmemGenericAllocationHandle;
#elif defined(USE_MUSA)
using VirPtr = MUdeviceptr;
using PhyMemHandle = MUmemGenericAllocationHandle;
#elif defined(USE_DCU)
using VirPtr = void*;
using PhyMemHandle = hipMemGenericAllocationHandle_t;
#endif

template <typename T>
inline uintptr_t vir_ptr_to_uintptr_impl(T ptr) {
  if constexpr (std::is_pointer_v<T>) {
    return reinterpret_cast<uintptr_t>(ptr);
  } else {
    return static_cast<uintptr_t>(ptr);
  }
}

inline uintptr_t vir_ptr_to_uintptr(VirPtr ptr) {
  return vir_ptr_to_uintptr_impl(ptr);
}

template <typename T>
inline T uintptr_to_vir_ptr_impl(uintptr_t ptr) {
  if constexpr (std::is_pointer_v<T>) {
    return reinterpret_cast<T>(ptr);
  } else {
    return static_cast<T>(ptr);
  }
}

inline VirPtr uintptr_to_vir_ptr(uintptr_t ptr) {
  return uintptr_to_vir_ptr_impl<VirPtr>(ptr);
}

inline VirPtr add_vir_ptr_offset(VirPtr base, size_t offset_bytes) {
  return uintptr_to_vir_ptr(vir_ptr_to_uintptr(base) + offset_bytes);
}

inline void* vir_ptr_to_void_ptr(VirPtr ptr) {
  return reinterpret_cast<void*>(vir_ptr_to_uintptr(ptr));
}

inline bool is_null_vir_ptr(VirPtr ptr) { return vir_ptr_to_uintptr(ptr) == 0; }

namespace vmm {

// get the recommended granularity size for physical memory allocation
size_t get_recommended_granularity(int32_t device_id);

// create a physical memory handle for a specific device
void create_phy_mem_handle(PhyMemHandle& phy_mem_handle, int32_t device_id);

// create a virtual memory pointer with a specific aligned size
void create_vir_ptr(VirPtr& vir_ptr, size_t aligned_size);

// release a physical memory handle
void release_phy_mem_handle(PhyMemHandle& phy_mem_handle);

// release a virtual memory pointer with a specific aligned size
void release_vir_ptr(VirPtr& vir_ptr, size_t aligned_size);

// map a virtual memory pointer to a physical memory handle
void map(VirPtr& vir_ptr, PhyMemHandle& phy_mem_handle, int32_t device_id = 0);
void map(VirPtr& vir_ptr,
         PhyMemHandle& phy_mem_handle,
         size_t granularity_size,
         int32_t device_id = 0);

// unmap a virtual memory pointer with a specific aligned size
void unmap(VirPtr& vir_ptr, size_t aligned_size);

}  // namespace vmm
}  // namespace xllm
