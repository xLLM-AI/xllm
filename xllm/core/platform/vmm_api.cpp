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

#include "vmm_api.h"

#include <glog/logging.h>

#include <cstdint>

#include "common/global_flags.h"
#include "core/framework/config/kv_cache_config.h"

namespace xllm {
namespace vmm {

size_t get_recommended_granularity(int32_t device_id) {
  size_t granularity_size = 0;
#if defined(USE_NPU)
  aclrtPhysicalMemProp prop = {};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_HUGE;  // 2MB
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  prop.reserve = 0;

  // get the recommended granularity size
  int ret = aclrtMemGetAllocationGranularity(
      &prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity_size);
  CHECK_EQ(ret, 0) << "Failed to get allocation granularity";
#elif defined(USE_MLU)
  CNmemAllocationProp prop = {};
  // The memory allocation type requested, which must be
  // CN_MEM_ALLOCATION_TYPE_DEFAULT currently according to cndrv developer
  // guide.
  prop.type =
      CN_MEM_ALLOCATION_TYPE_DEFAULT;  //  same as CU_MEM_ALLOCATION_TYPE_PINNED
                                       //  in CUDA
  prop.location.type = CN_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  prop.requestedHandleTypes = CN_MEM_HANDLE_TYPE_NONE;
  prop.allocFlags.compressionType = CN_MEM_ALLOCATION_COMP_NONE;

  // get the recommended granularity size
  int ret = cnMemGetAllocationGranularity(
      &granularity_size, &prop, CN_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  CHECK_EQ(ret, 0) << "Failed to get allocation granularity";
#elif defined(USE_CUDA) || defined(USE_ILU)
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;

  // get the recommended granularity size FIRST
  int ret = cuMemGetAllocationGranularity(
      &granularity_size, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  CHECK_EQ(ret, 0) << "Failed to get allocation granularity";
#elif defined(USE_DCU)
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device_id;

  // get the recommended granularity size FIRST
  int ret = hipMemGetAllocationGranularity(
      &granularity_size, &prop, hipMemAllocationGranularityRecommended);
  CHECK_EQ(ret, 0) << "Failed to get allocation granularity";
#else
  (void)device_id;
#endif
  VLOG(10) << "Granularity size for physical page: " << granularity_size
           << "Bytes";
  return granularity_size;
}

void create_phy_mem_handle(PhyMemHandle& phy_mem_handle, int32_t device_id) {
  int ret = 0;
  const size_t granularity_size = get_recommended_granularity(device_id);
#if defined(USE_NPU)
  aclrtPhysicalMemProp prop = {};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_HUGE;  // 2MB
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  prop.reserve = 0;

  ret = aclrtMallocPhysical(&phy_mem_handle, granularity_size, &prop, 0);
#elif defined(USE_MLU)
  CNmemAllocationProp prop = {};
  // The memory allocation type requested, which must be
  // CN_MEM_ALLOCATION_TYPE_DEFAULT currently according to cndrv developer
  // guide.
  prop.type =
      CN_MEM_ALLOCATION_TYPE_DEFAULT;  //  same as CU_MEM_ALLOCATION_TYPE_PINNED
                                       //  in CUDA
  prop.location.type = CN_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  prop.requestedHandleTypes = CN_MEM_HANDLE_TYPE_NONE;
  prop.allocFlags.compressionType = CN_MEM_ALLOCATION_COMP_NONE;

  ret = cnMemCreate(&phy_mem_handle, granularity_size, &prop, 0);

  CNmemAccessDesc accessDesc = {};
  accessDesc.location.type = CN_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = device_id;
  accessDesc.accessFlags = CN_MEM_ACCESS_FLAGS_PROT_READWRITE;
  ret = cnMemSetAccess(phy_mem_handle, granularity_size, &accessDesc, 1);
#elif defined(USE_CUDA) || defined(USE_ILU)
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;

  // Now create physical memory with the correct granularity size
  ret = cuMemCreate(&phy_mem_handle, granularity_size, &prop, 0);
  // Note: cuMemSetAccess is called in map() after cuMemMap, not here
#elif defined(USE_DCU)
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device_id;

  ret = hipMemCreate(&phy_mem_handle, granularity_size, &prop, 0);
#endif
  CHECK_EQ(ret, 0) << "Failed to create physical memory handle";
  ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size(
      granularity_size);
}

void create_vir_ptr(VirPtr& vir_ptr, size_t aligned_size) {
  int ret = 0;
#if defined(USE_NPU)
  ret = aclrtReserveMemAddress(&vir_ptr, aligned_size, 0, nullptr, 0);
#elif defined(USE_MLU)
  ret = cnMemAddressReserve(&vir_ptr, aligned_size, 0, 0, 0);
#elif defined(USE_CUDA) || defined(USE_ILU)
  ret = cuMemAddressReserve(&vir_ptr, aligned_size, 0, 0, 0);
#elif defined(USE_DCU)
  ret = hipMemAddressReserve(&vir_ptr, aligned_size, 0, 0, 0);
#elif defined(USE_MUSA)
  ret = muMemAddressReserve(&vir_ptr, aligned_size, 0, 0, 0);
#endif
  CHECK_EQ(ret, 0) << "Failed to create virtual memory handle";
}

void release_phy_mem_handle(PhyMemHandle& phy_mem_handle) {
  int ret = 0;
#if defined(USE_NPU)
  ret = aclrtFreePhysical(phy_mem_handle);
#elif defined(USE_MLU)
  ret = cnMemRelease(phy_mem_handle);
#elif defined(USE_CUDA) || defined(USE_ILU)
  ret = cuMemRelease(phy_mem_handle);
#elif defined(USE_DCU)
  ret = hipMemRelease(phy_mem_handle);
#elif defined(USE_MUSA)
  ret = muMemRelease(phy_mem_handle);
#endif
  CHECK_EQ(ret, 0) << "Failed to release physical memory handle";
}

void release_vir_ptr(VirPtr& vir_ptr, size_t aligned_size) {
  int ret = 0;
#if defined(USE_NPU)
  ret = aclrtReleaseMemAddress(vir_ptr);
#elif defined(USE_MLU)
  ret = cnMemAddressFree(vir_ptr, aligned_size);
#elif defined(USE_CUDA) || defined(USE_ILU)
  ret = cuMemAddressFree(vir_ptr, aligned_size);
#elif defined(USE_DCU)
  ret = hipMemAddressFree(vir_ptr, aligned_size);
#elif defined(USE_MUSA)
  ret = muMemAddressFree(vir_ptr, aligned_size);
#endif
  CHECK_EQ(ret, 0) << "Failed to release virtual memory handle";
}

void map(VirPtr& vir_ptr, PhyMemHandle& phy_mem_handle, int32_t device_id) {
  map(vir_ptr,
      phy_mem_handle,
      static_cast<size_t>(
          ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size()),
      device_id);
}

void map(VirPtr& vir_ptr,
         PhyMemHandle& phy_mem_handle,
         size_t granularity_size,
         int32_t device_id) {
  int ret = 0;
#if defined(USE_NPU)
  ret = aclrtMapMem(vir_ptr, granularity_size, 0, phy_mem_handle, 0);
#elif defined(USE_MLU)
  ret = cnMemMap(vir_ptr, granularity_size, 0, phy_mem_handle, 0);
#elif defined(USE_CUDA) || defined(USE_ILU)
  ret = cuMemMap(vir_ptr, granularity_size, 0, phy_mem_handle, 0);
  CHECK_EQ(ret, 0) << "Failed to map virtual memory to physical memory";

  // Set access permissions on the mapped virtual address range
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = device_id;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  ret = cuMemSetAccess(vir_ptr, granularity_size, &accessDesc, 1);
#elif defined(USE_MUSA)
  ret = muMemMap(vir_ptr, granularity_size, 0, phy_mem_handle, 0);
  CHECK_EQ(ret, 0) << "Failed to map virtual memory to physical memory";

  // Set access permissions on the mapped virtual address range
  MUmemAccessDesc accessDesc = {};
  accessDesc.location.type = MU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = device_id;
  accessDesc.flags = MU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  ret = muMemSetAccess(vir_ptr, granularity_size, &accessDesc, 1);
#elif defined(USE_DCU)
  ret = hipMemMap(vir_ptr, granularity_size, 0, phy_mem_handle, 0);
  CHECK_EQ(ret, 0) << "Failed to map virtual memory to physical memory";

  // Set access permissions on the mapped virtual address range
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device_id;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  ret = hipMemSetAccess(vir_ptr, granularity_size, &accessDesc, 1);
#endif
  CHECK_EQ(ret, 0) << "Failed to set memory access permissions";
}

void unmap(VirPtr& vir_ptr, size_t aligned_size) {
#if defined(USE_NPU)
  // For NPU, `aclrtUnmapMem` unmaps the range previously mapped by
  // `aclrtMapMem` at the given virtual address. Since we map per-physical-page
  // (granularity_size) at different offsets, we must unmap each mapped segment
  // before releasing the reserved virtual address range.
  size_t granularity_size = static_cast<size_t>(
      ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size());
  if (granularity_size == 0) {
    granularity_size = aligned_size;
  }
  auto* base = reinterpret_cast<std::uint8_t*>(vir_ptr);
  for (size_t offset = 0; offset < aligned_size; offset += granularity_size) {
    void* addr = base + offset;
    int ret = aclrtUnmapMem(addr);
    CHECK_EQ(ret, 0) << "Failed to unmap virtual memory from physical memory";
  }
  return;
#elif defined(USE_MLU)
  int ret = 0;
  ret = cnMemUnmap(vir_ptr, aligned_size);
  CHECK_EQ(ret, 0) << "Failed to unmap virtual memory from physical memory";
#elif defined(USE_CUDA) || defined(USE_ILU)
  int ret = 0;
  ret = cuMemUnmap(vir_ptr, aligned_size);
  CHECK_EQ(ret, 0) << "Failed to unmap virtual memory from physical memory";
#elif defined(USE_DCU)
  int ret = 0;
  ret = hipMemUnmap(vir_ptr, aligned_size);
  CHECK_EQ(ret, 0) << "Failed to unmap virtual memory from physical memory";
#endif
}

}  // namespace vmm
}  // namespace xllm
