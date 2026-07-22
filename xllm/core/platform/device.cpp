/* Copyright 2025-2026 The xLLM Authors.

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

#include "device.h"

#include "core/platform/platform.h"
#if defined(USE_NPU)
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#elif defined(USE_MLU)
#include <cn_api.h>
#include <framework/core/caching_allocator.h>
#include <framework/core/device.h>
#include <framework/core/device_utils.h>
#include <framework/generator/generator_impl.h>
#elif defined(USE_MUSA)
#include <c10/musa/MUSAGuard.h>
#include <musa.h>
#elif defined(USE_CUDA) || defined(USE_ILU)
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#elif defined(USE_DCU)
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime_api.h>
#endif

namespace xllm {

Device::Device(const torch::Device& device) : device_(device) {
#if defined(USE_MUSA) || defined(USE_CUDA)
  Platform::init_capabilities(device.index());
#endif
}

Device::Device(const int32_t device_index)
    : device_(torch::Device(Platform::type_torch(), device_index)) {
#if defined(USE_MUSA) || defined(USE_CUDA)
  Platform::init_capabilities(device_index);
#endif
}

Device::operator torch::Device() const { return unwrap(); }

void Device::set_device() const {
#if defined(USE_NPU)
  c10_npu::set_device(index());
#elif defined(USE_MLU)
  torch_mlu::setDevice(index());
#elif defined(USE_MUSA)
  c10::musa::set_device(index());
#elif defined(USE_CUDA) || defined(USE_ILU)
  c10::cuda::set_device(index());
#elif defined(USE_DCU)
  c10::hip::set_device(index());
#endif
}

void Device::set_seed(uint64_t seed) const {
  torch::manual_seed(seed);
#if defined(USE_NPU)
  auto gen = at_npu::detail::getDefaultNPUGenerator(index());
  gen.set_current_seed(seed);
#elif defined(USE_MLU)
  auto gen = torch_mlu::getDefaultMLUGenerator(index());
  {
    std::lock_guard<std::mutex> lock(gen.mutex());
    gen.set_current_seed(seed);
  }
#elif defined(USE_MUSA)
  torch::manual_seed(seed);
#elif defined(USE_CUDA) || defined(USE_DCU)
  torch::cuda::manual_seed(seed);
#endif
}

const torch::Device& Device::unwrap() const { return device_; }

int32_t Device::index() const { return device_.index(); }

// set device before init device context
void Device::init_device_context() const {
#if defined(USE_NPU)
  torch_npu::init_npu(index());
#endif
}

// set device before get device mem
Device::DeviceMem Device::get_device_mem() const {
  DeviceMem device_mem;
  size_t total_memory = 0;
  size_t free_memory = 0;
#if defined(USE_NPU)
  aclrtGetMemInfo(ACL_HBM_MEM, &free_memory, &total_memory);
#elif defined(USE_MLU)
  cnrtMemGetInfo(&free_memory, &total_memory);
#elif defined(USE_MUSA)
  musaMemGetInfo(&free_memory, &total_memory);
#elif defined(USE_CUDA) || defined(USE_ILU)
  cudaMemGetInfo(&free_memory, &total_memory);
#elif defined(USE_DCU)
  hipMemGetInfo(&free_memory, &total_memory);
#endif
  device_mem.total_memory = static_cast<int64_t>(total_memory);
  device_mem.free_memory = static_cast<int64_t>(free_memory);
  return device_mem;
}

int64_t Device::total_memory() { return get_device_mem().total_memory; }

void Device::empty_cache(int32_t device_index) {
  (void)device_index;
#if defined(USE_NPU)
  c10_npu::NPUCachingAllocator::emptyCache();
  c10_npu::NPUCachingAllocator::FreeDeviceCachedMemory(device_index);
#elif defined(USE_MLU)
  torch_mlu::MLUCachingAllocator::emptyCache();
#elif defined(USE_MUSA)
  c10::musa::MUSACachingAllocator::emptyCache();
#elif defined(USE_CUDA) || defined(USE_ILU)
  c10::cuda::CUDACachingAllocator::emptyCache();
#elif defined(USE_DCU)
  c10::hip::HIPCachingAllocator::emptyCache();
#endif
}

int64_t Device::free_memory() { return get_device_mem().free_memory; }

int Device::synchronize_default_stream() {
#if defined(USE_NPU)
  return aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(index()).stream());
#elif defined(USE_MLU)
  torch_mlu::getCurrentMLUStream(index()).synchronize();
#elif defined(USE_MUSA)
  c10::musa::getCurrentMUSAStream().synchronize();
#elif defined(USE_CUDA) || defined(USE_ILU)
  c10::cuda::getCurrentCUDAStream().synchronize();
#elif defined(USE_DCU)
  c10::hip::getCurrentHIPStream().synchronize();
#endif
  return 0;
}

std::unique_ptr<Stream> Device::get_stream_from_pool(const int32_t timeout) {
  return std::make_unique<Stream>(timeout);
}

std::unique_ptr<Stream> Device::current_stream() const {
#if defined(USE_NPU)
  auto current_s = c10_npu::getCurrentNPUStream(index());
#elif defined(USE_MLU)
  auto current_s = torch_mlu::getCurrentMLUStream(index());
#elif defined(USE_MUSA)
  auto current_s = c10::musa::getCurrentMUSAStream(index());
#elif defined(USE_CUDA) || defined(USE_ILU)
  auto current_s = c10::cuda::getCurrentCUDAStream(index());
#elif defined(USE_DCU)
  auto current_s = c10::hip::getCurrentHIPStream(index());
#endif
  return std::make_unique<Stream>(current_s);
}

}  // namespace xllm
