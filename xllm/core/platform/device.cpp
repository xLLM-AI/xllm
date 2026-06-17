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

#include "device.h"
#if defined(USE_NPU)
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#elif defined(USE_MLU)
#include <cn_api.h>
#include <framework/core/caching_allocator.h>
#include <framework/core/device.h>
#include <framework/core/device_utils.h>
#include <framework/generator/generator_impl.h>
#elif defined(USE_CUDA) || defined(USE_ILU)
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>

#include "cuda/cuda_utils.h"
#elif defined(USE_MUSA)
#include <c10/musa/MUSAGuard.h>
#include <musa.h>
#elif defined(USE_DCU)
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime_api.h>
#endif

namespace xllm {

int32_t Device::sm_count_ = 0;
bool Device::enable_pdl_ = false;
bool Device::support_sm90a_ = false;
bool Device::support_sm100a_ = false;
bool Device::support_sm100f_ = false;
bool Device::support_sm120a_ = false;

Device::Device(const torch::Device& device) : device_(device) {
#if defined(USE_CUDA)
  static int32_t sm_count = cuda::get_device_sm_count(device.index());
  sm_count_ = sm_count;

  static bool enable_pdl = cuda::support_pdl(device.index());
  enable_pdl_ = enable_pdl;

  static bool is_sm90a_supported = cuda::support_sm90a(device.index());
  support_sm90a_ = is_sm90a_supported;

  static bool is_sm100a_supported = cuda::support_sm100a(device.index());
  support_sm100a_ = is_sm100a_supported;

  static bool is_sm100f_supported = cuda::support_sm100f(device.index());
  support_sm100f_ = is_sm100f_supported;

  static bool is_sm120a_supported = cuda::support_sm120a(device.index());
  support_sm120a_ = is_sm120a_supported;
#endif
}

Device::Device(const int32_t device_index)
    : device_(torch::Device(type_torch(), device_index)) {
#if defined(USE_CUDA)
  static int32_t sm_count = cuda::get_device_sm_count(device_index);
  sm_count_ = sm_count;

  static bool enable_pdl = cuda::support_pdl(device_index);
  enable_pdl_ = enable_pdl;

  static bool is_sm90a_supported = cuda::support_sm90a(device_index);
  support_sm90a_ = is_sm90a_supported;

  static bool is_sm100a_supported = cuda::support_sm100a(device_index);
  support_sm100a_ = is_sm100a_supported;

  static bool is_sm100f_supported = cuda::support_sm100f(device_index);
  support_sm100f_ = is_sm100f_supported;

  static bool is_sm120a_supported = cuda::support_sm120a(device_index);
  support_sm120a_ = is_sm120a_supported;
#endif
}

Device::operator torch::Device() const { return unwrap(); }

void Device::set_device() const {
#if defined(USE_NPU)
  c10_npu::set_device(index());
#elif defined(USE_MLU)
  torch_mlu::setDevice(index());
#elif defined(USE_CUDA) || defined(USE_ILU)
  c10::cuda::set_device(index());
#elif defined(USE_MUSA)
  c10::musa::set_device(index());
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
#elif defined(USE_CUDA) || defined(USE_DCU)
  torch::cuda::manual_seed(seed);
#elif defined(USE_MUSA)
  torch::manual_seed(seed);
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

int Device::device_count() {
#if defined(USE_NPU)
  return c10_npu::device_count();
#elif defined(USE_MLU)
  return torch_mlu::device_count();
#elif defined(USE_CUDA) || defined(USE_ILU)
  return c10::cuda::device_count();
#elif defined(USE_MUSA)
  return c10::musa::device_count();
#elif defined(USE_DCU)
  return c10::hip::device_count();
#endif
}

std::string Device::type_str() {
#if defined(USE_NPU)
  return "npu";
#elif defined(USE_MLU)
  return "mlu";
#elif defined(USE_CUDA)
  return "cuda";
#elif defined(USE_ILU)
  return "ilu";
#elif defined(USE_MUSA)
  return "musa";
#elif defined(USE_DCU)
  return "dcu";
#endif
}

torch::DeviceType Device::type_torch() {
#if defined(USE_NPU) || defined(USE_MLU)
  return torch::kPrivateUse1;
#elif defined(USE_CUDA) || defined(USE_ILU) || defined(USE_DCU)
  return torch::kCUDA;
#elif defined(USE_MUSA)
  return torch::kMUSA;
#endif
}

int32_t Device::sm_count() { return sm_count_; }

bool Device::is_enable_pdl() { return enable_pdl_; }

bool Device::is_support_sm90a() { return support_sm90a_; }

bool Device::is_support_sm100a() { return support_sm100a_; }

bool Device::is_support_sm100f() { return support_sm100f_; }

bool Device::is_support_sm120a() { return support_sm120a_; }

// set device before get device mem
Device::DeviceMem Device::get_device_mem() const {
  DeviceMem device_mem;
  size_t total_memory = 0;
  size_t free_memory = 0;
#if defined(USE_NPU)
  aclrtGetMemInfo(ACL_HBM_MEM, &free_memory, &total_memory);
#elif defined(USE_MLU)
  cnrtMemGetInfo(&free_memory, &total_memory);
#elif defined(USE_CUDA) || defined(USE_ILU)
  cudaMemGetInfo(&free_memory, &total_memory);
#elif defined(USE_MUSA)
  musaMemGetInfo(&free_memory, &total_memory);
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
#elif defined(USE_CUDA) || defined(USE_ILU)
  c10::cuda::CUDACachingAllocator::emptyCache();
#elif defined(USE_MUSA)
  c10::musa::MUSACachingAllocator::emptyCache();
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
#elif defined(USE_CUDA) || defined(USE_ILU)
  c10::cuda::getCurrentCUDAStream().synchronize();
#elif defined(USE_MUSA)
  c10::musa::getCurrentMUSAStream().synchronize();
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
#elif defined(USE_CUDA) || defined(USE_ILU)
  auto current_s = c10::cuda::getCurrentCUDAStream(index());
#elif defined(USE_MUSA)
  auto current_s = c10::musa::getCurrentMUSAStream(index());
#elif defined(USE_DCU)
  auto current_s = c10::hip::getCurrentHIPStream(index());
#endif
  return std::make_unique<Stream>(current_s);
}

}  // namespace xllm
