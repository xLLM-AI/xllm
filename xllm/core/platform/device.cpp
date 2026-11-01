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

#include "device.h"

#include <torch/torch.h>

#include <limits>

#include "core/framework/config/model_config.h"
#include "core/platform/platform.h"
#if defined(USE_NPU)
#include <Python.h>
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/libs/init_npu.h>
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
#elif defined(USE_MUSA)
#include <c10/musa/MUSAGuard.h>
#include <musa.h>
#include <torch_musa/csrc/core/MUSACachingAllocator.h>
#elif defined(USE_DCU)
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime_api.h>
#endif

namespace xllm {

Device::Device(const torch::Device& device) : device_(device) {
#if defined(USE_CUDA) || defined(USE_MUSA)
  Platform::init_capabilities(device.index());
#endif
}

Device::Device(const int32_t device_index)
    : device_(torch::Device(Platform::type_torch(), device_index)) {
#if defined(USE_CUDA) || defined(USE_MUSA)
  Platform::init_capabilities(device_index);
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
  if (ModelConfig::is_python_model_impl(
          ModelConfig::get_instance().model_impl())) {
    // Python path: full NPU runtime already initialized in main() via
    // aclInit + torch_npu._C._npu_init(). Only switch device here.
    c10_npu::SetDevice(index());
  } else {
    // torch_npu lazily resolves python functions via
    // PyImport_ImportModule(). When a live host interpreter exists
    // (offline pybind inference), the calling worker thread must hold the
    // GIL, otherwise the import races with the main thread and corrupts
    // the interpreter state. The master constructors release the GIL
    // during engine init, so acquiring it here cannot deadlock. Standalone
    // binaries without a live interpreter keep the legacy behavior.
    if (Py_IsInitialized()) {
      PyGILState_STATE gil_state = PyGILState_Ensure();
      torch_npu::init_npu(index());
      PyGILState_Release(gil_state);
    } else {
      torch_npu::init_npu(index());
    }
  }
#endif
}

// set device before get device mem
std::optional<Device::DeviceMem> Device::query_device_mem() const {
  DeviceMem device_mem;
  size_t total_memory = 0;
  size_t free_memory = 0;
#if defined(USE_NPU)
  const aclError result =
      aclrtGetMemInfo(ACL_HBM_MEM, &free_memory, &total_memory);
  if (result != ACL_SUCCESS) {
    const char* error_message = aclGetRecentErrMsg();
    LOG(WARNING) << "Device memory query failed: api=aclrtGetMemInfo"
                 << " device_index=" << index()
                 << " driver_error=" << static_cast<int32_t>(result)
                 << " driver_error_message="
                 << (error_message == nullptr ? "unavailable" : error_message);
    return std::nullopt;
  }
#elif defined(USE_MLU)
  const cnrtRet_t result = cnrtMemGetInfo(&free_memory, &total_memory);
  if (result != cnrtSuccess) {
    LOG(WARNING) << "Device memory query failed: api=cnrtMemGetInfo"
                 << " device_index=" << index()
                 << " driver_error=" << static_cast<int32_t>(result);
    return std::nullopt;
  }
#elif defined(USE_CUDA) || defined(USE_ILU)
  const cudaError_t result = cudaMemGetInfo(&free_memory, &total_memory);
  if (result != cudaSuccess) {
    LOG(WARNING) << "Device memory query failed: api=cudaMemGetInfo"
                 << " device_index=" << index()
                 << " driver_error=" << static_cast<int32_t>(result);
    return std::nullopt;
  }
#elif defined(USE_MUSA)
  const musaError_t result = musaMemGetInfo(&free_memory, &total_memory);
  if (result != musaSuccess) {
    LOG(WARNING) << "Device memory query failed: api=musaMemGetInfo"
                 << " device_index=" << index()
                 << " driver_error=" << static_cast<int32_t>(result);
    return std::nullopt;
  }
#elif defined(USE_DCU)
  const hipError_t result = hipMemGetInfo(&free_memory, &total_memory);
  if (result != hipSuccess) {
    LOG(WARNING) << "Device memory query failed: api=hipMemGetInfo"
                 << " device_index=" << index()
                 << " driver_error=" << static_cast<int32_t>(result);
    return std::nullopt;
  }
#endif
  device_mem.total_memory = static_cast<int64_t>(total_memory);
  device_mem.free_memory = static_cast<int64_t>(free_memory);
  return device_mem;
}

std::optional<int64_t> Device::try_free_memory() const {
  const auto device_mem = query_device_mem();
  if (!device_mem.has_value()) {
    return std::nullopt;
  }
  return device_mem->free_memory;
}

int64_t Device::total_memory() {
  const auto device_mem = query_device_mem();
  return device_mem.has_value() ? device_mem->total_memory : 0;
}

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

int64_t Device::free_memory() {
  const auto device_mem = query_device_mem();
  return device_mem.has_value() ? device_mem->free_memory : 0;
}

std::optional<AllocatorMemoryStats> Device::allocator_memory_stats() const {
  if (device_.type() != Platform::type_torch() || index() < 0) {
    return std::nullopt;
  }
#if defined(USE_NPU)
  auto* allocator = c10_npu::NPUCachingAllocator::get();
  if (allocator == nullptr || allocator->name() != "native" ||
      !allocator->initialized()) {
    return std::nullopt;
  }
  const auto stats = allocator->getDeviceStats(index());
  const size_t stat_index =
      static_cast<size_t>(c10_npu::NPUCachingAllocator::StatType::AGGREGATE);
#elif defined(USE_CUDA) || defined(USE_ILU)
  auto* allocator = c10::cuda::CUDACachingAllocator::get();
  if (allocator == nullptr || allocator->name() != "native" ||
      !allocator->initialized()) {
    return std::nullopt;
  }
  const auto stats = allocator->getDeviceStats(index());
  const size_t stat_index =
      static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
#elif defined(USE_MUSA)
  auto* allocator = c10::musa::MUSACachingAllocator::get();
  if (allocator == nullptr || allocator->name() != "native" ||
      !allocator->initialized()) {
    return std::nullopt;
  }
  const auto stats = allocator->getDeviceStats(index());
  const size_t stat_index =
      static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
#else
  return std::nullopt;
#endif
#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_ILU) || \
    defined(USE_MUSA)
  const auto& reserved = stats.reserved_bytes[stat_index];
  if (reserved.current < 0 || reserved.peak < reserved.current ||
      static_cast<uint64_t>(reserved.peak) >
          std::numeric_limits<size_t>::max()) {
    return std::nullopt;
  }
  return AllocatorMemoryStats{static_cast<size_t>(reserved.current),
                              static_cast<size_t>(reserved.peak)};
#endif
}

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
  return std::make_unique<Stream>(device_, timeout);
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
