/* Copyright 2026 The xLLM Authors.

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

#include "core/platform/platform.h"

#include <mutex>

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#elif defined(USE_MLU)
#include <framework/core/device.h>
#elif defined(USE_CUDA) || defined(USE_ILU)
#include <c10/cuda/CUDACachingAllocator.h>

#include "core/platform/cuda/cuda_utils.h"
#elif defined(USE_MUSA)
#include <c10/musa/MUSAGuard.h>
#elif defined(USE_DCU)
#include <c10/hip/HIPCachingAllocator.h>
#endif

namespace xllm {

namespace {
std::once_flag g_init_flag;
}  // namespace

int32_t Platform::sm_count_ = 0;
bool Platform::enable_pdl_ = false;
bool Platform::support_sm90a_ = false;
bool Platform::support_sm100a_ = false;
bool Platform::support_sm100f_ = false;
bool Platform::support_sm120a_ = false;

std::string Platform::type_str() {
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

torch::DeviceType Platform::type_torch() {
#if defined(USE_NPU) || defined(USE_MLU)
  return torch::kPrivateUse1;
#elif defined(USE_CUDA) || defined(USE_ILU) || defined(USE_DCU)
  return torch::kCUDA;
#elif defined(USE_MUSA)
  return torch::kMUSA;
#endif
}

int32_t Platform::device_count() {
#if defined(USE_NPU)
  return static_cast<int32_t>(c10_npu::device_count());
#elif defined(USE_MLU)
  return static_cast<int32_t>(torch_mlu::device_count());
#elif defined(USE_CUDA) || defined(USE_ILU)
  return static_cast<int32_t>(c10::cuda::device_count());
#elif defined(USE_MUSA)
  return static_cast<int32_t>(c10::musa::device_count());
#elif defined(USE_DCU)
  return static_cast<int32_t>(c10::hip::device_count());
#endif
}

int32_t Platform::current_device() {
#if defined(USE_NPU)
  return static_cast<int32_t>(c10_npu::current_device());
#elif defined(USE_MLU)
  return static_cast<int32_t>(torch_mlu::current_device());
#elif defined(USE_CUDA) || defined(USE_ILU)
  return static_cast<int32_t>(c10::cuda::current_device());
#elif defined(USE_MUSA)
  return static_cast<int32_t>(c10::musa::current_device());
#elif defined(USE_DCU)
  return static_cast<int32_t>(c10::hip::current_device());
#endif
}

bool Platform::is_enable_pdl() { return enable_pdl_; }

int32_t Platform::sm_count() { return sm_count_; }

bool Platform::is_support_sm90a() { return support_sm90a_; }

bool Platform::is_support_sm100a() { return support_sm100a_; }

bool Platform::is_support_sm100f() { return support_sm100f_; }

bool Platform::is_support_sm120a() { return support_sm120a_; }

void Platform::init_capabilities(int32_t device_index) {
#if defined(USE_CUDA)
  std::call_once(g_init_flag, [device_index]() {
    sm_count_ = cuda::get_device_sm_count(device_index);
    enable_pdl_ = cuda::support_pdl(device_index);
    support_sm90a_ = cuda::support_sm90a(device_index);
    support_sm100a_ = cuda::support_sm100a(device_index);
    support_sm100f_ = cuda::support_sm100f(device_index);
    support_sm120a_ = cuda::support_sm120a(device_index);
  });
#else
  (void)device_index;
#endif
}

}  // namespace xllm
