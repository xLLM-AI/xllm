/* Copyright 2026 The xLLM Authors.

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

#include "core/platform/platform.h"

#include <glog/logging.h>

#include <mutex>
#include <utility>

#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU) || \
    defined(USE_MUSA) || defined(USE_DCU)
#include "core/platform/cpu_binding.h"
#endif

#if defined(USE_NPU)
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>

#include "core/platform/npu/npu_cpu_topology.h"
#elif defined(USE_MLU)
#include <framework/core/device.h>
#elif defined(USE_CUDA) || defined(USE_ILU)
#include <c10/cuda/CUDACachingAllocator.h>

#include "core/platform/cuda/cuda_utils.h"
#elif defined(USE_MUSA)
#include <c10/musa/MUSAGuard.h>

#include "core/platform/musa/musa_utils.h"
#elif defined(USE_DCU)
#include <c10/hip/HIPCachingAllocator.h>
#include <torch/torch.h>
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

bool Platform::is_ascend950() {
#if defined(USE_NPU)
  const char* soc_name = aclrtGetSocName();
  return soc_name != nullptr &&
         std::string(soc_name).find("Ascend950") != std::string::npos;
#else
  return false;
#endif
}

void Platform::initialize_cpu_binding(int32_t device_index, bool bind_irq) {
  if (has_cpu_binding()) {
    return;
  }
#if defined(USE_NPU)
  const char* soc_name = aclrtGetSocName();
  auto binding = npu::get_npu_cpu_binding(device_index,
                                          soc_name == nullptr ? "" : soc_name);
  if (binding) {
    const auto irq_cpus = binding->plan.reserved_cpus;
    if (CpuBinding::get_instance().initialize(std::move(binding->plan)) &&
        bind_irq) {
      npu::bind_npu_irqs(irq_cpus, binding->device);
    }
  }
#elif defined(USE_CUDA) || defined(USE_MLU) || defined(USE_MUSA) || \
    defined(USE_DCU)
  CpuBindingTopology topology;
  topology.allowed_cpus = numa::get_thread_cpus();
  topology.cpu_nodes = numa::get_cpu_numa_nodes();
  const int32_t count = device_count();
  topology.device_ids.reserve(count);
  for (int32_t id = 0; id < count; ++id) {
    topology.device_ids.emplace_back(id);
    const int32_t node = numa::get_device_numa_node(id);
    if (node >= 0) {
      std::vector<int32_t> cpus;
      cpus.reserve(topology.allowed_cpus.size());
      for (int32_t cpu : topology.allowed_cpus) {
        const auto cpu_node = topology.cpu_nodes.find(cpu);
        if (cpu_node != topology.cpu_nodes.end() && cpu_node->second == node) {
          cpus.emplace_back(cpu);
        }
      }
      topology.affinity.emplace(id, std::move(cpus));
    }
  }
  CpuBindingOptions options;
  std::string error;
  auto plan = make_cpu_binding_plan(topology, device_index, options, &error);
  if (!plan) {
    LOG(WARNING) << "CPU binding skipped: " << error;
    return;
  }
  CpuBinding::get_instance().initialize(std::move(*plan));
#else
  LOG(WARNING) << "CPU binding is not supported on this platform";
#endif
}

bool Platform::has_cpu_binding() {
#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU) || \
    defined(USE_MUSA) || defined(USE_DCU)
  return CpuBinding::get_instance().initialized();
#else
  return false;
#endif
}

void Platform::refresh_cpu_binding() {
#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU) || \
    defined(USE_MUSA) || defined(USE_DCU)
  CpuBinding::get_instance().refresh_threads();
#endif
}

void Platform::refresh_cpu_binding_after_first_forward() {
#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU) || \
    defined(USE_MUSA) || defined(USE_DCU)
  CpuBinding::get_instance().refresh_after_first_forward();
#endif
}

void Platform::finish_cpu_binding_warmup() {
#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU) || \
    defined(USE_MUSA) || defined(USE_DCU)
  CpuBinding::get_instance().finish_warmup();
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
#elif defined(USE_MUSA)
  std::call_once(g_init_flag, [device_index]() {
    sm_count_ = musa::get_device_multiprocessor_count(device_index);
  });
#else
  (void)device_index;
#endif
}

}  // namespace xllm
