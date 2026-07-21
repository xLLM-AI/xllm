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

#pragma once

#include <c10/core/Device.h>
#include <torch/torch.h>

#include <cstdint>
#include <string>

namespace xllm {

class Platform final {
 public:
  static std::string type_str();
  static torch::DeviceType type_torch();

  static constexpr bool is_cuda() {
#if defined(USE_CUDA)
    return true;
#else
    return false;
#endif
  }

  static constexpr bool is_npu() {
#if defined(USE_NPU)
    return true;
#else
    return false;
#endif
  }

  static constexpr bool is_mlu() {
#if defined(USE_MLU)
    return true;
#else
    return false;
#endif
  }

  // Temporary compatibility boundary for MLU CP. Remove this capability
  // after MLU moves CP input preparation into WorkerImpl.
  static constexpr bool uses_model_cp_partition() { return is_mlu(); }

  static constexpr bool is_ilu() {
#if defined(USE_ILU)
    return true;
#else
    return false;
#endif
  }

  static constexpr bool is_musa() {
#if defined(USE_MUSA)
    return true;
#else
    return false;
#endif
  }

  static constexpr bool is_dcu() {
#if defined(USE_DCU)
    return true;
#else
    return false;
#endif
  }

  static int32_t device_count();
  // Returns the logical index of the device bound to the current thread.
  // Valid only after the device has been set (e.g. via Device::set_device).
  static int32_t current_device();
  static int32_t sm_count();
  static bool is_enable_pdl();
  static bool is_support_sm90a();
  static bool is_support_sm100a();
  static bool is_support_sm100f();
  static bool is_support_sm120a();

  // Initialize the per-device capability flags. Must be called once a
  // physical device index is known (typically from Device's constructor).
  static void init_capabilities(int32_t device_index);

 private:
  // Only used for cuda; cached after the first init_capabilities call.
  static int32_t sm_count_;
  static bool enable_pdl_;
  static bool support_sm90a_;
  static bool support_sm100a_;
  static bool support_sm100f_;
  static bool support_sm120a_;
};

}  // namespace xllm
