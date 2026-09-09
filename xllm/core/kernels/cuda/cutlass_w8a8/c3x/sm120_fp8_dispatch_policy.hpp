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

#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "cutlass_extensions/common.hpp"

namespace xllm {
namespace kernel {
namespace cuda {

struct SM120DispatchConfig {
  static constexpr uint32_t kSmallM = 16;
  static constexpr uint32_t kLargeM = 128;
  static constexpr uint32_t kLargeN = 8192;
  static constexpr uint32_t kLargeK = 4096;
  static constexpr float kEfficiencyMargin = 0.05f;
  static constexpr uint32_t kDefaultSMCount = 128;
};

enum class SM120DispatchKernel {
  kTile128x32Swap,
  kTile128x64,
  kTile128x64Swap,
  kTile128x128,
};

struct SM120DispatchDecision {
  SM120DispatchKernel kernel;
  uint32_t tile_m;
  uint32_t tile_n;
  uint32_t tile_k;
  bool swap_ab;
};

inline constexpr std::array<SM120DispatchKernel, 4> kSM120DispatchKernels = {
    SM120DispatchKernel::kTile128x32Swap,
    SM120DispatchKernel::kTile128x64Swap,
    SM120DispatchKernel::kTile128x64,
    SM120DispatchKernel::kTile128x128,
};

inline float compute_sm120_wave_efficiency(uint32_t m,
                                           uint32_t n,
                                           uint32_t tile_m,
                                           uint32_t tile_n,
                                           uint32_t num_sms) {
  if (num_sms == 0) {
    num_sms = SM120DispatchConfig::kDefaultSMCount;
  }
  uint32_t tiles_m = (m + tile_m - 1) / tile_m;
  uint32_t tiles_n = (n + tile_n - 1) / tile_n;
  uint32_t total_tiles = tiles_m * tiles_n;
  uint32_t num_waves = (total_tiles + num_sms - 1) / num_sms;
  return static_cast<float>(total_tiles) /
         static_cast<float>(num_waves * num_sms);
}

inline bool should_use_sm120_swap_ab(uint32_t m, uint32_t n, uint32_t k) {
  if (m <= SM120DispatchConfig::kSmallM) {
    return true;
  }
  if (m <= SM120DispatchConfig::kLargeM) {
    return n >= SM120DispatchConfig::kLargeN ||
           k >= SM120DispatchConfig::kLargeK;
  }
  return false;
}

inline bool should_use_sm120_tile_128x128(uint32_t m,
                                          uint32_t n,
                                          uint32_t num_sms) {
  float eff_128x128 = compute_sm120_wave_efficiency(m, n, 128, 128, num_sms);
  float eff_128x64 = compute_sm120_wave_efficiency(m, n, 128, 64, num_sms);
  return eff_128x128 >= eff_128x64 - SM120DispatchConfig::kEfficiencyMargin;
}

inline uint32_t get_sm120_num_sms_for_device(int device) {
  static std::mutex mutex;
  static std::unordered_map<int, uint32_t> cache;

  std::lock_guard<std::mutex> guard(mutex);
  auto it = cache.find(device);
  if (it != cache.end()) {
    return it->second;
  }

  uint32_t count = static_cast<uint32_t>(get_device_sm_count(device));
  if (count == 0) {
    count = SM120DispatchConfig::kDefaultSMCount;
  }
  cache.emplace(device, count);
  return count;
}

inline SM120DispatchDecision select_sm120_dispatch(uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   uint32_t num_sms) {
  if (m <= SM120DispatchConfig::kSmallM) {
    return {SM120DispatchKernel::kTile128x32Swap, 128, 32, 128, true};
  }
  if (m <= SM120DispatchConfig::kLargeM) {
    if (should_use_sm120_swap_ab(m, n, k)) {
      return {SM120DispatchKernel::kTile128x64Swap, 128, 64, 128, true};
    }
    return {SM120DispatchKernel::kTile128x64, 128, 64, 128, false};
  }
  if (should_use_sm120_tile_128x128(m, n, num_sms)) {
    return {SM120DispatchKernel::kTile128x128, 128, 128, 128, false};
  }
  return {SM120DispatchKernel::kTile128x64, 128, 64, 128, false};
}

inline SM120DispatchDecision select_sm120_dispatch_for_device(uint32_t m,
                                                              uint32_t n,
                                                              uint32_t k,
                                                              int device) {
  return select_sm120_dispatch(m, n, k, get_sm120_num_sms_for_device(device));
}

inline const char* get_sm120_dispatch_kernel_name(SM120DispatchKernel kernel) {
  switch (kernel) {
    case SM120DispatchKernel::kTile128x32Swap:
      return "128x32_swap";
    case SM120DispatchKernel::kTile128x64:
      return "128x64";
    case SM120DispatchKernel::kTile128x64Swap:
      return "128x64_swap";
    case SM120DispatchKernel::kTile128x128:
      return "128x128";
  }
  return "unknown";
}

inline const char* get_sm120_dispatch_tile_shape_desc(
    SM120DispatchKernel kernel) {
  switch (kernel) {
    case SM120DispatchKernel::kTile128x32Swap:
      return "128x32x128";
    case SM120DispatchKernel::kTile128x64:
    case SM120DispatchKernel::kTile128x64Swap:
      return "128x64x128";
    case SM120DispatchKernel::kTile128x128:
      return "128x128x128";
  }
  return "unknown";
}

inline bool is_sm120_swap_ab_kernel(SM120DispatchKernel kernel) {
  switch (kernel) {
    case SM120DispatchKernel::kTile128x32Swap:
    case SM120DispatchKernel::kTile128x64Swap:
      return true;
    case SM120DispatchKernel::kTile128x64:
    case SM120DispatchKernel::kTile128x128:
      return false;
  }
  return false;
}

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm
