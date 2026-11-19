/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <pto/pto-inst.hpp>

#include "kernel_operator.h"

namespace gdn_sync {
constexpr uint16_t kEventIdOffset = 8;
constexpr uint16_t kEventIdMask = 0xf;
constexpr uint16_t kVecCoreIdOffset = 16;

AICORE inline void InitAddress(uint64_t fftsAddr) {
#if defined(PTO_NPU_ARCH_A2A3)
  set_ffts_base_addr(fftsAddr);
#else
  (void)fftsAddr;
#endif
}

AICORE inline void VectorBarrier() {
#if defined(PTO_NPU_ARCH_A2A3)
  pipe_barrier(PIPE_V);
#else
  pipe_barrier(PIPE_ALL);
#endif
}

template <pipe_t Pipe>
AICORE inline void Signal(uint16_t config) {
#if defined(PTO_NPU_ARCH_A5)
  const uint16_t eventId = (config >> kEventIdOffset) & kEventIdMask;
  pipe_barrier(PIPE_ALL);
#if defined(__DAV_CUBE__)
  set_intra_block(Pipe, eventId);
  set_intra_block(Pipe, eventId + kVecCoreIdOffset);
#elif defined(__DAV_VEC__)
  set_intra_block(Pipe, eventId);
#endif
#else
  ffts_cross_core_sync(Pipe, config);
#endif
}

template <pipe_t Pipe>
AICORE inline void Wait(uint16_t eventId) {
#if defined(PTO_NPU_ARCH_A5)
  wait_intra_block(Pipe, eventId);
#if defined(__DAV_CUBE__)
  wait_intra_block(Pipe, eventId + kVecCoreIdOffset);
#endif
  pipe_barrier(PIPE_ALL);
#else
  wait_flag_dev(eventId);
#endif
}

AICORE inline void RecordVecGm(uint16_t config) {
#if defined(PTO_NPU_ARCH_A5)
  const uint16_t eventId = (config >> kEventIdOffset) & kEventIdMask;
  set_intra_block(PIPE_MTE3, eventId);
#else
  ffts_cross_core_sync(PIPE_MTE3, config);
#endif
}

AICORE inline void WaitVecGm(uint16_t eventId) {
#if defined(PTO_NPU_ARCH_A5)
  wait_intra_block(PIPE_MTE2, eventId);
#if defined(__DAV_CUBE__)
  wait_intra_block(PIPE_MTE2, eventId + kVecCoreIdOffset);
#endif
#else
  wait_flag_dev(eventId);
#endif
}

AICORE inline void AllocateVecGm(uint16_t eventId) {
#if defined(PTO_NPU_ARCH_A5)
  wait_intra_block(PIPE_MTE3, eventId);
#else
  wait_flag_dev(eventId);
#endif
}

AICORE inline void FreeVecGm(uint16_t config) {
#if defined(PTO_NPU_ARCH_A5)
  const uint16_t eventId = (config >> kEventIdOffset) & kEventIdMask;
  set_intra_block(PIPE_MTE1, eventId);
  set_intra_block(PIPE_MTE1, eventId + kVecCoreIdOffset);
#else
  ffts_cross_core_sync(PIPE_FIX, config);
#endif
}
}  // namespace gdn_sync
