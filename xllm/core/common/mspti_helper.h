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

#pragma once

#include <cstdint>
#include <iostream>

#ifdef USE_MSPTI
#include "mspti/mspti.h"
#endif
#include "mstx/ms_tools_ext.h"

namespace xllm {
class MstxRange {
 public:
  explicit MstxRange(const char* name);

  ~MstxRange();

 private:
  std::string name_;
  uint64_t mstx_id_;
  aclrtStream stream_ = nullptr;
};

#define CONCATENATE(x, y) x##y

#define LLM_MSTX_RANGE() \
  MstxRange CONCATENATE(llm_mstx_range_, __LINE__) { __PRETTY_FUNCTION__ }

#ifdef USE_MSPTI
class MsptiMetrics {
 public:
  explicit MsptiMetrics() = default;

  ~MsptiMetrics() = default;

  static void register_subscriber();

  static void release_subscriber();

  static void user_buffer_request(uint8_t** buffer,
                                  size_t* size,
                                  size_t* maxNumRecords);

  static void user_buffer_complete(uint8_t* buffer,
                                   size_t size,
                                   size_t validSize);

  static std::string handle_marker_event(msptiActivity* pRecord);

  static std::string handle_memory_event(msptiActivity* pRecord);

  static std::string handle_hccl_event(msptiActivity* pRecord);

  static std::string handle_kernel_event(msptiActivity* pRecord);

 private:
  static uint8_t* pBuffer_;
  static msptiSubscriberHandle subscriber_;
};
#endif
}  // namespace xllm
