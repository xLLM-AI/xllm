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

#pragma once

// clang-format off
#if defined(USE_NPU)
#include "graph/types.h"
#endif
// clang-format on

#include <c10/core/Event.h>
#include <c10/core/Stream.h>
#include <c10/core/StreamGuard.h>

#include <cstdint>
#include <ostream>
#if defined(USE_NPU)
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/torch_npu.h>
#elif defined(USE_MLU)
#include <framework/core/MLUStream.h>
#elif defined(USE_MUSA)
#include <c10/musa/MUSAGuard.h>
#elif defined(USE_CUDA) || defined(USE_ILU)
#include <c10/cuda/CUDAStream.h>
#elif defined(USE_DCU)
#include <c10/hip/HIPStream.h>
#endif

#include "core/platform/stream_event.h"

namespace xllm {

#if defined(USE_NPU)
using PlatformStream = c10_npu::NPUStream;
#elif defined(USE_MLU)
using PlatformStream = torch_mlu::MLUStream;
#elif defined(USE_MUSA)
using PlatformStream = c10::musa::MUSAStream;
#elif defined(USE_CUDA) || defined(USE_ILU)
using PlatformStream = c10::cuda::CUDAStream;
#elif defined(USE_DCU)
using PlatformStream = c10::hip::HIPStream;
#endif

class Stream {
 public:
  explicit Stream(const int32_t timeout = -1);
  ~Stream() = default;

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  Stream(Stream&&) = default;
  Stream& operator=(Stream&&) = default;

  explicit Stream(PlatformStream stream, const int32_t timeout = -1);

  int synchronize() const;
  c10::StreamGuard set_stream_guard() const;
  void wait_event(const c10::Event& event);
  PlatformStream* get_stream() { return &stream_; }
  const PlatformStream* get_stream() const { return &stream_; }
  void wait_stream(const Stream& other_stream);
  StreamEventPtr record_event() const;
  bool wait_event(const StreamEventPtr& event) const;

  // Support for LOG(INFO) output
  friend std::ostream& operator<<(std::ostream& os, const Stream& stream);

 private:
  PlatformStream stream_;
  const int32_t timeout_;
};

}  // namespace xllm
