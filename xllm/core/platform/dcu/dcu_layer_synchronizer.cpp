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

#include "platform/dcu/dcu_layer_synchronizer.h"

#include <c10/hip/HIPStream.h>
#include <glog/logging.h>

#include <chrono>
#include <cstddef>
#include <thread>

namespace xllm {
namespace {

constexpr int32_t kDefaultSynchronizeTimeoutMs = 300000;

}  // namespace

DCULayerSynchronizerImpl::DCULayerSynchronizerImpl(int64_t num_layers,
                                                   int32_t timeout_ms)
    : events_(static_cast<size_t>(num_layers)),
      event_record_flags_(static_cast<size_t>(num_layers)),
      timeout_ms_(timeout_ms) {
  size_t num_layers_size = static_cast<size_t>(num_layers);
  for (size_t i = 0; i < num_layers_size; ++i) {
    event_record_flags_[i].store(false, std::memory_order_relaxed);
    hipError_t ret =
        hipEventCreateWithFlags(&events_[i], hipEventDisableTiming);
    CHECK(ret == hipSuccess)
        << "hipEventCreateWithFlags failed: " << hipGetErrorString(ret);
  }
}

DCULayerSynchronizerImpl::~DCULayerSynchronizerImpl() {
  for (hipEvent_t& event : events_) {
    hipEventDestroy(event);
  }
}

bool DCULayerSynchronizerImpl::synchronize_layer(int64_t layer_index) {
  CHECK_GE(layer_index, 0) << "layer_index must be non-negative.";
  CHECK_LT(static_cast<size_t>(layer_index), events_.size())
      << "layer_index out of bounds.";
  size_t layer = static_cast<size_t>(layer_index);
  while (!event_record_flags_[layer].load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::milliseconds(timeout_ms_ > 0 ? timeout_ms_
                                                : kDefaultSynchronizeTimeoutMs);

  while (true) {
    hipError_t ret = hipEventQuery(events_[layer]);
    if (ret == hipSuccess) {
      return true;
    }
    if (ret != hipErrorNotReady) {
      LOG(ERROR) << "hipEventQuery failed for layer " << layer_index << ": "
                 << hipGetErrorString(ret);
      return false;
    }
    if (timeout_ms_ > 0 && std::chrono::steady_clock::now() > deadline) {
      LOG(ERROR) << "synchronize_layer timed out for layer " << layer_index;
      return false;
    }
    std::this_thread::yield();
  }
}

bool DCULayerSynchronizerImpl::record_current(int64_t layer_index,
                                              int32_t device_index) {
  CHECK_GE(layer_index, 0) << "layer_index must be non-negative.";
  CHECK_LT(static_cast<size_t>(layer_index), events_.size())
      << "layer_index out of bounds.";
  size_t layer = static_cast<size_t>(layer_index);
  c10::hip::HIPStream stream = c10::hip::getCurrentHIPStream(device_index);
  hipError_t ret = hipEventRecord(events_[layer], stream.stream());
  if (ret != hipSuccess) {
    LOG(ERROR) << "hipEventRecord failed for layer " << layer_index << ": "
               << hipGetErrorString(ret);
    return false;
  }
  event_record_flags_[layer].store(true, std::memory_order_release);
  return true;
}

uint32_t DCULayerSynchronizerImpl::get_event_size() const {
  return static_cast<uint32_t>(events_.size());
}

}  // namespace xllm
