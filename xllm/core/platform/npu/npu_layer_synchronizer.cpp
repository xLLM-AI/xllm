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

#include "npu_layer_synchronizer.h"

#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace xllm {

NPULayerSynchronizerImpl::NPULayerSynchronizerImpl(const int64_t num_layers,
                                                   const int32_t timeout)
    : events_(num_layers, nullptr),
      event_record_flags_(num_layers),
      timeout_(timeout) {
  uint32_t flags = ACL_EVENT_SYNC;
  for (int64_t i = 0; i < num_layers; ++i) {
    auto ret = aclrtCreateEventWithFlag(&events_[i], flags);
    CHECK(ret == ACL_SUCCESS) << "Create event failed:" << ret;
  }
}

NPULayerSynchronizerImpl::~NPULayerSynchronizerImpl() {
  for (int64_t i = 0; i < events_.size(); ++i) {
    aclrtDestroyEvent(events_[i]);
  }
}

aclrtEvent* NPULayerSynchronizerImpl::get_event(const int64_t layer_index) {
  return &events_[layer_index];
}

std::atomic<bool>* NPULayerSynchronizerImpl::get_event_flag(
    const int64_t layer_index) {
  return &event_record_flags_[layer_index];
}

bool NPULayerSynchronizerImpl::synchronize_layer(const int64_t layer_index) {
  while (!event_record_flags_[layer_index].load(std::memory_order_acquire)) {
    if (aborted_.load(std::memory_order_acquire)) {
      return false;
    }
  }
  if (aborted_.load(std::memory_order_acquire)) {
    return false;
  }
  auto ret = aclrtSynchronizeEventWithTimeout(events_[layer_index], timeout_);
  if (ret != ACL_SUCCESS) {
    LOG(ERROR) << "Synchronize event failed: " << ret;
    return false;
  }
  return true;
}

bool NPULayerSynchronizerImpl::record_event(const int64_t layer_index,
                                            const int32_t device_index) {
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_index).stream();
  auto ret = aclrtRecordEvent(events_[layer_index], stream);
  if (ret != ACL_SUCCESS) {
    LOG(ERROR) << "Record event failed: " << ret;
    return false;
  }
  event_record_flags_[layer_index].store(true, std::memory_order_release);
  return true;
}

void NPULayerSynchronizerImpl::abort() {
  // Set aborted_ before raising the flags so any thread released from the spin
  // observes the abort and reports failure rather than syncing a stale event.
  aborted_.store(true, std::memory_order_release);
  for (size_t i = 0; i < event_record_flags_.size(); ++i) {
    event_record_flags_[i].store(true, std::memory_order_release);
  }
}

}  // namespace xllm
