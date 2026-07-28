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

#include "platform/mlu/mlu_layer_synchronizer.h"

#include <framework/core/MLUStream.h>
#include <glog/logging.h>

#include <exception>
#include <thread>

namespace xllm {
namespace {

size_t checked_layer_count(int64_t num_layers) {
  CHECK_GT(num_layers, 0) << "MLU layer synchronizer size must be positive.";
  return static_cast<size_t>(num_layers);
}

}  // namespace

MLULayerSynchronizerImpl::MLULayerSynchronizerImpl(int64_t num_layers)
    : events_(checked_layer_count(num_layers)),
      event_record_flags_(static_cast<size_t>(num_layers)) {
  for (std::atomic<bool>& flag : event_record_flags_) {
    flag.store(false, std::memory_order_relaxed);
  }
}

bool MLULayerSynchronizerImpl::valid_index(int64_t layer_index) const {
  if (layer_index < 0 ||
      static_cast<size_t>(layer_index) >= event_record_flags_.size()) {
    LOG(ERROR) << "MLU layer synchronizer index out of range: index="
               << layer_index << ", size=" << event_record_flags_.size();
    return false;
  }
  return true;
}

bool MLULayerSynchronizerImpl::synchronize_layer(int64_t layer_index) {
  if (!valid_index(layer_index)) {
    return false;
  }
  while (!event_record_flags_[layer_index].load(std::memory_order_acquire)) {
    if (aborted_.load(std::memory_order_acquire)) {
      return false;
    }
    std::this_thread::yield();
  }
  if (aborted_.load(std::memory_order_acquire) ||
      events_[layer_index] == nullptr) {
    return false;
  }
  try {
    events_[layer_index]->c10_event().synchronize();
    return true;
  } catch (const std::exception& error) {
    LOG(ERROR) << "MLU layer event synchronize failed: range=" << layer_index
               << ", error=" << error.what();
  } catch (...) {
    LOG(ERROR) << "MLU layer event synchronize failed: range=" << layer_index
               << ", unknown error.";
  }
  abort();
  return false;
}

bool MLULayerSynchronizerImpl::record_stream(int64_t layer_index,
                                             Stream* stream) {
  if (!valid_index(layer_index) || stream == nullptr ||
      aborted_.load(std::memory_order_acquire)) {
    abort();
    return false;
  }
  StreamEventPtr event = stream->record_event();
  if (event == nullptr) {
    LOG(ERROR) << "Failed to record MLU copy event for range=" << layer_index;
    abort();
    return false;
  }
  events_[layer_index] = std::move(event);
  event_record_flags_[layer_index].store(true, std::memory_order_release);
  return true;
}

bool MLULayerSynchronizerImpl::record_current(int64_t layer_index,
                                              int32_t device_index) {
  try {
    Stream current_stream(torch_mlu::getCurrentMLUStream(device_index));
    return record_stream(layer_index, &current_stream);
  } catch (const std::exception& error) {
    LOG(ERROR) << "Failed to get current MLU stream: device=" << device_index
               << ", error=" << error.what();
  } catch (...) {
    LOG(ERROR) << "Failed to get current MLU stream: device=" << device_index
               << ", unknown error.";
  }
  abort();
  return false;
}

void MLULayerSynchronizerImpl::abort() {
  aborted_.store(true, std::memory_order_release);
  for (std::atomic<bool>& flag : event_record_flags_) {
    flag.store(true, std::memory_order_release);
  }
}

}  // namespace xllm
