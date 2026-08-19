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

#include "rolling_load_manager.h"

#include <acl/acl_rt.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

#include <algorithm>

#include "base_loader.h"
#include "platform/stream.h"

namespace xllm {

RollingLoadManager::RollingLoadManager(
    std::vector<layer::BaseLoader*> loaders,
    std::shared_ptr<layer::RollingWeightBuffer> rolling_buffer,
    Stream* load_stream,
    Stream* compute_stream,
    int32_t requested_rolling_slots)
    : loaders_(std::move(loaders)),
      rolling_buffer_(std::move(rolling_buffer)),
      load_stream_(load_stream),
      compute_stream_(compute_stream),
      num_slots_(rolling_buffer_->num_slots()),
      preload_count_(std::min<int32_t>(num_slots_, num_layers())),
      requested_rolling_slots_(requested_rolling_slots),
      layer_to_slot_(static_cast<size_t>(num_layers()), -1),
      next_in_slot_(static_cast<size_t>(num_layers()), -1),
      layers_in_slot_(static_cast<size_t>(num_slots_)),
      dirty_slots_(static_cast<size_t>(num_slots_), false),
      refilled_slots_(static_cast<size_t>(num_slots_), false) {
  CHECK(!loaders_.empty()) << "RollingLoadManager: loaders must not be empty";
  CHECK(rolling_buffer_ != nullptr)
      << "RollingLoadManager: rolling_buffer must not be null";
  CHECK(load_stream_ != nullptr)
      << "RollingLoadManager: load_stream must not be null";

  // Mixed-slot policy:
  // - Keep startup layers [0, preload_count_) preloaded in slots [0, preload).
  // - Use configurable first K slots as rolling slots for layers
  //   [preload_count_, num_layers).
  // - Remaining startup slots stay fixed and never overwritten.
  if (requested_rolling_slots_ < 0) {
    rolling_slots_ = std::min<int32_t>(2, preload_count_);
  } else {
    rolling_slots_ = std::min(requested_rolling_slots_, preload_count_);
  }
  const int32_t n = num_layers();
  if (n > preload_count_) {
    CHECK_GT(rolling_slots_, 0)
        << "Need rolling_slots > 0 when decoder layers exceed cached slots. "
        << "num_layers=" << n << ", preload_count=" << preload_count_
        << ", requested_rolling_slots=" << requested_rolling_slots_;
  }

  for (int32_t i = 0; i < n; ++i) {
    int32_t slot = -1;
    if (i < preload_count_) {
      slot = i;
    } else {
      slot = (i - preload_count_) % rolling_slots_;
    }
    layer_to_slot_[static_cast<size_t>(i)] = slot;
    layers_in_slot_[static_cast<size_t>(slot)].push_back(i);
  }
  for (const auto& slot_layers : layers_in_slot_) {
    for (size_t i = 0; i + 1 < slot_layers.size(); ++i) {
      next_in_slot_[static_cast<size_t>(slot_layers[i])] = slot_layers[i + 1];
    }
  }

  uint32_t flags = ACL_EVENT_SYNC;
  compute_events_.resize(n, nullptr);
  h2d_events_.resize(n, nullptr);
  for (int32_t i = 0; i < n; ++i) {
    CHECK_EQ(aclrtCreateEventWithFlag(&compute_events_[i], flags), ACL_SUCCESS)
        << "Failed to create compute_event[" << i << "]";
    CHECK_EQ(aclrtCreateEventWithFlag(&h2d_events_[i], flags), ACL_SUCCESS)
        << "Failed to create h2d_event[" << i << "]";
  }

  LOG(INFO) << "RollingLoadManager: initialized with " << n << " layers, "
            << num_slots_ << " slots (preload=" << preload_count_
            << ", rolling=" << rolling_slots_
            << ", requested_rolling=" << requested_rolling_slots_
            << ", fixed=" << (preload_count_ - rolling_slots_) << "), " << n
            << " events each";
}

RollingLoadManager::~RollingLoadManager() {
  for (auto& e : compute_events_) {
    if (e) aclrtDestroyEvent(e);
  }
  for (auto& e : h2d_events_) {
    if (e) aclrtDestroyEvent(e);
  }
}

void RollingLoadManager::kick_h2d(int32_t layer_index) {
  aclrtStream ls = load_stream_->get_stream()->stream();
  loaders_[layer_index]->copy_weights_to_device_async(ls);
}

int32_t RollingLoadManager::slot_for_layer(int32_t layer_index) const {
  CHECK_GE(layer_index, 0) << "layer_index must be >= 0";
  CHECK_LT(layer_index, num_layers()) << "layer_index out of range";
  return layer_to_slot_[static_cast<size_t>(layer_index)];
}

void RollingLoadManager::init_rolling_load() {
  aclrtStream ls = load_stream_->get_stream()->stream();
  std::fill(dirty_slots_.begin(), dirty_slots_.end(), false);
  std::fill(refilled_slots_.begin(), refilled_slots_.end(), false);

  // Device slots and AT/ATB tensor bindings are refreshed in
  // model_->init_rolling_model_state() before this preload.

  // Pre-load the first min(N, num_layers) layers into their slots.
  for (int32_t i = 0; i < preload_count_; ++i) {
    kick_h2d(i);
    CHECK_EQ(aclrtRecordEvent(h2d_events_[i], ls), ACL_SUCCESS)
        << "Failed to record h2d_event[" << i << "] during init";
  }

  LOG(INFO) << "RollingLoadManager: pre-loaded " << preload_count_
            << " layers into " << num_slots_ << " slots";
}

void RollingLoadManager::refresh_rolling_buffer_address() {
  CHECK(rolling_buffer_ != nullptr)
      << "RollingLoadManager: rolling_buffer must not be null";
  rolling_buffer_->refresh_address();
}

void RollingLoadManager::wait_layer_h2d_ready(int32_t layer_index) {
  aclrtStream cs = compute_stream_ ? compute_stream_->get_stream()->stream()
                                   : c10_npu::getCurrentNPUStream().stream();

  // compute_stream waits for H2D of this layer to finish (GPU-side,
  // non-blocking CPU).
  CHECK_EQ(aclrtStreamWaitEvent(cs, h2d_events_[layer_index]), ACL_SUCCESS)
      << "aclrtStreamWaitEvent(compute_stream, h2d_event[" << layer_index
      << "]) failed";
  // Reset event after wait so it can be re-recorded later.
  CHECK_EQ(aclrtResetEvent(h2d_events_[layer_index], cs), ACL_SUCCESS)
      << "aclrtResetEvent(h2d_event[" << layer_index << "]) failed";
}

void RollingLoadManager::schedule_next_layer_h2d(int32_t layer_index) {
  aclrtStream cs = compute_stream_ ? compute_stream_->get_stream()->stream()
                                   : c10_npu::getCurrentNPUStream().stream();
  aclrtStream ls = load_stream_->get_stream()->stream();

  // Record compute completion for this layer.
  CHECK_EQ(aclrtRecordEvent(compute_events_[layer_index], cs), ACL_SUCCESS)
      << "aclrtRecordEvent(compute_event[" << layer_index << "]) failed";

  const int32_t slot = slot_for_layer(layer_index);
  // Kick H2D for the layer that will next use the same slot.
  const int32_t next_layer = next_in_slot_[static_cast<size_t>(layer_index)];
  if (next_layer >= 0) {
    // load_stream waits for compute on this layer to finish before overwriting
    // the slot.
    CHECK_EQ(aclrtStreamWaitEvent(ls, compute_events_[layer_index]),
             ACL_SUCCESS)
        << "aclrtStreamWaitEvent(load_stream, compute_event[" << layer_index
        << "]) failed";
    CHECK_EQ(aclrtResetEvent(compute_events_[layer_index], ls), ACL_SUCCESS)
        << "aclrtResetEvent(compute_event[" << layer_index << "]) failed";

    // H2D: next_layer host -> mapped slot for this slot chain.
    kick_h2d(next_layer);
    dirty_slots_[static_cast<size_t>(slot)] = true;

    // Record H2D completion event for next_layer.
    CHECK_EQ(aclrtRecordEvent(h2d_events_[next_layer], ls), ACL_SUCCESS)
        << "aclrtRecordEvent(h2d_event[" << next_layer << "]) failed";
  } else if (slot < preload_count_ && dirty_slots_[static_cast<size_t>(slot)] &&
             !refilled_slots_[static_cast<size_t>(slot)]) {
    // Tail layer of this slot chain. Refill startup layer early to overlap
    // with remaining compute of later layers (if any).
    CHECK_EQ(aclrtStreamWaitEvent(ls, compute_events_[layer_index]),
             ACL_SUCCESS)
        << "aclrtStreamWaitEvent(load_stream, compute_event[" << layer_index
        << "]) failed for early refill";
    CHECK_EQ(aclrtResetEvent(compute_events_[layer_index], ls), ACL_SUCCESS)
        << "aclrtResetEvent(compute_event[" << layer_index
        << "]) failed for early refill";

    const int32_t startup_layer = slot;
    kick_h2d(startup_layer);
    CHECK_EQ(aclrtRecordEvent(h2d_events_[startup_layer], ls), ACL_SUCCESS)
        << "aclrtRecordEvent(h2d_event[" << startup_layer
        << "]) failed for early refill";

    dirty_slots_[static_cast<size_t>(slot)] = false;
    refilled_slots_[static_cast<size_t>(slot)] = true;
  }
}

int32_t RollingLoadManager::last_executed_layer_in_slot(
    int32_t slot,
    int32_t capped_last) const {
  if (slot < 0 || slot >= static_cast<int32_t>(layers_in_slot_.size())) {
    return -1;
  }
  const auto& slot_layers = layers_in_slot_[static_cast<size_t>(slot)];
  auto it =
      std::upper_bound(slot_layers.begin(), slot_layers.end(), capped_last);
  if (it == slot_layers.begin()) {
    return -1;
  }
  --it;
  return *it;
}

void RollingLoadManager::finalize(int32_t last_executed_layer) {
  if (last_executed_layer < 0) {
    return;
  }
  int32_t capped_last = std::min(last_executed_layer, num_layers() - 1);
  aclrtStream ls = load_stream_->get_stream()->stream();

  for (int32_t s = 0; s < preload_count_; ++s) {
    if (refilled_slots_[static_cast<size_t>(s)]) {
      continue;
    }
    const int32_t last = last_executed_layer_in_slot(s, capped_last);

    // If there is no queued load-stream wait for this slot after `last`,
    // finalize must wait on the last compute completion before touching slot s.
    if (last >= 0 && next_in_slot_[static_cast<size_t>(last)] < 0) {
      CHECK_EQ(aclrtStreamWaitEvent(ls, compute_events_[last]), ACL_SUCCESS)
          << "aclrtStreamWaitEvent(load_stream, compute_event[" << last
          << "]) failed in finalize";
      CHECK_EQ(aclrtResetEvent(compute_events_[last], ls), ACL_SUCCESS)
          << "aclrtResetEvent(compute_event[" << last
          << "]) failed in finalize";
    }

    // Restore only dirty slots; clean slots already hold startup layer s.
    if (dirty_slots_[static_cast<size_t>(s)]) {
      kick_h2d(s);
    }

    // Record h2d_events_[s] for next forward's wait_layer_h2d_ready(s).
    CHECK_EQ(aclrtRecordEvent(h2d_events_[s], ls), ACL_SUCCESS)
        << "aclrtRecordEvent(h2d_event[" << s << "]) failed in finalize";
  }
  std::fill(dirty_slots_.begin(), dirty_slots_.end(), false);
  std::fill(refilled_slots_.begin(), refilled_slots_.end(), false);
}

}  // namespace xllm
