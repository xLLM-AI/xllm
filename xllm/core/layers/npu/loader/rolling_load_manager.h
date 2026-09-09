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

#include <acl/acl.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "rolling_weight_buffer.h"

namespace xllm {

class Stream;

namespace layer {
class BaseLoader;
}  // namespace layer

// RollingLoadManager orchestrates the rolling H2D weight pipeline for decoder
// layers. It keeps N slots in HBM and pipelines compute/load using two sets of
// ACL events, one per layer (not per slot):
//
//   h2d_events_[i]:     recorded on load_stream after H2D for layer i
//   completes.
//                       compute_stream waits on this before computing layer i.
//   compute_events_[i]: recorded on compute_stream after layer i forward.
//                       load_stream waits on this before overwriting the same
//                       slot with that slot's next mapped layer.
//
// Per-layer events avoid event reuse within a forward pass, which would cause
// races on ACL platforms where a "signaled" event can be incorrectly seen as
// already satisfied by a later StreamWaitEvent on the same event object.
class RollingLoadManager {
 public:
  RollingLoadManager(std::vector<layer::BaseLoader*> loaders,
                     std::shared_ptr<layer::RollingWeightBuffer> rolling_buffer,
                     Stream* load_stream,
                     Stream* compute_stream,
                     int32_t requested_rolling_slots);
  ~RollingLoadManager();

  // Non-copyable
  RollingLoadManager(const RollingLoadManager&) = delete;
  RollingLoadManager& operator=(const RollingLoadManager&) = delete;

  // Pre-load layers [0, N) into device slots and record h2d_events_[0..N-1].
  // Must be called once after all loaders have host_pinned_storage_ ready.
  void init_rolling_load();

  // Refresh rolling buffer base address after wakeup.
  void refresh_rolling_buffer_address();

  // Called before computing layer i.
  // compute_stream waits for h2d_events_[i] (GPU-side, non-blocking CPU).
  void wait_layer_h2d_ready(int32_t layer_index);

  // Called after computing layer i.
  // Records compute_events_[i] on compute_stream.
  // If this slot has a next layer: load_stream waits compute_events_[i], kicks
  // H2D(next_layer_in_same_slot), records h2d_events_[next_layer].
  void schedule_next_layer_h2d(int32_t layer_index);

  // Re-loads layers [0, N) into slots for the next forward pass.
  // Re-loads only remaining dirty slots (slots overwritten during this
  // forward and not already restored in schedule_next_layer_h2d), and ensures
  // startup h2d_events_ are ready for the next forward.
  //
  // last_executed_layer indicates the last layer index that actually finished
  // compute in current forward. This supports partial forward exits.
  void finalize(int32_t last_executed_layer);

  int32_t num_layers() const { return static_cast<int32_t>(loaders_.size()); }
  int32_t num_slots() const { return rolling_buffer_->num_slots(); }
  int32_t slot_for_layer(int32_t layer_index) const;

 private:
  // Kick async H2D from pinned host to device slot for the given layer.
  void kick_h2d(int32_t layer_index);

  // Return the largest executed layer mapped to this slot, or -1 if none.
  int32_t last_executed_layer_in_slot(int32_t slot, int32_t capped_last) const;

  std::vector<layer::BaseLoader*> loaders_;  // non-owning
  std::shared_ptr<layer::RollingWeightBuffer> rolling_buffer_;
  Stream* load_stream_;     // non-owning
  Stream* compute_stream_;  // non-owning

  // One event per layer (size = num_layers).
  // compute_events_[i]: signaled after layer i's forward completes.
  std::vector<aclrtEvent> compute_events_;
  // h2d_events_[i]: signaled after H2D for layer i completes.
  // Size = num_layers (reused for layers 0..N-1 in finalize; safe since
  // finalize runs after all layers are computed and all events are idle).
  std::vector<aclrtEvent> h2d_events_;

  int32_t num_slots_;
  int32_t preload_count_ = 0;
  int32_t rolling_slots_ = 0;
  int32_t requested_rolling_slots_ = 0;
  // Static layer->slot assignment for this model lifetime.
  std::vector<int32_t> layer_to_slot_;
  // For each layer, the next layer that reuses the same slot; -1 if none.
  std::vector<int32_t> next_in_slot_;
  // Layers grouped by slot (ascending).
  std::vector<std::vector<int32_t>> layers_in_slot_;
  // Slot-level dirty state: true if slot was overwritten by rolling H2D in
  // current forward and must be restored in finalize.
  std::vector<bool> dirty_slots_;
  // true if startup layer refill for this slot has already been queued and
  // h2d_events_[startup_layer] is already recorded in
  // schedule_next_layer_h2d().
  std::vector<bool> refilled_slots_;
};

// RAII helper for rolling load coordination in decoder layer loops.
class RollingLayerGuard final {
 public:
  explicit RollingLayerGuard(RollingLoadManager*& rolling_mgr)
      : rolling_mgr_(rolling_mgr) {}

  RollingLayerGuard(const RollingLayerGuard&) = delete;
  RollingLayerGuard& operator=(const RollingLayerGuard&) = delete;

  ~RollingLayerGuard() noexcept {
    if (rolling_mgr_ != nullptr) {
      rolling_mgr_->finalize(last_executed_layer_);
    }
  }

  void before_layer(int32_t layer_index) {
    if (rolling_mgr_ != nullptr) {
      rolling_mgr_->wait_layer_h2d_ready(layer_index);
    }
  }

  void after_layer(int32_t layer_index) {
    last_executed_layer_ = layer_index;
    if (rolling_mgr_ != nullptr) {
      rolling_mgr_->schedule_next_layer_h2d(layer_index);
    }
  }

 private:
  RollingLoadManager*& rolling_mgr_;
  int32_t last_executed_layer_ = -1;
};

}  // namespace xllm
