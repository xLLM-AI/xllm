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

#include <acl/acl.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/layers/npu/loader/rolling_weight_buffer.h"
#include "core/platform/stream.h"

namespace xllm {
namespace dit {

using layer::RollingWeightBuffer;

// Per-block weight data holder: packs parameters into a host pinned buffer,
// creates device tensor views from a rolling buffer slot, and copies H2D.
class BlockWeightLoader final {
 public:
  BlockWeightLoader() = default;
  ~BlockWeightLoader() { free_host_weights(); }

  // Extract all block parameters into a contiguous host pinned buffer.
  void build_from_module(torch::nn::Module& block);

  // Bind this block's device views to a rolling buffer slot.
  void set_rolling_buffer(std::shared_ptr<RollingWeightBuffer> buf,
                          int32_t slot_index);

  // Async H2D: host pinned → device slot. Must be called on the load stream.
  void copy_to_device_async(aclrtStream stream);

  size_t storage_size() const { return storage_size_; }
  void* host_pinned_storage() const { return host_pinned_storage_; }
  void set_layer_index(int32_t idx) { layer_index_ = idx; }
  int32_t layer_index() const { return layer_index_; }

 private:
  void refresh_slot_views();
  void free_host_weights();
  struct ParamInfo {
    std::string name;
    std::vector<int64_t> sizes;
    torch::ScalarType dtype;
    uint64_t host_offset;
    torch::nn::Module* owner;
  };

  size_t storage_size_ = 0;
  void* host_pinned_storage_ = nullptr;
  std::vector<ParamInfo> params_;
  std::vector<torch::Tensor> slot_tensors_;
  std::shared_ptr<RollingWeightBuffer> rolling_buffer_ = nullptr;
  int32_t slot_index_ = -1;
  int32_t layer_index_ = -1;
};

// Orchestrates streaming of block weights through N device-memory slots.
// Two event sets synchronize load/compute: h2d_events_ (H2D done → compute
// can start) and compute_events_ (compute done → slot can be overwritten).
// Tail-of-slot layers trigger early refill to restore startup layers.
class DitRollingLoadManager final {
 public:
  DitRollingLoadManager() = default;

  ~DitRollingLoadManager() {
    destroy_events(h2d_events_);
    destroy_events(compute_events_);
  }

  DitRollingLoadManager(const DitRollingLoadManager&) = delete;
  DitRollingLoadManager& operator=(const DitRollingLoadManager&) = delete;

  void init(std::vector<BlockWeightLoader*> loaders,
            std::shared_ptr<RollingWeightBuffer> buffer,
            int32_t num_slots) {
    int32_t num_layers = static_cast<int32_t>(loaders.size());
    CHECK_GE(num_slots, 2);
    CHECK(!loaders.empty());
    CHECK(buffer != nullptr);

    loaders_ = std::move(loaders);
    buffer_ = std::move(buffer);
    num_slots_ = num_slots;
    preload_count_ = std::min(num_slots_, num_layers);

    if (load_stream_ == nullptr) {
      load_stream_ = std::make_unique<Stream>();
    }

    destroy_events(h2d_events_);
    destroy_events(compute_events_);
    h2d_events_.resize(num_layers, nullptr);
    compute_events_.resize(num_layers, nullptr);
    for (int32_t i = 0; i < num_layers; ++i) {
      CHECK_EQ(aclrtCreateEventWithFlag(&h2d_events_[i], ACL_EVENT_SYNC),
               ACL_SUCCESS);
      CHECK_EQ(aclrtCreateEventWithFlag(&compute_events_[i], ACL_EVENT_SYNC),
               ACL_SUCCESS);
    }

    for (int32_t i = 0; i < num_layers; ++i) {
      int32_t slot = i % num_slots_;
      loaders_[i]->set_layer_index(i);
      loaders_[i]->set_rolling_buffer(buffer_, slot);
    }

    next_layer_in_slot_.assign(num_layers, -1);
    std::vector<int32_t> last_in_slot(num_slots_, -1);
    for (int32_t i = num_layers - 1; i >= 0; --i) {
      int32_t slot = i % num_slots_;
      next_layer_in_slot_[i] = last_in_slot[slot];
      last_in_slot[slot] = i;
    }
    refilled_slots_.assign(num_slots_, false);
  }

  void preload() {
    aclrtStream ls = load_stream_->get_stream()->stream();
    for (int32_t i = 0; i < preload_count_; ++i) {
      if (refilled_slots_[i]) {
        continue;
      }
      loaders_[i]->copy_to_device_async(ls);
      CHECK_EQ(aclrtRecordEvent(h2d_events_[i], ls), ACL_SUCCESS);
    }
    std::fill(refilled_slots_.begin(), refilled_slots_.end(), false);
  }

  void wait_h2d(int32_t layer_index) {
    auto cs = c10_npu::getCurrentNPUStream().stream();
    CHECK_EQ(aclrtStreamWaitEvent(cs, h2d_events_[layer_index]), ACL_SUCCESS);
    CHECK_EQ(aclrtResetEvent(h2d_events_[layer_index], cs), ACL_SUCCESS);
  }

  void schedule_next_h2d(int32_t layer_index) {
    auto cs = c10_npu::getCurrentNPUStream().stream();
    auto ls = load_stream_->get_stream()->stream();

    CHECK_EQ(aclrtRecordEvent(compute_events_[layer_index], cs), ACL_SUCCESS);

    int32_t slot = layer_index % num_slots_;
    int32_t next = next_layer_in_slot_[layer_index];

    CHECK_EQ(aclrtStreamWaitEvent(ls, compute_events_[layer_index]),
             ACL_SUCCESS);
    CHECK_EQ(aclrtResetEvent(compute_events_[layer_index], ls), ACL_SUCCESS);

    if (next >= 0) {
      loaders_[next]->copy_to_device_async(ls);
      CHECK_EQ(aclrtRecordEvent(h2d_events_[next], ls), ACL_SUCCESS);
    } else {
      loaders_[slot]->copy_to_device_async(ls);
      CHECK_EQ(aclrtRecordEvent(h2d_events_[slot], ls), ACL_SUCCESS);
      refilled_slots_[slot] = true;
    }
  }

  bool ensure_model(std::vector<BlockWeightLoader*> loaders,
                    std::shared_ptr<RollingWeightBuffer> buffer = nullptr,
                    int32_t num_slots = 0) {
    // Compare by content (first slot address), not by vector identity —
    // get_block_weight_loaders() returns a new vector each call.
    if (!loaders_.empty() && loaders.size() == loaders_.size() &&
        loaders[0] == loaders_[0]) {
      return false;
    }
    if (buffer) {
      buffer_ = std::move(buffer);
    }
    if (num_slots > 0) {
      num_slots_ = num_slots;
    }
    init(std::move(loaders), buffer_, num_slots_);
    preload();
    return true;
  }

  int32_t preload_count() const { return preload_count_; }

 private:
  static void destroy_events(std::vector<aclrtEvent>& events) {
    for (auto& e : events) {
      if (e) {
        aclrtDestroyEvent(e);
      }
    }
  }

  std::vector<BlockWeightLoader*> loaders_;
  std::shared_ptr<RollingWeightBuffer> buffer_;
  std::unique_ptr<Stream> load_stream_;
  int32_t num_slots_ = 0;
  int32_t preload_count_ = 0;

  std::vector<aclrtEvent> h2d_events_;
  std::vector<aclrtEvent> compute_events_;
  std::vector<int32_t> next_layer_in_slot_;
  std::vector<bool> refilled_slots_;
};

}  // namespace dit
}  // namespace xllm
