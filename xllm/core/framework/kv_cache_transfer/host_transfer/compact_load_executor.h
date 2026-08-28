/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/kv_cache_transfer/host_transfer/layout.h"
#include "framework/kv_cache_transfer/host_transfer/transfer_utils.h"
#include "platform/device.h"
#include "platform/stream.h"

namespace xllm {

class BatchMemcpy;
class LayerSynchronizer;

class CompactLoadExecutor final {
 public:
  CompactLoadExecutor(const HostKVLayout& layout,
                      const Device& device,
                      uint32_t layer_copy_batches,
                      BatchMemcpy& batch_memcpy,
                      size_t target_bytes);
  ~CompactLoadExecutor();

  CompactLoadExecutor(const CompactLoadExecutor&) = delete;
  CompactLoadExecutor& operator=(const CompactLoadExecutor&) = delete;

  bool execute(const HostKVRequest& request,
               const std::shared_ptr<LayerSynchronizer>& synchronizer);
  uint32_t event_count() const;
  uint32_t layers_per_event() const;
  void drain();

 private:
  using GroupRoleKey = std::pair<int32_t, KVCacheTensorRole::Value>;

  struct RoleState {
    torch::Tensor host_tensor;
    size_t block_bytes = 0;
    int64_t max_span_layers = 0;
  };

  struct LayerState {
    int64_t absolute_layer_id = -1;
    int64_t group_layer_slot = -1;
    std::map<KVCacheTensorRole::Value, torch::Tensor> device_roles;
  };

  struct RangeState {
    int64_t slot_begin = 0;
    int64_t slot_end = 0;
    std::vector<KVCacheTensorRole::Value> active_roles;
  };

  struct GroupState {
    int64_t blocks_per_chunk = 0;
    size_t max_span_bytes = 0;
    std::map<KVCacheTensorRole::Value, RoleState> roles;
    std::vector<LayerState> layers;
    std::vector<RangeState> ranges;
  };

  struct SlotState {
    std::map<GroupRoleKey, torch::Tensor> device_staging;
    torch::Tensor host_indices;
    HostPageAlignedRegion index_region;
    torch::Tensor device_indices;
    StreamEventPtr completion_event;
  };

  void init(const HostKVLayout& layout);
  GroupState init_group(const HostKVLayout& layout,
                        const HostKVGroupLayout& group) const;
  RangeState init_range(const GroupState& group, const LayerRange& range) const;
  bool submit_range(int32_t group_id,
                    const GroupState& group,
                    size_t range_index,
                    const std::vector<HostKVMapping>& mappings);
  SlotState& next_slot();
  void wait_slot(const SlotState& slot) const;
  void fill_indices(SlotState& slot,
                    const std::vector<HostKVMapping>& mappings,
                    size_t offset,
                    size_t count) const;
  bool enqueue_h2d(SlotState& slot,
                   int32_t group_id,
                   const GroupState& group,
                   const RangeState& range,
                   const std::vector<HostKVMapping>& mappings,
                   size_t offset,
                   size_t count);
  void scatter(SlotState& slot,
               int32_t group_id,
               const GroupState& group,
               size_t range_index,
               const RangeState& range,
               int64_t count) const;
  void log_init() const;
  void drain_or_die(const char* reason);

  Device device_;
  size_t target_bytes_ = 0;
  uint32_t layers_per_event_ = 1;
  std::vector<LayerRange> ranges_;
  BatchMemcpy& batch_memcpy_;
  std::map<int32_t, GroupState> groups_;
  std::array<SlotState, 2> slots_;
  size_t next_slot_index_ = 0;
  std::unique_ptr<Stream> copy_stream_;
  std::mutex mutex_;
  bool drained_ = false;
};

}  // namespace xllm
