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
#include <vector>

#include "framework/kv_cache_transfer/host_transfer/layout.h"
#include "platform/device.h"
#include "platform/stream.h"

namespace xllm {

class BatchMemcpy;

class CompactOffloadExecutor final {
 public:
  CompactOffloadExecutor(const HostKVLayout& layout,
                         const Device& device,
                         BatchMemcpy& batch_memcpy,
                         size_t target_bytes);
  ~CompactOffloadExecutor();

  CompactOffloadExecutor(const CompactOffloadExecutor&) = delete;
  CompactOffloadExecutor& operator=(const CompactOffloadExecutor&) = delete;

  void execute(const HostKVRequest& request, const Stream& compute_stream);
  void drain();

 private:
  struct SourceState {
    torch::Tensor tensor;
    size_t destination_offset = 0;
  };

  struct RoleState {
    torch::Tensor host;
    std::vector<SourceState> sources;
    size_t block_bytes = 0;
    size_t packed_offset = 0;
    size_t padded_bytes = 0;
  };

  struct GroupState {
    std::vector<RoleState> roles;
    size_t packed_bytes = 0;
    int64_t blocks_per_tile = 0;
  };

  struct SlotState {
    torch::Tensor packed;
    torch::Tensor host_block_ids;
    torch::Tensor device_block_ids;
    int32_t last_group_id = -1;
    StreamEventPtr d2h_complete;
  };

  void append_role(GroupState* group,
                   const HostKVLayout& layout,
                   const HostKVGroupLayout& layout_group,
                   KVCacheTensorRole::Value role);
  GroupState make_group(const HostKVLayout& layout,
                        const HostKVGroupLayout& layout_group);
  void init(const HostKVLayout& layout);
  SlotState& next_slot();
  void reuse_slot(SlotState& slot);
  void pack_blocks(const GroupState& group,
                   const torch::Tensor& block_ids,
                   const torch::Tensor& packed);
  void submit(SlotState& slot,
              int32_t group_id,
              const GroupState& group,
              const std::vector<HostKVMapping>& mappings,
              size_t mapping_offset,
              size_t mapping_count);
  void submit_d2h(SlotState& slot,
                  const GroupState& group,
                  const std::vector<HostKVMapping>& mappings,
                  size_t mapping_offset,
                  size_t mapping_count);
  void flush_slots();
  void log_init(size_t slot_bytes,
                size_t total_slot_bytes,
                bool expanded_for_mapping) const;

  Device device_;
  BatchMemcpy& batch_memcpy_;
  size_t target_bytes_ = 0;
  std::map<int32_t, GroupState> groups_;
  std::array<SlotState, 2> slots_;
  size_t next_slot_index_ = 0;
  std::unique_ptr<Stream> pack_stream_;
  std::unique_ptr<Stream> d2h_stream_;
  std::mutex mutex_;
  bool drained_ = false;
};

}  // namespace xllm
