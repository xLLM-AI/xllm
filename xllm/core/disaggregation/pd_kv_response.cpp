/* Copyright 2026 The xLLM Authors.

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

#include "core/disaggregation/pd_kv_response.h"

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/framework/request/sequence_kv_state.h"
#include "core/kv_cache/block/block.h"

namespace xllm {

void append_pd_kv_transfer_groups(
    const KVCacheState& kv_state,
    bool include_linear_state,
    proto::DisaggResponse* response,
    std::vector<int32_t>* virtual_memory_block_ids) {
  if (kv_state.has_multi_block_export()) {
    // Iterate borrowed slices directly, without allocating an export-view
    // vector or copying reference-counted Block handles.
    for (const BlockType block_type : kMultiBlockExportOrder) {
      const auto blocks = kv_state.blocks(block_type);
      if (blocks.empty()) {
        continue;
      }
      proto::KVTransferGroup* group = response->add_groups();
      group->set_group_id(cache_group_id(block_type));
      // P uses this cursor to skip grouped blocks already cached on D.
      group->set_remote_shared_num(
          static_cast<uint32_t>(kv_state.shared_blocks_num(block_type)));
      group->mutable_ids()->Reserve(static_cast<int32_t>(blocks.size()));
      for (const auto& block : blocks) {
        const bool valid = block.is_valid();
        CHECK(valid || block_type == BlockType::SWA)
            << "Decode allocated an invalid grouped KV block under a "
            << "non-SWA leaf, group_id=" << cache_group_id(block_type);
        // Expired SWA slots must preserve their logical indices on the wire.
        const int64_t block_id = valid ? block.id() : -1;
        group->add_ids(static_cast<uint64_t>(block_id));
      }
    }
  } else {
    const auto blocks = kv_state.blocks(BlockType::KV);
    const size_t shared_num = kv_state.shared_blocks_num(BlockType::KV);
    CHECK_LE(shared_num, blocks.size());
    proto::KVTransferGroup* group = response->add_groups();
    group->set_group_id(cache_group_id(BlockType::KV));
    group->set_remote_shared_num(static_cast<uint32_t>(shared_num));
    group->mutable_ids()->Reserve(
        static_cast<int32_t>(blocks.size() - shared_num));
    if (virtual_memory_block_ids != nullptr) {
      virtual_memory_block_ids->reserve(virtual_memory_block_ids->size() +
                                        blocks.size() - shared_num);
    }
    for (size_t i = shared_num; i < blocks.size(); ++i) {
      const int32_t block_id = blocks[i].id();
      group->add_ids(static_cast<uint64_t>(block_id));
      if (virtual_memory_block_ids != nullptr) {
        virtual_memory_block_ids->push_back(block_id);
      }
    }
  }

  if (include_linear_state) {
    const int32_t linear_state_id = kv_state.get_linear_block_id();
    CHECK_GE(linear_state_id, 0)
        << "Decode did not allocate a linear-state slot.";
    proto::KVTransferGroup* group = response->add_groups();
    group->set_group_id(cache_group_id(BlockType::LINEAR));
    group->add_ids(static_cast<uint64_t>(linear_state_id));
  }
}

}  // namespace xllm
