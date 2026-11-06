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

#include <glog/logging.h>
#include <xxhash.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

#include "core/common/types.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/kv_cache/transfer/kv_transfer_completion.h"
#include "core/platform/device.h"

namespace xllm {
std::vector<std::string> reduce_failed_request_ids(
    const std::unordered_set<std::string>& local_failed_request_ids,
    const std::vector<std::string>& canonical_request_ids,
    const std::array<ProcessGroup*, 2>& groups,
    const Device& device) {
  // Request batches are replicated across TP/CP ranks, but differ between DP
  // replicas. Reducing on the DP group would therefore combine unrelated
  // request-ID bitmaps and can deadlock when their shapes differ. Callers skip
  // this function for empty transfer lists, which are group-consistent because
  // transfer entries are derived from the replicated request batch.
  std::array<ProcessGroup*, 2> active_groups{};
  size_t num_groups = 0;
  for (ProcessGroup* group : groups) {
    if (group != nullptr && group->world_size() > 1 &&
        std::find(active_groups.begin(),
                  active_groups.begin() + num_groups,
                  group) == active_groups.begin() + num_groups) {
      active_groups[num_groups++] = group;
    }
  }
  if (num_groups == 0) {
    std::vector<std::string> failed_request_ids;
    for (const std::string& request_id : canonical_request_ids) {
      if (local_failed_request_ids.find(request_id) !=
          local_failed_request_ids.end()) {
        failed_request_ids.emplace_back(request_id);
      }
    }
    return failed_request_ids;
  }

  const size_t kPayloadMetadataSize = 2;
  const size_t payload_size =
      kPayloadMetadataSize + canonical_request_ids.size();
  torch::Tensor payload_host = torch::zeros(
      {static_cast<int64_t>(payload_size)},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* payload_data = payload_host.data_ptr<int64_t>();
  payload_data[0] = static_cast<int64_t>(canonical_request_ids.size());
  uint64_t request_ids_hash = 0;
  for (const auto& request_id : canonical_request_ids) {
    request_ids_hash = XXH3_64bits_withSeed(
        request_id.data(), request_id.size(), request_ids_hash);
  }
  std::memcpy(&payload_data[1], &request_ids_hash, sizeof(request_ids_hash));
  for (size_t index = 0; index < canonical_request_ids.size(); ++index) {
    if (local_failed_request_ids.find(canonical_request_ids[index]) !=
        local_failed_request_ids.end()) {
      payload_data[kPayloadMetadataSize + index] = 1;
    }
  }
  // Orthogonal TP/CP groups carry the cumulative bitmap forward so a failure
  // found by an earlier group also reaches ranks that only share the later
  // group. Metadata stays unchanged, so the existing payload is reused.
  torch::Tensor payload = payload_host.to(device);
  for (size_t group_index = 0; group_index < num_groups; ++group_index) {
    ProcessGroup* group = active_groups[group_index];
    torch::Tensor gathered =
        group->allgather_base_sync(payload).to(torch::kCPU);
    CHECK_EQ(gathered.numel(),
             static_cast<int64_t>(group->world_size() * payload.numel()))
        << "unexpected KV transfer request payload shape";
    const int64_t* gathered_data = gathered.data_ptr<int64_t>();
    for (int32_t rank = 0; rank < group->world_size(); ++rank) {
      const int64_t rank_offset = rank * payload.numel();
      CHECK_EQ(gathered_data[rank_offset], payload_data[0])
          << "KV transfer request count differs across reduction ranks";
      CHECK_EQ(gathered_data[rank_offset + 1], payload_data[1])
          << "KV transfer request IDs differ across reduction ranks";
      for (size_t index = 0; index < canonical_request_ids.size(); ++index) {
        if (gathered_data[rank_offset + kPayloadMetadataSize + index] > 0) {
          payload_data[kPayloadMetadataSize + index] = 1;
        }
      }
    }
    if (group_index + 1 < num_groups) {
      payload.copy_(payload_host);
    }
  }
  std::vector<std::string> failed_request_ids;
  for (size_t index = 0; index < canonical_request_ids.size(); ++index) {
    if (payload_data[kPayloadMetadataSize + index] > 0) {
      failed_request_ids.emplace_back(canonical_request_ids[index]);
    }
  }
  return failed_request_ids;
}

std::vector<std::string> finalize_kv_push_failures(
    KVTransferCompletion& kv_transfers,
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    const std::string& kv_cache_transfer_mode,
    const std::array<ProcessGroup*, 2>& groups,
    const Device& device) {
  const std::unordered_set<std::string> local_failed_request_ids =
      kv_transfers.wait();
  if (kv_cache_transfer_mode != "PUSH" || transfer_kv_infos.empty()) {
    return {};
  }
  std::vector<std::string> canonical_request_ids =
      unique_transfer_request_ids(transfer_kv_infos);
  std::sort(canonical_request_ids.begin(), canonical_request_ids.end());
  return reduce_failed_request_ids(
      local_failed_request_ids, canonical_request_ids, groups, device);
}

}  // namespace xllm
