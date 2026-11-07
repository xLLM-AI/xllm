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
  // Only TP/CP replicas share a request set; DP groups carry unrelated batches.
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

  std::array<int64_t, 2> metadata{
      static_cast<int64_t>(canonical_request_ids.size()), 0};
  uint64_t request_ids_hash = 0;
  for (const auto& request_id : canonical_request_ids) {
    request_ids_hash = XXH3_64bits_withSeed(
        request_id.data(), request_id.size(), request_ids_hash);
  }
  std::memcpy(&metadata[1], &request_ids_hash, sizeof(request_ids_hash));
  const torch::Tensor metadata_payload =
      torch::from_blob(metadata.data(), {2}, torch::kInt64).to(device);
  // Every rank validates the request set before skipping an empty bitmap.
  for (size_t group_index = 0; group_index < num_groups; ++group_index) {
    ProcessGroup* group = active_groups[group_index];
    const torch::Tensor gathered =
        group->allgather_base_sync(metadata_payload).to(torch::kCPU);
    const auto* gathered_data = gathered.data_ptr<int64_t>();
    for (int64_t rank = 0; rank < group->world_size(); ++rank) {
      CHECK_EQ(gathered_data[rank * 2], metadata[0])
          << "KV transfer request count differs across reduction ranks";
      CHECK_EQ(gathered_data[rank * 2 + 1], metadata[1])
          << "KV transfer request IDs differ across reduction ranks";
    }
  }
  if (canonical_request_ids.empty()) {
    return {};
  }

  torch::Tensor bitmap_host = torch::zeros(
      {static_cast<int64_t>(canonical_request_ids.size())}, torch::kInt32);
  auto* bitmap_data = bitmap_host.data_ptr<int32_t>();
  for (size_t index = 0; index < canonical_request_ids.size(); ++index) {
    bitmap_data[index] =
        local_failed_request_ids.count(canonical_request_ids[index]) != 0;
  }
  torch::Tensor bitmap = bitmap_host.to(device);
  for (size_t group_index = 0; group_index < num_groups; ++group_index) {
    active_groups[group_index]->allreduce(bitmap);
  }
  bitmap_host = bitmap.to(torch::kCPU);
  bitmap_data = bitmap_host.data_ptr<int32_t>();
  std::vector<std::string> failed_request_ids;
  for (size_t index = 0; index < canonical_request_ids.size(); ++index) {
    if (bitmap_data[index] != 0) {
      failed_request_ids.emplace_back(canonical_request_ids[index]);
    }
  }
  return failed_request_ids;
}

std::vector<std::string> finalize_kv_push_failures(
    KVTransferCompletion& kv_transfers,
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    bool enable_kv_push,
    const std::array<ProcessGroup*, 2>& groups,
    const Device& device) {
  const std::unordered_set<std::string> local_failed_request_ids =
      kv_transfers.wait();
  if (!enable_kv_push) {
    return {};
  }
  std::vector<std::string> canonical_request_ids =
      unique_transfer_request_ids(transfer_kv_infos);
  std::sort(canonical_request_ids.begin(), canonical_request_ids.end());
  return reduce_failed_request_ids(
      local_failed_request_ids, canonical_request_ids, groups, device);
}

}  // namespace xllm
