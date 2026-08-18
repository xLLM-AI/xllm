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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xllm {
namespace layer {
namespace dsv4_eplb {

struct StagingTensorSpec {
  int64_t storage_bytes;
  bool requires_format_cast;
};

class StagingBufferKey final {
 public:
  int64_t numel;
  int32_t scalar_type;
  int64_t npu_format;

  bool operator==(const StagingBufferKey& other) const {
    return numel == other.numel && scalar_type == other.scalar_type &&
           npu_format == other.npu_format;
  }
};

inline int32_t missing_staging_buffer_count(
    const std::vector<StagingBufferKey>& available,
    const StagingBufferKey& required,
    int32_t required_count) {
  if (required_count <= 0 || required.numel <= 0) {
    return 0;
  }
  int32_t available_count = 0;
  for (const StagingBufferKey& candidate : available) {
    if (candidate == required) {
      ++available_count;
    }
  }
  return std::max(required_count - available_count, 0);
}

inline bool should_reserve_staging_buffers(bool enable_eplb,
                                           std::string_view model_type) {
  return enable_eplb && !model_type.ends_with("_mtp");
}

inline int64_t dispatch_ffn_max_output_size(int64_t local_tokens,
                                            int64_t topk,
                                            int64_t ep_world_size) {
  if (local_tokens <= 0 || topk <= 0 || ep_world_size <= 0) {
    return 0;
  }
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  if (local_tokens > kMax / topk) {
    return 0;
  }
  const int64_t routed_tokens = local_tokens * topk;
  if (routed_tokens > kMax / ep_world_size) {
    return 0;
  }
  return routed_tokens * ep_world_size;
}

inline int64_t calculate_staging_reservation_bytes(
    const std::vector<StagingTensorSpec>& tensor_specs) {
  int64_t reservation_bytes = 0;
  for (const StagingTensorSpec& tensor_spec : tensor_specs) {
    if (tensor_spec.storage_bytes <= 0) {
      continue;
    }
    reservation_bytes += tensor_spec.storage_bytes;
    if (tensor_spec.requires_format_cast) {
      reservation_bytes += tensor_spec.storage_bytes;
    }
  }
  return reservation_bytes;
}

inline std::vector<int32_t> build_initial_expert_ids(
    int32_t num_total_experts,
    int32_t ep_world_size,
    int32_t device_experts_num,
    int32_t redundant_experts_num) {
  std::vector<int32_t> expert_ids;
  if (num_total_experts <= 0 || ep_world_size <= 0 ||
      device_experts_num <= redundant_experts_num) {
    return expert_ids;
  }
  const int32_t routed_experts_num = device_experts_num - redundant_experts_num;
  expert_ids.reserve(static_cast<size_t>(ep_world_size * device_experts_num));
  for (int32_t rank = 0; rank < ep_world_size; ++rank) {
    const int32_t base = rank * routed_experts_num;
    for (int32_t slot = 0; slot < routed_experts_num; ++slot) {
      expert_ids.push_back(base + slot);
    }
    const int32_t duplicate_expert = base + routed_experts_num - 1;
    for (int32_t slot = 0; slot < redundant_experts_num; ++slot) {
      expert_ids.push_back(duplicate_expert);
    }
  }
  return expert_ids;
}

inline std::vector<int32_t> slice_rank_expert_ids(
    const std::vector<int32_t>& expert_ids,
    int32_t ep_rank,
    int32_t device_experts_num) {
  std::vector<int32_t> rank_expert_ids;
  if (ep_rank < 0 || device_experts_num <= 0) {
    return rank_expert_ids;
  }
  const size_t begin =
      static_cast<size_t>(ep_rank) * static_cast<size_t>(device_experts_num);
  const size_t end = begin + static_cast<size_t>(device_experts_num);
  if (end > expert_ids.size()) {
    return rank_expert_ids;
  }
  rank_expert_ids.assign(expert_ids.begin() + begin, expert_ids.begin() + end);
  return rank_expert_ids;
}

// Resolves each logical expert id to the physical slot this consumer should
// route to. When an expert has multiple duplicates in `expert_ids`, the pick
// is rotated by `(ep_rank + moe_tp_rank_in_group)` so that ranks sharing the
// same EP position but different moe_tp positions land on different
// duplicates. That spreads the read pressure of hot experts across all
// copies instead of stacking every TP consumer onto the first one -- the same
// rotation trick vLLM-ascend uses for its dynamic EPLB. When
// `moe_tp_rank_in_group` is 0 (pure EP, no moe_tp), the rotation degenerates
// to the previous `ep_rank % duplicate_count` behaviour, so callers that do
// not know their moe_tp rank keep their current pick.
inline std::vector<int32_t> build_log2phy_map(
    const std::vector<int32_t>& expert_ids,
    int32_t num_total_experts,
    int32_t ep_rank,
    int32_t moe_tp_rank_in_group = 0) {
  std::vector<int32_t> log2phy_map(static_cast<size_t>(num_total_experts), -1);
  if (num_total_experts <= 0 || expert_ids.empty()) {
    return log2phy_map;
  }
  const int32_t rotation_base =
      (ep_rank >= 0 ? ep_rank : 0) +
      (moe_tp_rank_in_group >= 0 ? moe_tp_rank_in_group : 0);
  std::vector<int32_t> duplicate_counts(static_cast<size_t>(num_total_experts),
                                        0);
  for (int32_t expert_id : expert_ids) {
    if (expert_id >= 0 && expert_id < num_total_experts) {
      ++duplicate_counts[static_cast<size_t>(expert_id)];
    }
  }
  std::vector<int32_t> selected_duplicates(
      static_cast<size_t>(num_total_experts), -1);
  for (int32_t expert_id = 0; expert_id < num_total_experts; ++expert_id) {
    const int32_t duplicate_count =
        duplicate_counts[static_cast<size_t>(expert_id)];
    if (duplicate_count > 0) {
      selected_duplicates[static_cast<size_t>(expert_id)] =
          rotation_base % duplicate_count;
    }
  }
  std::vector<int32_t> duplicate_indices(static_cast<size_t>(num_total_experts),
                                         0);
  for (int32_t physical_id = 0;
       physical_id < static_cast<int32_t>(expert_ids.size());
       ++physical_id) {
    const int32_t expert_id = expert_ids[static_cast<size_t>(physical_id)];
    if (expert_id < 0 || expert_id >= num_total_experts) {
      continue;
    }
    const size_t expert_index = static_cast<size_t>(expert_id);
    if (duplicate_indices[expert_index] == selected_duplicates[expert_index]) {
      log2phy_map[expert_index] = physical_id;
    }
    ++duplicate_indices[expert_index];
  }
  return log2phy_map;
}

inline std::vector<int32_t> find_slot_sources(
    const std::vector<int32_t>& active_expert_ids,
    const std::vector<int32_t>& target_expert_ids) {
  std::vector<int32_t> source_slots;
  source_slots.reserve(target_expert_ids.size());
  // Tolerate missing targets: sentinel -1 replaces the failed slot instead of
  // clearing the whole vector.
  std::unordered_map<int32_t, int32_t> expert_to_slot;
  expert_to_slot.reserve(active_expert_ids.size());
  for (int32_t slot = 0; slot < static_cast<int32_t>(active_expert_ids.size());
       ++slot) {
    expert_to_slot.emplace(active_expert_ids[static_cast<size_t>(slot)], slot);
  }
  for (int32_t target_expert_id : target_expert_ids) {
    auto it = expert_to_slot.find(target_expert_id);
    source_slots.push_back(it == expert_to_slot.end() ? -1 : it->second);
  }
  return source_slots;
}

inline std::vector<int32_t> collect_changed_slots(
    const std::vector<int32_t>& active_expert_ids,
    const std::vector<int32_t>& pending_expert_ids) {
  if (active_expert_ids.size() != pending_expert_ids.size()) {
    return {};
  }
  std::vector<int32_t> changed_slots;
  changed_slots.reserve(pending_expert_ids.size());
  for (int32_t slot = 0; slot < static_cast<int32_t>(pending_expert_ids.size());
       ++slot) {
    if (active_expert_ids[static_cast<size_t>(slot)] !=
        pending_expert_ids[static_cast<size_t>(slot)]) {
      changed_slots.emplace_back(slot);
    }
  }
  return changed_slots;
}

// Point-to-point transfer op emitted by compute_p2p_transfer_plan. `local_slot`
// is a slot index on the emitting rank; `peer_slot` is the paired slot index
// on `peer_rank`. Semantics:
//   * When emitted as a recv, `local_slot` is the destination pending slot on
//     the emitting rank and `peer_slot` is the source slot on `peer_rank`.
//   * When emitted as a send, `local_slot` is the source slot on the emitting
//     rank and `peer_slot` is the destination slot on `peer_rank`.
// Tensor payload is added by the caller when it walks all 18 weight tensors.
struct P2POp {
  int32_t peer_rank;
  int32_t local_slot;
  int32_t peer_slot;
};

// Three-way categorization of a (rank, dst_slot) pending assignment. Buckets
// are mutually exclusive; every (rank, dst_slot) lands in exactly one.
// UNCHANGED / SAME_GPU never emit a P2P op. Every remote-rank transfer uses the
// dedicated HCCL process group's send/recv API over the all-connected HCCS
// super-node fabric and therefore lands in HCCS.
enum class EplbP2PBucket : int8_t {
  UNCHANGED = 0,  // slot's expert id unchanged after this rebalance round
  SAME_GPU = 1,   // migration served locally on the owning rank
  HCCS = 2,       // cross-rank transfer over the HCCS super-node fabric
};

// Aggregate counts per bucket. Optional out-param of the P2P planner.
struct EplbP2PBucketStats {
  int64_t unchanged = 0;
  int64_t same_gpu = 0;
  int64_t hccs = 0;
};

inline bool compute_p2p_transfer_plan(
    const std::vector<int32_t>& global_active_expert_ids,
    const std::vector<int32_t>& global_pending_expert_ids,
    int32_t ep_rank,
    int32_t device_experts_num,
    std::vector<P2POp>& recv_ops,
    std::vector<P2POp>& send_ops,
    EplbP2PBucketStats* stats = nullptr) {
  EplbP2PBucketStats ignored_stats;
  EplbP2PBucketStats& output_stats = stats == nullptr ? ignored_stats : *stats;
  recv_ops.clear();
  send_ops.clear();
  output_stats = EplbP2PBucketStats{};
  if (device_experts_num <= 0 || ep_rank < 0) {
    return false;
  }
  if (global_active_expert_ids.size() != global_pending_expert_ids.size()) {
    return false;
  }
  const int32_t world_size =
      static_cast<int32_t>(global_active_expert_ids.size()) /
      device_experts_num;
  if (world_size <= 0 ||
      static_cast<int64_t>(world_size) * device_experts_num !=
          static_cast<int64_t>(global_active_expert_ids.size())) {
    return false;
  }
  if (ep_rank >= world_size) {
    return false;
  }

  // Reverse index: for every logical expert currently active in the cluster,
  // remember one (rank, slot) that owns it. Built in a single O(W * E) sweep
  // so subsequent per-destination lookups run in O(1) instead of the naive
  // O(W * E) scan repeated O(W * E) times (v1 was O((W*E)^2)). Prefer keeping
  // the *first* occurrence in same-rank-first order so a local slot rewrite
  // beats a cross-rank move when both hold the same expert.
  //
  // We seed the map only with fresh unique inserts (`emplace`), so ties fall
  // to the earliest slot found in rank-major order. To keep the owner-rank
  // local-preference guarantee, we do an owner-local fast path first inside
  // the destination loop before consulting the reverse index.
  std::unordered_map<int32_t, std::pair<int32_t, int32_t>> expert_to_source;
  expert_to_source.reserve(static_cast<size_t>(world_size) *
                           static_cast<size_t>(device_experts_num));
  for (int32_t rank = 0; rank < world_size; ++rank) {
    const size_t begin =
        static_cast<size_t>(rank) * static_cast<size_t>(device_experts_num);
    for (int32_t slot = 0; slot < device_experts_num; ++slot) {
      const int32_t expert =
          global_active_expert_ids[begin + static_cast<size_t>(slot)];
      expert_to_source.emplace(expert, std::make_pair(rank, slot));
    }
  }

  auto find_source = [&](int32_t owner_rank,
                         int32_t target_expert,
                         int32_t& out_rank,
                         int32_t& out_slot) -> bool {
    // Owner-local fast path: identical to v1's "look at this rank first" so a
    // pure local slot rewrite still wins over any cross-rank match, matching
    // the tests in deepseek_v4_eplb_test that pin same-rank preference.
    const size_t local_begin = static_cast<size_t>(owner_rank) *
                               static_cast<size_t>(device_experts_num);
    for (int32_t slot = 0; slot < device_experts_num; ++slot) {
      if (global_active_expert_ids[local_begin + static_cast<size_t>(slot)] ==
          target_expert) {
        out_rank = owner_rank;
        out_slot = slot;
        return true;
      }
    }
    auto it = expert_to_source.find(target_expert);
    if (it == expert_to_source.end()) {
      return false;
    }
    out_rank = it->second.first;
    out_slot = it->second.second;
    return true;
  };

  for (int32_t owner_rank = 0; owner_rank < world_size; ++owner_rank) {
    const size_t pending_begin = static_cast<size_t>(owner_rank) *
                                 static_cast<size_t>(device_experts_num);
    for (int32_t dst_slot = 0; dst_slot < device_experts_num; ++dst_slot) {
      const int32_t target_expert =
          global_pending_expert_ids[pending_begin +
                                    static_cast<size_t>(dst_slot)];
      // UNCHANGED bucket: the (rank, dst_slot) already holds this expert.
      // active_expert_ids and pending_expert_ids share the same rank-major
      // layout, so pending_begin also indexes into the active view.
      if (global_active_expert_ids[pending_begin +
                                   static_cast<size_t>(dst_slot)] ==
          target_expert) {
        ++output_stats.unchanged;
        continue;
      }
      int32_t src_rank = -1;
      int32_t src_slot = -1;
      if (!find_source(owner_rank, target_expert, src_rank, src_slot)) {
        recv_ops.clear();
        send_ops.clear();
        output_stats = EplbP2PBucketStats{};
        return false;
      }
      if (src_rank == owner_rank) {
        // Same rank as destination: a local slot rewrite. `find_source` only
        // returns a src_slot whose active expert equals `target_expert`, and
        // we already skipped the UNCHANGED case above where
        // active[dst_slot] == target_expert, so src_slot and dst_slot are
        // guaranteed distinct here — this is a genuine same-GPU copy, never a
        // no-op.
        ++output_stats.same_gpu;
        continue;
      }
      ++output_stats.hccs;
      if (owner_rank == ep_rank) {
        recv_ops.push_back(P2POp{src_rank, dst_slot, src_slot});
      } else if (src_rank == ep_rank) {
        send_ops.push_back(P2POp{owner_rank, src_slot, dst_slot});
      }
    }
  }
  return true;
}

}  // namespace dsv4_eplb
}  // namespace layer
}  // namespace xllm
