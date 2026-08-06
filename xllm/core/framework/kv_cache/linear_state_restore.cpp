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

#include "framework/kv_cache/linear_state_restore.h"

#include <glog/logging.h>

#include <algorithm>

#include "core/common/constants.h"

namespace xllm {

namespace {

int32_t discover_num_slots(const std::vector<KVCache>& kv_caches) {
  int32_t num_slots = 0;
  for (const KVCache& kv_cache : kv_caches) {
    const torch::Tensor conv_cache = kv_cache.get_conv_cache();
    const torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
    if (!conv_cache.defined() && !ssm_cache.defined()) {
      continue;
    }
    CHECK(conv_cache.defined() && ssm_cache.defined())
        << "linear-attention layers must provide both conv and ssm caches";
    CHECK_GT(conv_cache.size(0), kPaddingLinearStateId)
        << "linear-attention cache must include the reserved padding slot";
    CHECK_GT(ssm_cache.size(0), 0)
        << "linear-attention ssm cache must contain checkpoint rows";
    CHECK_EQ(ssm_cache.size(0) % conv_cache.size(0), 0)
        << "ssm cache checkpoint layout mismatch, ssm_rows="
        << ssm_cache.size(0) << ", conv_rows=" << conv_cache.size(0);
    if (num_slots == 0) {
      num_slots = static_cast<int32_t>(conv_cache.size(0));
      continue;
    }
    CHECK_EQ(num_slots, static_cast<int32_t>(conv_cache.size(0)))
        << "linear-attention cache slot count must match across layers";
  }
  return num_slots;
}

struct SlotRange {
  int64_t start = 0;
  int64_t length = 0;
};

std::vector<SlotRange> coalesce_slot_ranges(std::vector<int32_t> slot_ids) {
  CHECK(!slot_ids.empty());
  std::sort(slot_ids.begin(), slot_ids.end());
  slot_ids.erase(std::unique(slot_ids.begin(), slot_ids.end()), slot_ids.end());

  std::vector<SlotRange> ranges;
  ranges.reserve(slot_ids.size());
  int64_t range_start = slot_ids.front();
  int64_t previous_slot = range_start;
  for (size_t i = 1; i < slot_ids.size(); ++i) {
    const int64_t slot_id = slot_ids[i];
    if (slot_id == previous_slot + 1) {
      previous_slot = slot_id;
      continue;
    }
    ranges.push_back({range_start, previous_slot - range_start + 1});
    range_start = slot_id;
    previous_slot = slot_id;
  }
  ranges.push_back({range_start, previous_slot - range_start + 1});
  return ranges;
}

void zero_slots_across_layers(std::vector<KVCache>& kv_caches,
                              const std::vector<int32_t>& slot_ids) {
  const std::vector<SlotRange> ranges = coalesce_slot_ranges(slot_ids);
  bool cleared = false;
  for (const KVCache& kv_cache : kv_caches) {
    const torch::Tensor conv_cache = kv_cache.get_conv_cache();
    const torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
    if (!conv_cache.defined() && !ssm_cache.defined()) {
      continue;
    }
    CHECK(conv_cache.defined() && ssm_cache.defined());
    const int64_t checkpoint_stride = ssm_cache.size(0) / conv_cache.size(0);
    for (const SlotRange& range : ranges) {
      if (range.length == 1) {
        conv_cache.select(0, range.start).zero_();
      } else {
        conv_cache.narrow(0, range.start, range.length).zero_();
      }
      ssm_cache
          .narrow(0,
                  range.start * checkpoint_stride,
                  range.length * checkpoint_stride)
          .zero_();
    }
    cleared = true;
  }
  CHECK(cleared) << "linear-state reset found no recurrent cache to clear";
}

void copy_slot_across_layers(std::vector<KVCache>& kv_caches,
                             int32_t dst_slot_id,
                             int32_t src_slot_id) {
  bool copied = false;
  for (const KVCache& kv_cache : kv_caches) {
    const torch::Tensor conv_cache = kv_cache.get_conv_cache();
    const torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
    if (!conv_cache.defined() && !ssm_cache.defined()) {
      continue;
    }
    CHECK(conv_cache.defined() && ssm_cache.defined());
    const int64_t checkpoint_stride = ssm_cache.size(0) / conv_cache.size(0);
    conv_cache.select(0, dst_slot_id).copy_(conv_cache.select(0, src_slot_id));
    ssm_cache
        .narrow(0,
                static_cast<int64_t>(dst_slot_id) * checkpoint_stride,
                checkpoint_stride)
        .copy_(ssm_cache.narrow(
            0,
            static_cast<int64_t>(src_slot_id) * checkpoint_stride,
            checkpoint_stride));
    copied = true;
  }
  CHECK(copied) << "linear-state restore found no recurrent cache to copy";
}

}  // namespace

LinearStateValidityMask build_linear_state_mask(
    const std::vector<int32_t>& cached_tokens,
    int64_t active_rows) {
  CHECK(!cached_tokens.empty()) << "cached_tokens must not be empty";
  CHECK_GT(active_rows, 0) << "active_rows must be positive";
  const int64_t logical_rows = static_cast<int64_t>(cached_tokens.size());
  CHECK_EQ(active_rows % logical_rows, 0)
      << "logical rows must evenly divide active rows, logical_rows="
      << logical_rows << ", active_rows=" << active_rows;

  const int64_t repeat_count = active_rows / logical_rows;
  LinearStateValidityMask warm_mask;
  warm_mask.reserve(static_cast<size_t>(active_rows));
  for (int32_t num_tokens : cached_tokens) {
    const int64_t is_warm = num_tokens > 0 ? 1 : 0;
    for (int64_t repeat_idx = 0; repeat_idx < repeat_count; ++repeat_idx) {
      warm_mask.emplace_back(is_warm);
    }
  }
  return warm_mask;
}

void restore_linear_state_slots(
    std::vector<KVCache>& kv_caches,
    const std::vector<LinearStateCacheOp>& cache_ops,
    LinearStateValidityMask& validity_mask) {
  if (cache_ops.empty()) {
    return;
  }

  CHECK_EQ(cache_ops.size(), validity_mask.size())
      << "validity_mask must match the linear-state operation batch, "
      << "cache_ops=" << cache_ops.size()
      << ", validity_mask=" << validity_mask.size();

  const int32_t num_slots = discover_num_slots(kv_caches);
  CHECK_GT(num_slots, kPaddingLinearStateId)
      << "linear-state operations require an allocated recurrent cache";
  const auto is_real_slot = [num_slots](int32_t slot_id) {
    return slot_id > kPaddingLinearStateId && slot_id < num_slots;
  };

  for (const LinearStateCacheOp& cache_op : cache_ops) {
    const int32_t live_slot_id = cache_op.linear_state_id;
    CHECK(!(cache_op.reset_requested && cache_op.restore_requested))
        << "linear-state reset and restore are mutually exclusive";
    CHECK(is_real_slot(live_slot_id))
        << "linear-state live slot must be a real non-padding slot, slot="
        << live_slot_id << ", num_slots=" << num_slots;
    if (cache_op.reset_requested) {
      CHECK_LT(cache_op.restore_src_slot_id, 0)
          << "linear-state reset must not carry a restore source";
      continue;
    }
    if (!cache_op.restore_requested) {
      CHECK_LT(cache_op.restore_src_slot_id, 0)
          << "linear-state source requires restore_requested=true";
      continue;
    }
    const int32_t src_slot_id = cache_op.restore_src_slot_id;
    CHECK(is_real_slot(src_slot_id))
        << "linear-state restore source must be a real non-padding slot, slot="
        << src_slot_id << ", num_slots=" << num_slots;
  }

  std::vector<int32_t> pending_reset_slots;
  std::vector<size_t> pending_reset_rows;
  pending_reset_slots.reserve(cache_ops.size());
  pending_reset_rows.reserve(cache_ops.size());
  const auto flush_resets = [&]() {
    if (pending_reset_slots.empty()) {
      return;
    }
    zero_slots_across_layers(kv_caches, pending_reset_slots);
    for (const size_t row : pending_reset_rows) {
      validity_mask[row] = 0;
    }
    pending_reset_slots.clear();
    pending_reset_rows.clear();
  };

  for (size_t i = 0; i < cache_ops.size(); ++i) {
    const LinearStateCacheOp& cache_op = cache_ops[i];
    if (cache_op.reset_requested) {
      pending_reset_slots.push_back(cache_op.linear_state_id);
      pending_reset_rows.push_back(i);
      continue;
    }
    if (!cache_op.restore_requested) {
      continue;
    }

    flush_resets();
    const int32_t live_slot_id = cache_op.linear_state_id;
    const int32_t src_slot_id = cache_op.restore_src_slot_id;
    copy_slot_across_layers(kv_caches, live_slot_id, src_slot_id);
    validity_mask[i] = 1;
    VLOG(1) << "Qwen3.5 linear state checkpoint restored; live_slot_id="
            << live_slot_id << ", src_slot_id=" << src_slot_id;
  }
  flush_resets();
}

}  // namespace xllm
