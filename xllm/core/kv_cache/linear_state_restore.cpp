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

#include "core/kv_cache/linear_state_restore.h"

#include <glog/logging.h>

#include <algorithm>

#include "core/common/constants.h"
#include "core/kv_cache/storage/kv_cache_utils.h"

namespace xllm {

namespace {

int32_t discover_num_slots(const std::vector<KVCache>& kv_caches) {
  int32_t num_slots = 0;
  for (const KVCache& kv_cache : kv_caches) {
    const torch::Tensor kpool_tail = kv_cache.get_kpool_tail();
    const torch::Tensor conv_cache = kv_cache.get_conv_cache();
    const torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
    if (kpool_tail.defined()) {
      CHECK_GT(kpool_tail.size(0), kPaddingLinearStateId);
      if (num_slots == 0) {
        num_slots = static_cast<int32_t>(kpool_tail.size(0));
      }
      CHECK_EQ(num_slots, kpool_tail.size(0));
    }
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

void restore_linear_state_slots(std::vector<KVCache>& kv_caches,
                                const std::vector<int32_t>& write_ids,
                                const std::vector<int32_t>& read_ids,
                                LinearStateValidityMask& validity_mask,
                                bool reads_distinct_state) {
  if (write_ids.empty() || validity_mask.empty()) {
    return;
  }

  const auto& source_ids = read_ids.empty() ? write_ids : read_ids;
  CHECK_EQ(source_ids.size(), write_ids.size())
      << "linear-state read/write rows must match";
  CHECK_GE(validity_mask.size(), write_ids.size())
      << "linear-state validity mask must cover every logical row";
  CHECK_EQ(validity_mask.size() % write_ids.size(), 0u)
      << "linear-state validity mask must evenly expand logical rows";
  const int32_t num_slots = discover_num_slots(kv_caches);
  CHECK_GT(num_slots, kPaddingLinearStateId)
      << "linear-state restore requires an allocated recurrent cache";
  for (size_t row = 0; row < write_ids.size(); ++row) {
    CHECK_GE(write_ids[row], kPaddingLinearStateId);
    CHECK_LT(write_ids[row], num_slots);
    CHECK_GE(source_ids[row], kPaddingLinearStateId);
    CHECK_LT(source_ids[row], num_slots);
    CHECK((write_ids[row] == kPaddingLinearStateId) ==
          (source_ids[row] == kPaddingLinearStateId))
        << "padding must not be used as a real linear-state source or target";
  }
  for (int64_t validity : validity_mask) {
    CHECK(validity == 0 || validity == 1)
        << "linear-state validity entries must be 0 or 1";
  }

  const size_t repeat = validity_mask.size() / write_ids.size();
  std::vector<int32_t> pending_reset_slots;
  pending_reset_slots.reserve(write_ids.size());
  // Flush resets before the next checkpoint copy, including the final batch.
  for (size_t row = 0; row <= write_ids.size(); ++row) {
    if (row < write_ids.size()) {
      const auto mask_begin = validity_mask.begin() + row * repeat;
      if (write_ids[row] == kPaddingLinearStateId) {
        std::fill(mask_begin, mask_begin + repeat, 0);
        continue;
      }
      if (source_ids[row] == write_ids[row]) {
        if (*mask_begin == 0) {
          pending_reset_slots.push_back(write_ids[row]);
        }
        continue;
      }
    }
    std::sort(pending_reset_slots.begin(), pending_reset_slots.end());
    pending_reset_slots.erase(
        std::unique(pending_reset_slots.begin(), pending_reset_slots.end()),
        pending_reset_slots.end());
    for (size_t begin = 0; begin < pending_reset_slots.size();) {
      size_t end = begin + 1;
      while (end < pending_reset_slots.size() &&
             pending_reset_slots[end] == pending_reset_slots[end - 1] + 1) {
        ++end;
      }
      for (const KVCache& kv_cache : kv_caches) {
        const torch::Tensor kpool_tail = kv_cache.get_kpool_tail();
        if (!kpool_tail.defined()) {
          continue;
        }
        const torch::Tensor tail_slots = kpool_tail.narrow(
            0,
            pending_reset_slots[begin],
            pending_reset_slots[end - 1] - pending_reset_slots[begin] + 1);
        tail_slots.zero_();
        fill_kpool_tail_gate_sentinel(tail_slots);
      }
      begin = end;
    }
    pending_reset_slots.clear();
    if (row == write_ids.size()) {
      break;
    }
    if (!reads_distinct_state) {
      for (const KVCache& kv_cache : kv_caches) {
        const torch::Tensor kpool_tail = kv_cache.get_kpool_tail();
        const torch::Tensor conv_cache = kv_cache.get_conv_cache();
        const torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
        if (kpool_tail.defined()) {
          kpool_tail.select(0, write_ids[row])
              .copy_(kpool_tail.select(0, source_ids[row]));
        }
        if (!conv_cache.defined()) {
          continue;
        }
        const int64_t stride = ssm_cache.size(0) / conv_cache.size(0);
        conv_cache.select(0, write_ids[row])
            .copy_(conv_cache.select(0, source_ids[row]));
        ssm_cache
            .narrow(0, static_cast<int64_t>(write_ids[row]) * stride, stride)
            .copy_(ssm_cache.narrow(
                0, static_cast<int64_t>(source_ids[row]) * stride, stride));
      }
    }
    const auto mask_begin = validity_mask.begin() + row * repeat;
    std::fill(mask_begin, mask_begin + repeat, 1);
  }
}

}  // namespace xllm
