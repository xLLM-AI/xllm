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

#include <cstdint>
#include <vector>

#include "core/util/hash_util.h"

namespace xllm {

using LinearStatePrefixHash = PrefixHash;
using LinearStateValidityMask = std::vector<int64_t>;

class LinearStateCacheOp final {
 public:
  // Live slot the sequence advances its recurrent state in.
  int32_t linear_state_id = -1;
  // A newly admitted sequence has no recurrent history. The physical slot may
  // have been used by an earlier request, so the worker must clear it before
  // the first forward instead of relying on allocator contents.
  bool reset_requested = false;
  // Checkpoint source slot resolved by the scheduler. With
  // `restore_requested=true`, the worker copies it into `linear_state_id`
  // before forward. With `restore_requested=false`, a valid source denotes
  // direct read, only for consumers that support separate read/write slots:
  // forward reads this checkpoint in place and writes to the live slot.
  // A source-less row is a continued request/no-op. The
  // source is invalid for cold-start reset rows and mandatory for physical
  // restore rows.
  bool restore_requested = false;
  int32_t restore_src_slot_id = -1;

  bool is_direct_read() const {
    return restore_src_slot_id >= 0 && !restore_requested && !reset_requested;
  }

  bool is_valid_after(bool prev_valid) const {
    return restore_requested || is_direct_read() ||
           (prev_valid && !reset_requested);
  }
};

}  // namespace xllm
