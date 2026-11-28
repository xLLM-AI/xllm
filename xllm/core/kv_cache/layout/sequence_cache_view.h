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

#pragma once

#include <cstddef>
#include <optional>

#include "core/kv_cache/layout/cache_partition.h"

namespace xllm {

enum class CacheRetention {
  HISTORY,
  WINDOW,
  RECURRENT,
  SEQUENCE,
};

// An ownership-free snapshot of one partition on a sequence. Logical positions
// include SWA holes and sparse recurrent checkpoints; resident_blocks counts
// only valid physical handles. Taking a snapshot does not pin those resources.
struct SequenceCacheView {
  size_t logical_blocks = 0;
  size_t resident_blocks = 0;
  size_t token_stride = 0;
  size_t shared_logical_blocks = 0;
  size_t published_logical_blocks = 0;
  // Deepest retained shared recurrent checkpoint, not the live output slot.
  std::optional<size_t> restore_tokens;
};

}  // namespace xllm
