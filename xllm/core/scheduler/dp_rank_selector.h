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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>

#include "core/common/types.h"

namespace xllm {

// Constraints are borrowed for one selection call; the selector never retains
// them.
struct DpPlacementConstraints {
  int32_t assigned_rank = -1;
  // Empty permits every rank; otherwise one entry per rank, zero excludes it.
  std::span<const uint8_t> eligible_ranks;
};

// Selects by admissible KV headroom, rotating ties. Request limits filter the
// candidates before scoring; an existing binding is never moved to another
// rank.
class DpRankSelector final {
 public:
  // Returns -1 when no eligible rank exists, including a blocked assigned rank.
  int32_t select(std::span<const DpRankLoad> loads,
                 const DpPlacementConstraints& constraints = {}) const;

 private:
  mutable std::atomic<size_t> cursor_{0};
};

}  // namespace xllm
