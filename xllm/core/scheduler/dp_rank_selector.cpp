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

#include "core/scheduler/dp_rank_selector.h"

#include <glog/logging.h>

#include <limits>
#include <vector>

namespace xllm {

int32_t DpRankSelector::select(
    std::span<const DpRankLoad> loads,
    const DpPlacementConstraints& constraints) const {
  CHECK_LE(loads.size(),
           static_cast<size_t>(std::numeric_limits<int32_t>::max()));
  CHECK_GE(constraints.assigned_rank, -1);
  CHECK(constraints.eligible_ranks.empty() ||
        constraints.eligible_ranks.size() == loads.size());

  const auto is_eligible = [&constraints](size_t rank) {
    return constraints.eligible_ranks.empty() ||
           constraints.eligible_ranks[rank] != 0;
  };
  if (constraints.assigned_rank >= 0) {
    const size_t rank = static_cast<size_t>(constraints.assigned_rank);
    CHECK_LT(rank, loads.size());
    return is_eligible(rank) ? constraints.assigned_rank : -1;
  }

  size_t max_headroom = 0;
  std::vector<int32_t> candidates;
  candidates.reserve(loads.size());
  for (size_t rank = 0; rank < loads.size(); ++rank) {
    if (!is_eligible(rank)) {
      continue;
    }
    const auto& load = loads[rank];
    const size_t headroom = load.admission_capacity > load.used_blocks
                                ? load.admission_capacity - load.used_blocks
                                : 0;
    if (candidates.empty() || headroom > max_headroom) {
      max_headroom = headroom;
      candidates.clear();
      candidates.emplace_back(static_cast<int32_t>(rank));
    } else if (headroom == max_headroom) {
      candidates.emplace_back(static_cast<int32_t>(rank));
    }
  }
  if (candidates.empty()) {
    return -1;
  }
  if (candidates.size() == 1) {
    return candidates.front();
  }
  const size_t offset =
      cursor_.fetch_add(/*arg=*/1, std::memory_order_relaxed) %
      candidates.size();
  return candidates[offset];
}

}  // namespace xllm
