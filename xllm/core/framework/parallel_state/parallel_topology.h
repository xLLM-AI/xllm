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

#include <cstdint>
#include <optional>
#include <string>

#include "core/framework/parallel_state/rank_generator.h"

namespace xllm::parallel_state {

// Native LLM rank layout: DP x PCP x TP; MoE EP overlays the same world.
// CP/DCP retain the public cp_size/kv_split_size configuration names. DCP
// partitions PCP, spans a DP-local domain, or partitions TP without PCP.
class ParallelTopology final {
 public:
  ParallelTopology(int32_t global_rank,
                   int32_t world_size,
                   int32_t dp_size,
                   int32_t ep_size,
                   int32_t pcp_size,
                   int32_t dcp_size);

  static std::optional<std::string> validate_dcp_tp(int32_t world_size,
                                                    int32_t dp_size,
                                                    int32_t dcp_size);

  int32_t dp_rank() const { return dp_.rank; }
  int32_t tp_size() const { return static_cast<int32_t>(tp_.ranks.size()); }
  int32_t tp_rank() const { return tp_.rank; }
  int32_t pcp_size() const { return static_cast<int32_t>(pcp_.ranks.size()); }
  int32_t pcp_rank() const { return pcp_.rank; }
  int32_t dcp_size() const { return static_cast<int32_t>(dcp_.ranks.size()); }
  int32_t dcp_rank() const { return dcp_.rank; }
  int32_t moe_tp_size() const {
    return static_cast<int32_t>(moe_tp_.ranks.size());
  }

  const RankGroup& tp() const { return tp_; }
  const RankGroup& dp() const { return dp_; }
  const RankGroup& pcp() const { return pcp_; }
  const RankGroup& dcp() const { return dcp_; }
  const RankGroup& moe_tp() const { return moe_tp_; }
  const RankGroup& moe_ep() const { return moe_ep_; }

  const std::vector<int32_t>& pcp_group_ranks() const { return pcp_.ranks; }
  const std::vector<int32_t>& dcp_group_ranks() const { return dcp_.ranks; }

 private:
  RankGroup tp_;
  RankGroup dp_;
  RankGroup pcp_;
  RankGroup dcp_;
  RankGroup moe_tp_;
  RankGroup moe_ep_;
};

}  // namespace xllm::parallel_state
