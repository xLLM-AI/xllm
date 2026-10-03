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

#include "core/framework/parallel_state/parallel_topology.h"

#include <glog/logging.h>

#include <string>
#include <vector>

namespace xllm::parallel_state {

std::optional<std::string> ParallelTopology::validate_dcp_tp(int32_t world_size,
                                                             int32_t dp_size,
                                                             int32_t dcp_size) {
  if (dcp_size < 1) {
    return "kv_split_size must resolve to a value greater than or equal to 1";
  }
  if (dcp_size == 1) {
    return std::nullopt;
  }
  if (world_size < 1 || dp_size < 1) {
    return "DCP requires positive world_size and dp_size";
  }
  if (world_size % dp_size != 0) {
    return "DCP requires world_size divisible by dp_size";
  }
  const int32_t tp_size = world_size / dp_size;
  if (dcp_size > tp_size || tp_size % dcp_size != 0) {
    return "DCP kv_split_size must divide the TP size within each DP "
           "replica";
  }
  return std::nullopt;
}

ParallelTopology::ParallelTopology(int32_t global_rank,
                                   int32_t world_size,
                                   int32_t dp_size,
                                   int32_t ep_size,
                                   int32_t pcp_size,
                                   int32_t dcp_size) {
  CHECK_GT(world_size, 0) << "world_size must be positive";
  CHECK_GT(dp_size, 0) << "dp_size must be positive";
  CHECK_GT(ep_size, 0) << "ep_size must be positive";
  CHECK_GT(pcp_size, 0) << "pcp_size must be positive";
  CHECK_GE(global_rank, 0) << "global_rank must be non-negative";
  CHECK_LT(global_rank, world_size) << "global_rank must be in the world";
  CHECK_EQ(world_size % (static_cast<int64_t>(dp_size) * pcp_size), 0)
      << "world_size must be divisible by dp_size * pcp_size";
  CHECK_EQ(world_size % ep_size, 0) << "ep_size must divide world_size";
  const int32_t tp_size = world_size / dp_size / pcp_size;
  const int32_t moe_tp_size = world_size / ep_size;
  const bool partitions_pcp =
      dcp_size > 0 && dcp_size <= pcp_size && pcp_size % dcp_size == 0;
  const bool partitions_tp =
      pcp_size == 1 &&
      !validate_dcp_tp(world_size, dp_size, dcp_size).has_value();
  const int32_t dp_stride = world_size / dp_size;
  CHECK(partitions_pcp || partitions_tp || dcp_size == dp_stride)
      << "dcp_size must divide pcp_size, divide tp_size with pcp_size=1, "
         "or equal pcp_size * tp_size";

  const RankGenerator generator(world_size);
  const std::vector<int32_t> attention_sizes{tp_size, pcp_size, dp_size};
  const std::vector<std::string> attention_order{"tp", "cp", "dp"};
  tp_ = generator.get_rank_group(
      "tp", attention_sizes, attention_order, global_rank);
  dp_ = generator.get_rank_group(
      "dp", attention_sizes, attention_order, global_rank);
  pcp_ = generator.get_rank_group(
      "cp", attention_sizes, attention_order, global_rank);
  if (partitions_pcp) {
    dcp_ = generator.get_rank_group(
        "dcp",
        {tp_size, pcp_size / dcp_size, dcp_size, dp_size},
        {"tp", "replica", "dcp", "dp"},
        global_rank);
  } else if (partitions_tp) {
    dcp_ = generator.get_rank_group("dcp",
                                    {dcp_size, tp_size / dcp_size, dp_size},
                                    {"dcp", "replica", "dp"},
                                    global_rank);
  } else {
    dcp_ = generator.get_rank_group(
        "dcp", {dp_stride, dp_size}, {"dcp", "dp"}, global_rank);
  }
  const std::vector<int32_t> moe_sizes{moe_tp_size, ep_size};
  const std::vector<std::string> moe_order{"tp", "ep"};
  moe_tp_ = generator.get_rank_group("tp", moe_sizes, moe_order, global_rank);
  moe_ep_ = generator.get_rank_group("ep", moe_sizes, moe_order, global_rank);
}

}  // namespace xllm::parallel_state
