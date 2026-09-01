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

#include "core/framework/parallel_state/context_parallel_topology.h"

#include <gtest/gtest.h>

#include <vector>

namespace xllm::parallel_state {
namespace {

TEST(ContextParallelTopologyTest, SeparatesTpPcpAndFullDomainDcp) {
  const ContextParallelTopology topology(/*global_rank=*/3,
                                         /*world_size=*/8,
                                         /*dp_size=*/2,
                                         /*pcp_size=*/2,
                                         /*dcp_size=*/4);

  EXPECT_EQ(topology.dp_rank(), 0);
  EXPECT_EQ(topology.tp_size(), 2);
  EXPECT_EQ(topology.tp_rank(), 1);
  EXPECT_EQ(topology.pcp_rank(), 1);
  EXPECT_EQ(topology.dcp_rank(), 3);
  EXPECT_EQ(topology.pcp_group_ranks(), (std::vector<int32_t>{1, 3}));
  EXPECT_EQ(topology.dcp_group_ranks(), (std::vector<int32_t>{0, 1, 2, 3}));
}

TEST(ContextParallelTopologyTest, PartitionsFactorDcpInsidePcpGroup) {
  const ContextParallelTopology first_partition(/*global_rank=*/5,
                                                /*world_size=*/16,
                                                /*dp_size=*/2,
                                                /*pcp_size=*/4,
                                                /*dcp_size=*/2);
  EXPECT_EQ(first_partition.dp_rank(), 0);
  EXPECT_EQ(first_partition.pcp_group_ranks(),
            (std::vector<int32_t>{1, 3, 5, 7}));
  EXPECT_EQ(first_partition.dcp_rank(), 1);
  EXPECT_EQ(first_partition.dcp_group_ranks(), (std::vector<int32_t>{1, 5}));
  EXPECT_EQ(first_partition.dcp_group_ranks()[first_partition.dcp_rank()], 5);

  const ContextParallelTopology second_dp_partition(/*global_rank=*/13,
                                                    /*world_size=*/16,
                                                    /*dp_size=*/2,
                                                    /*pcp_size=*/4,
                                                    /*dcp_size=*/2);
  EXPECT_EQ(second_dp_partition.dp_rank(), 1);
  EXPECT_EQ(second_dp_partition.pcp_group_ranks(),
            (std::vector<int32_t>{9, 11, 13, 15}));
  EXPECT_EQ(second_dp_partition.dcp_rank(), 1);
  EXPECT_EQ(second_dp_partition.dcp_group_ranks(),
            (std::vector<int32_t>{9, 13}));
  EXPECT_EQ(
      second_dp_partition.dcp_group_ranks()[second_dp_partition.dcp_rank()],
      13);
}

TEST(ContextParallelTopologyTest, SupportsPcpBoundaryDcpLayouts) {
  const ContextParallelTopology single_rank_dcp(/*global_rank=*/5,
                                                /*world_size=*/16,
                                                /*dp_size=*/2,
                                                /*pcp_size=*/4,
                                                /*dcp_size=*/1);
  EXPECT_EQ(single_rank_dcp.dcp_rank(), 0);
  EXPECT_EQ(single_rank_dcp.dcp_group_ranks(), (std::vector<int32_t>{5}));

  const ContextParallelTopology pcp_wide_dcp(/*global_rank=*/5,
                                             /*world_size=*/16,
                                             /*dp_size=*/2,
                                             /*pcp_size=*/4,
                                             /*dcp_size=*/4);
  EXPECT_EQ(pcp_wide_dcp.dcp_rank(), 2);
  EXPECT_EQ(pcp_wide_dcp.dcp_group_ranks(), (std::vector<int32_t>{1, 3, 5, 7}));
}

TEST(ContextParallelTopologyTest, SupportsFullDpLocalDcpLayout) {
  const ContextParallelTopology topology(/*global_rank=*/13,
                                         /*world_size=*/16,
                                         /*dp_size=*/2,
                                         /*pcp_size=*/4,
                                         /*dcp_size=*/8);
  EXPECT_EQ(topology.dp_rank(), 1);
  EXPECT_EQ(topology.dcp_rank(), 5);
  EXPECT_EQ(topology.dcp_group_ranks(),
            (std::vector<int32_t>{8, 9, 10, 11, 12, 13, 14, 15}));
  EXPECT_EQ(topology.dcp_group_ranks()[topology.dcp_rank()], 13);
}

TEST(ContextParallelTopologyTest, RejectsNonFactorDcpLayout) {
  EXPECT_DEATH(ContextParallelTopology(/*global_rank=*/0,
                                       /*world_size=*/8,
                                       /*dp_size=*/1,
                                       /*pcp_size=*/4,
                                       /*dcp_size=*/3),
               "dcp_size must divide pcp_size");
}

TEST(ContextParallelTopologyTest, LocalRankIndexesEveryLegalDcpGroup) {
  const std::vector<int32_t> dcp_sizes{1, 2, 4, 8};
  for (const int32_t dcp_size : dcp_sizes) {
    for (int32_t global_rank = 0; global_rank < 16; ++global_rank) {
      const ContextParallelTopology topology(global_rank,
                                             /*world_size=*/16,
                                             /*dp_size=*/2,
                                             /*pcp_size=*/4,
                                             dcp_size);
      EXPECT_EQ(topology.dcp_group_ranks()[topology.dcp_rank()], global_rank);
      for (const int32_t member_rank : topology.dcp_group_ranks()) {
        EXPECT_EQ(member_rank / 8, topology.dp_rank());
      }
    }
  }
}

}  // namespace
}  // namespace xllm::parallel_state
