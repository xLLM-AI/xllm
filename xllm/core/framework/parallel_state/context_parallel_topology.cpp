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

#include <glog/logging.h>

namespace xllm::parallel_state {

ContextParallelTopology::ContextParallelTopology(int32_t global_rank,
                                                 int32_t world_size,
                                                 int32_t dp_size,
                                                 int32_t pcp_size,
                                                 int32_t dcp_size)
    : pcp_size_(pcp_size), dcp_size_(dcp_size) {
  CHECK_GT(world_size, 0) << "world_size must be positive";
  CHECK_GT(dp_size, 0) << "dp_size must be positive";
  CHECK_GT(pcp_size_, 0) << "pcp_size must be positive";
  CHECK_GE(global_rank, 0) << "global_rank must be non-negative";
  CHECK_LT(global_rank, world_size) << "global_rank must be in the world";
  CHECK_EQ(world_size % (dp_size * pcp_size_), 0)
      << "world_size must be divisible by dp_size * pcp_size";

  tp_size_ = world_size / (dp_size * pcp_size_);
  const int32_t dp_stride = pcp_size_ * tp_size_;
  dp_rank_ = global_rank / dp_stride;
  const int32_t dp_local_rank = global_rank % dp_stride;
  pcp_rank_ = dp_local_rank / tp_size_;
  tp_rank_ = dp_local_rank % tp_size_;

  const bool partitions_pcp =
      dcp_size_ > 0 && dcp_size_ <= pcp_size_ && pcp_size_ % dcp_size_ == 0;
  const bool supported_dcp_size = partitions_pcp || dcp_size_ == dp_stride;
  CHECK(supported_dcp_size)
      << "dcp_size must divide pcp_size or equal pcp_size * tp_size";

  const int32_t dp_group_start = dp_rank_ * dp_stride;
  pcp_group_ranks_.reserve(static_cast<size_t>(pcp_size_));
  for (int32_t rank = 0; rank < pcp_size_; ++rank) {
    pcp_group_ranks_.emplace_back(dp_group_start + rank * tp_size_ + tp_rank_);
  }

  if (partitions_pcp) {
    const int32_t pcp_per_dcp = pcp_size_ / dcp_size_;
    dcp_rank_ = pcp_rank_ / pcp_per_dcp;
    const int32_t dcp_replica_rank = pcp_rank_ % pcp_per_dcp;
    dcp_group_ranks_.reserve(static_cast<size_t>(dcp_size_));
    for (int32_t owner_rank = 0; owner_rank < dcp_size_; ++owner_rank) {
      const int32_t owner_pcp_rank =
          dcp_replica_rank + owner_rank * pcp_per_dcp;
      dcp_group_ranks_.emplace_back(dp_group_start + owner_pcp_rank * tp_size_ +
                                    tp_rank_);
    }
    return;
  }

  dcp_group_ranks_.reserve(static_cast<size_t>(dp_stride));
  for (int32_t rank = 0; rank < dp_stride; ++rank) {
    dcp_group_ranks_.emplace_back(dp_group_start + rank);
  }
  dcp_rank_ = dp_local_rank;
}

}  // namespace xllm::parallel_state
