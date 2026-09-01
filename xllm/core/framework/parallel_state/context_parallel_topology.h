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

#pragma once

#include <cstdint>
#include <vector>

namespace xllm::parallel_state {

// Rank layout for prefill context parallelism (PCP) and decode context
// parallelism (DCP).
//
// The public configuration names map to the topology terms as follows:
//   pcp_size = cp_size
//   dcp_size = kv_split_size_effective()
// Accordingly, cp_rank is the PCP rank and kv_split_rank is the DCP rank.
// cp_size and kv_split_size are retained as the public configuration names for
// compatibility, while PCP and DCP describe their runtime topology semantics.
//
// A PCP group varies pcp_rank while keeping dp_rank and tp_rank fixed. DCP
// describes KV-cache ownership and decode merging. A DCP group either
// partitions a PCP group when dcp_size divides pcp_size, or spans the complete
// DP-local PCP x TP domain when dcp_size equals pcp_size * tp_size.
class ContextParallelTopology final {
 public:
  ContextParallelTopology(int32_t global_rank,
                          int32_t world_size,
                          int32_t dp_size,
                          int32_t pcp_size,
                          int32_t dcp_size);

  int32_t dp_rank() const { return dp_rank_; }
  int32_t tp_size() const { return tp_size_; }
  int32_t tp_rank() const { return tp_rank_; }
  int32_t pcp_size() const { return pcp_size_; }
  int32_t pcp_rank() const { return pcp_rank_; }
  int32_t dcp_size() const { return dcp_size_; }
  int32_t dcp_rank() const { return dcp_rank_; }

  const std::vector<int32_t>& pcp_group_ranks() const {
    return pcp_group_ranks_;
  }
  const std::vector<int32_t>& dcp_group_ranks() const {
    return dcp_group_ranks_;
  }

 private:
  int32_t dp_rank_ = 0;
  int32_t tp_size_ = 1;
  int32_t tp_rank_ = 0;
  int32_t pcp_size_ = 1;
  int32_t pcp_rank_ = 0;
  int32_t dcp_size_ = 1;
  int32_t dcp_rank_ = 0;
  std::vector<int32_t> pcp_group_ranks_;
  std::vector<int32_t> dcp_group_ranks_;
};

}  // namespace xllm::parallel_state
