/* Copyright 2025-2026 The xLLM Authors.

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

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

#include "process_group.h"

namespace xllm {

class ProcessGroupImpl : public ProcessGroup {
 public:
  ProcessGroupImpl(int32_t global_rank,
                   int32_t world_size,
                   int32_t rank_size,
                   int32_t port,
                   bool trans,
                   const std::string& host,
                   const std::string& group_name,
                   const torch::Device& device)
      : ProcessGroup(global_rank, world_size, device) {
    c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> pg_options =
        c10d::ProcessGroupNCCL::Options::create();
    pg_options->group_name = group_name;
    int32_t rank = global_rank;
    if (world_size != rank_size) {
      auto [local_rank, group_ranks] =
          get_group_rank(world_size, global_rank, rank_size, trans);
      pg_options->global_ranks_in_group = group_ranks;
      rank = local_rank;
    }

    auto store = create_tcp_store(host, port, rank);
    pg_ = std::make_unique<c10d::ProcessGroupNCCL>(
        store, rank, rank_size, pg_options);
  }

  ProcessGroupImpl(int32_t global_rank,
                   int32_t local_rank,
                   const std::vector<int32_t>& group_ranks,
                   int32_t world_size,
                   int32_t rank_size,
                   int32_t port,
                   const std::string& host,
                   const std::string& group_name,
                   const torch::Device& device)
      : ProcessGroup(global_rank, world_size, device) {
    c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> pg_options =
        c10d::ProcessGroupNCCL::Options::create();
    pg_options->group_name = group_name;
    std::vector<uint64_t> ranks_unsigned;
    ranks_unsigned.reserve(group_ranks.size());
    for (int32_t rank : group_ranks) {
      ranks_unsigned.push_back(static_cast<uint64_t>(rank));
    }
    pg_options->global_ranks_in_group = ranks_unsigned;

    auto store = create_tcp_store(host, port, local_rank);
    pg_ = std::make_unique<c10d::ProcessGroupNCCL>(
        store, local_rank, rank_size, pg_options);
  }
};

}  // namespace xllm
