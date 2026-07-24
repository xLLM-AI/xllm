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

#include "parallel_args.h"
#include "process_group.h"

namespace xllm {

// Forward declaration
namespace runtime {
struct Options;
}

namespace parallel_state {

struct GatherAsyncCtx {
  torch::Tensor input;
  torch::Tensor stacked;
  c10::intrusive_ptr<c10d::Work> work;
  std::vector<int32_t> token_num_list;
};

struct ReduceAsyncCtx {
  torch::Tensor tensor;
  c10::intrusive_ptr<c10d::Work> work;
};

std::optional<ParallelArgs> get_dp_attn_parallel_args(
    const ParallelArgs& parallel_args);

torch::Tensor gather(const torch::Tensor& input,
                     ProcessGroup* process_group,
                     int32_t dim = -1);

torch::Tensor gather(const torch::Tensor& input,
                     ProcessGroup* process_group,
                     const std::vector<int32_t>& token_num_list);

GatherAsyncCtx launch_gather(const torch::Tensor& input,
                             ProcessGroup* process_group,
                             const std::vector<int32_t>& token_num_list);

torch::Tensor finish_gather(GatherAsyncCtx ctx);

ReduceAsyncCtx launch_reduce(torch::Tensor input, ProcessGroup* process_group);

torch::Tensor finish_reduce(ReduceAsyncCtx ctx);

torch::Tensor all_gather_interleaved(const torch::Tensor& input,
                                     ProcessGroup* process_group);

torch::Tensor reduce(torch::Tensor& input, ProcessGroup* process_group);

torch::Tensor reduce_scatter(const torch::Tensor& input,
                             ProcessGroup* process_group);

// Global ranks in this rank's CP group, ordered by CP rank.
std::vector<int32_t> compute_cp_group_ranks(int32_t global_rank,
                                            int32_t world_size,
                                            int32_t dp_size,
                                            int32_t cp_size);

torch::Tensor scatter(torch::Tensor input,
                      ProcessGroup* process_group,
                      int dim = -1);

std::function<torch::Tensor()> all_to_all_4D(const torch::Tensor& input_,
                                             int32_t scatter_idx,
                                             int32_t gather_idx,
                                             bool is_sync,
                                             ProcessGroup* pg);

// Create a process group where each process has a single device
// devices: list of devices to create process groups on.
std::vector<std::unique_ptr<ProcessGroup>> create_npu_process_groups(
    const std::vector<torch::Device>& devices);

// Create process groups for local (single-node) scenarios
// Supports GPU (CUDA/MLU) and NPU, including single-device case
// Parse port from options.master_node_addr() to support multiple instances
std::vector<std::unique_ptr<ProcessGroup>> create_local_process_groups(
    const std::vector<torch::Device>& devices,
    const runtime::Options& options);

}  // namespace parallel_state
}  // namespace xllm
