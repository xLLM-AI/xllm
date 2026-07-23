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

#include "flash_comm1_context.h"
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

torch::Tensor scatter(torch::Tensor input,
                      ProcessGroup* process_group,
                      int dim = -1);

// FlashComm1 sequence-parallel primitives.
//
// shard_dim0_padded: pad `input` along dim0 up to a multiple of world_size,
// then return this rank's contiguous 1/world_size shard. The padding rows live
// only on the last shard(s); callers restore the original token count with
// all_gather_dim0_unpad. A no-op when process_group is null or world_size == 1.
torch::Tensor shard_dim0_padded(const torch::Tensor& input,
                                int32_t rank,
                                int32_t world_size);

// all_gather_dim0_unpad: all-gather the per-rank shards back into the full
// (padded) token dimension along dim0, then slice off padding so the result
// has exactly `original_num_tokens` rows. When original_num_tokens < 0 the
// gathered (still padded) tensor is returned unsliced. A no-op when
// process_group is null or world_size == 1.
torch::Tensor all_gather_dim0_unpad(const torch::Tensor& input,
                                    ProcessGroup* process_group,
                                    int64_t original_num_tokens);

// reduce_scatter_padded_dim0: reduce-scatter along dim0, padding dim0 up to a
// multiple of world_size first. Unlike reduce_scatter(), the trailing padding
// is INTENTIONALLY kept so every rank returns an identically-shaped
// [chunk_size, ...] shard that matches shard_dim0_padded (chunk_size =
// ceil(num_tokens / world_size)). This keeps the FlashComm1 sharded residual
// stream uniform across ranks; padding rows are stripped by the final
// all_gather_dim0_unpad. A no-op when process_group is null or world_size == 1.
torch::Tensor reduce_scatter_padded_dim0(const torch::Tensor& input,
                                         ProcessGroup* process_group);

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
