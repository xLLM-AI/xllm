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

#include <torch/torch.h>

#include <vector>

#include "framework/parallel_state/parallel_args.h"

namespace xllm {

namespace layer {
struct AttentionMetadata;
}

enum class RowParallelReduceMode : int8_t {
    NONE = 0,
    ALL_REDUCE = 1,
    REDUCE_SCATTER = 2,
    MATMUL_REDUCE_SCATTER = 3,
};

struct FlashComm1Context {
    bool enabled = false;
    int32_t tp_rank = 0;
    int32_t tp_world_size = 1;
    int32_t original_num_tokens = 0;
    int32_t padded_num_tokens = 0;
    int32_t local_num_tokens = 0;
    int32_t padded_local_num_tokens = 0;
    int32_t pad_size = 0;
    bool is_prefill = true;
    bool enable_mmrs_fusion = false;
    std::string mmrs_comm_mode = "aiv";
    ProcessGroup* tp_group = nullptr;

    FlashComm1Context() = default;

    bool is_sequence_sharded() const { return enabled && tp_world_size > 1; }

    int32_t local_num_tokens_for_rank(int32_t rank) const;
    std::vector<int32_t> token_num_list() const;

    int64_t get_shard_start() const;
    int64_t get_shard_end() const;
};

struct FlashComm1Options {
    bool enable_flashcomm1 = false;
    int32_t min_prefill_tokens = 1000;
    int32_t min_decode_tokens = 128;
    bool enable_mmrs_fusion = false;
    std::string mmrs_comm_mode = "aiv";
};

FlashComm1Context build_flash_comm1_context(
    int32_t num_tokens,
    bool is_prefill,
    const ParallelArgs& parallel_args);

FlashComm1Context build_flash_comm1_context(
    int32_t num_tokens,
    bool is_prefill,
    const ParallelArgs& parallel_args,
    const FlashComm1Options& options);

torch::Tensor shard_sequence(const torch::Tensor& input, const FlashComm1Context& ctx);

torch::Tensor gather_sequence(const torch::Tensor& input, const FlashComm1Context& ctx);

torch::Tensor gather_and_unpad_sequence(const torch::Tensor& input, const FlashComm1Context& ctx);

torch::Tensor maybe_pad_for_reduce(const torch::Tensor& input, const FlashComm1Context& ctx);

torch::Tensor maybe_pad_and_reduce(torch::Tensor input,
                                    const FlashComm1Context& ctx,
                                    RowParallelReduceMode mode);

RowParallelReduceMode row_parallel_reduce_mode_for_fc1(
    const FlashComm1Context& ctx);

torch::Tensor maybe_chunk_residual(const torch::Tensor& residual,
                                    int32_t tp_rank,
                                    int32_t tp_world_size);

torch::Tensor maybe_shard_residual(const torch::Tensor& residual,
                                   const FlashComm1Context& ctx);

}  // namespace xllm
