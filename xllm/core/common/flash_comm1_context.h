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

#include "framework/parallel_state/parallel_args.h"

namespace xllm {

namespace layer {
struct AttentionMetadata;
}

enum class RowParallelReduceMode : int8_t {
    NONE = 0,
    ALL_REDUCE = 1,
    REDUCE_SCATTER = 2,
};

struct FlashComm1Context {
    bool enabled = false;
    int32_t tp_rank = 0;
    int32_t tp_world_size = 1;
    int32_t original_num_tokens = 0;
    int32_t padded_num_tokens = 0;
    int32_t local_num_tokens = 0;
    int32_t pad_size = 0;
    bool is_prefill = true;
    ProcessGroup* tp_group = nullptr;

    FlashComm1Context() = default;

    bool is_sequence_sharded() const { return enabled && tp_world_size > 1; }

    int64_t get_shard_start() const { return tp_rank * local_num_tokens; }
    int64_t get_shard_end() const { return (tp_rank + 1) * local_num_tokens; }
};

FlashComm1Context build_flash_comm1_context(
    int32_t num_tokens,
    bool is_prefill,
    const ParallelArgs& parallel_args);

torch::Tensor shard_sequence(const torch::Tensor& input, const FlashComm1Context& ctx);

torch::Tensor gather_sequence(const torch::Tensor& input, const FlashComm1Context& ctx);

torch::Tensor gather_and_unpad_sequence(const torch::Tensor& input, const FlashComm1Context& ctx);

torch::Tensor maybe_pad_for_reduce(const torch::Tensor& input, const FlashComm1Context& ctx);

torch::Tensor maybe_pad_and_reduce(torch::Tensor input,
                                    const FlashComm1Context& ctx,
                                    RowParallelReduceMode mode);

torch::Tensor maybe_chunk_residual(const torch::Tensor& residual,
                                    int32_t tp_rank,
                                    int32_t tp_world_size);

}  // namespace xllm