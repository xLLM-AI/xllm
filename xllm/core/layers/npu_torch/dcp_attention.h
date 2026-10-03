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

#include <torch/types.h>

#include <cstdint>

#include "layers/common/dcp_attention_merge.h"

namespace xllm {

class ProcessGroup;

namespace layer {

struct AttentionMetadata;

namespace detail {

DcpAttentionResult merge_dcp_tnd_attention_shards(
    const torch::Tensor& partial_outputs,
    const torch::Tensor& partial_lse);

}  // namespace detail

// Replicated-GQA attention with cache pages striped across the DCP group.
// The caller validates that group members own the same KV head.
void dcp_decode(const torch::Tensor& query,
                torch::Tensor& output,
                const torch::Tensor& key_cache,
                const torch::Tensor& value_cache,
                const AttentionMetadata& attention_metadata,
                int64_t num_kv_heads,
                double scale,
                ProcessGroup& dcp_group);

void dcp_chunked_prefill(const torch::Tensor& query,
                         const torch::Tensor& key,
                         const torch::Tensor& value,
                         torch::Tensor& output,
                         const torch::Tensor& key_cache,
                         const torch::Tensor& value_cache,
                         const AttentionMetadata& attention_metadata,
                         int64_t num_kv_heads,
                         double scale,
                         ProcessGroup& dcp_group);

}  // namespace layer
}  // namespace xllm
