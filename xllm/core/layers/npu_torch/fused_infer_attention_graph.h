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
#include <memory>
#include <vector>

namespace xllm::npu {
class AclGraphTaskUpdateContext;
enum class FusedInferAttentionGraphBranch;
}  // namespace xllm::npu

namespace xllm::layer::detail {

torch::Tensor run_fused_infer_attention_graph(
    const std::shared_ptr<npu::AclGraphTaskUpdateContext>& graph_context,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& block_table,
    const std::vector<int64_t>& actual_seq_lengths,
    const std::vector<int64_t>& actual_seq_lengths_kv,
    int64_t num_key_value_heads,
    double scale,
    int32_t dcp_size,
    int32_t dcp_rank,
    npu::FusedInferAttentionGraphBranch branch,
    torch::Tensor& output);

}  // namespace xllm::layer::detail
