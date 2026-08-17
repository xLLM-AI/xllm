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

#include <cstdint>
#include <vector>

namespace xllm::layer::detail {

inline constexpr int64_t kMaxDcpChunkedPrefillQueryLen = 2048;

std::vector<int64_t> compute_dcp_local_kv_seq_lens(
    const std::vector<int64_t>& global_kv_seq_lens,
    int32_t dcp_size,
    int32_t dcp_rank,
    int64_t block_size);

// Graph replay must derive the empty-shard mask from the live device KV
// lengths. A host-side branch would be frozen at capture time.
void normalize_zero_dcp_partials_for_graph(
    torch::Tensor& partial_out,
    torch::Tensor& partial_lse,
    const torch::Tensor& global_kv_seq_lens,
    int32_t dcp_rank,
    int64_t block_size);

// Per-request cached-context length for chunked prefill: the KV that precedes
// the current chunk, derived as global_kv_seq_len - current_chunk_query_len.
// q_cu_seq_lens is the cumulative host query length per request.
std::vector<int64_t> compute_dcp_context_lens(
    const std::vector<int64_t>& q_cu_seq_lens,
    const std::vector<int64_t>& global_kv_seq_lens);

// Chunked-prefill counterpart of the decode length validator: query tokens per
// request may exceed one, so the partial tensors are indexed by token rather
// than by request.
std::vector<int64_t> validate_dcp_chunked_lengths(
    const std::vector<int64_t>& q_cu_seq_lens,
    const std::vector<int64_t>& global_kv_seq_lens,
    int64_t token_count);

void validate_dcp_chunked_block_table(const torch::Tensor& local_block_table,
                                      int64_t request_count);

// Zero the context partial for requests whose local context shard is empty on
// this rank. Unlike the decode variant, the partial's leading dimension is the
// flattened token count, so each empty request zeroes its own token range
// [previous_q_end, q_cu_seq_lens[request]).
void normalize_zero_dcp_chunked_partials(
    torch::Tensor& partial_out,
    torch::Tensor& partial_lse,
    const std::vector<int64_t>& local_context_lens,
    const std::vector<int64_t>& q_cu_seq_lens);

torch::Tensor merge_dcp_partials(const torch::Tensor& all_partial_out,
                                 const torch::Tensor& all_partial_lse);

}  // namespace xllm::layer::detail
