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

#include "layers/npu_torch/dcp_attention_utils.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <tuple>

namespace xllm::layer::detail {

std::vector<int64_t> compute_dcp_local_kv_seq_lens(
    const std::vector<int64_t>& global_kv_seq_lens,
    int32_t dcp_size,
    int32_t dcp_rank,
    int64_t block_size) {
  CHECK_GT(dcp_size, 1);
  CHECK_GE(dcp_rank, 0);
  CHECK_LT(dcp_rank, dcp_size);
  CHECK_GT(block_size, 0);

  std::vector<int64_t> local_kv_seq_lens;
  local_kv_seq_lens.reserve(global_kv_seq_lens.size());
  for (const int64_t global_kv_seq_len : global_kv_seq_lens) {
    CHECK_GE(global_kv_seq_len, 0);
    const int64_t base = global_kv_seq_len / block_size / dcp_size * block_size;
    const int64_t remainder = global_kv_seq_len - base * dcp_size;
    const int64_t rank_offset = static_cast<int64_t>(dcp_rank) * block_size;
    const int64_t local_remainder =
        std::clamp(remainder - rank_offset, int64_t{0}, block_size);
    local_kv_seq_lens.emplace_back(base + local_remainder);
  }
  return local_kv_seq_lens;
}

void normalize_zero_dcp_partials_for_graph(
    torch::Tensor& partial_out,
    torch::Tensor& partial_lse,
    const torch::Tensor& global_kv_seq_lens,
    int32_t dcp_rank,
    int64_t block_size) {
  CHECK(global_kv_seq_lens.defined());
  CHECK_EQ(global_kv_seq_lens.dim(), 1);
  CHECK_EQ(global_kv_seq_lens.size(0), partial_out.size(0));
  const int64_t first_local_block_offset =
      static_cast<int64_t>(dcp_rank) * block_size;
  const torch::Tensor zero_local_kv_mask =
      global_kv_seq_lens.le(first_local_block_offset)
          .view({global_kv_seq_lens.size(0), 1, 1});
  partial_out.masked_fill_(zero_local_kv_mask, 0.0);
  partial_lse.masked_fill_(zero_local_kv_mask,
                           -std::numeric_limits<float>::infinity());
}

std::vector<int64_t> compute_dcp_context_lens(
    const std::vector<int64_t>& q_cu_seq_lens,
    const std::vector<int64_t>& global_kv_seq_lens) {
  CHECK(!q_cu_seq_lens.empty())
      << "chunked prefill requires cumulative query lengths";
  CHECK_EQ(q_cu_seq_lens.size(), global_kv_seq_lens.size())
      << "chunked prefill requires one query and KV length per request";

  std::vector<int64_t> context_lens;
  context_lens.reserve(global_kv_seq_lens.size());
  int64_t previous_q_end = 0;
  for (size_t request_index = 0; request_index < global_kv_seq_lens.size();
       ++request_index) {
    const int64_t q_end = q_cu_seq_lens[request_index];
    const int64_t query_len = q_end - previous_q_end;
    const int64_t context_len = global_kv_seq_lens[request_index] - query_len;
    CHECK_GE(context_len, 0)
        << "chunked prefill context length must be non-negative";
    context_lens.emplace_back(context_len);
    previous_q_end = q_end;
  }
  return context_lens;
}

std::vector<int64_t> validate_dcp_chunked_lengths(
    const std::vector<int64_t>& q_cu_seq_lens,
    const std::vector<int64_t>& global_kv_seq_lens,
    int64_t token_count) {
  CHECK(!q_cu_seq_lens.empty())
      << "DCP chunked prefill requires host cumulative query lengths.";

  const size_t query_begin = q_cu_seq_lens.front() == 0 ? 1 : 0;
  CHECK_LT(query_begin, q_cu_seq_lens.size())
      << "DCP chunked prefill requires at least one request.";
  const std::vector<int64_t> normalized_q_cu_seq_lens(
      q_cu_seq_lens.begin() + query_begin, q_cu_seq_lens.end());

  CHECK_EQ(normalized_q_cu_seq_lens.size(), global_kv_seq_lens.size())
      << "DCP chunked prefill requires one query and KV length per request.";

  int64_t previous_q_end = 0;
  for (size_t request_index = 0;
       request_index < normalized_q_cu_seq_lens.size();
       ++request_index) {
    const int64_t q_end = normalized_q_cu_seq_lens[request_index];
    const int64_t query_len = q_end - previous_q_end;
    CHECK_GE(query_len, 1)
        << "DCP chunked prefill requires at least one query token per request.";
    CHECK_LE(query_len, kMaxDcpChunkedPrefillQueryLen)
        << "DCP chunked prefill does not yet support chunked query length "
           "above "
        << kMaxDcpChunkedPrefillQueryLen << ".";
    CHECK_GE(global_kv_seq_lens[request_index], query_len)
        << "DCP chunked prefill KV length must cover the current chunk query.";
    previous_q_end = q_end;
  }
  CHECK_EQ(previous_q_end, token_count)
      << "DCP chunked cumulative query lengths do not match query tokens.";
  return normalized_q_cu_seq_lens;
}

void validate_dcp_chunked_block_table(const torch::Tensor& local_block_table,
                                      int64_t request_count) {
  CHECK(local_block_table.defined())
      << "DCP chunked local block table must be defined.";
  CHECK_EQ(local_block_table.dim(), 2)
      << "DCP chunked local block table must be two-dimensional.";
  CHECK_EQ(local_block_table.size(0), request_count)
      << "DCP local block table batch size does not match request count.";
}

void normalize_zero_dcp_chunked_partials(
    torch::Tensor& partial_out,
    torch::Tensor& partial_lse,
    const std::vector<int64_t>& local_context_lens,
    const std::vector<int64_t>& q_cu_seq_lens) {
  CHECK_EQ(partial_out.scalar_type(), torch::kFloat32);
  CHECK_EQ(partial_lse.scalar_type(), torch::kFloat32);
  CHECK_EQ(partial_out.dim(), 3);
  CHECK_EQ(partial_lse.dim(), 3);
  CHECK_EQ(partial_out.size(0), partial_lse.size(0));
  CHECK_EQ(partial_out.size(1), partial_lse.size(1));
  CHECK_EQ(partial_lse.size(2), 1);
  CHECK_EQ(local_context_lens.size(), q_cu_seq_lens.size());

  int64_t previous_q_end = 0;
  for (size_t request_index = 0; request_index < local_context_lens.size();
       ++request_index) {
    const int64_t q_end = q_cu_seq_lens[request_index];
    const int64_t query_len = q_end - previous_q_end;
    CHECK_GE(query_len, 1);
    CHECK_GE(local_context_lens[request_index], 0);
    if (local_context_lens[request_index] == 0) {
      partial_out.narrow(0, previous_q_end, query_len).zero_();
      partial_lse.narrow(0, previous_q_end, query_len)
          .fill_(-std::numeric_limits<float>::infinity());
    }
    previous_q_end = q_end;
  }
  CHECK_EQ(previous_q_end, partial_out.size(0));
}

torch::Tensor merge_dcp_partials(const torch::Tensor& all_partial_out,
                                 const torch::Tensor& all_partial_lse) {
  CHECK(all_partial_out.scalar_type() == torch::kFloat32 ||
        all_partial_out.scalar_type() == torch::kFloat64);
  CHECK_EQ(all_partial_lse.scalar_type(), all_partial_out.scalar_type());
  CHECK_EQ(all_partial_out.dim(), 4);
  CHECK_EQ(all_partial_lse.dim(), 4);
  CHECK_EQ(all_partial_out.size(0), all_partial_lse.size(0));
  CHECK_EQ(all_partial_out.size(1), all_partial_lse.size(1));
  CHECK_EQ(all_partial_out.size(2), all_partial_lse.size(2));
  CHECK_EQ(all_partial_lse.size(3), 1);

  const torch::Tensor finite_lse = torch::isfinite(all_partial_lse);
  const torch::Tensor max_lse = std::get<0>(all_partial_lse.max(0));
  const torch::Tensor max_lse_is_finite = torch::isfinite(max_lse);
  const torch::Tensor safe_max_lse =
      torch::where(max_lse_is_finite, max_lse, torch::zeros_like(max_lse));
  const torch::Tensor weights =
      torch::where(finite_lse,
                   torch::exp(all_partial_lse - safe_max_lse),
                   torch::zeros_like(all_partial_lse));
  const torch::Tensor safe_partial_out =
      torch::where(finite_lse.expand_as(all_partial_out),
                   all_partial_out,
                   torch::zeros_like(all_partial_out));
  const torch::Tensor denominator = weights.sum(0);
  const torch::Tensor safe_denominator = torch::where(
      denominator.gt(0), denominator, torch::ones_like(denominator));
  const torch::Tensor merged_out =
      (weights * safe_partial_out).sum(0) / safe_denominator;
  return torch::where(max_lse_is_finite.expand_as(merged_out),
                      merged_out,
                      torch::zeros_like(merged_out));
}

}  // namespace xllm::layer::detail
