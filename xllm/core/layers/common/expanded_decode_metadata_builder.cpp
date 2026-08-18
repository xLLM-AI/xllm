/* Copyright 2026 The xLLM Authors.

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

#include "layers/common/expanded_decode_metadata_builder.h"

#include <glog/logging.h>

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "framework/model/model_input_params.h"

namespace xllm::layer {
namespace {

std::vector<int32_t> expand_host_kv_seq_lens(
    const std::vector<int32_t>& kv_seq_lens) {
  std::vector<int32_t> expanded;
  expanded.reserve(kv_seq_lens.size() * 2);
  for (int32_t kv_seq_len : kv_seq_lens) {
    CHECK_GT(kv_seq_len, 0) << "KV sequence length must be positive";
    expanded.emplace_back(kv_seq_len - 1);
    expanded.emplace_back(kv_seq_len);
  }
  return expanded;
}

std::vector<int32_t> build_page_counts(const std::vector<int32_t>& kv_seq_lens,
                                       int32_t block_size,
                                       int64_t block_table_capacity) {
  std::vector<int32_t> page_counts;
  page_counts.reserve(kv_seq_lens.size());
  for (int32_t kv_seq_len : kv_seq_lens) {
    const int32_t effective_kv_seq_len = std::max(kv_seq_len, 1);
    const int32_t page_count =
        (effective_kv_seq_len + block_size - 1) / block_size;
    CHECK_LE(page_count, block_table_capacity)
        << "Expanded KV length exceeds block-table capacity";
    page_counts.emplace_back(page_count);
  }
  return page_counts;
}

std::vector<int32_t> build_indptr(const std::vector<int32_t>& page_counts) {
  std::vector<int32_t> indptr;
  indptr.reserve(page_counts.size() + 1);
  indptr.emplace_back(0);
  for (int32_t page_count : page_counts) {
    CHECK_GT(page_count, 0);
    indptr.emplace_back(indptr.back() + page_count);
  }
  return indptr;
}

std::vector<int32_t> build_last_page_lens(
    const std::vector<int32_t>& kv_seq_lens,
    int32_t block_size) {
  std::vector<int32_t> last_page_lens;
  last_page_lens.reserve(kv_seq_lens.size());
  for (int32_t kv_seq_len : kv_seq_lens) {
    const int32_t effective_kv_seq_len = std::max(kv_seq_len, 1);
    last_page_lens.emplace_back((effective_kv_seq_len - 1) % block_size + 1);
  }
  return last_page_lens;
}

}  // namespace

void ExpandedDecodeMetadataBuilder::populate(ModelInputParams& target,
                                             const ModelInputParams& source,
                                             const torch::Tensor& kv_seq_lens,
                                             int32_t block_size) {
  const torch::Tensor& source_block_tables =
      source.attention.device.block_tables;
  CHECK(source_block_tables.defined());
  CHECK_EQ(source_block_tables.dim(), 2);
  CHECK_EQ(kv_seq_lens.dim(), 1);
  CHECK_EQ(source_block_tables.size(0), kv_seq_lens.size(0));
  CHECK_GT(block_size, 0);
  CHECK_GT(source_block_tables.size(1), 0);

  const std::vector<int32_t>& source_host_kv_seq_lens =
      source.attention.host.kv_seq_lens;
  CHECK_EQ(source_host_kv_seq_lens.size(),
           static_cast<size_t>(kv_seq_lens.numel()))
      << "Scheduler host KV lengths must match device sequence count";
  std::vector<int32_t> expanded_host_kv_seq_lens =
      expand_host_kv_seq_lens(source_host_kv_seq_lens);
  torch::Tensor expanded_kv_seq_lens =
      torch::stack({kv_seq_lens - 1, kv_seq_lens}, /*dim=*/1).flatten();
  torch::Tensor expanded_block_tables =
      source_block_tables.repeat_interleave(/*repeats=*/2, /*dim=*/0);
  populate_expanded_layout(target,
                           expanded_kv_seq_lens,
                           expanded_block_tables,
                           std::move(expanded_host_kv_seq_lens),
                           block_size);

  target.attention.device.block_tables = expanded_block_tables.contiguous();
  target.attention.device.paged_kv_indptr =
      target.graph.expanded_paged_kv_indptr;
  target.attention.device.paged_kv_indices =
      target.graph.expanded_paged_kv_indices;
  target.attention.device.paged_kv_last_page_len =
      target.graph.expanded_paged_kv_last_page_len;
  if (source.attention.host.block_tables.defined() &&
      source.attention.host.block_tables.device().is_cpu()) {
    target.attention.host.block_tables =
        source.attention.host.block_tables.repeat_interleave(/*repeats=*/2,
                                                             /*dim=*/0);
  }
}

void ExpandedDecodeMetadataBuilder::populate_expanded_layout(
    ModelInputParams& target,
    const torch::Tensor& expanded_kv_seq_lens,
    const torch::Tensor& expanded_block_tables,
    std::vector<int32_t> expanded_host_kv_seq_lens,
    int32_t block_size) {
  CHECK(expanded_kv_seq_lens.defined());
  CHECK(expanded_block_tables.defined());
  CHECK_EQ(expanded_kv_seq_lens.dim(), 1);
  CHECK_EQ(expanded_block_tables.dim(), 2);
  CHECK_EQ(expanded_block_tables.size(0), expanded_kv_seq_lens.numel());
  CHECK_EQ(expanded_host_kv_seq_lens.size(),
           static_cast<size_t>(expanded_kv_seq_lens.numel()));
  CHECK_GT(expanded_block_tables.size(1), 0);
  CHECK_GT(block_size, 0);

  const std::vector<int32_t> page_counts = build_page_counts(
      expanded_host_kv_seq_lens, block_size, expanded_block_tables.size(1));
  const std::vector<int32_t> indptr = build_indptr(page_counts);
  const std::vector<int32_t> last_page_lens =
      build_last_page_lens(expanded_host_kv_seq_lens, block_size);
  CHECK_EQ(indptr.front(), 0);
  CHECK(std::is_sorted(indptr.begin(), indptr.end()));
  CHECK_EQ(indptr.back(),
           std::accumulate(page_counts.begin(), page_counts.end(), int32_t{0}));
  for (int32_t last_page_len : last_page_lens) {
    CHECK_GE(last_page_len, 1);
    CHECK_LE(last_page_len, block_size);
  }

  torch::Tensor page_counts_tensor = torch::tensor(
      page_counts, expanded_block_tables.options().dtype(torch::kInt32));
  torch::Tensor page_offsets = torch::arange(expanded_block_tables.size(1),
                                             expanded_block_tables.options());
  torch::Tensor valid_pages =
      page_offsets.unsqueeze(0) < page_counts_tensor.unsqueeze(1);

  target.graph.expanded_paged_kv_indices =
      expanded_block_tables.masked_select(valid_pages).contiguous();
  target.graph.expanded_paged_kv_indptr = torch::tensor(
      indptr, expanded_block_tables.options().dtype(torch::kInt32));
  target.graph.expanded_paged_kv_last_page_len = torch::tensor(
      last_page_lens, expanded_block_tables.options().dtype(torch::kInt32));

  target.graph.use_expanded_decode_for_spec_verify_attention = true;
  target.graph.expanded_kv_seq_lens = expanded_kv_seq_lens;
  target.graph.expanded_block_tables = expanded_block_tables;
  target.graph.expanded_tiling_data = torch::Tensor();
  target.graph.expanded_kv_seq_lens_vec = std::move(expanded_host_kv_seq_lens);

  ExpandedDecodeMetadataBuilder::validate(
      ExpandedDecodeMetadataBuilder::build(target),
      expanded_kv_seq_lens.numel(),
      block_size);
}

std::vector<int32_t> ExpandedDecodeMetadataBuilder::build_tokenwise_kv_seq_lens(
    const std::vector<int32_t>& q_seq_lens,
    const std::vector<int32_t>& kv_seq_lens) {
  CHECK_EQ(q_seq_lens.size(), kv_seq_lens.size())
      << "q/kv sequence lengths must both be sequence-scoped";
  std::vector<int32_t> expanded_kv_seq_lens;
  for (size_t seq_idx = 0; seq_idx < q_seq_lens.size(); ++seq_idx) {
    const int32_t q_len = q_seq_lens[seq_idx];
    const int32_t kv_len = kv_seq_lens[seq_idx];
    CHECK_GE(q_len, 1) << "query sequence length must be positive";
    CHECK_GE(kv_len, q_len) << "KV length must include query tokens";
    for (int32_t token_idx = 0; token_idx < q_len; ++token_idx) {
      expanded_kv_seq_lens.emplace_back(kv_len - q_len + token_idx + 1);
    }
  }
  return expanded_kv_seq_lens;
}

ExpandedDecodeMetadata ExpandedDecodeMetadataBuilder::build(
    const ModelInputParams& params) {
  ExpandedDecodeMetadata metadata;
  metadata.enabled = params.graph.use_expanded_decode_for_spec_verify_attention;
  if (!metadata.enabled) {
    return metadata;
  }

  metadata.kv_seq_lens = params.graph.expanded_kv_seq_lens;
  metadata.block_table = params.graph.expanded_block_tables;
  metadata.paged_kv_indptr = params.graph.expanded_paged_kv_indptr;
  metadata.paged_kv_indices = params.graph.expanded_paged_kv_indices;
  metadata.paged_kv_last_page_len =
      params.graph.expanded_paged_kv_last_page_len;
  metadata.paged_attention_tiling_data = params.graph.expanded_tiling_data;
  metadata.kv_seq_lens_host_vec = params.graph.expanded_kv_seq_lens_vec;
  if (!params.graph.expanded_kv_seq_lens_vec.empty()) {
    metadata.kv_seq_lens_host =
        torch::tensor(params.graph.expanded_kv_seq_lens_vec, torch::kInt32);
  }
  validate(metadata);
  return metadata;
}

void ExpandedDecodeMetadataBuilder::validate(
    const ExpandedDecodeMetadata& metadata,
    int64_t expected_sequence_count,
    int32_t block_size) {
  if (!metadata.enabled) {
    return;
  }

  CHECK(metadata.kv_seq_lens.defined());
  CHECK(metadata.block_table.defined());
  CHECK(metadata.paged_kv_indptr.defined());
  CHECK(metadata.paged_kv_indices.defined());
  CHECK(metadata.paged_kv_last_page_len.defined());
  CHECK_EQ(metadata.kv_seq_lens.dim(), 1);
  CHECK_EQ(metadata.block_table.dim(), 2);
  const int64_t sequence_count = metadata.kv_seq_lens.numel();
  if (expected_sequence_count >= 0) {
    CHECK_EQ(sequence_count, expected_sequence_count);
  }
  CHECK_EQ(metadata.block_table.size(0), sequence_count);
  CHECK_EQ(metadata.paged_kv_indptr.dim(), 1);
  CHECK_EQ(metadata.paged_kv_indptr.numel(), sequence_count + 1);
  CHECK_EQ(metadata.paged_kv_indices.dim(), 1);
  CHECK_GT(metadata.paged_kv_indices.numel(), 0);
  CHECK_EQ(metadata.paged_kv_last_page_len.dim(), 1);
  CHECK_EQ(metadata.paged_kv_last_page_len.numel(), sequence_count);
  if (metadata.kv_seq_lens_host.defined()) {
    CHECK(metadata.kv_seq_lens_host.device().is_cpu());
    CHECK_EQ(metadata.kv_seq_lens_host.dim(), 1);
    CHECK_EQ(metadata.kv_seq_lens_host.numel(), sequence_count);
  }
  if (metadata.paged_kv_indptr.device().is_cpu()) {
    const torch::Tensor& host_indptr = metadata.paged_kv_indptr;
    CHECK_EQ(host_indptr[0].item<int32_t>(), 0);
    const int64_t offset_count = host_indptr.numel() - 1;
    CHECK(torch::all(host_indptr.narrow(0, 1, offset_count) >=
                     host_indptr.narrow(0, 0, offset_count))
              .item<bool>())
        << "paged_kv_indptr must be monotonic";
    CHECK_EQ(host_indptr[host_indptr.numel() - 1].item<int32_t>(),
             metadata.paged_kv_indices.numel());
  }
  if (metadata.paged_kv_last_page_len.device().is_cpu()) {
    const torch::Tensor& host_last_page_len = metadata.paged_kv_last_page_len;
    CHECK(torch::all(host_last_page_len >= 1).item<bool>());
    if (block_size > 0) {
      CHECK(torch::all(host_last_page_len <= block_size).item<bool>());
    }
  }
  if (!metadata.kv_seq_lens_host_vec.empty()) {
    CHECK_EQ(metadata.kv_seq_lens_host_vec.size(),
             static_cast<size_t>(sequence_count));
    for (int32_t kv_seq_len : metadata.kv_seq_lens_host_vec) {
      CHECK_GE(kv_seq_len, 0);
      if (block_size > 0) {
        const int32_t effective_kv_seq_len = std::max(kv_seq_len, 1);
        const int32_t page_count =
            (effective_kv_seq_len + block_size - 1) / block_size;
        CHECK_LE(page_count, metadata.block_table.size(1));
      }
    }
  }
}

}  // namespace xllm::layer
