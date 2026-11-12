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

#include "core/layers/common/expanded_decode_metadata_builder.h"

#include <glog/logging.h>

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "core/framework/model/llm_model_params.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/model/seq_lens.h"
#include "core/framework/model/vlm_model_params.h"
#include "core/util/tensor_helper.h"

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

torch::Tensor expand_block_tables(const torch::Tensor& block_tables,
                                  const std::vector<int32_t>& q_seq_lens) {
  CHECK_EQ(block_tables.dim(), 2);
  CHECK_GE(block_tables.size(0), static_cast<int64_t>(q_seq_lens.size()))
      << "block table rows are fewer than sequences";

  const int64_t num_tokens =
      std::accumulate(q_seq_lens.begin(), q_seq_lens.end(), int64_t{0});
  CHECK_GT(num_tokens, 0) << "tokenwise expansion requires query tokens";
  const bool uniform = std::all_of(
      q_seq_lens.begin(),
      q_seq_lens.end(),
      [width = q_seq_lens.front()](int32_t len) { return len == width; });
  if (uniform) {
    return block_tables.repeat_interleave(q_seq_lens.front(), /*dim=*/0);
  }
  return block_tables.repeat_interleave(
      async_h2d_tensor(q_seq_lens, block_tables.device()),
      /*dim=*/0,
      /*output_size=*/num_tokens);
}

}  // namespace

template <typename Source>
void ExpandedDecodeMetadataBuilder::populate(ModelParamsView target,
                                             const Source& source,
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
    ModelParamsView target,
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

  std::vector<int32_t> page_counts;
  std::vector<int32_t> last_page_lens;
  std::vector<int32_t> indptr = {0};
  page_counts.reserve(expanded_host_kv_seq_lens.size());
  last_page_lens.reserve(expanded_host_kv_seq_lens.size());
  indptr.reserve(expanded_host_kv_seq_lens.size() + 1);
  for (const int32_t kv_seq_len : expanded_host_kv_seq_lens) {
    page_counts.emplace_back(
        util::append_paged_kv_indptr(kv_seq_len,
                                     expanded_block_tables.size(1),
                                     block_size,
                                     indptr,
                                     last_page_lens));
  }

  const torch::Tensor page_counts_tensor = torch::tensor(
      page_counts, expanded_block_tables.options().dtype(torch::kInt32));
  const torch::Tensor page_offsets = torch::arange(
      expanded_block_tables.size(1), expanded_block_tables.options());
  const torch::Tensor valid_pages =
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
    const ModelParamsView& params) {
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
        CHECK_LE(util::kv_page_count(kv_seq_len, block_size),
                 metadata.block_table.size(1));
      }
    }
  }
}

template void ExpandedDecodeMetadataBuilder::populate(ModelParamsView,
                                                      const ModelParamsView&,
                                                      const torch::Tensor&,
                                                      int32_t);
template void ExpandedDecodeMetadataBuilder::populate(ModelParamsView,
                                                      const LlmModelParams&,
                                                      const torch::Tensor&,
                                                      int32_t);
template void ExpandedDecodeMetadataBuilder::populate(ModelParamsView,
                                                      const VlmModelParams&,
                                                      const torch::Tensor&,
                                                      int32_t);

void ExpandedDecodeMetadataBuilder::reset_expanded_decode_metadata(
    ModelParamsView target) {
  target.graph.use_expanded_decode_for_spec_verify_attention = false;
  target.graph.expanded_kv_seq_lens = torch::Tensor();
  target.graph.expanded_block_tables = torch::Tensor();
  target.graph.expanded_paged_kv_indptr = torch::Tensor();
  target.graph.expanded_paged_kv_indices = torch::Tensor();
  target.graph.expanded_paged_kv_last_page_len = torch::Tensor();
  target.graph.expanded_tiling_data = torch::Tensor();
  target.graph.expanded_kv_seq_lens_vec.clear();
}

bool ExpandedDecodeMetadataBuilder::update_expanded_decode_host_metadata(
    ModelParamsView target) {
  reset_expanded_decode_metadata(target);
  if (target.attention.host.q_seq_lens.empty() ||
      target.attention.host.kv_seq_lens.empty()) {
    return false;
  }
  std::vector<int32_t> expanded_kv_seq_lens = build_tokenwise_kv_seq_lens(
      target.attention.host.q_seq_lens, target.attention.host.kv_seq_lens);
  target.graph.use_expanded_decode_for_spec_verify_attention = true;
  target.graph.expanded_kv_seq_lens_vec = std::move(expanded_kv_seq_lens);
  return true;
}

void ExpandedDecodeMetadataBuilder::update_expanded_decode_metadata(
    ModelParamsView target,
    const torch::Device& device,
    int32_t block_size) {
  if (!update_expanded_decode_host_metadata(target)) {
    return;
  }
  populate_expanded_layout(
      target,
      async_h2d_tensor(target.graph.expanded_kv_seq_lens_vec, device),
      expand_block_tables(target.attention.device.block_tables,
                          target.attention.host.q_seq_lens),
      target.graph.expanded_kv_seq_lens_vec,
      block_size);
}

}  // namespace xllm::layer
