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

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

constexpr int64_t kInt8QuantMode = 2;
constexpr int64_t kKeyHeads = 1;
constexpr int64_t kNoReturnValue = 0;
constexpr int64_t kRightDownCausal = 3;
constexpr int64_t kCandidateSource = 1;

}  // namespace

torch::Tensor quant_lightning_indexer_v2_metadata(
    const torch::Tensor& query_offsets,
    const torch::Tensor& committed_lengths,
    const std::optional<torch::Tensor>& residual_lengths,
    int64_t query_heads,
    int64_t head_dim,
    int64_t topk,
    int64_t max_query_len,
    int64_t max_committed_len,
    int64_t compression_ratio) {
  auto metadata = query_offsets.new_empty({kDsaMetadataBufferElements});
  auto empty = query_offsets.new_empty({0});
  const auto& residual =
      residual_lengths.has_value() ? *residual_lengths : empty;
  const int64_t batch_size = committed_lengths.numel();
  char query_layout[] = "TND";
  char key_layout[] = "PA_BBND";
  EXEC_NPU_CMD(aclnnQuantLightningIndexerV2MetadataVa,
               query_offsets,
               /*cu_seqlens_k=*/empty,
               /*seqused_q=*/empty,
               committed_lengths,
               residual,
               query_heads,
               kKeyHeads,
               head_dim,
               topk,
               kInt8QuantMode,
               batch_size,
               max_query_len,
               max_committed_len,
               query_layout,
               key_layout,
               kRightDownCausal,
               compression_ratio,
               metadata);
  return metadata;
}

std::tuple<torch::Tensor, torch::Tensor> quant_lightning_indexer_v2(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& weights,
    const torch::Tensor& query_scale,
    const torch::Tensor& key_scale,
    const torch::Tensor& query_offsets,
    const torch::Tensor& committed_lengths,
    const std::optional<torch::Tensor>& residual_lengths,
    const torch::Tensor& block_table,
    const torch::Tensor& metadata,
    const std::optional<torch::Tensor>& candidate_blocks,
    int64_t topk,
    int64_t max_query_len,
    int64_t compression_ratio,
    int64_t candidate_mode,
    int64_t candidate_count,
    int64_t candidate_block_size) {
  const int64_t rows = query.size(0);
  const int64_t key_heads = key.size(2);
  auto output_options = query.options().dtype(torch::kInt32);
  auto indices = torch::empty({rows, key_heads, topk}, output_options);
  auto values = torch::empty({0}, query.options().dtype(torch::kBFloat16));
  auto candidates =
      candidate_mode == kCandidateSource
          ? torch::empty({rows, key_heads, candidate_count}, output_options)
          : torch::empty({0}, output_options);
  const std::optional<torch::Tensor> absent;
  const int64_t key_page_stride = key.stride(0);
  const int64_t scale_page_stride = key_scale.stride(0);
  char query_layout[] = "TND";
  char key_layout[] = "PA_BBND";
  EXEC_NPU_CMD(aclnnQuantLightningIndexerV2Va,
               query,
               key,
               weights,
               query_scale,
               key_scale,
               query_offsets,
               /*cu_seqlens_k=*/absent,
               /*seqused_q=*/absent,
               committed_lengths,
               residual_lengths,
               block_table,
               /*output_idx_offset=*/absent,
               metadata,
               candidate_blocks,
               topk,
               kInt8QuantMode,
               max_query_len,
               query_layout,
               key_layout,
               kRightDownCausal,
               compression_ratio,
               kNoReturnValue,
               candidate_mode,
               candidate_count,
               candidate_block_size,
               key_page_stride,
               scale_page_stride,
               indices,
               values,
               candidates);
  return {indices, candidates};
}

}  // namespace xllm::kernel::npu
