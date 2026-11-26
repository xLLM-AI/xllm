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

#include <optional>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

constexpr int64_t kMetadataSize = 2048;
constexpr int64_t kSwaMaskMode = 4;
constexpr int64_t kNonCausalMaskMode = 0;
constexpr int64_t kNoCompression = 0;
constexpr int64_t kNumKvHeads = 1;
constexpr int64_t kNoTopk = 0;
constexpr int64_t kCmpCausalMaskMode = 3;
constexpr int64_t kWinRight = 0;
constexpr int64_t kTopkValueMode = 1;
constexpr bool kHasOriKv = true;
constexpr bool kReturnSoftmaxLse = false;

}  // namespace

torch::Tensor sparse_flash_mla_metadata(
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& seqused_kv,
    const std::optional<torch::Tensor>& seqused_cmp_kv,
    const std::optional<torch::Tensor>& cmp_residual_kv,
    int64_t num_heads_q,
    int64_t head_dim,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    int64_t max_seqlen_cmp_kv,
    int64_t cmp_topk,
    int64_t cmp_ratio,
    int64_t window_size,
    const std::optional<torch::Tensor>& ori_topk_length,
    int64_t ori_topk) {
  auto output = cu_seqlens_q.new_empty({kMetadataSize});
  auto empty = cu_seqlens_q.new_empty({0});
  const int64_t batch_size = seqused_kv.numel();
  const int64_t win_left = window_size - 1;
  const bool has_cmp_kv = cmp_ratio > 0;
  const int64_t cmp_mask_mode = has_cmp_kv ? kCmpCausalMaskMode : 0;
  char layout_q[] = "TND";
  char layout_kv[] = "PA_BBND";
  const auto cmp_lengths = seqused_cmp_kv.value_or(empty);
  const auto residual = cmp_residual_kv.value_or(empty);
  const auto ori_lengths = ori_topk_length.value_or(empty);
  const int64_t ori_mask_mode =
      ori_topk > 0 ? kNonCausalMaskMode : kSwaMaskMode;
  EXEC_NPU_CMD(aclnnSparseFlashMlaMetadata,
               cu_seqlens_q,
               /*cu_seqlens_ori_kv=*/empty,
               /*cu_seqlens_cmp_kv=*/empty,
               /*seqused_q=*/empty,
               seqused_kv,
               cmp_lengths,
               residual,
               ori_lengths,
               /*cmp_topk_length=*/empty,
               num_heads_q,
               kNumKvHeads,
               head_dim,
               batch_size,
               max_seqlen_q,
               max_seqlen_kv,
               max_seqlen_cmp_kv,
               ori_topk,
               cmp_topk,
               cmp_ratio,
               ori_mask_mode,
               cmp_mask_mode,
               win_left,
               kWinRight,
               layout_q,
               layout_kv,
               kHasOriKv,
               has_cmp_kv,
               output);
  return output;
}

torch::Tensor sparse_flash_mla(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& block_table,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& seqused_kv,
    const std::optional<torch::Tensor>& cmp_kv,
    const std::optional<torch::Tensor>& cmp_block_table,
    const std::optional<torch::Tensor>& cmp_indices,
    const std::optional<torch::Tensor>& seqused_cmp_kv,
    const std::optional<torch::Tensor>& cmp_residual_kv,
    const std::optional<torch::Tensor>& sinks,
    const torch::Tensor& metadata,
    double softmax_scale,
    int64_t cmp_ratio,
    int64_t window_size,
    const std::optional<torch::Tensor>& ori_sparse_indices,
    const std::optional<torch::Tensor>& ori_topk_length) {
  auto output = torch::empty_like(q);
  auto softmax_lse = torch::empty({0}, q.options().dtype(torch::kFloat32));
  const int64_t win_left = window_size - 1;
  const bool has_cmp_kv = cmp_ratio > 0;
  const int64_t cmp_mask_mode = has_cmp_kv ? kCmpCausalMaskMode : 0;
  const std::optional<torch::Tensor> absent;
  const int64_t ori_mask_mode =
      ori_sparse_indices.has_value() ? kNonCausalMaskMode : kSwaMaskMode;
  char layout_q[] = "TND";
  char layout_kv[] = "PA_BBND";
  EXEC_NPU_CMD(aclnnSparseFlashMla,
               q,
               kv,
               cmp_kv,
               ori_sparse_indices,
               cmp_indices,
               block_table,
               cmp_block_table,
               cu_seqlens_q,
               /*cu_seqlens_ori_kv=*/absent,
               /*cu_seqlens_cmp_kv=*/absent,
               /*seqused_q=*/absent,
               seqused_kv,
               seqused_cmp_kv,
               cmp_residual_kv,
               ori_topk_length,
               /*cmp_topk_length=*/absent,
               sinks,
               metadata,
               softmax_scale,
               cmp_ratio,
               ori_mask_mode,
               cmp_mask_mode,
               win_left,
               kWinRight,
               layout_q,
               layout_kv,
               kTopkValueMode,
               kReturnSoftmaxLse,
               output,
               softmax_lse);
  return output;
}

torch::Tensor sparse_flash_mla_swa_metadata(const torch::Tensor& cu_seqlens_q,
                                            const torch::Tensor& seqused_kv,
                                            int64_t num_heads_q,
                                            int64_t head_dim,
                                            int64_t max_seqlen_q,
                                            int64_t max_seqlen_kv,
                                            int64_t window_size) {
  return sparse_flash_mla_metadata(cu_seqlens_q,
                                   seqused_kv,
                                   /*seqused_cmp_kv=*/std::nullopt,
                                   /*cmp_residual_kv=*/std::nullopt,
                                   num_heads_q,
                                   head_dim,
                                   max_seqlen_q,
                                   max_seqlen_kv,
                                   /*max_seqlen_cmp_kv=*/0,
                                   kNoTopk,
                                   kNoCompression,
                                   window_size);
}

torch::Tensor sparse_flash_mla_swa(const torch::Tensor& q,
                                   const torch::Tensor& kv,
                                   const torch::Tensor& block_table,
                                   const torch::Tensor& cu_seqlens_q,
                                   const torch::Tensor& seqused_kv,
                                   const std::optional<torch::Tensor>& sinks,
                                   const torch::Tensor& metadata,
                                   double softmax_scale,
                                   int64_t window_size) {
  return sparse_flash_mla(q,
                          kv,
                          block_table,
                          cu_seqlens_q,
                          seqused_kv,
                          /*cmp_kv=*/std::nullopt,
                          /*cmp_block_table=*/std::nullopt,
                          /*cmp_indices=*/std::nullopt,
                          /*seqused_cmp_kv=*/std::nullopt,
                          /*cmp_residual_kv=*/std::nullopt,
                          sinks,
                          metadata,
                          softmax_scale,
                          kNoCompression,
                          window_size);
}

}  // namespace xllm::kernel::npu
