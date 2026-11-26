/* Copyright 2025-2026 The xLLM Authors.

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

#include <dlfcn.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cmath>
#include <optional>
#include <string>
#include <string_view>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/util/utils.h"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

void check_sparse_attn_sharedkv_shape_and_dtype(const at::Tensor& q,
                                                c10::string_view layout_q,
                                                c10::string_view layout_kv) {
  TORCH_CHECK(q.dim() >= 1,
              "Input tensor q's dim num should be at least 1, actual ",
              q.dim(),
              ".");
  TORCH_CHECK(q.dtype() == at::kHalf || q.dtype() == at::kBFloat16,
              "q should be FLOAT16 or BFLOAT16.");
  TORCH_CHECK(!layout_q.empty(), "layout_q should not be empty.");
  TORCH_CHECK(!layout_kv.empty(), "layout_kv should not be empty.");
}

at::Tensor construct_sparse_attn_sharedkv_attn_out_tensor(const at::Tensor& q) {
  return at::empty(q.sizes(), q.options().dtype(q.dtype()));
}

at::Tensor construct_sparse_attn_sharedkv_softmax_lse_tensor(
    const at::Tensor& q,
    bool return_softmax_lse) {
  if (!return_softmax_lse) {
    return at::empty({0}, q.options().dtype(at::kFloat));
  }
  auto softmax_lse_shape = q.sizes().vec();
  softmax_lse_shape.back() = 1;
  return at::empty(softmax_lse_shape, q.options().dtype(at::kFloat));
}

int64_t get_sparse_attn_sharedkv_kv_stride(
    const c10::optional<at::Tensor>& kv) {
  if (!kv.has_value() || !kv.value().defined() || kv.value().dim() == 0) {
    return 0;
  }
  return kv.value().stride(0);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> sparse_attn_sharedkv(
    const at::Tensor& q,
    const c10::optional<at::Tensor>& ori_kv,
    const c10::optional<at::Tensor>& cmp_kv,
    const c10::optional<at::Tensor>& ori_sparse_indices,
    const c10::optional<at::Tensor>& cmp_sparse_indices,
    const c10::optional<at::Tensor>& ori_block_table,
    const c10::optional<at::Tensor>& cmp_block_table,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_ori_kv,
    const c10::optional<at::Tensor>& cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_kv,
    const c10::optional<at::Tensor>& sinks,
    const c10::optional<at::Tensor>& metadata,
    double softmax_scale,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool return_softmax_lse) {
  check_sparse_attn_sharedkv_shape_and_dtype(q, layout_q, layout_kv);
  at::Tensor attn_out = construct_sparse_attn_sharedkv_attn_out_tensor(q);
  at::Tensor softmax_lse =
      construct_sparse_attn_sharedkv_softmax_lse_tensor(q, return_softmax_lse);

  std::string layout_q_str = std::string(layout_q);
  std::string layout_kv_str = std::string(layout_kv);
  auto layout_q_arg = const_cast<char*>(layout_q_str.c_str());
  auto layout_kv_arg = const_cast<char*>(layout_kv_str.c_str());
  const int64_t ori_kv_stride = get_sparse_attn_sharedkv_kv_stride(ori_kv);
  const int64_t cmp_kv_stride = get_sparse_attn_sharedkv_kv_stride(cmp_kv);

  EXEC_NPU_CMD(aclnnSparseAttnSharedkv,
               q,
               ori_kv,
               cmp_kv,
               ori_sparse_indices,
               cmp_sparse_indices,
               ori_block_table,
               cmp_block_table,
               cu_seqlens_q,
               cu_seqlens_ori_kv,
               cu_seqlens_cmp_kv,
               seqused_q,
               seqused_kv,
               sinks,
               metadata,
               softmax_scale,
               cmp_ratio,
               ori_mask_mode,
               cmp_mask_mode,
               ori_kv_stride,
               cmp_kv_stride,
               ori_win_left,
               ori_win_right,
               layout_q_arg,
               layout_kv_arg,
               return_softmax_lse,
               attn_out,
               softmax_lse);
  return std::make_tuple(attn_out, softmax_lse);
}

namespace {

constexpr int64_t kHeadDim = 512;
constexpr int64_t kSparseMaskMode = 0;
constexpr int64_t kNoCompression = 1;
constexpr int64_t kWindowOffset = 0;

}  // namespace

bool supports_sparse_flash_mla_swa() {
  const char* soc = aclrtGetSocName();
  if (soc == nullptr ||
      !std::string_view(soc).starts_with(/*x=*/"Ascend910_93")) {
    return false;
  }
  static const bool supported = [] {
    char package[] = "ops_transformer";
    char version[128] = {};
    using GetVersionString = aclError (*)(char*, char*);
    const auto get_version_string = reinterpret_cast<GetVersionString>(
        dlsym(RTLD_DEFAULT, /*symbol=*/"aclsysGetVersionStr"));
    if (get_version_string == nullptr ||
        get_version_string(package, version) != ACL_SUCCESS) {
      return false;
    }
    // Early 9.2 builds export the same symbols but reject explicit SWA top-k.
    const std::string_view package_version(version);
    if (package_version != "9.2.0" && package_version != "9.2.1") {
      return false;
    }
    using aclnn::detail::get_op_api_func_addr;
    return get_op_api_func_addr(/*api_name=*/"aclnnSparseFlashMla") &&
           get_op_api_func_addr(
               /*api_name=*/"aclnnSparseFlashMlaGetWorkspaceSize") &&
           get_op_api_func_addr(/*api_name=*/"aclnnSparseFlashMlaMetadata") &&
           get_op_api_func_addr(
               /*api_name=*/"aclnnSparseFlashMlaMetadataGetWorkspaceSize");
  }();
  return supported;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sparse_flash_mla_swa_metadata(const torch::Tensor& cu_seqlens_q,
                              const torch::Tensor& kv_seq_lens,
                              const torch::Tensor& ori_block_table,
                              int64_t max_kv_len,
                              int64_t num_heads,
                              int64_t page_size,
                              int64_t visible_window) {
  const torch::DeviceGuard guard(kv_seq_lens.device());
  constexpr int64_t kIndexAlignment = 128;
  const int64_t ori_topk = util::align_up(visible_window, kIndexAlignment);
  auto kv_ends = kv_seq_lens.unsqueeze(/*dim=*/1);
  const auto no_query =
      cu_seqlens_q.slice(/*dim=*/0, /*start=*/1) <=
      cu_seqlens_q.narrow(/*dim=*/0, /*start=*/0, kv_seq_lens.size(/*dim=*/0));
  // SWA tables right-align the live pages after any evicted prefix.
  const auto live_pages =
      ori_block_table.ge(/*other=*/0)
          .sum(
              /*dim=*/1, /*keepdim=*/true, /*dtype=*/torch::kInt32);
  const auto first_live =
      (torch::floor_divide(kv_ends + page_size - 1, page_size) - live_pages)
          .clamp_min(/*min=*/0) *
      page_size;
  auto ori_topk_length =
      (kv_ends - first_live)
          .clamp(/*min=*/0, visible_window)
          .masked_fill(no_query.unsqueeze(/*dim=*/1), /*value=*/0);
  auto offsets = torch::arange(ori_topk, kv_seq_lens.options());
  auto ori_sparse_indices = (kv_ends - ori_topk_length + offsets)
                                .masked_fill(offsets >= ori_topk_length,
                                             /*value=*/-1)
                                .unsqueeze(/*dim=*/1);

  constexpr int64_t kMetadataElements = 1024;
  auto metadata = torch::zeros({kMetadataElements}, kv_seq_lens.options());
  const std::optional<torch::Tensor> absent;
  const char* query_layout = "TND";
  const char* paged_layout = "PA_BBND";
  constexpr int64_t kNumKvHeads = 1;
  constexpr int64_t kMaxQueryLen = 1;
  constexpr int64_t kMaxCompressedKvLen = 0;
  constexpr int64_t kCompressedTopk = 0;
  constexpr bool kHasOriKv = true;
  constexpr bool kHasCmpKv = false;
  const int64_t batch_size = kv_seq_lens.numel();
  EXEC_NPU_CMD(aclnnSparseFlashMlaMetadata,
               cu_seqlens_q,
               /*cuSeqlensOriKvOptional=*/absent,
               /*cuSeqlensCmpKvOptional=*/absent,
               /*sequsedQOptional=*/absent,
               kv_seq_lens,
               /*sequsedCmpKvOptional=*/absent,
               /*cmpResidualKvOptional=*/absent,
               ori_topk_length,
               /*cmpTopkLengthOptional=*/absent,
               num_heads,
               kNumKvHeads,
               kHeadDim,
               batch_size,
               kMaxQueryLen,
               max_kv_len,
               kMaxCompressedKvLen,
               ori_topk,
               kCompressedTopk,
               kNoCompression,
               kSparseMaskMode,
               kSparseMaskMode,
               kWindowOffset,
               kWindowOffset,
               query_layout,
               paged_layout,
               kHasOriKv,
               kHasCmpKv,
               metadata);
  return {ori_sparse_indices, ori_topk_length, metadata};
}

torch::Tensor sparse_flash_mla_swa(const torch::Tensor& q,
                                   const torch::Tensor& ori_kv,
                                   const torch::Tensor& ori_block_table,
                                   const torch::Tensor& cu_seqlens_q,
                                   const torch::Tensor& kv_seq_lens,
                                   const torch::Tensor& ori_sparse_indices,
                                   const torch::Tensor& ori_topk_length,
                                   const std::optional<torch::Tensor>& sinks,
                                   const torch::Tensor& metadata) {
  const torch::DeviceGuard guard(q.device());
  const double softmax_scale = 1.0 / std::sqrt(q.size(/*dim=*/-1));
  const std::optional<torch::Tensor> absent;
  auto output = torch::empty_like(q);
  auto lse =
      torch::empty(/*size=*/{0}, q.options().dtype(/*dtype=*/torch::kFloat32));
  char query_layout[] = "TND";
  char paged_layout[] = "PA_BBND";
  constexpr int64_t kLogicalIndices = 1;
  constexpr bool kReturnSoftmaxLse = false;
  EXEC_NPU_CMD(aclnnSparseFlashMla,
               q,
               ori_kv,
               /*cmpKvOptional=*/absent,
               ori_sparse_indices,
               /*cmpSparseIndicesOptional=*/absent,
               ori_block_table,
               /*cmpBlockTableOptional=*/absent,
               cu_seqlens_q,
               /*cuSeqlensOriKvOptional=*/absent,
               /*cuSeqlensCmpKvOptional=*/absent,
               /*sequsedQOptional=*/absent,
               kv_seq_lens,
               /*sequsedCmpKvOptional=*/absent,
               /*cmpResidualKvOptional=*/absent,
               ori_topk_length,
               /*cmpTopkLengthOptional=*/absent,
               sinks,
               metadata,
               softmax_scale,
               kNoCompression,
               kSparseMaskMode,
               kSparseMaskMode,
               kWindowOffset,
               kWindowOffset,
               query_layout,
               paged_layout,
               kLogicalIndices,
               kReturnSoftmaxLse,
               output,
               lse);
  return output;
}

}  // namespace xllm::kernel::npu
