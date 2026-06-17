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

#include <torch/library.h>

#include <string>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
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

}  // namespace xllm::kernel::npu
