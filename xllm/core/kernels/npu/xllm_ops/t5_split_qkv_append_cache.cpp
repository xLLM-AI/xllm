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

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
t5_split_qkv_append_cache(const torch::Tensor& qkv,
                          const torch::Tensor& past_key,
                          const torch::Tensor& past_value,
                          int64_t head_num,
                          int64_t head_dim) {
  TORCH_CHECK(qkv.dim() == 3, "t5_split_qkv_append_cache: qkv must be 3D");
  TORCH_CHECK(
      past_key.dim() == 4 && past_key.sizes() == past_value.sizes(),
      "t5_split_qkv_append_cache: pastKey/pastValue must share a 4D shape");
  TORCH_CHECK(
      qkv.device() == past_key.device() && qkv.device() == past_value.device(),
      "t5_split_qkv_append_cache: inputs must share a device");
  TORCH_CHECK(qkv.scalar_type() == past_key.scalar_type() &&
                  qkv.scalar_type() == past_value.scalar_type(),
              "t5_split_qkv_append_cache: inputs must share a dtype");
  TORCH_CHECK(
      qkv.scalar_type() == at::kHalf || qkv.scalar_type() == at::kBFloat16,
      "t5_split_qkv_append_cache: only fp16/bf16 are supported");
  TORCH_CHECK(qkv.is_contiguous() && past_key.is_contiguous() &&
                  past_value.is_contiguous(),
              "t5_split_qkv_append_cache: inputs must be contiguous");

  const int64_t batch = past_key.size(0);
  const int64_t heads = past_key.size(1);
  const int64_t history = past_key.size(2);
  const int64_t dim = past_key.size(3);
  TORCH_CHECK(qkv.size(0) == batch && qkv.size(1) == 1 &&
                  qkv.size(2) == 3 * heads * dim,
              "t5_split_qkv_append_cache: packed qkv shape mismatch");
  TORCH_CHECK(
      head_num == heads && head_dim == dim,
      "t5_split_qkv_append_cache: headNum/headDim must match the cache");

  torch::Tensor query = torch::empty({batch, heads, 1, dim}, qkv.options());
  torch::Tensor key =
      torch::empty({batch, heads, history + 1, dim}, qkv.options());
  torch::Tensor value = torch::empty_like(key);

  EXEC_NPU_CMD(aclnnT5SplitQkvAppendCache,
               qkv,
               past_key,
               past_value,
               head_num,
               head_dim,
               query,
               key,
               value);
  return {query, key, value};
}

}  // namespace xllm::kernel::npu
