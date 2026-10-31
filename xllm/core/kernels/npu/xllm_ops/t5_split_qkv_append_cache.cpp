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

#include <tuple>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
t5_split_qkv_append_cache(const torch::Tensor& qkv,
                          const torch::Tensor& past_key,
                          const torch::Tensor& past_value,
                          int64_t head_num,
                          int64_t head_dim) {
  CHECK_EQ(qkv.dim(), 3) << "qkv must be [batch, 1, 3 * head_num * head_dim].";
  CHECK_EQ(qkv.size(1), 1) << "qkv carries the current token only.";
  CHECK_EQ(past_key.dim(), 4)
      << "past_key must be [batch, num_heads, cache_len, head_dim].";
  CHECK_EQ(past_value.sizes(), past_key.sizes())
      << "past_value shape must match past_key.";
  CHECK_GT(head_num, 0) << "head_num must be positive.";
  CHECK_GT(head_dim, 0) << "head_dim must be positive.";
  CHECK_EQ(past_key.size(0), qkv.size(0)) << "batch size mismatch.";
  CHECK_EQ(past_key.size(1), head_num) << "KV head count mismatch.";
  CHECK_EQ(past_key.size(3), head_dim) << "KV head dimension mismatch.";
  CHECK_EQ(qkv.size(2), 3 * head_num * head_dim)
      << "qkv last dimension must be three packed head blocks.";

  CHECK(qkv.is_contiguous()) << "qkv must be contiguous.";
  CHECK(past_key.is_contiguous()) << "past_key must be contiguous.";
  CHECK(past_value.is_contiguous()) << "past_value must be contiguous.";
  CHECK(qkv.scalar_type() == torch::kFloat16 ||
        qkv.scalar_type() == torch::kBFloat16)
      << "t5_split_qkv_append_cache supports FP16 and BF16.";
  CHECK_EQ(qkv.scalar_type(), past_key.scalar_type());
  CHECK_EQ(qkv.scalar_type(), past_value.scalar_type());
  CHECK_EQ(qkv.device(), past_key.device());
  CHECK_EQ(qkv.device(), past_value.device());
  CHECK(qkv.device().is_privateuseone())
      << "t5_split_qkv_append_cache inputs must be NPU tensors.";

  const int64_t batch = past_key.size(0);
  const int64_t cache_len = past_key.size(2);
  torch::Tensor query = torch::empty({batch, head_num, 1, head_dim},
                                     qkv.options());
  torch::Tensor key =
      torch::empty({batch, head_num, cache_len + 1, head_dim}, qkv.options());
  torch::Tensor value =
      torch::empty({batch, head_num, cache_len + 1, head_dim}, qkv.options());

  // The kernel copies bit by bit, so an empty cache is handled by the same path
  // as a populated one and needs no special case here.
  //
  // EXEC_NPU_CMD resolves aclnnT5SplitQkvAppendCache through dlsym, so this
  // file compiles and the library loads even when the pinned third_party/xllm_ops
  // revision predates the operator; the call then fails the CHECK inside the
  // macro with an explicit "not in libcust_opapi" message. Building a working
  // binary therefore requires an xllm_ops revision that ships
  // xllm_ops/t5_split_qkv_append_cache.
  EXEC_NPU_CMD(aclnnT5SplitQkvAppendCache,
               qkv,
               past_key,
               past_value,
               head_num,
               head_dim,
               query,
               key,
               value);
  return std::make_tuple(query, key, value);
}

}  // namespace xllm::kernel::npu
