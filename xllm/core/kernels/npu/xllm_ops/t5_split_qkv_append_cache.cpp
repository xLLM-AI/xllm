/* Copyright 2026 The xLLM Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/torch.h>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

namespace xllm::kernel::npu {
namespace {

void check_t5_split_qkv_append_cache_inputs(const torch::Tensor& qkv,
                                            const torch::Tensor& past_key,
                                            const torch::Tensor& past_value,
                                            int64_t head_num,
                                            int64_t head_dim) {
  CHECK_EQ(qkv.dim(), 3) << "qkv must have shape [B, 1, 3*H*D].";
  CHECK(past_key.dim() == 4 && past_value.sizes() == past_key.sizes())
      << "pastKey and pastValue must have matching [B, H, L, D] shapes.";
  const int64_t batch = past_key.size(0);
  const int64_t heads = past_key.size(1);
  const int64_t length = past_key.size(2);
  const int64_t dim = past_key.size(3);
  CHECK(batch >= 1 && batch <= 64 && heads >= 1 && heads <= 128 &&
        length >= 0 && length <= 8192 && dim >= 1 && dim <= 256)
      << "input dimensions are outside the supported task range.";
  CHECK(head_num == heads && head_dim == dim && qkv.size(0) == batch &&
        qkv.size(1) == 1 && qkv.size(2) == 3 * heads * dim)
      << "qkv dimensions and head attributes must match the past cache.";
  CHECK_LE(static_cast<uint64_t>(batch) * heads * dim *
               (2 * static_cast<uint64_t>(length) + 3),
           (1ULL << 28))
      << "output size exceeds the supported limit.";
  CHECK(qkv.scalar_type() == torch::kFloat16 ||
        qkv.scalar_type() == torch::kBFloat16)
      << "qkv must use float16 or bfloat16.";
  CHECK(qkv.scalar_type() == past_key.scalar_type() &&
        qkv.scalar_type() == past_value.scalar_type())
      << "qkv and past cache dtypes must match.";
  CHECK(qkv.device() == past_key.device() &&
        qkv.device() == past_value.device() && qkv.device().is_privateuseone())
      << "all inputs must be on the same NPU device.";
  CHECK(qkv.is_contiguous() && past_key.is_contiguous() &&
        past_value.is_contiguous())
      << "all inputs must be contiguous.";
}

}  // namespace

bool has_t5_split_qkv_append_cache() {
  static const bool available =
      aclnn::detail::get_op_api_func_addr(
          "aclnnT5SplitQkvAppendCacheGetWorkspaceSize") != nullptr &&
      aclnn::detail::get_op_api_func_addr("aclnnT5SplitQkvAppendCache") !=
          nullptr;
  return available;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
t5_split_qkv_append_cache(const torch::Tensor& qkv,
                          const torch::Tensor& past_key,
                          const torch::Tensor& past_value,
                          int64_t head_num,
                          int64_t head_dim) {
  check_t5_split_qkv_append_cache_inputs(
      qkv, past_key, past_value, head_num, head_dim);
  CHECK(has_t5_split_qkv_append_cache())
      << "T5SplitQkvAppendCache requires the A3 xllm_ops vendor package.";
  const c10_npu::NPUGuard device_guard(qkv.device());
  const int64_t batch = past_key.size(0);
  const int64_t heads = past_key.size(1);
  const int64_t length = past_key.size(2);
  const int64_t dim = past_key.size(3);
  auto query = torch::empty({batch, heads, 1, dim}, qkv.options());
  auto key = torch::empty({batch, heads, length + 1, dim}, qkv.options());
  auto value = torch::empty({batch, heads, length + 1, dim}, qkv.options());
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
