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

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

void check_moe_gating_top_k_hash_shape_and_dtype(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& input_ids,
    const c10::optional<at::Tensor>& tid2eid,
    int64_t k) {
  TORCH_CHECK(x.dim() >= 1,
              "Input tensor x's dim num should be at least 1, actual ",
              x.dim(),
              ".");
  TORCH_CHECK(x.size(x.dim() - 1) > 0,
              "Input tensor x's last dim should be positive, actual ",
              x.size(x.dim() - 1),
              ".");
  TORCH_CHECK(k > 0, "Attribute k should be greater than 0, actual ", k, ".");
  TORCH_CHECK(k <= x.size(x.dim() - 1),
              "Attribute k should be no greater than x.shape[-1], actual k is ",
              k,
              ", x.shape[-1] is ",
              x.size(x.dim() - 1),
              ".");

  TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kHalf ||
                  x.dtype() == at::kBFloat16,
              "x should be FLOAT16, BFLOAT16, or FLOAT32.");

  if (bias.has_value()) {
    const auto& bias_tensor = bias.value();
    TORCH_CHECK(bias_tensor.dtype() == x.dtype(),
                "bias's dtype should be equal to x's dtype.");
  }
  if (input_ids.has_value()) {
    const auto& input_ids_tensor = input_ids.value();
    TORCH_CHECK(input_ids_tensor.dtype() == at::kInt ||
                    input_ids_tensor.dtype() == at::kLong,
                "input_ids should be INT32 or INT64.");
  }
  if (tid2eid.has_value()) {
    const auto& tid2eid_tensor = tid2eid.value();
    TORCH_CHECK(tid2eid_tensor.dtype() == at::kInt ||
                    tid2eid_tensor.dtype() == at::kLong,
                "tid2eid should be INT32 or INT64.");
  }
}

at::Tensor construct_moe_gating_top_k_hash_y_tensor(const at::Tensor& x,
                                                    int64_t k) {
  auto y_shape = x.sizes().vec();
  y_shape.back() = k;
  return at::empty(y_shape, x.options().dtype(x.dtype()));
}

at::Tensor construct_moe_gating_top_k_hash_expert_idx_tensor(
    const at::Tensor& y) {
  return at::empty(y.sizes(), y.options().dtype(at::kInt));
}

at::Tensor construct_moe_gating_top_k_hash_out_tensor(const at::Tensor& x) {
  return at::empty(x.sizes(), x.options().dtype(at::kFloat));
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k_hash(
    const at::Tensor& x,
    int64_t k,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& input_ids,
    const c10::optional<at::Tensor>& tid2eid,
    int64_t k_group,
    int64_t group_count,
    double routed_scaling_factor,
    double eps,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag) {
  check_moe_gating_top_k_hash_shape_and_dtype(x, bias, input_ids, tid2eid, k);
  at::Tensor y = construct_moe_gating_top_k_hash_y_tensor(x, k);
  at::Tensor expert_idx = construct_moe_gating_top_k_hash_expert_idx_tensor(y);
  at::Tensor out = construct_moe_gating_top_k_hash_out_tensor(x);

  EXEC_NPU_CMD(aclnnMoeGatingTopKHash,
               x,
               bias,
               input_ids,
               tid2eid,
               k,
               k_group,
               group_count,
               routed_scaling_factor,
               eps,
               group_select_mode,
               renorm,
               norm_type,
               out_flag,
               y,
               expert_idx,
               out);
  return std::make_tuple(y, expert_idx, out);
}

}  // namespace xllm::kernel::npu
