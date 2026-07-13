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

#include <optional>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

void check_moe_gating_top_k_shape_and_dtype(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& bias,
    int64_t k) {
  CHECK_EQ(x.dim(), 2) << "Input tensor x's dim num should be 2, actual "
                       << x.dim() << ".";
  CHECK_GT(x.size(1), 0)
      << "Input tensor x's expert dim should be positive, actual " << x.size(1)
      << ".";
  CHECK_GT(k, 0) << "Attribute k should be greater than 0, actual " << k << ".";
  CHECK_LE(k, x.size(1))
      << "Attribute k should be no greater than x.shape[-1], actual k is " << k
      << ", x.shape[-1] is " << x.size(1) << ".";
  CHECK(x.dtype() == torch::kFloat || x.dtype() == torch::kHalf ||
        x.dtype() == torch::kBFloat16)
      << "x should be FLOAT16, BFLOAT16, or FLOAT32.";

  if (bias.has_value()) {
    const torch::Tensor& bias_tensor = bias.value();
    CHECK_EQ(bias_tensor.dtype(), x.dtype())
        << "bias's dtype should be equal to x's dtype.";
    CHECK_EQ(bias_tensor.dim(), 1)
        << "bias's dim num should be 1, actual " << bias_tensor.dim() << ".";
    CHECK_EQ(bias_tensor.size(0), x.size(1))
        << "bias's first dim should be equal to x's expert dim.";
  }
}

std::optional<torch::Tensor> defined_tensor_or_nullopt(
    const std::optional<torch::Tensor>& tensor) {
  if (tensor.has_value() && tensor->defined()) {
    return tensor.value();
  }
  return std::nullopt;
}

torch::Tensor construct_moe_gating_top_k_y_tensor(const torch::Tensor& x,
                                                  int64_t k) {
  return torch::empty({x.size(0), k}, x.options().dtype(x.dtype()));
}

torch::Tensor construct_moe_gating_top_k_expert_idx_tensor(
    const torch::Tensor& y) {
  return torch::empty(y.sizes(), y.options().dtype(torch::kInt));
}

torch::Tensor construct_moe_gating_top_k_out_tensor(const torch::Tensor& x) {
  return torch::empty(x.sizes(), x.options().dtype(torch::kFloat));
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_gating_top_k(
    const torch::Tensor& x,
    int64_t k,
    const std::optional<torch::Tensor>& bias,
    int64_t k_group,
    int64_t group_count,
    double routed_scaling_factor,
    double eps,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag) {
  const std::optional<torch::Tensor> bias_opt = defined_tensor_or_nullopt(bias);
  check_moe_gating_top_k_shape_and_dtype(x, bias_opt, k);
  torch::Tensor y = construct_moe_gating_top_k_y_tensor(x, k);
  torch::Tensor expert_idx = construct_moe_gating_top_k_expert_idx_tensor(y);
  torch::Tensor out = construct_moe_gating_top_k_out_tensor(x);
  const c10::optional<torch::Tensor> bias_for_aclnn =
      bias_opt.has_value() ? c10::optional<torch::Tensor>(bias_opt.value())
                           : c10::nullopt;

  EXEC_NPU_CMD(aclnnMoeGatingTopK,
               x,
               bias_for_aclnn,
               k,
               k_group,
               group_count,
               group_select_mode,
               renorm,
               norm_type,
               out_flag,
               routed_scaling_factor,
               eps,
               y,
               expert_idx,
               out);
  return std::make_tuple(y, expert_idx, out);
}

}  // namespace xllm::kernel::npu
