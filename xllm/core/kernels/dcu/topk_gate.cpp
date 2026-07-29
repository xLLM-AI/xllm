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

#include <glog/logging.h>

#include <cstdint>
#include <string>

#include "kernels/cuda/cuda_ops_api.h"
#include "kernels/dcu/dcu_ops_api.h"

namespace aiter {
namespace native {

void grouped_topk(torch::Tensor& gating_output,
                  torch::Tensor& topk_weights,
                  torch::Tensor& topk_ids,
                  int num_expert_group,
                  int topk_group,
                  bool need_renorm,
                  bool is_softmax,
                  float routed_scaling_factor);

void biased_grouped_topk(torch::Tensor& gating_output,
                         torch::Tensor& correction_bias,
                         torch::Tensor& topk_weights,
                         torch::Tensor& topk_ids,
                         int num_expert_group,
                         int topk_group,
                         bool need_renorm,
                         float routed_scaling_factor);

}  // namespace native
}  // namespace aiter

namespace xllm {
namespace kernel {
namespace dcu {

namespace {

bool is_supported_scoring_func(const std::string& scoring_func) {
  return scoring_func == "softmax" || scoring_func == "sigmoid";
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> moe_grouped_topk(
    torch::Tensor& gating_output,
    int64_t topk,
    int64_t num_expert_group,
    int64_t topk_group,
    bool renormalize,
    const std::optional<torch::Tensor>& correction_bias,
    const std::string& scoring_func,
    double routed_scaling_factor) {
  CHECK(gating_output.defined()) << "dcu::moe_grouped_topk: input is undefined";
  CHECK_EQ(gating_output.dim(), 2)
      << "dcu::moe_grouped_topk: input must be [num_tokens, num_experts]";
  CHECK_GT(topk, 0) << "dcu::moe_grouped_topk: topk must be positive";
  CHECK_GT(num_expert_group, 1)
      << "dcu::moe_grouped_topk requires num_expert_group > 1";
  CHECK_GT(topk_group, 0)
      << "dcu::moe_grouped_topk: topk_group must be positive";
  CHECK_LE(topk_group, num_expert_group)
      << "dcu::moe_grouped_topk: topk_group must not exceed "
         "num_expert_group";
  CHECK(is_supported_scoring_func(scoring_func))
      << "dcu::moe_grouped_topk: unsupported scoring function " << scoring_func;

  const int64_t num_tokens = gating_output.size(0);
  torch::Tensor topk_weights = torch::empty(
      {num_tokens, topk},
      torch::dtype(torch::kFloat32).device(gating_output.device()));
  torch::Tensor topk_ids =
      torch::empty({num_tokens, topk},
                   torch::dtype(torch::kInt32).device(gating_output.device()));

  const int32_t num_expert_group_i32 = static_cast<int32_t>(num_expert_group);
  const int32_t topk_group_i32 = static_cast<int32_t>(topk_group);
  const float routed_scaling_factor_f32 =
      static_cast<float>(routed_scaling_factor);

  if (correction_bias.has_value()) {
    CHECK_EQ(scoring_func, "sigmoid")
        << "dcu::moe_grouped_topk: correction bias is supported only for "
           "sigmoid scoring";
    torch::Tensor bias = correction_bias.value();
    // AITER grouped topk requires correction_bias to match gating_output dtype.
    if (bias.scalar_type() != gating_output.scalar_type()) {
      bias = bias.to(gating_output.scalar_type());
    }
    aiter::native::biased_grouped_topk(gating_output,
                                       bias,
                                       topk_weights,
                                       topk_ids,
                                       num_expert_group_i32,
                                       topk_group_i32,
                                       renormalize,
                                       routed_scaling_factor_f32);
    return std::make_tuple(topk_weights, topk_ids);
  }

  const bool is_softmax = scoring_func == "softmax";
  aiter::native::grouped_topk(gating_output,
                              topk_weights,
                              topk_ids,
                              num_expert_group_i32,
                              topk_group_i32,
                              renormalize,
                              is_softmax,
                              routed_scaling_factor_f32);
  return std::make_tuple(topk_weights, topk_ids);
}

std::tuple<torch::Tensor, torch::Tensor> moe_active_topk(
    torch::Tensor& gating_output,
    int64_t topk,
    int64_t num_expert_group,
    int64_t topk_group,
    bool renormalize,
    const std::optional<torch::Tensor>& correction_bias,
    const std::string& scoring_func,
    double routed_scaling_factor) {
  if (num_expert_group > 1) {
    return moe_grouped_topk(gating_output,
                            topk,
                            num_expert_group,
                            topk_group,
                            renormalize,
                            correction_bias,
                            scoring_func,
                            routed_scaling_factor);
  }
  return cuda::moe_fused_topk(
      gating_output, topk, renormalize, correction_bias, scoring_func);
}

}  // namespace dcu
}  // namespace kernel
}  // namespace xllm
