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

#include "activation.h"

#include <glog/logging.h>

#include <cmath>

#include "kernels/ops_api.h"
namespace xllm {
namespace layer {

namespace {

bool has_effective_swiglu_limit(double swiglu_limit) {
  return std::isfinite(swiglu_limit) && swiglu_limit > 0.0 &&
         swiglu_limit < 1000000.0;
}

torch::Tensor swiglu_with_clamp(const torch::Tensor& input,
                                double swiglu_limit) {
  CHECK(input.defined()) << "SwiGLU input is undefined.";
  CHECK_GT(input.dim(), 0) << "SwiGLU input must have at least one dimension.";
  const int64_t last_dim = input.size(-1);
  CHECK_EQ(last_dim % 2, 0)
      << "SwiGLU input last dimension must be even, got " << last_dim;
  const int64_t half_dim = last_dim / 2;
  // Align with DeepSeek-V4 official Expert.forward: cast to fp32 before
  // clamp/silu/mul to avoid precision loss near the swiglu_limit boundary,
  // then cast the result back to the input dtype.
  const auto in_dtype = input.scalar_type();
  auto gate = input.slice(/*dim=*/-1, /*start=*/0, /*end=*/half_dim)
                  .to(torch::kFloat32);
  auto up = input.slice(/*dim=*/-1, /*start=*/half_dim, /*end=*/last_dim)
                .to(torch::kFloat32);
  gate = torch::clamp_max(gate, swiglu_limit);
  up = torch::clamp(up, -swiglu_limit, swiglu_limit);
  auto out = torch::silu(gate) * up;
  return out.to(in_dtype);
}

}  // namespace

ActivationImpl::ActivationImpl(const std::string& act_mode,
                               bool is_gated,
                               double swiglu_limit)
    : act_mode_(act_mode), is_gated_(is_gated), swiglu_limit_(swiglu_limit) {}

void ActivationImpl::forward(torch::Tensor& input, torch::Tensor& output) {
  if (is_gated_ && (act_mode_ == "silu" || act_mode_ == "swiglu") &&
      has_effective_swiglu_limit(swiglu_limit_)) {
    output = swiglu_with_clamp(input, swiglu_limit_);
    return;
  }

  xllm::kernel::ActivationParams activation_params;
  activation_params.input = input;
  activation_params.output = output;
  activation_params.act_mode = act_mode_;
  activation_params.is_gated = is_gated_;
  xllm::kernel::active(activation_params);
  // Unified assignment: NPU returns new tensor, others modify in-place (no-op
  // assignment)
  output = activation_params.output;
}

}  // namespace layer
}  // namespace xllm
