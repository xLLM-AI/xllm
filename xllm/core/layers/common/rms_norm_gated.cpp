/* Copyright 2025-2026 The xLLM Authors.

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

#include "rms_norm_gated.h"

#if defined(USE_NPU)
#include <ATen/ops/silu.h>

#include <cstdlib>
#include <cstring>
#endif

#include <glog/logging.h>

#include "framework/state_dict/utils.h"
#if defined(USE_NPU)
#include "xllm/core/kernels/npu/npu_ops_api.h"
#endif
#include "xllm/core/kernels/ops_api.h"

namespace xllm {
namespace layer {

#if defined(USE_NPU)
namespace {

bool enable_native_split_gated_rms_norm() {
  static const bool enabled = []() {
    const char* value = std::getenv("XLLM_NATIVE_SPLIT_GATED_RMS_NORM");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

}  // namespace
#endif

RmsNormGatedImpl::RmsNormGatedImpl(int64_t dim,
                                   double eps,
                                   const torch::TensorOptions& options)
    : norm_dim_(dim), eps_(eps) {
  weight_ = register_parameter(
      "weight", torch::empty({dim}, options), /*requires_grad=*/false);
}

torch::Tensor RmsNormGatedImpl::forward(torch::Tensor& input,
                                        std::optional<torch::Tensor> gate) {
#if defined(USE_NPU)
  if (enable_native_split_gated_rms_norm() && gate.has_value() &&
      input.dim() == 2 && input.size(0) >= 1024 && input.size(1) == norm_dim_) {
    auto normalized =
        xllm::kernel::npu::rms_norm(input, weight_, eps_, "rmsnorm");
    auto gate_value = gate.value();
    at::silu_(gate_value);
    normalized.mul_(gate_value);
    return normalized;
  }
#endif
  xllm::kernel::GatedLayerNormParams params;
  params.x = input;
  params.weight = weight_;
  torch::Tensor bias;
  params.bias = bias;
  params.eps = eps_;
  if (gate.has_value()) {
    params.z = gate;
  }
  params.group_size = input.size(-1);
  params.is_rms_norm = true;
  auto ret = xllm::kernel::gated_layer_norm(params);
  return ret;
}

void RmsNormGatedImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(weight);
}

}  // namespace layer
}  // namespace xllm
