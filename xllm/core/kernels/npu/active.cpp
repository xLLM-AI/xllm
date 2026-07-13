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

#include <glog/logging.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "core/kernels/ops_api.h"
#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace xllm::kernel::npu {
namespace {

constexpr double kSwigluOaiAlpha = 1.702;
constexpr double kSwigluOaiLimit = 7.0;

torch::Tensor swiglu_oai(const torch::Tensor& input) {
  int64_t input_dim = input.dim();
  int64_t last_dim = input.size(input_dim - 1);
  CHECK(last_dim % 2 == 0)
      << "swigluoai expects the last dimension to be even, got " << last_dim;

  int64_t split_size = last_dim / 2;
  torch::Tensor gate =
      input.narrow(input_dim - 1, 0, split_size).clamp_max(kSwigluOaiLimit);
  torch::Tensor up = input.narrow(input_dim - 1, split_size, split_size)
                         .clamp(-kSwigluOaiLimit, kSwigluOaiLimit);
  return (up + 1.0) * gate * torch::sigmoid(gate * kSwigluOaiAlpha);
}

}  // namespace

torch::Tensor active(const torch::Tensor& input, const std::string& act_mode) {
  if (act_mode == xllm::kernel::kActModeSwigluOai) {
    return swiglu_oai(input);
  }

  if (act_mode != xllm::kernel::kActModeSilu && act_mode != "swiglu") {
    LOG(FATAL) << "Only swiglu activation is supported in NPU active";
  }
  return at_npu::native::custom_ops::npu_swiglu(input);
}
}  // namespace xllm::kernel::npu
