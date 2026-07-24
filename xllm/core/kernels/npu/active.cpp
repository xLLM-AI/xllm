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

#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace xllm::kernel::npu {

torch::Tensor active(const torch::Tensor& input, const std::string& act_mode) {
  if (act_mode == "gelu" || act_mode == "gelu_pytorch_tanh") {
    const auto approximate = act_mode == "gelu_pytorch_tanh" ? "tanh" : "none";
    return at_npu::native::custom_ops::npu_gelu(input, approximate);
  }
  if (act_mode == "silu" || act_mode == "swiglu") {
    return at_npu::native::custom_ops::npu_swiglu(input);
  }
  LOG(FATAL) << "Unsupported NPU activation: " << act_mode;
}
}  // namespace xllm::kernel::npu
