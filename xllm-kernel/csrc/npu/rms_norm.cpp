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

#include <torch/library.h>
#include <torch/types.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include <tuple>

namespace {

torch::Tensor rms_norm(const torch::Tensor& input,
                       const torch::Tensor& weight,
                       double eps) {
  return std::get<0>(
      at_npu::native::custom_ops::npu_rms_norm(input, weight, eps));
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("rms_norm(Tensor input, Tensor weight, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("rms_norm", &rms_norm);
}
