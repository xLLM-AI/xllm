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

#include "dcu_ops_api.h"

namespace xllm::kernel::dcu {

// Torch fallback: DCU does not have a fused gemma_rms_norm kernel yet.
// Applies RMSNorm with a `(1 + gamma)` weight (Gemma-style) and writes
// the result into `norm_out`.
torch::Tensor gemma_rms_norm(const torch::Tensor& x,
                             const torch::Tensor& gamma,
                             double eps,
                             torch::Tensor& norm_out) {
  auto input_dtype = x.dtype();
  auto x_fp32 = x.to(torch::kFloat32);
  auto gamma_fp32 = gamma.to(torch::kFloat32);

  auto variance = torch::mean(torch::pow(x_fp32, 2), -1, true);
  auto normalized = x_fp32 * torch::rsqrt(variance + eps);
  auto output = normalized * (1.0f + gamma_fp32);
  norm_out = output.to(input_dtype);
  return norm_out;
}

}  // namespace xllm::kernel::dcu
