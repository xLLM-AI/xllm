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

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {

torch::Tensor matmul(const torch::Tensor& a,
                     const torch::Tensor& b,
                     const std::optional<torch::Tensor>& bias,
                     const std::optional<torch::Tensor>& c,
                     double alpha,
                     double beta) {
  return tmo::torch_api::matmul(a,
                                b,
                                bias,
                                c,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                "none",
                                alpha,
                                beta,
                                true,
                                true,
                                1.0,
                                1.0,
                                false,
                                true);
}

torch::Tensor batch_matmul(const torch::Tensor& a,
                           const torch::Tensor& b,
                           bool trans_a,
                           bool trans_b) {
  return tmo::torch_api::batch_matmul(a,
                                      b,
                                      /*c=*/std::nullopt,
                                      /*bias=*/std::nullopt,
                                      /*dtype=*/std::nullopt,
                                      /*a_scale_tensor=*/std::nullopt,
                                      /*b_scale_tensor=*/std::nullopt,
                                      /*act_mode=*/"none",
                                      /*alpha=*/1.0,
                                      /*beta=*/0.0,
                                      /*a_scale=*/1.0,
                                      /*b_scale=*/1.0,
                                      /*trans_a=*/trans_a,
                                      /*trans_b=*/trans_b,
                                      /*use_hp_active=*/false,
                                      /*approximate=*/false);
}

}  // namespace xllm::kernel::mlu
