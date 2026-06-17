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
#if defined(USE_DCU)
#include "kernels/dcu/dcu_ops_api.h"
#else
#include "kernels/cuda/cuda_ops_api.h"
#endif
#include "moe_topk_sigmoid_kernels.cuh"
#include "moe_topk_softmax_kernels.cuh"

namespace xllm::kernel::cuda {

std::tuple<torch::Tensor, torch::Tensor> moe_fused_topk(
    torch::Tensor& gating_output,
    int64_t topk,
    bool renormalize,
    const std::optional<torch::Tensor>& correction_bias,
    const std::string& scoring_func) {
  int64_t num_tokens = gating_output.size(0);

  torch::Tensor topk_weights = torch::empty(
      {num_tokens, topk},
      torch::dtype(torch::kFloat32).device(gating_output.device()));
  torch::Tensor topk_ids =
      torch::empty({num_tokens, topk},
                   torch::dtype(torch::kInt32).device(gating_output.device()));

  if (scoring_func == "softmax") {
    std::optional<torch::Tensor> none_correction_bias = std::nullopt;
    topk_softmax(topk_weights,
                 topk_ids,
                 gating_output,
                 renormalize,
                 /*moe_softcapping=*/0.0,
                 none_correction_bias);
  } else if (scoring_func == "sigmoid") {
    topk_sigmoid(
        topk_weights, topk_ids, gating_output, renormalize, correction_bias);
  } else {
    LOG(FATAL) << "Unsupported scoring function for moe topk: " << scoring_func
               << "only softmax and sigmoid are supported";
  }

  return std::make_tuple(topk_weights, topk_ids);
}

}  // namespace xllm::kernel::cuda
