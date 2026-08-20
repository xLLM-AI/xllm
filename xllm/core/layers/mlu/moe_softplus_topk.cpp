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

#include "layers/mlu/moe_softplus_topk.h"

#include <tuple>

#include "kernels/mlu/mlu_ops_api.h"
namespace xllm {
namespace layer {

MOESoftPlusTopKImpl::MOESoftPlusTopKImpl(int64_t n_routed_experts,
                                         int64_t n_activated_experts,
                                         float route_scale,
                                         int64_t vocab_size,
                                         bool use_hash,
                                         const torch::TensorOptions& options)
    : n_routed_experts_(n_routed_experts),
      topk_(n_activated_experts),
      route_scale_(route_scale),
      vocab_size_(vocab_size),
      use_hash_(use_hash) {
  if (use_hash_) {
    tid2eid_ = register_parameter(
        "tid2eid",
        torch::empty({vocab_size_, topk_}, options.dtype(torch::kInt32)),
        /*requires_grad=*/false);
    tid2eid_opt_ = tid2eid_;
  } else {
    bias_ = register_parameter(
        "bias",
        torch::empty({n_routed_experts_}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    bias_opt_ = bias_;
  }
}

std::tuple<torch::Tensor, torch::Tensor> MOESoftPlusTopKImpl::forward(
    const torch::Tensor& scores,
    const std::optional<torch::Tensor>& input_ids) {
  auto [weights, indices] = xllm::kernel::mlu::moe_softplus_topk(
      scores, topk_, input_ids, tid2eid_opt_, bias_opt_, route_scale_);
  return {weights, indices};
}

void MOESoftPlusTopKImpl::load_state_dict(const StateDict& state_dict) {
  if (use_hash_) {
    LOAD_WEIGHT(tid2eid);
    tid2eid_opt_ = tid2eid_;
  } else {
    LOAD_WEIGHT(bias);
    bias_opt_ = bias_;
  }
}

}  // namespace layer
}  // namespace xllm
