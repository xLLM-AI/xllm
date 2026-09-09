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

#include <glog/logging.h>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/npu_ops_api.h"
#include "core/kernels/npu/utils.h"

namespace {

c10::optional<torch::Tensor> to_c10_optional_tensor(
    const std::optional<torch::Tensor>& tensor_opt) {
  if (tensor_opt.has_value() && tensor_opt.value().defined()) {
    return tensor_opt.value();
  }
  return c10::nullopt;
}

}  // namespace

namespace xllm::kernel::npu {

torch::Tensor npu_recurrent_gated_delta_rule(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& state,
    const std::optional<torch::Tensor>& beta,
    const std::optional<double> scale,
    const std::optional<torch::Tensor>& actual_seq_lengths,
    const std::optional<torch::Tensor>& ssm_state_indices,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const std::optional<torch::Tensor>& g,
    const std::optional<torch::Tensor>& gk) {
  check_tensor(query, "query", "recurrent_gated_delta_rule");
  check_tensor(key, "key", "recurrent_gated_delta_rule");
  check_tensor(value, "value", "recurrent_gated_delta_rule");
  check_tensor(state, "state", "recurrent_gated_delta_rule");
  CHECK(scale.has_value())
      << "recurrent_gated_delta_rule requires a valid scale value";

  c10::optional<torch::Tensor> beta_tensor = to_c10_optional_tensor(beta);
  c10::optional<torch::Tensor> actual_seq_lengths_tensor =
      to_c10_optional_tensor(actual_seq_lengths);
  c10::optional<torch::Tensor> ssm_state_indices_tensor =
      to_c10_optional_tensor(ssm_state_indices);
  c10::optional<torch::Tensor> num_accepted_tokens_tensor =
      to_c10_optional_tensor(num_accepted_tokens);
  c10::optional<torch::Tensor> g_tensor = to_c10_optional_tensor(g);
  c10::optional<torch::Tensor> gk_tensor = to_c10_optional_tensor(gk);
  float scale_value = static_cast<float>(scale.value());
  torch::Tensor output = torch::empty_like(value);

  EXEC_NPU_CMD(aclnnRecurrentGatedDeltaRule,
               query,
               key,
               value,
               beta_tensor,
               state,
               actual_seq_lengths_tensor,
               ssm_state_indices_tensor,
               g_tensor,
               gk_tensor,
               num_accepted_tokens_tensor,
               scale_value,
               output);
  return output;
}

}  // namespace xllm::kernel::npu
