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

void active(const torch::Tensor& input,
            torch::Tensor& output,
            const std::optional<torch::Tensor>& bias,
            const std::optional<torch::Tensor>& cusum_token_count,
            const std::string& act_mode,
            bool is_gated,
            int64_t start_expert_id,
            int64_t expert_size) {
  std::string hidden_act = act_mode;
  std::string gelu_approximate = "none";
  if (act_mode == "gelu_pytorch_tanh") {
    hidden_act = "gelu";
    gelu_approximate = "tanh";
  }
  tmo::torch_api::active(input,
                         output,
                         bias,
                         cusum_token_count,
                         hidden_act,
                         is_gated,
                         start_expert_id,
                         expert_size,
                         /*active_coef=*/1.0,
                         /*high_precision=*/false,
                         gelu_approximate);
}
}  // namespace xllm::kernel::mlu
