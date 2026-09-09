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

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {

void gather_split(const torch::Tensor& input,
                  const torch::Tensor& gather_index,
                  const torch::Tensor& valid_token_num,
                  const torch::Tensor& output_head,
                  const torch::Tensor& output_tail) {
  tmo::torch_api::gather_split(
      output_head, output_tail, input, gather_index, valid_token_num);
}

}  // namespace xllm::kernel::mlu
