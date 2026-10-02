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

#include <tuple>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
npu_mega_gdn_decode(const torch::Tensor& qkv,
                    const torch::Tensor& z,
                    const torch::Tensor& b,
                    const torch::Tensor& a,
                    const torch::Tensor& conv_weight,
                    torch::Tensor& conv_state,
                    const torch::Tensor& a_log,
                    const torch::Tensor& dt_bias,
                    torch::Tensor& ssm_state,
                    const torch::Tensor& read_state_indices,
                    const torch::Tensor& write_state_indices,
                    const torch::Tensor& norm_weight,
                    bool fla_ssm_state_layout) {
  torch::Tensor conv_out = torch::empty_like(qkv);
  torch::Tensor out = torch::empty_like(z);
  torch::Tensor& conv_state_out = conv_state;
  torch::Tensor& ssm_state_out = ssm_state;

  EXEC_NPU_CMD(aclnnMegaGdnDecode,
               qkv,
               z,
               b,
               a,
               conv_weight,
               conv_state,
               a_log,
               dt_bias,
               ssm_state,
               read_state_indices,
               write_state_indices,
               norm_weight,
               fla_ssm_state_layout,
               conv_out,
               conv_state_out,
               ssm_state_out,
               out);

  return {conv_out, conv_state_out, ssm_state_out, out};
}

}  // namespace xllm::kernel::npu
