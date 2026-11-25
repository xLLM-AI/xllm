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

#include "laser_attention.h"

#include <torch/library.h>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/utils.h"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {
torch::Tensor laser_attention(const torch::Tensor& q_bnsd,
                              const torch::Tensor& k_bnsd,
                              const torch::Tensor& v_bnsd,
                              double scale_value,
                              int64_t head_num) {
  check_tensor(q_bnsd, "query", "laser_attention");
  check_tensor(k_bnsd, "key", "laser_attention");
  check_tensor(v_bnsd, "value", "laser_attention");
  return xllm_kernel::run_laser_attention(
      q_bnsd, k_bnsd, v_bnsd, scale_value, head_num, [](auto&... args) {
        EXEC_NPU_CMD(aclnnXllmKernelLaserAttention, args...);
      });
}

namespace {
TORCH_LIBRARY_FRAGMENT(xllm_ops, m) {
  m.def(
      "laser_attention(Tensor query, Tensor key, Tensor value, float scale) -> "
      "Tensor");
}

TORCH_LIBRARY_IMPL(xllm_ops, PrivateUse1, m) {
  m.impl("laser_attention",
         [](const torch::Tensor& query,
            const torch::Tensor& key,
            const torch::Tensor& value,
            double scale) {
           return laser_attention(query, key, value, scale, query.size(1));
         });
}
}  // namespace

}  // namespace xllm::kernel::npu
