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

#include <torch/library.h>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

at::Tensor construct_hc_pre_inv_rms_output_tensor(const at::Tensor& x) {
  if (x.dim() == 4) {
    return at::empty({x.size(0), x.size(1), 1}, x.options().dtype(at::kFloat));
  }
  TORCH_CHECK(x.dim() == 3,
              "Input tensor x's dim num should be 3 or 4, actual ",
              x.dim(),
              ".");
  return at::empty({x.size(0), 1}, x.options().dtype(at::kFloat));
}

void check_hc_pre_inv_rms_shape_and_dtype(const at::Tensor& x, double epsilon) {
  TORCH_CHECK(x.dim() == 3 || x.dim() == 4,
              "Input tensor x's dim num should be 3 or 4, actual ",
              x.dim(),
              ".");
  for (int64_t index = 0; index < x.dim(); ++index) {
    TORCH_CHECK(x.size(index) > 0,
                "Input tensor x's shape should be positive, but x.shape[",
                index,
                "] is ",
                x.size(index),
                ".");
  }

  TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kHalf ||
                  x.dtype() == at::kBFloat16,
              "x should be FLOAT16, BFLOAT16, or FLOAT32.");
  TORCH_CHECK(epsilon > 0.0, "epsilon should be greater than 0.");
}

}  // namespace

at::Tensor hc_pre_inv_rms(const at::Tensor& x, double epsilon) {
  check_hc_pre_inv_rms_shape_and_dtype(x, epsilon);
  at::Tensor out = construct_hc_pre_inv_rms_output_tensor(x);
  EXEC_NPU_CMD(aclnnHcPreInvRms, x, epsilon, out);
  return out;
}

}  // namespace xllm::kernel::npu
