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

void scatter_nd_update(torch::Tensor& var,
                       const torch::Tensor& indices,
                       const torch::Tensor& updates) {
  at::IntArrayRef var_stride = var.strides();
  EXEC_NPU_CMD(aclnnScatterNdUpdateV2, var, indices, updates, var_stride);
}

}  // namespace xllm::kernel::npu
