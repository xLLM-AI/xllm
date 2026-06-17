/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <unordered_map>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

void npu_inplace_partial_rotary_mul(torch::Tensor& x,
                                    const torch::Tensor& r1,
                                    const torch::Tensor& r2,
                                    c10::string_view rotary_mode,
                                    at::IntArrayRef partial_slice) {
  static const std::unordered_map<std::string, int64_t> mode_map = {
      {"half", 0},
      {"interleave", 1},
      {"quarter", 2},
      {"interleave-half", 3},
  };
  const std::string rotary_mode_str(rotary_mode);
  auto it = mode_map.find(rotary_mode_str);
  TORCH_CHECK(it != mode_map.end(),
              "Unsupported rotary_mode=",
              rotary_mode_str,
              ". Supported modes: half/interleave/quarter/interleave-half");
  TORCH_CHECK(x.dim() == 4,
              "npu_inplace_partial_rotary_mul expects x dim=4, got ",
              x.dim());
  EXEC_NPU_CMD(
      aclnnInplacePartialRotaryMul, x, r1, r2, it->second, partial_slice);
}

}  // namespace xllm::kernel::npu
