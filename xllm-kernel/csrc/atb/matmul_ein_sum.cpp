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

#include <ATen/MemoryOverlap.h>
#include <atb/atb_infer.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/Logging.h>
#include <torch/library.h>
#include <torch/types.h>

#include "csrc/atb/atb_operation.h"
#include "csrc/atb/utils.h"

namespace xllm_kernel {
namespace {

void run_projection(const torch::Tensor& input,
                    const torch::Tensor& weight,
                    torch::Tensor& output) {
  torch::Tensor packed = input.contiguous();
  atb::VariantPack pack;
  pack.inTensors = {to_atb_tensor(packed), to_atb_tensor(weight)};
  pack.outTensors = {to_atb_tensor(output)};
  atb::infer::LinearParam param;
  param.transposeA = false;
  param.transposeB = false;
  param.hasBias = false;
  param.enAccum = false;
  param.outDataType = ACL_DT_UNDEFINED;
  param.matmulType = atb::infer::LinearParam::MATMUL_EIN_SUM;
  run_atb_operation("XllmKernelMatmulEinSum",
                    param,
                    /*variant=*/0,
                    std::move(pack),
                    {packed, weight, output});
}

torch::Tensor matmul_ein_sum(const torch::Tensor& input,
                             const torch::Tensor& weight) {
  check_operands(input, weight);
  const torch::DeviceGuard guard(input.device());
  torch::Tensor output = torch::empty(
      {input.size(0), input.size(1), weight.size(2)}, input.options());
  run_projection(input, weight, output);
  return output;
}

torch::Tensor matmul_ein_sum_out(const torch::Tensor& input,
                                 const torch::Tensor& weight,
                                 torch::Tensor& output) {
  check_operands(input, weight);
  CHECK(output.device() == input.device());
  CHECK(output.scalar_type() == input.scalar_type());
  CHECK(output.dim() == 3);
  CHECK(output.size(0) == input.size(0));
  CHECK(output.size(1) == input.size(1));
  CHECK(output.size(2) == weight.size(2));
  CHECK(output.is_contiguous());
  check_base_format(output);
  torch::assert_no_overlap(output, input);
  torch::assert_no_overlap(output, weight);
  const torch::DeviceGuard guard(input.device());
  run_projection(input, weight, output);
  return output;
}

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("matmul_ein_sum(Tensor input, Tensor weight) -> Tensor");
  m.def(
      "matmul_ein_sum_out(Tensor input, Tensor weight, Tensor(a!) out) -> "
      "Tensor(a!)");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("matmul_ein_sum", TORCH_FN(matmul_ein_sum));
  m.impl("matmul_ein_sum_out", TORCH_FN(matmul_ein_sum_out));
}

}  // namespace
}  // namespace xllm_kernel
