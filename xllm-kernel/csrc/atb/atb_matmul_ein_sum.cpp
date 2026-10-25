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

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>
#include <torch/types.h>

#include "custom_functions_npu/atb_common.h"

namespace xllm::kernel::npu {
namespace {

void check_operands(const torch::Tensor& input, const torch::Tensor& weight) {
  CHECK_EQ(input.device().type(), c10::DeviceType::PrivateUse1)
      << "ATB EIN_SUM requires NPU input";
  CHECK_EQ(input.device(), weight.device())
      << "ATB EIN_SUM input and weight must use the same NPU";
  CHECK_EQ(input.scalar_type(), torch::kBFloat16)
      << "ATB EIN_SUM requires bf16 input";
  CHECK_EQ(weight.scalar_type(), input.scalar_type())
      << "ATB EIN_SUM input and weight dtypes must match";
  CHECK_EQ(input.dim(), 3) << "ATB EIN_SUM input must be [T,H,D]";
  CHECK_EQ(weight.dim(), 3) << "ATB EIN_SUM weight must be [H,D,O]";
  CHECK_GT(input.size(0), 0) << "ATB EIN_SUM needs at least one token";
  CHECK_GT(input.size(1), 0) << "ATB EIN_SUM needs at least one head";
  CHECK_GT(input.size(2), 0) << "ATB EIN_SUM reduction dimension is empty";
  CHECK_EQ(input.size(1), weight.size(0)) << "ATB EIN_SUM head mismatch";
  CHECK_EQ(input.size(2), weight.size(1)) << "ATB EIN_SUM reduction mismatch";
  CHECK_GT(weight.size(2), 0) << "ATB EIN_SUM output dimension is empty";
  CHECK(weight.is_contiguous()) << "ATB EIN_SUM weight must be contiguous";
  CHECK_EQ(atb::utils::get_format_for_atb(weight), ACL_FORMAT_ND)
      << "ATB EIN_SUM weight must have ND format";
}

void check_output(const torch::Tensor& input,
                  const torch::Tensor& weight,
                  const torch::Tensor& output) {
  CHECK_EQ(output.device(), input.device())
      << "ATB EIN_SUM output must use the input NPU";
  CHECK_EQ(output.scalar_type(), input.scalar_type())
      << "ATB EIN_SUM output dtype must match input";
  CHECK_EQ(output.dim(), 3) << "ATB EIN_SUM output must be [T,H,O]";
  CHECK_EQ(output.size(0), input.size(0)) << "ATB EIN_SUM token mismatch";
  CHECK_EQ(output.size(1), input.size(1)) << "ATB EIN_SUM head mismatch";
  CHECK_EQ(output.size(2), weight.size(2))
      << "ATB EIN_SUM output dimension mismatch";
  CHECK(output.is_contiguous()) << "ATB EIN_SUM output must be contiguous";
  CHECK_EQ(atb::utils::get_format_for_atb(output), ACL_FORMAT_ND)
      << "ATB EIN_SUM output must have ND format";
}

void run_atb_matmul_ein_sum(const torch::Tensor& input,
                            const torch::Tensor& weight,
                            torch::Tensor& output) {
  torch::Tensor contiguous_input = input.contiguous();
  CHECK_EQ(atb::utils::get_format_for_atb(contiguous_input), ACL_FORMAT_ND)
      << "ATB EIN_SUM input must have ND format";

  atb::infer::LinearParam param;
  param.transposeA = false;
  param.transposeB = false;
  param.hasBias = false;
  param.enAccum = false;
  param.outDataType = ACL_DT_UNDEFINED;
  param.matmulType = atb::infer::LinearParam::MATMUL_EIN_SUM;

  auto& cache = atb::OpParamCache<atb::infer::LinearParam>::getInstance();
  atb::Operation* operation = cache.get_operation(param, "LinearOperation");
  atb::ParamSetter setter;
  setter.Input(contiguous_input).Input(weight).Output(output);
  atb::run_atb_cmd(operation, setter, "LinearOperation");
}

}  // namespace

torch::Tensor atb_matmul_ein_sum(const torch::Tensor& input,
                                 const torch::Tensor& weight) {
  check_operands(input, weight);
  const c10::DeviceGuard device_guard(input.device());
  torch::Tensor output = torch::empty(
      {input.size(0), input.size(1), weight.size(2)}, input.options());
  CHECK_EQ(atb::utils::get_format_for_atb(output), ACL_FORMAT_ND)
      << "ATB EIN_SUM output must have ND format";
  run_atb_matmul_ein_sum(input, weight, output);
  return output;
}

torch::Tensor atb_matmul_ein_sum_out(const torch::Tensor& input,
                                     const torch::Tensor& weight,
                                     torch::Tensor& output) {
  check_operands(input, weight);
  check_output(input, weight, output);
  const c10::DeviceGuard device_guard(input.device());
  run_atb_matmul_ein_sum(input, weight, output);
  return output;
}

}  // namespace xllm::kernel::npu
