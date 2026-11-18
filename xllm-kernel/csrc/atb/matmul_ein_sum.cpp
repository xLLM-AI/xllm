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
#include <glog/logging.h>
#include <torch/library.h>
#include <torch/types.h>
#include <torch_npu/csrc/core/npu/NPUGraphsUtils.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <memory>
#include <unordered_map>

#include "csrc/atb/atb_projection.h"
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
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
  bool capturing = static_cast<int32_t>(
                       c10_npu::currentStreamCaptureStatusMayInitCtx()) != 0;
  if (capturing) {
    // Match the existing Python bridge: setup and graph-pool allocation happen
    // on the capture thread; the operation is released after submission.
    auto projection = std::make_shared<AtbProjection>(stream);
    uint64_t workspace_size = projection->setup(pack);
    torch::Tensor workspace =
        torch::empty({static_cast<int64_t>(workspace_size)},
                     input.options().dtype(torch::kUInt8));
    auto call =
        [projection, pack, workspace, workspace_size, packed, weight, output]()
        -> int {
      projection->execute(pack, workspace.data_ptr(), workspace_size);
      return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("XllmKernelMatmulEinSum", call);
    return;
  }

  auto call = [pack, stream, packed, weight, output]() -> int {
    // ATB setup mutates the operation. Keep the eager cache local to the
    // execution thread and stream; graph captures own separate operations.
    static thread_local std::unordered_map<aclrtStream,
                                           std::unique_ptr<AtbProjection>>
        cache;
    auto& projection = cache[stream];
    if (!projection) {
      projection = std::make_unique<AtbProjection>(stream);
    }
    uint64_t workspace_size = projection->setup(pack);
    torch::Tensor workspace;
    void* workspace_ptr = nullptr;
    if (workspace_size != 0) {
      workspace = at_npu::native::allocate_workspace(workspace_size, stream);
      workspace_ptr = workspace.data_ptr();
    }
    projection->execute(pack, workspace_ptr, workspace_size);
    return 0;
  };
  at_npu::native::OpCommand::RunOpApiV2("XllmKernelMatmulEinSum", call);
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
  CHECK_EQ(output.device(), input.device());
  CHECK_EQ(output.scalar_type(), input.scalar_type());
  CHECK_EQ(output.dim(), 3);
  CHECK_EQ(output.size(0), input.size(0));
  CHECK_EQ(output.size(1), input.size(1));
  CHECK_EQ(output.size(2), weight.size(2));
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
