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

#include <glog/logging.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "core/framework/parallel_state/process_group.h"
#include "npu_ops_api.h"

namespace xllm::kernel::npu {
namespace {

bool can_call_torch_npu_mmrs(const torch::Tensor& a,
                             const torch::Tensor& b,
                             const std::optional<torch::Tensor>& bias,
                             ProcessGroup* process_group) {
  if (process_group == nullptr) {
    LOG_FIRST_N(WARNING, 8)
        << "FC1 MMRS torch_npu skipped: process_group is null.";
    return false;
  }
  if (!a.defined() || !b.defined()) {
    LOG_FIRST_N(WARNING, 8)
        << "FC1 MMRS torch_npu skipped: input tensor is missing. a_defined="
        << a.defined() << ", b_defined=" << b.defined();
    return false;
  }
  if (a.dim() != 2 || b.dim() != 2) {
    LOG_FIRST_N(WARNING, 8)
        << "FC1 MMRS torch_npu skipped: expected 2D tensors, got a_dim="
        << a.dim() << ", b_dim=" << b.dim();
    return false;
  }
  if (a.scalar_type() != at::kHalf && a.scalar_type() != at::kBFloat16) {
    LOG_FIRST_N(WARNING, 8)
        << "FC1 MMRS torch_npu skipped: unsupported input dtype="
        << a.scalar_type();
    return false;
  }
  if (a.scalar_type() != b.scalar_type()) {
    LOG_FIRST_N(WARNING, 8)
        << "FC1 MMRS torch_npu skipped: dtype mismatch. a=" << a.scalar_type()
        << ", b=" << b.scalar_type();
    return false;
  }
  if (bias.has_value() && bias->defined() &&
      bias->scalar_type() != a.scalar_type()) {
    LOG_FIRST_N(WARNING, 8)
        << "FC1 MMRS torch_npu skipped: bias dtype mismatch. bias="
        << bias->scalar_type() << ", input=" << a.scalar_type();
    return false;
  }
  if (a.size(1) != b.size(0)) {
    LOG_FIRST_N(WARNING, 8)
        << "FC1 MMRS torch_npu skipped: matmul K mismatch. a=" << a.sizes()
        << ", b=" << b.sizes();
    return false;
  }
  return true;
}

}  // namespace

torch::Tensor matmul_reduce_scatter(const torch::Tensor& a,
                                    const torch::Tensor& b,
                                    const std::optional<torch::Tensor>& bias,
                                    ProcessGroup* process_group,
                                    const std::string& reduce_op,
                                    int64_t comm_turn,
                                    const std::string& comm_mode) {
  if (!can_call_torch_npu_mmrs(a, b, bias, process_group)) {
    return torch::Tensor();
  }

  std::string group = process_group->hccl_comm_name(/*init_comm=*/true);
  if (group.empty()) {
    LOG_FIRST_N(WARNING, 8)
        << "FC1 MMRS torch_npu skipped: HCCL group name is empty; fallback to "
           "matmul + reduce_scatter path.";
    return torch::Tensor();
  }

  std::optional<c10::string_view> torch_comm_mode = std::nullopt;
  if (comm_mode == "ai_cpu" || comm_mode == "aiv") {
    torch_comm_mode = c10::string_view(comm_mode);
  } else if (!comm_mode.empty() && comm_mode != "none") {
    LOG_FIRST_N(WARNING, 8)
        << "FC1 MMRS torch_npu unsupported comm_mode=" << comm_mode
        << "; using torch_npu default comm_mode.";
  }
  return at_npu::native::custom_ops::npu_mm_reduce_scatter_base(
      a,
      b,
      group,
      process_group->world_size(),
      reduce_op.empty() ? c10::string_view("sum") : c10::string_view(reduce_op),
      bias,
      /*x1_scale=*/std::nullopt,
      /*x2_scale=*/std::nullopt,
      comm_turn,
      /*output_dtype=*/std::nullopt,
      torch_comm_mode);
}

}  // namespace xllm::kernel::npu
