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

#include "npu_ops_api.h"

#include <glog/logging.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include <sstream>

#include "core/framework/parallel_state/process_group.h"

namespace xllm::kernel::npu {
namespace {

std::string matmul_reduce_scatter_reject_reason(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& output,
    ProcessGroup* process_group) {
  if (process_group == nullptr) {
    return "process_group is null";
  }
  if (!output.has_value() || !output->defined()) {
    return "output tensor is missing";
  }
  if (a.dim() != 2 || b.dim() != 2 || output->dim() != 2) {
    std::ostringstream oss;
    oss << "expected 2D tensors, got a_dim=" << a.dim()
        << ", b_dim=" << b.dim() << ", output_dim=" << output->dim();
    return oss.str();
  }
  if (a.scalar_type() != at::kHalf && a.scalar_type() != at::kBFloat16) {
    std::ostringstream oss;
    oss << "unsupported input dtype=" << a.scalar_type();
    return oss.str();
  }
  if (a.scalar_type() != b.scalar_type() ||
      a.scalar_type() != output->scalar_type()) {
    std::ostringstream oss;
    oss << "dtype mismatch: a=" << a.scalar_type()
        << ", b=" << b.scalar_type()
        << ", output=" << output->scalar_type();
    return oss.str();
  }
  if (bias.has_value() && bias->defined() &&
      bias->scalar_type() != a.scalar_type()) {
    std::ostringstream oss;
    oss << "bias dtype mismatch: bias=" << bias->scalar_type()
        << ", input=" << a.scalar_type();
    return oss.str();
  }
  if (a.size(1) != b.size(0)) {
    std::ostringstream oss;
    oss << "matmul K mismatch: a=" << a.sizes() << ", b=" << b.sizes();
    return oss.str();
  }
  if (output->size(1) != b.size(1)) {
    std::ostringstream oss;
    oss << "output N mismatch: output=" << output->sizes()
        << ", b=" << b.sizes();
    return oss.str();
  }
  return "";
}

torch::Tensor matmul_kn(const torch::Tensor& a,
                        const torch::Tensor& b,
                        const std::optional<torch::Tensor>& bias) {
  torch::Tensor out = torch::matmul(a, b);
  if (bias.has_value() && bias->defined()) {
    out = out + bias.value();
  }
  return out;
}

bool should_use_ai_cpu_for_aiv_risky_shape(const torch::Tensor& a,
                                           const torch::Tensor& b,
                                           ProcessGroup* process_group) {
  if (process_group == nullptr || a.scalar_type() != at::kBFloat16 ||
      a.dim() != 2 || b.dim() != 2 || a.size(0) != 2048 ||
      b.size(1) != 5120) {
    return false;
  }
  const int64_t world_size = process_group->world_size();
  const int64_t k = a.size(1);
  if (world_size == 2) {
    return k == 3072 || k == 8704;
  }
  if (world_size == 4) {
    return k == 4352;
  }
  if (world_size == 8) {
    return k == 2176;
  }
  return false;
}

}  // namespace

torch::Tensor matmul_reduce_scatter(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& output,
    ProcessGroup* process_group,
    const std::string& reduce_op,
    int64_t comm_turn,
    int64_t stream_mode,
    const std::string& comm_mode) {
  const std::string reject_reason =
      matmul_reduce_scatter_reject_reason(a, b, bias, output, process_group);
  if (!reject_reason.empty()) {
    LOG_FIRST_N(WARNING, 8)
        << "FC1 MMRS torch_npu skipped: " << reject_reason
        << "; fallback to matmul + reduce_scatter path. a=" << a.sizes()
        << ", b=" << b.sizes()
        << ", output="
        << (output.has_value() && output->defined() ? output->sizes()
                                                    : c10::IntArrayRef{});
    return matmul_kn(a, b, bias);
  }

  std::string group = process_group->hccl_comm_name(/*init_comm=*/true);
  if (group.empty()) {
    LOG_FIRST_N(WARNING, 8)
        << "FC1 MMRS torch_npu skipped: HCCL group name is empty; fallback to "
           "matmul + reduce_scatter path.";
    return matmul_kn(a, b, bias);
  }

  std::string effective_comm_mode = comm_mode;
  if ((comm_mode.empty() || comm_mode == "aiv") &&
      should_use_ai_cpu_for_aiv_risky_shape(a, b, process_group)) {
    effective_comm_mode = "ai_cpu";
    LOG_FIRST_N(WARNING, 16)
        << "FC1 MMRS comm_mode switched from aiv to ai_cpu for known risky "
           "torch_npu shape: a="
        << a.sizes() << ", b=" << b.sizes()
        << ", world_size=" << process_group->world_size()
        << ", dtype=" << a.scalar_type();
  }

  LOG_FIRST_N(INFO, 16)
      << "FC1 MMRS torch_npu hit: a=" << a.sizes() << ", b=" << b.sizes()
      << ", expected_out=" << output->sizes() << ", dtype=" << a.scalar_type()
      << ", rank=" << process_group->rank()
      << ", world_size=" << process_group->world_size()
      << ", reduce_op=" << (reduce_op.empty() ? "sum" : reduce_op)
      << ", comm_turn=" << comm_turn << ", stream_mode=" << stream_mode
      << ", comm_mode=" << (comm_mode.empty() ? "none" : comm_mode)
      << ", effective_comm_mode="
      << (effective_comm_mode.empty() ? "none" : effective_comm_mode);

  std::optional<c10::string_view> torch_comm_mode = std::nullopt;
  if (effective_comm_mode == "ai_cpu" || effective_comm_mode == "aiv") {
    torch_comm_mode = c10::string_view(effective_comm_mode);
  } else if (!effective_comm_mode.empty() && effective_comm_mode != "none") {
    LOG_FIRST_N(WARNING, 8)
        << "FC1 MMRS torch_npu unsupported comm_mode=" << effective_comm_mode
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
