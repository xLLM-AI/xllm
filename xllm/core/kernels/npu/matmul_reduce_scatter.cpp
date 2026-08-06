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
                             ProcessGroup* process_group,
                             const std::optional<torch::Tensor>& x1_scale,
                             const std::optional<torch::Tensor>& x2_scale) {
  const bool is_quant = x1_scale.has_value() && x2_scale.has_value();
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
  if (is_quant) {
    // Quantized (w8a8_dynamic) path: int8 inputs + float32 dequant scales.
    if (a.scalar_type() != at::kChar || b.scalar_type() != at::kChar) {
      LOG_FIRST_N(WARNING, 8)
          << "FC1 MMRS torch_npu skipped: quant path needs int8 a/b, got a="
          << a.scalar_type() << ", b=" << b.scalar_type();
      return false;
    }
    if (!x1_scale->defined() || !x2_scale->defined() ||
        x1_scale->scalar_type() != at::kFloat ||
        x2_scale->scalar_type() != at::kFloat) {
      LOG_FIRST_N(WARNING, 8) << "FC1 MMRS torch_npu skipped: quant scales "
                                 "must be defined float32.";
      return false;
    }
    if (bias.has_value() && bias->defined()) {
      LOG_FIRST_N(WARNING, 8)
          << "FC1 MMRS torch_npu skipped: non-zero bias unsupported in quant "
             "path.";
      return false;
    }
  } else {
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

torch::Tensor matmul_reduce_scatter(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const std::optional<torch::Tensor>& bias,
    ProcessGroup* process_group,
    const std::string& reduce_op,
    int64_t comm_turn,
    const std::string& comm_mode,
    const std::optional<torch::Tensor>& x1_scale,
    const std::optional<torch::Tensor>& x2_scale,
    const std::optional<at::ScalarType>& output_dtype) {
  if (!can_call_torch_npu_mmrs(a, b, bias, process_group, x1_scale, x2_scale)) {
    return torch::Tensor();
  }
  const bool is_quant = x1_scale.has_value() && x2_scale.has_value();

  std::string group = process_group->hccl_comm_name(/*init_comm=*/true);
  if (group.empty()) {
    LOG_FIRST_N(WARNING, 8)
        << "FC1 MMRS torch_npu skipped: HCCL group name is empty; fallback to "
           "matmul + reduce_scatter path.";
    return torch::Tensor();
  }

  // Quantized MMRS is only supported in aiv comm mode; force it when needed.
  const std::string effective_comm_mode =
      (is_quant && comm_mode != "aiv") ? std::string("aiv") : comm_mode;
  if (is_quant && comm_mode != "aiv") {
    LOG_FIRST_N(WARNING, 8)
        << "FC1 MMRS quant path requires comm_mode=aiv; overriding requested "
           "comm_mode="
        << comm_mode;
  }
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
      x1_scale,
      x2_scale,
      comm_turn,
      output_dtype,
      torch_comm_mode);
}

}  // namespace xllm::kernel::npu
