// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <acl/acl.h>
#include <glog/logging.h>
#include <torch/library.h>
#include <torch/types.h>

#include <array>
#include <mutex>
#include <optional>
#include <tuple>

#include "csrc/npu/ascendc_api.h"
#include "op_plugin/utils/op_api_common_base.h"

namespace xllm_kernel {
namespace {

namespace gated {

const AscendcApi& get_api() {
  static const auto api =
      load_ascendc_api("aclnnXllmKernelLayerNormFwd",
                       "aclnnXllmKernelLayerNormFwdGetWorkspaceSize");
  return api;
}

void GetApiFunc(const char* /*api_name*/,
                const char* /*workspace_name*/,
                void*& launch,
                void*& workspace) {
  static std::once_flag initialize;
  std::call_once(initialize, [&] {
    const auto& api = get_api();
    launch = api.launch;
    workspace = api.workspace;
  });
}

torch::Tensor rms_norm_gated(const torch::Tensor& input,
                             const torch::Tensor& gate,
                             const torch::Tensor& weight,
                             double eps) {
  CHECK_GT(input.numel(), 0);
  CHECK(input.sizes() == gate.sizes());
  CHECK(weight.dim() == 1 && weight.size(0) == input.size(-1));

  const int64_t width = input.size(-1);
  const auto input_2d = input.reshape({-1, width}).contiguous();
  const auto gate_2d = gate.reshape({-1, width}).contiguous();
  const auto weight_contiguous = weight.contiguous();
  auto output = torch::empty_like(input_2d);
  auto mean = torch::empty({0}, input.options().dtype(torch::kFloat32));
  auto rstd =
      torch::empty({input_2d.size(0)}, input.options().dtype(torch::kFloat32));

  const torch::DeviceGuard guard(input.device());
  const std::optional<torch::Tensor> bias = std::nullopt;
  const bool norm_before_gate = true;
  const bool is_rms_norm = true;
  EXEC_NPU_CMD_EXT(aclnnXllmKernelLayerNormFwd,
                   input_2d,
                   weight_contiguous,
                   bias,
                   gate_2d,
                   eps,
                   /*group_size=*/width,
                   norm_before_gate,
                   is_rms_norm,
                   output,
                   mean,
                   rstd);
  return output.view(input.sizes());
}

}  // namespace gated

namespace quant {

const AscendcApi& get_api() {
  static const auto api =
      load_ascendc_api("aclnnXllmKernelRmsNormDynamicQuant",
                       "aclnnXllmKernelRmsNormDynamicQuantGetWorkspaceSize");
  return api;
}

void GetApiFunc(const char* /*api_name*/,
                const char* /*workspace_name*/,
                void*& launch,
                void*& workspace) {
  static std::once_flag initialize;
  std::call_once(initialize, [&] {
    const auto& api = get_api();
    launch = api.launch;
    workspace = api.workspace;
  });
}

std::tuple<torch::Tensor, torch::Tensor> rms_norm_dynamic_quant(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    double eps) {
  CHECK_GT(input.numel(), 0);
  CHECK(weight.dim() == 1 && weight.size(0) == input.size(-1));

  auto output =
      torch::empty(input.sizes(), input.options().dtype(torch::kInt8));
  auto scale = torch::empty(input.sizes().slice(0, input.dim() - 1),
                            input.options().dtype(torch::kFloat32));
  const std::optional<torch::Tensor> absent = std::nullopt;
  const std::array<bool, 2> output_mask{true, false};

  // The ACLNN schema requires both sets of output descriptors.
  auto unused_output = torch::empty({1}, output.options());
  auto unused_scale = torch::empty({1}, scale.options());
  const auto input_contiguous = input.contiguous();
  const auto weight_contiguous = weight.contiguous();
  const int64_t dst_type = ACL_INT8;
  const torch::DeviceGuard guard(input.device());
  EXEC_NPU_CMD_EXT(aclnnXllmKernelRmsNormDynamicQuant,
                   input_contiguous,
                   weight_contiguous,
                   /*smooth_scale1=*/absent,
                   /*smooth_scale2=*/absent,
                   /*beta=*/absent,
                   eps,
                   output_mask,
                   dst_type,
                   output,
                   unused_output,
                   scale,
                   unused_scale);
  return {output, scale};
}

}  // namespace quant

void prepare_ascendc_normalization() {
  gated::get_api();
  quant::get_api();
}

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("prepare_ascendc_normalization() -> ()",
        TORCH_FN(prepare_ascendc_normalization));
  m.def(
      "rms_norm_gated(Tensor input, Tensor gate, Tensor weight, float eps) -> "
      "Tensor");
  m.def(
      "rms_norm_dynamic_quant(Tensor input, Tensor weight, float eps) -> "
      "(Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("rms_norm_gated", TORCH_FN(gated::rms_norm_gated));
  m.impl("rms_norm_dynamic_quant", TORCH_FN(quant::rms_norm_dynamic_quant));
}

}  // namespace
}  // namespace xllm_kernel
