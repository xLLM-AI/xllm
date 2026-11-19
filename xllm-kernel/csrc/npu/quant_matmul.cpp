// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <acl/acl.h>
#include <torch/library.h>
#include <torch/types.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

#include <mutex>
#include <optional>

#include "csrc/npu/ascendc_api.h"
#include "op_plugin/utils/op_api_common_base.h"

namespace xllm_kernel {
namespace {

namespace nz_decode {

const AscendcApi& get_api() {
  static const auto api =
      find_ascendc_api("aclnnXllmKernelQuantMatmulNzDecode",
                       "aclnnXllmKernelQuantMatmulNzDecodeGetWorkspaceSize");
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

void run(const torch::Tensor& input,
         const torch::Tensor& weight,
         const torch::Tensor& scale,
         const torch::Tensor& bias,
         torch::Tensor& out) {
  EXEC_NPU_CMD_EXT(
      aclnnXllmKernelQuantMatmulNzDecode, input, weight, scale, bias, out);
}

}  // namespace nz_decode

void prepare_quant_matmul_out() { nz_decode::get_api(); }

bool supports_nz_decode(const torch::Tensor& input,
                        const torch::Tensor& weight,
                        const torch::Tensor& scale,
                        const torch::Tensor& bias,
                        const torch::Tensor& out) {
  // The device kernel has fixed projection dimensions and packs up to 16 rows.
  if (input.dim() != 2 || weight.dim() != 2 || input.size(0) < 1 ||
      input.size(0) > 16) {
    return false;
  }
  const int64_t k = input.size(1);
  const int64_t n = weight.size(1);
  const bool projection =
      (k == 5120 && (n == 6400 || n == 1280)) || (k == 3200 && n == 5120);
  const auto& api = nz_decode::get_api();
  return projection && api.launch && api.workspace &&
         input.scalar_type() == torch::kInt8 &&
         weight.scalar_type() == torch::kInt8 && scale.dim() == 1 &&
         scale.numel() == n && scale.scalar_type() == torch::kFloat32 &&
         bias.dim() == 1 && bias.numel() == n &&
         bias.scalar_type() == torch::kInt32 &&
         out.scalar_type() == torch::kBFloat16 &&
         at_npu::native::get_npu_format(weight) == ACL_FORMAT_FRACTAL_NZ;
}

torch::Tensor quant_matmul_out(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    bool transpose2,
    const torch::Tensor& scale,
    const std::optional<torch::Tensor>& offset,
    const std::optional<torch::Tensor>& pertoken_scale,
    const std::optional<torch::Tensor>& bias,
    std::optional<torch::ScalarType> output_dtype,
    torch::Tensor& out) {
  CHECK_GE(input.dim(), 2);
  CHECK_GE(weight.dim(), 2);
  CHECK_EQ(input.size(-1), weight.size(transpose2 ? -1 : -2));
  auto shape = input.sizes().vec();
  shape.back() = weight.size(transpose2 ? 0 : 1);
  CHECK(out.sizes().equals(shape));
  CHECK(out.is_contiguous());
  CHECK_EQ(out.scalar_type(), output_dtype.value_or(torch::kInt8));

  const torch::DeviceGuard guard(input.device());
  if (!transpose2 && !offset && !pertoken_scale && bias &&
      supports_nz_decode(input, weight, scale, *bias, out)) {
    nz_decode::run(input, weight, scale, *bias, out);
  } else {
    const bool transpose1 = false;
    EXEC_NPU_CMD_EXT(aclnnQuantMatmulV4,
                     input,
                     weight,
                     scale,
                     offset,
                     pertoken_scale,
                     bias,
                     transpose1,
                     transpose2,
                     out);
  }
  return out;
}

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("prepare_quant_matmul_out() -> ()", TORCH_FN(prepare_quant_matmul_out));
  m.def(
      "quant_matmul_out(Tensor input, Tensor weight, bool transpose2, "
      "Tensor scale, Tensor? offset, Tensor? pertoken_scale, Tensor? bias, "
      "ScalarType? output_dtype, Tensor(a!) out) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("quant_matmul_out", TORCH_FN(quant_matmul_out));
}

}  // namespace
}  // namespace xllm_kernel
