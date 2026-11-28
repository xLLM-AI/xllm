// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <c10/util/Logging.h>
#include <dlfcn.h>
#include <torch/library.h>
#include <torch/types.h>

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <tuple>
#include <vector>

#include "op_plugin/utils/op_api_common_base.h"

namespace xllm_kernel {
namespace {

struct GammaAddRmsNormApi {
  void* run;
  void* workspace;
};

const GammaAddRmsNormApi& gamma_add_rms_norm_api() {
  static const auto api = [] {
    Dl_info binding{};
    CHECK(dladdr(reinterpret_cast<const void*>(&gamma_add_rms_norm_api),
                 &binding));
    const auto library_path =
        std::filesystem::path(binding.dli_fname).parent_path() /
        "ascendc_opp/xllm_kernel/op_api/lib/libcust_opapi.so";

    // The process retains the library and its embedded CANN registrations.
    void* library = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    CHECK(library) << library_path << ": " << dlerror();
    const GammaAddRmsNormApi functions{
        dlsym(library, "aclnnGammaAddRmsNorm"),
        dlsym(library, "aclnnGammaAddRmsNormGetWorkspaceSize")};
    CHECK(functions.run && functions.workspace) << library_path;
    return functions;
  }();
  return api;
}

void prepare_gamma_add_rms_norm() { gamma_add_rms_norm_api(); }

// EXEC_NPU_CMD_EXT manages the current torch_npu stream and workspace.
// Resolve its two entry points from this package instead of the OPP search
// path.
void GetApiFunc(const char* /*api_name*/,
                const char* /*workspace_name*/,
                void*& run,
                void*& workspace) {
  static std::once_flag initialize;
  std::call_once(initialize, [&] {
    const auto& api = gamma_add_rms_norm_api();
    run = api.run;
    workspace = api.workspace;
  });
}

std::tuple<torch::Tensor, torch::Tensor> fused_add_gemma_rms_norm(
    const torch::Tensor& input,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps) {
  CHECK(input.numel() > 0);
  CHECK(weight.numel() > 0);
  CHECK(input.sizes() == residual.sizes());
  CHECK(input.dim() >= weight.dim());

  auto normalized = torch::empty_like(input);
  auto summed = torch::empty_like(input);
  std::vector<int64_t> rstd_shape(input.dim(), 1);
  const int64_t leading_dims = input.dim() - weight.dim();
  std::copy_n(input.sizes().begin(), leading_dims, rstd_shape.begin());
  auto rstd = torch::empty(rstd_shape, input.options().dtype(torch::kFloat32));

  const torch::DeviceGuard guard(input.device());
  const bool add_gamma_offset = true;
  EXEC_NPU_CMD_EXT(aclnnXllmKernelGammaAddRmsNorm,
                   input,
                   residual,
                   weight,
                   eps,
                   add_gamma_offset,
                   normalized,
                   rstd,
                   summed);
  return {normalized, summed};
}

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("prepare_gamma_add_rms_norm() -> ()",
        TORCH_FN(prepare_gamma_add_rms_norm));
  m.def(
      "fused_add_gemma_rms_norm(Tensor input, Tensor residual, Tensor weight, "
      "float eps) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("fused_add_gemma_rms_norm", TORCH_FN(fused_add_gemma_rms_norm));
}

}  // namespace
}  // namespace xllm_kernel
