// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <torch/library.h>
#include <torch/types.h>

#include <array>
#include <mutex>

#include "csrc/npu/ascendc_api.h"
#include "op_plugin/utils/op_api_common_base.h"

namespace xllm_kernel {
namespace {

const AscendcApi& get_api() {
  static const auto api = load_ascendc_api(
      "aclnnXllmKernelInplacePartialRotaryMul",
      "aclnnXllmKernelInplacePartialRotaryMulGetWorkspaceSize");
  return api;
}

void prepare_inplace_partial_rotary_mul() { get_api(); }

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

void inplace_partial_rotary_mul(torch::Tensor& value,
                                const torch::Tensor& cosine,
                                const torch::Tensor& sine,
                                int64_t start,
                                int64_t end) {
  CHECK_EQ(value.dim(), 4);
  CHECK(0 <= start && start < end && end <= value.size(-1));
  CHECK_EQ((end - start) % 2, 0);

  const torch::DeviceGuard guard(value.device());
  const int64_t interleave = 1;
  const std::array<int64_t, 2> bounds{start, end};
  const at::IntArrayRef partial_slice(bounds.data(), bounds.size());
  EXEC_NPU_CMD_EXT(aclnnXllmKernelInplacePartialRotaryMul,
                   value,
                   cosine,
                   sine,
                   interleave,
                   partial_slice);
}

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("prepare_inplace_partial_rotary_mul() -> ()",
        TORCH_FN(prepare_inplace_partial_rotary_mul));
  m.def(
      "inplace_partial_rotary_mul(Tensor(a!) value, Tensor cosine, Tensor "
      "sine, int start, int end) -> ()");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("inplace_partial_rotary_mul", TORCH_FN(inplace_partial_rotary_mul));
}

}  // namespace
}  // namespace xllm_kernel
