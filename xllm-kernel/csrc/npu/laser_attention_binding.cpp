// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <torch/library.h>

#include <mutex>

#include "csrc/npu/ascendc_api.h"
#include "csrc/npu/laser_attention.h"
#include "op_plugin/utils/op_api_common_base.h"

namespace xllm_kernel {
namespace {

const AscendcApi& laser_api() {
  static const auto api =
      load_ascendc_api("aclnnXllmKernelLaserAttention",
                       "aclnnXllmKernelLaserAttentionGetWorkspaceSize");
  return api;
}

void GetApiFunc(const char* /*api_name*/,
                const char* /*workspace_name*/,
                void*& launch,
                void*& workspace) {
  static std::once_flag initialize;
  std::call_once(initialize, [&] {
    const auto& api = laser_api();
    launch = api.launch;
    workspace = api.workspace;
  });
}

torch::Tensor laser_attention(const torch::Tensor& query,
                              const torch::Tensor& key,
                              const torch::Tensor& value,
                              double scale) {
  const torch::DeviceGuard guard(query.device());
  return run_laser_attention(
      query, key, value, scale, query.size(1), [](auto&... args) {
        EXEC_NPU_CMD_EXT(aclnnXllmKernelLaserAttention, args...);
      });
}

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("prepare_laser_attention() -> ()", [] { laser_api(); });
  m.def(
      "laser_attention(Tensor query, Tensor key, Tensor value, float scale) -> "
      "Tensor");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("laser_attention", TORCH_FN(laser_attention));
}

}  // namespace
}  // namespace xllm_kernel
