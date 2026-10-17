/* Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. */
#include <cmath>

#include "csrc/xlite/common.h"
using Launcher = uint32_t(uint32_t,
                          aclrtStream,
                          void*,
                          void*,
                          void*,
                          void*,
                          void*,
                          uint32_t,
                          uint32_t,
                          float,
                          int,
                          uint32_t,
                          uint32_t,
                          uint32_t,
                          uint32_t,
                          uint32_t,
                          bool,
                          void*,
                          uint32_t,
                          bool);
extern "C" Launcher aclrtlaunch_norm_bfloat16_t;
extern "C" Launcher aclrtlaunch_norm_float16_t;

#include <tuple>
namespace {
using namespace xllm_kernel::xlite;
std::tuple<torch::Tensor, torch::Tensor> run(const torch::Tensor& value,
                                             const torch::Tensor& residual,
                                             const torch::Tensor& weight,
                                             double eps) {
  check_tensor(value);
  check_pair(value, weight);
  check_pair(value, residual);
  CHECK(value.dim() == 2 && residual.sizes() == value.sizes() &&
        weight.dim() == 1);
  CHECK(value.size(1) == weight.numel() && weight.numel() > 0 &&
        weight.numel() <= 8192 && weight.numel() % 64 == 0);
  CHECK(std::isfinite(eps) && eps >= 0 &&
        eps <= std::numeric_limits<float>::max());
  const c10::DeviceGuard guard(value.device());
  torch::Tensor x = value.contiguous(), w = weight.contiguous();
  torch::Tensor sum = residual.contiguous().clone();
  torch::Tensor out = torch::empty_like(x);
  if (x.numel() == 0) {
    return {out, sum};
  }
  const uint32_t count = cores(x);
  const aclrtStream stream =
      c10_npu::getCurrentNPUStream(x.device().index()).stream();
  Launcher* launcher = x.scalar_type() == torch::kBFloat16
                           ? aclrtlaunch_norm_bfloat16_t
                           : aclrtlaunch_norm_float16_t;
  at_npu::native::OpCommand::RunOpApi(
      "XliteAddRmsNorm",
      [x, w, sum, out, count, stream, launcher, eps]() -> int {
        const uint32_t width = static_cast<uint32_t>(x.size(1));
        return checked_status(launcher(count,
                                       stream,
                                       x.data_ptr(),
                                       sum.data_ptr(),
                                       w.data_ptr(),
                                       nullptr,
                                       out.data_ptr(),
                                       static_cast<uint32_t>(x.size(0)),
                                       width,
                                       static_cast<float>(eps),
                                       0,
                                       1,
                                       width,
                                       width,
                                       0,
                                       0,
                                       true,
                                       nullptr,
                                       1,
                                       false));
      });
  return {out, sum};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def(
      "xlite_fused_add_rms_norm(Tensor value, Tensor residual, Tensor weight, "
      "float eps) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("xlite_fused_add_rms_norm", &run);
}
