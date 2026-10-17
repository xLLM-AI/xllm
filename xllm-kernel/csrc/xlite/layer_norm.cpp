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

namespace {
using namespace xllm_kernel::xlite;
torch::Tensor run(const torch::Tensor& value,
                  const torch::Tensor& weight,
                  const torch::Tensor& bias,
                  double eps) {
  check_tensor(value);
  check_pair(value, weight);
  check_pair(value, bias);
  CHECK(value.dim() == 2 && weight.dim() == 1 &&
        bias.sizes() == weight.sizes());
  CHECK(value.size(1) == weight.numel() && weight.numel() > 0 &&
        weight.numel() <= 8192 && weight.numel() % 64 == 0);
  CHECK(std::isfinite(eps) && eps >= 0 &&
        eps <= std::numeric_limits<float>::max());
  const c10::DeviceGuard guard(value.device());
  torch::Tensor x = value.contiguous(), w = weight.contiguous(),
                b = bias.contiguous();
  torch::Tensor out = torch::empty_like(x);
  if (x.numel() == 0) {
    return out;
  }
  const uint32_t count = cores(x);
  const aclrtStream stream =
      c10_npu::getCurrentNPUStream(x.device().index()).stream();
  Launcher* launcher = x.scalar_type() == torch::kBFloat16
                           ? aclrtlaunch_norm_bfloat16_t
                           : aclrtlaunch_norm_float16_t;
  at_npu::native::OpCommand::RunOpApi(
      "XliteLayerNorm", [x, w, b, out, count, stream, launcher, eps]() -> int {
        const uint32_t width = static_cast<uint32_t>(x.size(1));
        return checked_status(launcher(count,
                                       stream,
                                       x.data_ptr(),
                                       nullptr,
                                       w.data_ptr(),
                                       b.data_ptr(),
                                       out.data_ptr(),
                                       static_cast<uint32_t>(x.size(0)),
                                       width,
                                       static_cast<float>(eps),
                                       /*NormKind::Layer=*/1,
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
  return out;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def(
      "xlite_layer_norm(Tensor value, Tensor weight, Tensor bias, float eps) "
      "-> Tensor");
}
TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("xlite_layer_norm", &run);
}
