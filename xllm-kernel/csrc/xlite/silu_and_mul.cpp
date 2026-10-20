/* Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. */
#include "csrc/xlite/common.h"

// GVirt kernels/silu_and_mul.h.
using Launcher =
    uint32_t(uint32_t, aclrtStream, void*, void*, void*, uint32_t, uint32_t);
extern "C" Launcher aclrtlaunch_silu_and_mul_bfloat16_t;
extern "C" Launcher aclrtlaunch_silu_and_mul_float16_t;
namespace {
using namespace xllm_kernel::xlite;
torch::Tensor silu_and_mul(const torch::Tensor& value) {
  check_tensor(value);
  CHECK(value.dim() == 2);
  CHECK(value.size(1) > 0 && value.size(1) <= 16384 &&
        value.size(1) % 128 == 0);
  const c10::DeviceGuard guard(value.device());
  torch::Tensor x = value.contiguous();
  torch::Tensor out = torch::empty({x.size(0), x.size(1) / 2}, x.options());
  if (x.numel() == 0) {
    return out;
  }
  const uint32_t count = cores(x);
  const aclrtStream stream =
      c10_npu::getCurrentNPUStream(x.device().index()).stream();
  Launcher* launcher = x.scalar_type() == torch::kBFloat16
                           ? aclrtlaunch_silu_and_mul_bfloat16_t
                           : aclrtlaunch_silu_and_mul_float16_t;
  at_npu::native::OpCommand::RunOpApi(
      "XliteSiluAndMul", [x, out, count, stream, launcher]() -> int {
        return checked_status(launcher(count,
                                       stream,
                                       x.data_ptr(),
                                       out.data_ptr(),
                                       nullptr,
                                       static_cast<uint32_t>(x.size(0)),
                                       static_cast<uint32_t>(out.size(1))));
      });
  return out;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("xlite_silu_and_mul(Tensor value) -> Tensor");
}
TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("xlite_silu_and_mul", &silu_and_mul);
}
