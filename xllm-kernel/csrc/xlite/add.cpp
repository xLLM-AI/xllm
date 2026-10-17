/* Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. */
#include "csrc/xlite/common.h"

// GVirt kernels/add.h at the package-pinned revision.
using Launcher =
    uint32_t(uint32_t, aclrtStream, void*, void*, void*, uint32_t, uint32_t);
extern "C" Launcher aclrtlaunch_add_bfloat16_t;
extern "C" Launcher aclrtlaunch_add_float16_t;
namespace {
using namespace xllm_kernel::xlite;
torch::Tensor add(const torch::Tensor& value, const torch::Tensor& residual) {
  check_tensor(value);
  check_pair(value, residual);
  CHECK(value.dim() == 2 && value.sizes() == residual.sizes());
  CHECK(value.size(1) > 0 && value.size(1) <= 8192 && value.size(1) % 64 == 0);
  const c10::DeviceGuard guard(value.device());
  torch::Tensor x = value.contiguous();
  torch::Tensor y = residual.contiguous();
  torch::Tensor out = torch::empty_like(x);
  if (value.numel() == 0) {
    return out;
  }
  const uint32_t count = cores(value);
  const aclrtStream stream =
      c10_npu::getCurrentNPUStream(value.device().index()).stream();
  Launcher* launcher = value.scalar_type() == torch::kBFloat16
                           ? aclrtlaunch_add_bfloat16_t
                           : aclrtlaunch_add_float16_t;
  at_npu::native::OpCommand::RunOpApi(
      "XliteAdd", [x, y, out, count, stream, launcher]() -> int {
        return checked_status(launcher(count,
                                       stream,
                                       x.data_ptr(),
                                       y.data_ptr(),
                                       out.data_ptr(),
                                       static_cast<uint32_t>(x.size(0)),
                                       static_cast<uint32_t>(x.size(1))));
      });
  return out;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("xlite_add(Tensor value, Tensor residual) -> Tensor");
}
TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) { m.impl("xlite_add", &add); }
