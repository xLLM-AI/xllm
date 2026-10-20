/* Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. */
#include "csrc/xlite/common.h"

// GVirt kernels/matmul.h: ND [N,K] weight, transpose=0, no bias/dequantization.
using Launcher = uint32_t(uint32_t,
                          aclrtStream,
                          void*,
                          void*,
                          void*,
                          uint64_t,
                          uint64_t,
                          uint64_t,
                          uint64_t,
                          uint64_t,
                          uint64_t,
                          uint64_t,
                          uint64_t,
                          uint64_t,
                          void*,
                          void*);
extern "C" Launcher aclrtlaunch_matmul_bfloat16_t;
extern "C" Launcher aclrtlaunch_matmul_float16_t;
namespace {
using namespace xllm_kernel::xlite;
torch::Tensor run(const torch::Tensor& value, const torch::Tensor& weight) {
  check_tensor(value);
  check_pair(value, weight);
  CHECK(value.dim() == 2 && weight.dim() == 2 &&
        value.size(1) == weight.size(1));
  CHECK(weight.size(0) > 0 && weight.size(1) > 0 && weight.size(0) % 16 == 0 &&
        weight.size(1) % 16 == 0);
  CHECK(value.size(0) * weight.size(0) <= std::numeric_limits<int32_t>::max());
  const c10::DeviceGuard guard(value.device());
  torch::Tensor x = value.contiguous(), w = weight.contiguous();
  torch::Tensor out = torch::empty({x.size(0), w.size(0)}, x.options());
  if (x.size(0) == 0) {
    return out;
  }
  const uint32_t count = cores(x, true);
  const aclrtStream stream =
      c10_npu::getCurrentNPUStream(x.device().index()).stream();
  Launcher* launcher = x.scalar_type() == torch::kBFloat16
                           ? aclrtlaunch_matmul_bfloat16_t
                           : aclrtlaunch_matmul_float16_t;
  at_npu::native::OpCommand::RunOpApi(
      "XliteLinear", [x, w, out, count, stream, launcher]() -> int {
        const uint64_t default_tile = std::numeric_limits<uint64_t>::max();
        return checked_status(launcher(count,
                                       stream,
                                       x.data_ptr(),
                                       w.data_ptr(),
                                       out.data_ptr(),
                                       static_cast<uint64_t>(x.size(0)),
                                       static_cast<uint64_t>(w.size(0)),
                                       static_cast<uint64_t>(x.size(1)),
                                       0,
                                       0,
                                       default_tile,
                                       default_tile,
                                       default_tile,
                                       0x600,
                                       nullptr,
                                       nullptr));
      });
  return out;
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("xlite_linear(Tensor value, Tensor weight) -> Tensor");
}
TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("xlite_linear", &run);
}
