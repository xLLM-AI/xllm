/* Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. */
#include "csrc/xlite/common.h"

// GVirt kernels/einsum_mht_htd_mhd.h. T/D=-1 select full packed heads.
using Launcher = uint32_t(uint32_t,
                          aclrtStream,
                          void*,
                          void*,
                          void*,
                          uint32_t,
                          uint32_t,
                          uint32_t,
                          uint32_t,
                          uint64_t,
                          uint64_t,
                          uint64_t,
                          bool,
                          uint64_t,
                          int,
                          int);
extern "C" Launcher aclrtlaunch_einsum_mht_htd_mhd_bfloat16_t;
namespace {
using namespace xllm_kernel::xlite;
torch::Tensor run(const torch::Tensor& value, const torch::Tensor& weight) {
  check_tensor(value);
  check_pair(value, weight);
  CHECK(value.scalar_type() == torch::kBFloat16);
  CHECK(value.dim() == 3 && weight.dim() == 3 &&
        value.size(1) == weight.size(0) && value.size(2) == weight.size(1));
  CHECK(weight.size(1) > 0 && weight.size(2) > 0 && weight.size(1) % 16 == 0 &&
        weight.size(2) % 16 == 0);
  CHECK(value.size(0) * value.size(1) * weight.size(2) <=
        std::numeric_limits<int32_t>::max());
  const c10::DeviceGuard guard(value.device());
  torch::Tensor x = value.contiguous(), w = weight.contiguous();
  torch::Tensor out =
      torch::empty({x.size(0), x.size(1), w.size(2)}, x.options());
  if (x.numel() == 0) {
    return out;
  }
  const uint32_t count = cores(x, true);
  const aclrtStream stream =
      c10_npu::getCurrentNPUStream(x.device().index()).stream();
  at_npu::native::OpCommand::RunOpApi(
      "XliteEinsum", [x, w, out, count, stream]() -> int {
        const uint64_t tile = std::numeric_limits<uint64_t>::max();
        return checked_status(aclrtlaunch_einsum_mht_htd_mhd_bfloat16_t(
            count,
            stream,
            x.data_ptr(),
            w.data_ptr(),
            out.data_ptr(),
            static_cast<uint32_t>(x.size(0)),
            static_cast<uint32_t>(x.size(1)),
            static_cast<uint32_t>(x.size(2)),
            static_cast<uint32_t>(w.size(2)),
            tile,
            tile,
            tile,
            false,
            0x600,
            -1,
            -1));
      });
  return out;
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("xlite_einsum(Tensor value, Tensor weight) -> Tensor");
}
TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("xlite_einsum", &run);
}
