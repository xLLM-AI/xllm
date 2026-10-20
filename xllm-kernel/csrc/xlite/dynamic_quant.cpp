/* Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. */

#include "csrc/xlite/common.h"
extern "C" uint32_t aclrtlaunch_quant_bf16_to_i8_dynamic(uint32_t,
                                                         aclrtStream,
                                                         void*,
                                                         void*,
                                                         void*,
                                                         void*,
                                                         uint32_t,
                                                         uint32_t);
namespace {
std::tuple<torch::Tensor, torch::Tensor> run(const torch::Tensor& value) {
  using namespace xllm_kernel::xlite;
  check_tensor(value);
  CHECK(value.dim() == 2 && value.scalar_type() == torch::kBFloat16);
  CHECK(value.size(1) > 0 && value.size(1) <= 8192 && value.size(1) % 64 == 0);
  const c10::DeviceGuard guard(value.device());
  torch::Tensor x = value.contiguous();
  torch::Tensor output = torch::empty_like(x, x.options().dtype(torch::kInt8));
  torch::Tensor scale =
      torch::empty({x.size(0)}, x.options().dtype(torch::kFloat32));
  if (x.numel() == 0) {
    return {output, scale};
  }
  const uint32_t count = cores(x);
  const aclrtStream stream =
      c10_npu::getCurrentNPUStream(x.device().index()).stream();
  at_npu::native::OpCommand::RunOpApi(
      "XliteDynamicQuant", [x, output, scale, count, stream]() -> int {
        return checked_status(aclrtlaunch_quant_bf16_to_i8_dynamic(
            count,
            stream,
            x.data_ptr(),
            scale.data_ptr(),
            output.data_ptr(),
            nullptr,
            static_cast<uint32_t>(x.size(0)),
            static_cast<uint32_t>(x.size(1))));
      });
  // The raw launcher divides by zero for all-zero rows. Include the repair in
  // the operation (and timing), without a host read or a stream
  // synchronization.
  output =
      torch::where(scale.unsqueeze(-1) == 0, torch::zeros_like(output), output);
  return {output, scale};
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("xlite_dynamic_quant(Tensor value) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("xlite_dynamic_quant", &run);
}
