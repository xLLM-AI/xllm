/* Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. */

#include "csrc/xlite/common.h"
using Launcher = uint32_t(uint32_t,
                          aclrtStream,
                          uint32_t,
                          uint32_t,
                          uint32_t,
                          uint32_t,
                          uint32_t,
                          uint32_t,
                          void*,
                          void*,
                          uint32_t,
                          uint32_t,
                          void*,
                          void*,
                          uint32_t,
                          void*,
                          void*,
                          uint32_t,
                          uint32_t);
extern "C" Launcher aclrtlaunch_rope_complex_and_cache_bfloat16_t;
extern "C" Launcher aclrtlaunch_rope_complex_and_cache_float16_t;
namespace {
torch::Tensor run(const torch::Tensor& value,
                  const torch::Tensor& cosine,
                  const torch::Tensor& sine) {
  using namespace xllm_kernel::xlite;
  check_tensor(value);
  check_pair(value, cosine);
  check_pair(value, sine);
  CHECK(value.dim() == 3 && cosine.dim() == 4 &&
        cosine.sizes() == sine.sizes());
  CHECK(cosine.size(0) == value.size(0) && cosine.size(1) == 1 &&
        cosine.size(2) == 1 && cosine.size(3) == value.size(2));
  CHECK(value.size(2) == 64 || value.size(2) == 128);
  CHECK(value.size(1) > 0 && value.size(1) * value.size(2) <= 4096);
  const c10::DeviceGuard guard(value.device());
  torch::Tensor x = value.contiguous();
  torch::Tensor c = cosine.reshape({value.size(0), value.size(2)})
                        .slice(1, 0, value.size(2), 2);
  torch::Tensor s = sine.reshape({value.size(0), value.size(2)})
                        .slice(1, 0, value.size(2), 2);
  torch::Tensor freqs = torch::stack({c, s}, -1)
                            .reshape({value.size(0), value.size(2)})
                            .to(torch::kFloat32)
                            .contiguous();
  torch::Tensor positions =
      torch::arange(value.size(0), value.options().dtype(torch::kInt64));
  torch::Tensor out = torch::empty_like(x);
  if (x.numel() == 0) {
    return out;
  }
  const uint32_t count = cores(x);
  const aclrtStream stream =
      c10_npu::getCurrentNPUStream(x.device().index()).stream();
  Launcher* launcher = x.scalar_type() == torch::kBFloat16
                           ? aclrtlaunch_rope_complex_and_cache_bfloat16_t
                           : aclrtlaunch_rope_complex_and_cache_float16_t;
  at_npu::native::OpCommand::RunOpApi(
      "XliteInterleavedRoPE",
      [x, freqs, positions, out, count, stream, launcher]() -> int {
        const uint32_t width = static_cast<uint32_t>(x.size(2));
        return checked_status(launcher(count,
                                       stream,
                                       static_cast<uint32_t>(x.size(0)),
                                       static_cast<uint32_t>(x.size(1)),
                                       width,
                                       width,
                                       0,
                                       0,
                                       x.data_ptr(),
                                       out.data_ptr(),
                                       width,
                                       0,
                                       freqs.data_ptr(),
                                       positions.data_ptr(),
                                       0,
                                       nullptr,
                                       nullptr,
                                       0,
                                       1));
      });
  return out;
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def(
      "xlite_interleaved_rotary_embedding(Tensor value, Tensor cosine, Tensor "
      "sine) -> Tensor");
}
TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("xlite_interleaved_rotary_embedding", &run);
}
