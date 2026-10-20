/* Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. */
#include "csrc/xlite/common.h"

// GVirt kernels/embed_kernel.h reads uint32 IDs, never int64 pointers.
using Launcher = uint32_t(uint32_t,
                          aclrtStream,
                          void*,
                          void*,
                          void*,
                          uint32_t,
                          uint32_t,
                          uint32_t,
                          uint32_t,
                          uint32_t);
extern "C" Launcher aclrtlaunch_embed_kernel_bfloat16_t;
extern "C" Launcher aclrtlaunch_embed_kernel_float16_t;
namespace {
using namespace xllm_kernel::xlite;
torch::Tensor embedding(const torch::Tensor& indices,
                        const torch::Tensor& weight) {
  check_tensor(weight);
  CHECK(weight.dim() == 2 && indices.dim() == 1);
  CHECK(indices.scalar_type() == torch::kInt32 &&
        indices.device() == weight.device());
  CHECK(indices.numel() <= std::numeric_limits<int32_t>::max());
  CHECK(weight.size(0) > 0 && weight.size(1) > 0 && weight.size(1) <= 8192 &&
        weight.size(1) % 64 == 0);
  CHECK(indices.numel() * weight.size(1) <=
        std::numeric_limits<int32_t>::max());
  const c10::DeviceGuard guard(weight.device());
  torch::Tensor ids = indices.contiguous();
  torch::Tensor w = weight.contiguous();
  torch::Tensor out = torch::empty({ids.numel(), w.size(1)}, w.options());
  if (ids.numel() == 0) {
    return out;
  }
  const uint32_t count = cores(w);
  const aclrtStream stream =
      c10_npu::getCurrentNPUStream(w.device().index()).stream();
  Launcher* launcher = w.scalar_type() == torch::kBFloat16
                           ? aclrtlaunch_embed_kernel_bfloat16_t
                           : aclrtlaunch_embed_kernel_float16_t;
  at_npu::native::OpCommand::RunOpApi(
      "XliteEmbedding", [ids, w, out, count, stream, launcher]() -> int {
        return checked_status(launcher(count,
                                       stream,
                                       w.data_ptr(),
                                       ids.data_ptr(),
                                       out.data_ptr(),
                                       static_cast<uint32_t>(w.size(1)),
                                       static_cast<uint32_t>(ids.numel()),
                                       0,
                                       static_cast<uint32_t>(w.size(0)),
                                       1));
      });
  return out;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("xlite_embedding(Tensor indices, Tensor weight) -> Tensor");
}
TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("xlite_embedding", &embedding);
}
