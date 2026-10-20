/* Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. */

#include <cmath>

#include "csrc/xlite/common.h"
extern "C" uint32_t aclrtlaunch_sigmoid_topk_float(uint32_t,
                                                   aclrtStream,
                                                   void*,
                                                   void*,
                                                   void*,
                                                   float,
                                                   void*,
                                                   void*,
                                                   uint32_t,
                                                   uint32_t,
                                                   uint32_t,
                                                   uint32_t,
                                                   uint32_t,
                                                   bool);
namespace {
std::tuple<torch::Tensor, torch::Tensor> run(const torch::Tensor& logits,
                                             const torch::Tensor& bias,
                                             int64_t topk,
                                             int64_t groups,
                                             int64_t selected_groups,
                                             double scaling) {
  using namespace xllm_kernel::xlite;
  CHECK(logits.device().type() == c10::DeviceType::PrivateUse1 &&
        bias.device() == logits.device());
  CHECK(logits.scalar_type() == torch::kFloat32 &&
        bias.scalar_type() == torch::kFloat32);
  CHECK(logits.dim() == 2 && logits.size(1) == 256 && bias.dim() == 1 &&
        bias.numel() == 256);
  CHECK(logits.size(0) <= std::numeric_limits<int32_t>::max() / 256);
  CHECK((groups == 1 && selected_groups == 1) ||
        (groups == 8 && selected_groups == 4));
  CHECK(topk > 0 && topk <= 32 && std::isfinite(scaling) && scaling > 0 &&
        scaling <= std::numeric_limits<float>::max());
  CHECK(at_npu::native::get_npu_format(logits) == ACL_FORMAT_ND);
  CHECK(at_npu::native::get_npu_format(bias) == ACL_FORMAT_ND);
  const c10::DeviceGuard guard(logits.device());
  torch::Tensor x = logits.contiguous(), b = bias.contiguous();
  torch::Tensor ids = torch::arange(256, x.options().dtype(torch::kInt32));
  torch::Tensor dense = torch::empty_like(x);
  torch::Tensor bitmap = torch::empty({x.size(0), 8}, ids.options());
  const uint32_t count = cores(x);
  const aclrtStream stream =
      c10_npu::getCurrentNPUStream(x.device().index()).stream();
  if (x.size(0) > 0) {
    at_npu::native::OpCommand::RunOpApi(
        "XliteMoeGateRouting",
        [x,
         b,
         ids,
         dense,
         bitmap,
         count,
         stream,
         topk,
         groups,
         selected_groups,
         scaling]() -> int {
          return checked_status(aclrtlaunch_sigmoid_topk_float(
              count,
              stream,
              x.data_ptr(),
              ids.data_ptr(),
              b.data_ptr(),
              static_cast<float>(scaling),
              dense.data_ptr(),
              bitmap.data_ptr(),
              static_cast<uint32_t>(x.size(0)),
              256,
              static_cast<uint32_t>(groups),
              static_cast<uint32_t>(selected_groups),
              static_cast<uint32_t>(topk),
              true));
        });
  }
  // Decode the routing bitmap, not dense-weight magnitudes: probabilities may
  // underflow to zero even for selected experts. Conversion is part of timing.
  torch::Tensor index64 = ids.to(torch::kInt64);
  torch::Tensor words =
      bitmap.index_select(1, torch::floor_divide(index64, 32));
  torch::Tensor selected =
      torch::bitwise_and(
          torch::bitwise_right_shift(words, index64.remainder(32)), 1)
          .to(torch::kBool);
  torch::Tensor ordered = std::get<0>(
      torch::where(
          selected,
          index64.unsqueeze(0),
          torch::full_like(words, 256, words.options().dtype(torch::kInt64)))
          .topk(topk, 1, false, true));
  return {dense.gather(1, ordered), ordered.to(torch::kInt32)};
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def(
      "xlite_moe_gate_routing(Tensor logits, Tensor bias, int topk, int "
      "groups, int selected_groups, float scaling) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("xlite_moe_gate_routing", &run);
}
