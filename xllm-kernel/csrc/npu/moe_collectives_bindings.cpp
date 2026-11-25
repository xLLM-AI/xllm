// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <torch/library.h>

#include "csrc/npu/collectives.h"
#include "csrc/npu/mega_moe_binding.h"

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("has_mega_moe() -> bool", TORCH_FN(xllm::kernel::npu::has_mega_moe));
  m.def(
      "mega_moe(Tensor context, Tensor x, Tensor(a!) topk_ids, Tensor "
      "topk_weights, Tensor[] weight1, Tensor[] weight2, Tensor[] "
      "weight_scales1, Tensor[] weight_scales2, int moe_expert_num, int "
      "ep_world_size, int ccl_buffer_size, int num_max_tokens_per_rank, "
      "Tensor? x_active_mask=None) -> (Tensor, Tensor)");
  m.def("npu_all_reduce(Tensor(a!) x, int comm) -> ()");
  m.def("npu_all_gather(Tensor input, Tensor(a!) output, int comm) -> ()");
  m.def("npu_reduce_scatter(Tensor input, Tensor(a!) output, int comm) -> ()");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("mega_moe", TORCH_FN(xllm_kernel::mega_moe));
  m.impl("npu_all_reduce",
         TORCH_FN(xllm::kernel::npu::all_reduce_on_current_stream));
  m.impl("npu_all_gather",
         TORCH_FN(xllm::kernel::npu::all_gather_on_current_stream));
  m.impl("npu_reduce_scatter",
         TORCH_FN(xllm::kernel::npu::reduce_scatter_on_current_stream));
}
