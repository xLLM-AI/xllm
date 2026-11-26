// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <torch/library.h>

#include "csrc/npu/sampling.h"

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("top_k_top_p(Tensor(a!) logits, Tensor top_k, Tensor top_p) -> ()");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("top_k_top_p", TORCH_FN(xllm::kernel::npu::top_k_top_p));
}
