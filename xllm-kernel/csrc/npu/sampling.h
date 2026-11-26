// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#pragma once

#include <torch/types.h>

namespace xllm::kernel::npu {

void top_k_top_p(torch::Tensor& logits,
                 const torch::Tensor& top_k,
                 const torch::Tensor& top_p);

}  // namespace xllm::kernel::npu
