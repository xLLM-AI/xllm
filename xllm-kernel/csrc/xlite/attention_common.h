/* Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. */

#pragma once
#include "csrc/xlite/common.h"
namespace xllm_kernel::xlite {
inline void check_metadata(const torch::Tensor& value,
                           const torch::Tensor& reference,
                           int64_t rank) {
  CHECK(value.device() == reference.device() &&
        value.scalar_type() == torch::kInt32);
  CHECK(value.dim() == rank && value.is_contiguous());
  CHECK(value.numel() <= std::numeric_limits<int32_t>::max());
  CHECK(at_npu::native::get_npu_format(value) == ACL_FORMAT_ND);
}
}  // namespace xllm_kernel::xlite
