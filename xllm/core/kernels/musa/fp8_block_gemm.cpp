/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <tvm/ffi/extra/stl.h>
#include <tvm/ffi/string.h>

#include <optional>
#include <string>
#include <tuple>

#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {

// Mate groupwise FP8 GEMM layout:
//   a       [M, K]              float8_e4m3fn, contiguous
//   b       [N, K] (NT)         float8_e4m3fn, contiguous
//   a_scale [M, ceil(K/128)]    float32,       contiguous (K-major)
//   b_scale [ceil(N/128),
//            ceil(K/128)]        float32,       contiguous (K-major)
//   out     [M, N]              bf16/fp16
torch::Tensor gemm_fp8_nt_groupwise(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_scale,
    const torch::Tensor& b_scale,
    torch::ScalarType output_dtype,
    const std::optional<torch::Tensor>& output) {
  const int64_t m = a.size(0);
  const int64_t n = b.size(0);

  torch::Tensor out;
  if (output.has_value() && output.value().defined()) {
    out = output.value();
  } else {
    out = torch::empty({m, n}, a.options().dtype(output_dtype));
  }

  TvmffiStreamGuard stream_guard(a.device());

  static const std::string kGemmOpsUri = "gemm_ops";
  get_function(kGemmOpsUri, "gemm_fp8_nt_groupwise")(
      to_ffi_tensor(a),
      to_ffi_tensor(b),
      to_ffi_tensor(a_scale),
      to_ffi_tensor(b_scale),
      /*scale_major_mode=*/std::string("K"),
      /*mma_sm=*/static_cast<int64_t>(1),
      /*scale_granularity_mnk=*/
      std::make_tuple(static_cast<int64_t>(1),
                      static_cast<int64_t>(128),
                      static_cast<int64_t>(128)),
      to_ffi_tensor(out),
      /*backend=*/std::string("mudnn"));

  return out;
}

}  // namespace xllm::kernel::musa
