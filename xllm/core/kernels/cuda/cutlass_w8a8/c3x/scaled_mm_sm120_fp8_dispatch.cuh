/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ===========================================================================*/

// ref to:
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/c3x/scaled_mm_sm120_fp8_dispatch.cuh

#pragma once

// clang-format off
#include "scaled_mm.cuh"
#include "sm120_fp8_kernel_configs.cuh"
#include "sm120_fp8_dispatch_policy.hpp"
// clang-format on

namespace xllm {
namespace kernel {
namespace cuda {

// =============================================================================
// SM120 FP8 GEMM Caller with swap_ab Support
// =============================================================================
template <typename Gemm, typename... EpilogueArgs>
void cutlass_gemm_caller_sm120_fp8(torch::Tensor& out,
                                   torch::Tensor const& a,
                                   torch::Tensor const& b,
                                   EpilogueArgs&&... epilogue_params) {
  constexpr bool swap_ab = Gemm::swap_ab;
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;
  using GemmKernel = typename Gemm::GemmKernel;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;

  const int32_t m = a.size(0);
  const int32_t n = b.size(1);
  const int32_t k = a.size(1);

  // Problem shape: swap M and N dimensions when using swap_ab
  auto prob_shape =
      swap_ab ? cute::make_shape(n, m, k, 1) : cute::make_shape(m, n, k, 1);

  auto a_stride =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  auto b_stride =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  auto c_stride = cutlass::make_cute_packed_stride(
      StrideC{},
      swap_ab ? cute::make_shape(n, m, 1) : cute::make_shape(m, n, 1));

  auto* a_ptr = static_cast<ElementAB*>(a.data_ptr());
  auto* b_ptr = static_cast<ElementAB*>(b.data_ptr());
  auto* c_ptr = static_cast<ElementD*>(out.data_ptr());

  // Swap operands A and B when swap_ab is enabled
  typename GemmKernel::MainloopArguments mainloop_args =
      swap_ab ? typename GemmKernel::MainloopArguments{b_ptr,
                                                       b_stride,
                                                       a_ptr,
                                                       a_stride}
              : typename GemmKernel::MainloopArguments{
                    a_ptr, a_stride, b_ptr, b_stride};

  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(
          std::forward<EpilogueArgs>(epilogue_params)...),
      c_ptr,
      c_stride,
      c_ptr,
      c_stride};

  c3x::cutlass_gemm_caller<GemmKernel>(
      a.device(), prob_shape, mainloop_args, epilogue_args);
}

// =============================================================================
// SM120 FP8 Dispatch: Select Optimal Config Based on Problem Shape
// =============================================================================
// Strategy:
// 1. Small M (<=16): swap_ab + 128x32 tile
// 2. Medium M (17-128) with large N/K: swap_ab + 128x64 tile
// 3. Medium M (17-128) with small N/K: 128x64 tile (no swap)
// 4. Large M (>128): Select 128x64 vs 128x128 based on wave efficiency

template <typename InType,
          typename OutType,
          bool EnableBias,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm120_fp8_dispatch(torch::Tensor& out,
                                            torch::Tensor const& a,
                                            torch::Tensor const& b,
                                            torch::Tensor const& a_scales,
                                            torch::Tensor const& b_scales,
                                            EpilogueArgs&&... args) {
  static_assert(std::is_same_v<InType, cutlass::float_e4m3_t>);
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  // Kernel configurations
  using GemmDefault =
      typename sm120_fp8_config_default<InType, OutType, EnableBias>::
          Cutlass3xGemm;
  using Gemm128x64 =
      typename sm120_fp8_config_tile_128x64<InType, OutType, EnableBias>::
          Cutlass3xGemm;
  using Gemm128x64Swap =
      typename sm120_fp8_config_tile_128x64_swap<InType, OutType, EnableBias>::
          Cutlass3xGemm;
  using Gemm128x32Swap =
      typename sm120_fp8_config_tile_128x32_swap<InType, OutType, EnableBias>::
          Cutlass3xGemm;

  const uint32_t m = a.size(0);
  const uint32_t n = b.size(1);
  const uint32_t k = a.size(1);
  const auto decision =
      select_sm120_dispatch_for_device(m, n, k, a.device().index());

  switch (decision.kernel) {
    case SM120DispatchKernel::kTile128x32Swap:
      return cutlass_gemm_caller_sm120_fp8<Gemm128x32Swap>(
          out, a, b, b_scales, a_scales, std::forward<EpilogueArgs>(args)...);
    case SM120DispatchKernel::kTile128x64Swap:
      return cutlass_gemm_caller_sm120_fp8<Gemm128x64Swap>(
          out, a, b, b_scales, a_scales, std::forward<EpilogueArgs>(args)...);
    case SM120DispatchKernel::kTile128x64:
      return cutlass_gemm_caller_sm120_fp8<Gemm128x64>(
          out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
    case SM120DispatchKernel::kTile128x128:
      return cutlass_gemm_caller_sm120_fp8<GemmDefault>(
          out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
  }
  TORCH_CHECK(false, "Unsupported SM120 dispatch kernel");
}

// =============================================================================
// SM120 FP8 Scaled MM Entry Point
// =============================================================================
template <bool EnableBias, typename... EpilogueArgs>
void cutlass_scaled_mm_sm120_fp8_epilogue(torch::Tensor& out,
                                          torch::Tensor const& a,
                                          torch::Tensor const& b,
                                          torch::Tensor const& a_scales,
                                          torch::Tensor const& b_scales,
                                          EpilogueArgs&&... epilogue_args) {
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  if (out.dtype() == torch::kBFloat16) {
    cutlass_gemm_sm120_fp8_dispatch<cutlass::float_e4m3_t,
                                    cutlass::bfloat16_t,
                                    EnableBias>(
        out,
        a,
        b,
        a_scales,
        b_scales,
        std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    cutlass_gemm_sm120_fp8_dispatch<cutlass::float_e4m3_t,
                                    cutlass::half_t,
                                    EnableBias>(
        out,
        a,
        b,
        a_scales,
        b_scales,
        std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm
