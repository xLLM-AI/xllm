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

#pragma once

// clang-format off
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"
#include "cutlass_gemm_caller.cuh"
// clang-format on

namespace xllm {
namespace kernel {
namespace cuda {

template <typename ElementAB_,
          typename ElementD_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape,
          typename ClusterShape,
          typename KernelSchedule,
          typename EpilogueSchedule,
          bool swap_ab_ = false>
struct cutlass_3x_gemm_sm120_fp8 {
  using ElementAB = ElementAB_;
  using ElementC = ElementD_;
  using ElementD = ElementD_;
  using ElementAcc =
      std::conditional_t<std::is_same_v<ElementAB, int8_t>, int32_t, float>;

  using Epilogue = Epilogue_<ElementAcc, ElementD, TileShape>;
  using EVTCompute = typename Epilogue::EVTCompute;

  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD =
      128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr bool swap_ab = swap_ab_;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using LayoutC = LayoutD;

  using LayoutA_T = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_T = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
  using LayoutC_T = typename cutlass::layout::LayoutTranspose<LayoutC>::type;
  using LayoutD_T = typename cutlass::layout::LayoutTranspose<LayoutD>::type;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm120,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAcc,
          float,
          ElementC,
          conditional_t<swap_ab, LayoutC_T, LayoutC>,
          AlignmentCD,
          ElementD,
          conditional_t<swap_ab, LayoutD_T, LayoutD>,
          AlignmentCD,
          EpilogueSchedule,
          EVTCompute>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);
  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(CEStorageSize)>;

  using CollectiveMainloop =
      conditional_t<swap_ab,
                    typename cutlass::gemm::collective::CollectiveBuilder<
                        cutlass::arch::Sm120,
                        cutlass::arch::OpClassTensorOp,
                        ElementAB,
                        LayoutB_T,
                        AlignmentAB,
                        ElementAB,
                        LayoutA_T,
                        AlignmentAB,
                        ElementAcc,
                        TileShape,
                        ClusterShape,
                        Stages,
                        KernelSchedule>::CollectiveOp,
                    typename cutlass::gemm::collective::CollectiveBuilder<
                        cutlass::arch::Sm120,
                        cutlass::arch::OpClassTensorOp,
                        ElementAB,
                        LayoutA,
                        AlignmentAB,
                        ElementAB,
                        LayoutB,
                        AlignmentAB,
                        ElementAcc,
                        TileShape,
                        ClusterShape,
                        Stages,
                        KernelSchedule>::CollectiveOp>;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                           CollectiveMainloop,
                                           CollectiveEpilogue,
                                           void>;
};

template <typename InType,
          typename OutType,
          bool EnableBias,
          int TileM_,
          int TileN_,
          int TileK_,
          bool SwapAB_>
struct sm120_fp8_config_generic {
  static_assert(std::is_same_v<InType, cutlass::float_e4m3_t>,
                "InType must be float_e4m3_t");

  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileShape =
      Shape<cute::Int<TileM_>, cute::Int<TileN_>, cute::Int<TileK_>>;
  using ClusterShape = Shape<_1, _1, _1>;

  static constexpr bool SwapAB = SwapAB_;

  using Cutlass3xGemm = conditional_t<
      EnableBias,
      conditional_t<SwapAB,
                    cutlass_3x_gemm_sm120_fp8<InType,
                                              OutType,
                                              c3x::ScaledEpilogueColumnBias,
                                              TileShape,
                                              ClusterShape,
                                              KernelSchedule,
                                              EpilogueSchedule,
                                              SwapAB>,
                    cutlass_3x_gemm_sm120_fp8<InType,
                                              OutType,
                                              c3x::ScaledEpilogueBias,
                                              TileShape,
                                              ClusterShape,
                                              KernelSchedule,
                                              EpilogueSchedule,
                                              SwapAB>>,
      cutlass_3x_gemm_sm120_fp8<InType,
                                OutType,
                                c3x::ScaledEpilogue,
                                TileShape,
                                ClusterShape,
                                KernelSchedule,
                                EpilogueSchedule,
                                SwapAB>>;
};

template <typename InType, typename OutType, bool EnableBias>
using sm120_fp8_config_default =
    sm120_fp8_config_generic<InType, OutType, EnableBias, 128, 128, 128, false>;

template <typename InType, typename OutType, bool EnableBias>
using sm120_fp8_config_tile_128x64 =
    sm120_fp8_config_generic<InType, OutType, EnableBias, 128, 64, 128, false>;

template <typename InType, typename OutType, bool EnableBias>
using sm120_fp8_config_tile_128x64_swap =
    sm120_fp8_config_generic<InType, OutType, EnableBias, 128, 64, 128, true>;

template <typename InType, typename OutType, bool EnableBias>
using sm120_fp8_config_tile_128x32_swap =
    sm120_fp8_config_generic<InType, OutType, EnableBias, 128, 32, 128, true>;

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm
