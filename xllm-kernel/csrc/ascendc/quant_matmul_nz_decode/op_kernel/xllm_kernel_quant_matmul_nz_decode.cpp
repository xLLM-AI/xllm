/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_operator.h"
#include "quant_matmul_nz_workspace.hpp"

using namespace Catlass;

namespace quant_matmul_nz_decode_detail {

// Keep the 16-row decode tile's relaxed M alignment local to this operator.
template <uint32_t kTileN, uint32_t kTileK>
struct DecodeGemmShape : public GemmShape<16, kTileN, kTileK> {};

// QuantMatmulNzWorkspace requires compile-time tile metadata, while the
// official dynamic block receives those same tile shapes in its constructor.
template <class BlockMmad_, class L1TileShape_, class L0TileShape_>
class FixedShapeDynamicBlockMmad : public BlockMmad_ {
 public:
  using DynamicBlockMmad = BlockMmad_;
  using L1TileShape = L1TileShape_;
  using L0TileShape = L0TileShape_;
  using ArchTag = typename DynamicBlockMmad::ArchTag;

  CATLASS_DEVICE explicit FixedShapeDynamicBlockMmad(
      Arch::Resource<ArchTag>& resource)
      : DynamicBlockMmad(L1TileShape::ToCoord(),
                         L0TileShape::ToCoord(),
                         resource) {}
};

}  // namespace quant_matmul_nz_decode_detail

namespace Catlass::Gemm::helper {

template <uint32_t kL1TileN,
          uint32_t kL1TileK,
          uint32_t kL0TileN,
          uint32_t kL0TileK>
struct TileShapeAlignChecker<
    quant_matmul_nz_decode_detail::DecodeGemmShape<kL1TileN, kL1TileK>,
    quant_matmul_nz_decode_detail::DecodeGemmShape<kL0TileN, kL0TileK>,
    int8_t,
    int8_t,
    32> {
  static constexpr uint32_t _ALIGN = 256;

  static_assert(kL1TileK * SizeOfBits<int8_t>::value % _ALIGN == 0,
                "L1 K tile must be 32-byte aligned");
  static_assert(kL0TileK * SizeOfBits<int8_t>::value % _ALIGN == 0,
                "L0 K tile must be 32-byte aligned");
  static_assert(kL1TileN * SizeOfBits<int8_t>::value % _ALIGN == 0,
                "L1 N tile must be 32-byte aligned");
  static_assert(kL0TileN * SizeOfBits<int8_t>::value % _ALIGN == 0,
                "L0 N tile must be 32-byte aligned");
};

}  // namespace Catlass::Gemm::helper

template <uint32_t kL1TileN,
          uint32_t kL1TileK,
          uint32_t kL0TileN = 128,
          uint32_t kL0TileK = 256,
          bool kEnableShuffleK = false,
          uint32_t kProblemM = 0,
          uint32_t kProblemK = 0,
          uint32_t kProblemN = 0>
__aicore__ inline void RunQuantMatmulNzDecode(GM_ADDR x,
                                              GM_ADDR weight,
                                              GM_ADDR scale,
                                              GM_ADDR bias,
                                              GM_ADDR y,
                                              GM_ADDR workspace,
                                              uint32_t m,
                                              uint32_t k,
                                              uint32_t n) {
  using ArchTag = Arch::AtlasA2;
  static constexpr bool kUseDynamicTile = kL1TileN != kL0TileN;
  using L1TileShape =
      quant_matmul_nz_decode_detail::DecodeGemmShape<kL1TileN, kL1TileK>;
  using L0TileShape =
      quant_matmul_nz_decode_detail::DecodeGemmShape<kL0TileN, kL0TileK>;
  using AType = Gemm::GemmType<int8_t, layout::RowMajor>;
  using BType = Gemm::GemmType<int8_t, layout::zN>;
  using CType = Gemm::GemmType<int32_t, layout::RowMajor>;
  using DynamicDispatchPolicy =
      Gemm::MmadAtlasA2DynamicPreloadAsyncWithCallback<1,
                                                       2,
                                                       2,
                                                       2,
                                                       1,
                                                       false,
                                                       kEnableShuffleK>;
  using StaticDispatchPolicy =
      Gemm::MmadAtlasA2PreloadAsyncWithCallback<1,
                                                2,
                                                2,
                                                2,
                                                1,
                                                false,
                                                kEnableShuffleK>;
  using DispatchPolicy = std::conditional_t<kUseDynamicTile,
                                            DynamicDispatchPolicy,
                                            StaticDispatchPolicy>;
  using BaseBlockMmad = Gemm::Block::
      BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
  using DynamicBlockMmad = quant_matmul_nz_decode_detail::
      FixedShapeDynamicBlockMmad<BaseBlockMmad, L1TileShape, L0TileShape>;
  using BlockMmad =
      std::conditional_t<kUseDynamicTile, DynamicBlockMmad, BaseBlockMmad>;

  using EpiloguePolicy = Epilogue::EpilogueAtlasA2PerTokenDequant<2>;
  using ScaleType = Gemm::GemmType<float, layout::VectorLayout>;
  using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
  using DType = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
  using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
  using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
  using ColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
  using EpilogueTileShape = MatrixShape<16, 256>;
  using TileRowBroadcastMul = Epilogue::Tile::
      TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
  using TileBroadcastOneBlk = Epilogue::Tile::
      TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
  using TileColumnBroadcastMul =
      Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag,
                                                   ColumnBroadcastMulType,
                                                   EpilogueTileShape>;
  using TileCopy = Epilogue::Tile::
      TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
  using EpilogueScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;
  using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpiloguePolicy,
                                                       CType,
                                                       ScaleType,
                                                       PerTokenScaleType,
                                                       DType,
                                                       TileRowBroadcastMul,
                                                       TileBroadcastOneBlk,
                                                       TileColumnBroadcastMul,
                                                       TileCopy,
                                                       EpilogueScheduler>;
  using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
  using MatmulKernel = Gemm::Kernel::
      QuantMatmulNzWorkspace<BlockMmad, BlockEpilogue, BlockScheduler, 2>;

  const uint32_t problem_m = kProblemM == 0 ? m : kProblemM;
  const uint32_t problem_k = kProblemK == 0 ? k : kProblemK;
  const uint32_t problem_n = kProblemN == 0 ? n : kProblemN;
  GemmCoord problem_shape{problem_m, problem_n, problem_k};
  layout::RowMajor layout_x{problem_m, problem_k};
  auto layout_weight = layout::zN::MakeLayout<int8_t>(problem_k, problem_n);
  layout::VectorLayout layout_scale{problem_n};
  layout::VectorLayout unused_layout{problem_m};
  layout::RowMajor layout_y{problem_m, problem_n};
  typename MatmulKernel::Params params{problem_shape,
                                       x,
                                       layout_x,
                                       weight,
                                       layout_weight,
                                       scale,
                                       layout_scale,
                                       scale,
                                       unused_layout,
                                       bias,
                                       y,
                                       layout_y,
                                       AscendC::GetUserWorkspace(workspace)};
  MatmulKernel matmul;
  matmul(params);
}

extern "C" __global__ __aicore__ void xllm_kernel_quant_matmul_nz_decode(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR scale,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  GET_TILING_DATA(tiling_data, tiling);

  const uint32_t m = tiling_data.m;
  const uint32_t k = tiling_data.k;
  const uint32_t n = tiling_data.n;
  if (TILING_KEY_IS(6)) {
    RunQuantMatmulNzDecode<320, 512, 128, 256, false, 1, 5120, 6400>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(7)) {
    RunQuantMatmulNzDecode<320, 512, 128, 256, false, 2, 5120, 6400>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(8)) {
    RunQuantMatmulNzDecode<160, 512, 128, 256, false, 4, 5120, 6400>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(9)) {
    RunQuantMatmulNzDecode<160, 512, 128, 256, false, 8, 5120, 6400>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(10)) {
    RunQuantMatmulNzDecode<160, 512, 128, 256, false, 16, 5120, 6400>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(11)) {
    RunQuantMatmulNzDecode<128, 1536, 128, 256, true, 1, 5120, 1280>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(12)) {
    RunQuantMatmulNzDecode<128, 1536, 128, 256, true, 2, 5120, 1280>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(13)) {
    RunQuantMatmulNzDecode<128, 1536, 128, 256, true, 4, 5120, 1280>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(14)) {
    RunQuantMatmulNzDecode<128, 1536, 128, 256, true, 8, 5120, 1280>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(15)) {
    RunQuantMatmulNzDecode<128, 1536, 128, 256, true, 16, 5120, 1280>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(0)) {
    RunQuantMatmulNzDecode<320, 512>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(5)) {
    RunQuantMatmulNzDecode<160, 512>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(1)) {
    RunQuantMatmulNzDecode<64, 3200, 64, 512, true>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(3)) {
    RunQuantMatmulNzDecode<128, 1792, 128, 256, true>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(2)) {
    RunQuantMatmulNzDecode<256, 896, 128, 256, true>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(4)) {
    RunQuantMatmulNzDecode<128, 1536, 128, 256, true>(
        x, weight, scale, bias, y, workspace, m, k, n);
  }
}

// The generated mixed-core wrapper clears workspace through matmul helpers.
#include "lib/matmul_intf.h"

namespace AscendC {

__aicore__ inline void PrepareQuantMatmulNzMixedCoreWorkspace(__gm__ uint8_t*) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
  SetAtomicNone();
  SetMaskNorm();
  SetLoadDataBoundary(static_cast<uint64_t>(0));
  SetLoadDataPaddingValue(static_cast<uint64_t>(0));
  NotifyEvent<PIPE_MTE3>(WORKSPACE_SYNC_ID);
#endif
}

}  // namespace AscendC

// CATLASS uses FFTS cross-core flags directly and does not use the KFC message
// queues. Preserve the wrapper's completion notification, but skip clearing
// the unused 30 KiB KFC region on every AIC.
#define clearWorkspace(workspace) \
  PrepareQuantMatmulNzMixedCoreWorkspace(workspace)
