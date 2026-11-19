/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#pragma once

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_,
          class BlockEpilogue_,
          class BlockScheduler_,
          uint32_t WORKSPACE_STAGES_>
class QuantMatmulNzWorkspace {
 public:
  using BlockMmad = BlockMmad_;
  using ArchTag = typename BlockMmad::ArchTag;
  using L1TileShape = typename BlockMmad::L1TileShape;
  using ElementA = typename BlockMmad::ElementA;
  using LayoutA = typename BlockMmad::LayoutA;
  using ElementB = typename BlockMmad::ElementB;
  using LayoutB = typename BlockMmad::LayoutB;
  using ElementC = typename BlockMmad::ElementC;
  using LayoutC = typename BlockMmad::LayoutC;
  using ElementAccumulator = typename BlockMmad::ElementAccumulator;

  using BlockEpilogue = BlockEpilogue_;
  using ElementScale = typename BlockEpilogue::ElementScale;
  using LayoutScale = typename BlockEpilogue::LayoutScale;
  using ElementPerTokenScale = typename BlockEpilogue::ElementPerTokenScale;
  using LayoutPerTokenScale = typename BlockEpilogue::LayoutPerTokenScale;
  using ElementD = typename BlockEpilogue::ElementD;
  using LayoutD = typename BlockEpilogue::LayoutD;
  using EpilogueParams = typename BlockEpilogue::Params;

  using BlockScheduler = BlockScheduler_;
  static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

  /// Parameters structure
  struct Params {
    // Data members
    GemmCoord problemShape;
    __gm__ ElementA* ptrA;
    LayoutA layoutA;
    __gm__ ElementB* ptrB;
    LayoutB layoutB;
    __gm__ ElementScale* ptrScale;
    LayoutScale layoutScale;
    __gm__ ElementPerTokenScale* ptrPerTokenScale;
    LayoutPerTokenScale layoutPerTokenScale;
    __gm__ int32_t* ptrBias;
    __gm__ ElementD* ptrD;
    LayoutD layoutD;
    GM_ADDR ptrWorkspace;

    // Methods
    CATLASS_DEVICE
    Params() {}

    CATLASS_DEVICE
    Params(GemmCoord problemShape_,
           GM_ADDR ptrA_,
           LayoutA layoutA_,
           GM_ADDR ptrB_,
           LayoutB layoutB_,
           GM_ADDR ptrScale_,
           LayoutScale layoutScale_,
           GM_ADDR ptrPerTokenScale_,
           LayoutPerTokenScale layoutPerTokenScale_,
           GM_ADDR ptrBias_,
           GM_ADDR ptrD_,
           LayoutD layoutD_,
           GM_ADDR ptrWorkspace_)
        : problemShape(problemShape_),
          ptrA(reinterpret_cast<__gm__ ElementA*>(ptrA_)),
          layoutA(layoutA_),
          ptrB(reinterpret_cast<__gm__ ElementB*>(ptrB_)),
          layoutB(layoutB_),
          ptrScale(reinterpret_cast<__gm__ ElementScale*>(ptrScale_)),
          layoutScale(layoutScale_),
          ptrPerTokenScale(reinterpret_cast<__gm__ ElementPerTokenScale*>(
              ptrPerTokenScale_)),
          layoutPerTokenScale(layoutPerTokenScale_),
          ptrBias(reinterpret_cast<__gm__ int32_t*>(ptrBias_)),
          ptrD(reinterpret_cast<__gm__ ElementD*>(ptrD_)),
          layoutD(layoutD_),
          ptrWorkspace(ptrWorkspace_) {}
  };

  // Methods
  CATLASS_DEVICE
  QuantMatmulNzWorkspace() {
    Arch::FlagID flagId = 0;
    for (uint32_t stageId = 0; stageId < WORKSPACE_STAGES; ++stageId) {
      flagAicFinishStoreList[stageId] = Arch::CrossCoreFlag(flagId++);
      flagAivFinishComputeList[stageId] = Arch::CrossCoreFlag(flagId++);
      aicWaitFuncList[stageId] = {this, stageId};
      aicSetFuncList[stageId] = {this, stageId};
    }
  }

  template <int32_t CORE_TYPE = g_coreType>
  CATLASS_DEVICE void operator()(Params const& params);

  template <>
  CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params) {
    BlockScheduler blockScheduler;
    blockScheduler.Update(params.problemShape,
                          MakeCoord(L1TileShape::M, L1TileShape::N));
    uint32_t coreLoops = blockScheduler.GetCoreLoops();

    BlockMmad blockMmad(resource);

    // Represent the full gm
    AscendC::GlobalTensor<ElementA> gmA;
    gmA.SetGlobalBuffer(params.ptrA);
    AscendC::GlobalTensor<ElementB> gmB;
    gmB.SetGlobalBuffer(params.ptrB);
    if (params.problemShape.n() != 1280) {
      gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }

    uint32_t coreIdx = AscendC::GetBlockIdx();
    uint32_t coreNum = AscendC::GetBlockNum();

    AscendC::GlobalTensor<ElementC> gmC;
    gmC.SetGlobalBuffer(
        reinterpret_cast<__gm__ ElementC*>(params.ptrWorkspace));
    auto layoutC = layout::RowMajor{L1TileShape::M * coreNum * WORKSPACE_STAGES,
                                    L1TileShape::N};

    uint32_t stageId = 0;
    uint32_t stageUsed = 0;

    // Loop through the matmul of each groupIdx
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
      // Compute block location
      GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
      GemmCoord actualBlockShape =
          blockScheduler.GetActualBlockShape(blockCoord);

      Callback callbackBeforeFixpipe{};
      if (stageUsed == WORKSPACE_STAGES) {
        callbackBeforeFixpipe = MakeCallback(&aicWaitFuncList[stageId]);
      } else {
        ++stageUsed;
      }
      Callback callbackAfterFixpipe = MakeCallback(&aicSetFuncList[stageId]);

      // Compute initial location in logical coordinates
      MatrixCoord offsetA{blockCoord.m() * L1TileShape::M,
                          blockCoord.k() * L1TileShape::K};
      MatrixCoord offsetB{blockCoord.k() * L1TileShape::K,
                          blockCoord.n() * L1TileShape::N};
      MatrixCoord offsetC{(stageId * coreNum + coreIdx) * L1TileShape::M, 0};
      int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
      int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
      int64_t gmOffsetC = layoutC.GetOffset(offsetC);

      // Compute block-scoped matrix multiply-add
      if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
        blockMmad(gmA[gmOffsetA],
                  params.layoutA,
                  gmB[gmOffsetB],
                  params.layoutB,
                  gmC[gmOffsetC],
                  layoutC,
                  actualBlockShape,
                  callbackBeforeFixpipe,
                  callbackAfterFixpipe);
      } else {
        callbackBeforeFixpipe();
        blockMmad(gmA[gmOffsetA],
                  params.layoutA,
                  gmB[gmOffsetB],
                  params.layoutB,
                  gmC[gmOffsetC],
                  layoutC,
                  actualBlockShape);
        callbackAfterFixpipe();
      }

      stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
    }

    if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
      blockMmad.SynchronizeBlock();
    }

    while (stageUsed > 0) {
      uint32_t aivComputeStageId =
          (stageId >= stageUsed) ? (stageId - stageUsed)
                                 : (stageId + WORKSPACE_STAGES - stageUsed);
      Arch::CrossCoreWaitFlag(flagAivFinishComputeList[aivComputeStageId]);
      --stageUsed;
    }
  }

  template <>
  CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params) {
    BlockScheduler blockScheduler;
    uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    uint32_t coreNum = AscendC::GetBlockNum();

    AscendC::GlobalTensor<ElementC> gmC;
    gmC.SetGlobalBuffer(
        reinterpret_cast<__gm__ ElementC*>(params.ptrWorkspace));
    auto layoutC = layout::RowMajor{L1TileShape::M * coreNum * WORKSPACE_STAGES,
                                    L1TileShape::N};

    uint32_t stageId = 0;

    blockScheduler.Update(params.problemShape, L1TileShape::ToCoordMN());
    uint32_t coreLoops = blockScheduler.GetCoreLoops();

    GemmCoord blockShapeMNK = L1TileShape::ToCoord();
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
      GemmCoord blockCoordMNK = blockScheduler.GetBlockCoord(loopIdx);
      GemmCoord actualBlockShapeMNK =
          blockScheduler.GetActualBlockShape(blockCoordMNK);

      MatrixCoord offsetC{(stageId * coreNum + coreIdx) * L1TileShape::M, 0};
      int64_t gmOffsetC = layoutC.GetOffset(offsetC);
      auto gmBlockC = gmC[gmOffsetC];

      Arch::CrossCoreWaitFlag(flagAicFinishStoreList[stageId]);
      static_epilogue(params, blockCoordMNK, actualBlockShapeMNK, gmBlockC);
      Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComputeList[stageId]);

      stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
    }
  }

 private:
  CATLASS_DEVICE
  void static_epilogue(Params const& params,
                       GemmCoord const& block_coord,
                       GemmCoord const& actual_block_shape,
                       AscendC::GlobalTensor<ElementC> const& gm_block_c) {
    constexpr bool kSupportsTwoDimensionalEpilogue =
        L1TileShape::N == 128 || L1TileShape::N == 160 || L1TileShape::N == 256;
    constexpr uint32_t kMinTwoDimensionalRows = L1TileShape::N == 256 ? 8 : 2;
    if constexpr (kSupportsTwoDimensionalEpilogue) {
      if (actual_block_shape.m() >= kMinTwoDimensionalRows &&
          actual_block_shape.n() == L1TileShape::N) {
        if (AscendC::GetSubBlockIdx() == 0) {
          static_epilogue_two_dimensional(
              params, block_coord, actual_block_shape, gm_block_c);
        }
        return;
      }
    }
    static_epilogue_row(params, block_coord, actual_block_shape, gm_block_c);
  }

  CATLASS_DEVICE
  void static_epilogue_row(Params const& params,
                           GemmCoord const& blockCoord,
                           GemmCoord const& actualBlockShape,
                           AscendC::GlobalTensor<ElementC> const& gmBlockC) {
    constexpr uint32_t kEpilogueTileN = L1TileShape::N == 320 ? 320 : 256;
    uint32_t subblockIdx = AscendC::GetSubBlockIdx();
    uint32_t tileNOffset = subblockIdx * kEpilogueTileN;
    if (tileNOffset >= actualBlockShape.n()) {
      return;
    }
    uint32_t tileN = actualBlockShape.n() - tileNOffset;
    tileN = tileN < kEpilogueTileN ? tileN : kEpilogueTileN;
    uint32_t blockNOffset = blockCoord.n() * L1TileShape::N;

    size_t ubOffset = 0;
    auto ubC = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
    ubOffset += kEpilogueTileN * sizeof(int32_t);
    auto ubBias = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
    ubOffset += kEpilogueTileN * sizeof(int32_t);
    auto ubScale = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
    ubOffset += kEpilogueTileN * sizeof(float);
    auto ubFloat = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
    ubOffset += kEpilogueTileN * sizeof(float);
    auto ubD = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);

    AscendC::GlobalTensor<int32_t> gmBias;
    gmBias.SetGlobalBuffer(params.ptrBias);
    AscendC::GlobalTensor<ElementScale> gmScale;
    gmScale.SetGlobalBuffer(params.ptrScale);
    AscendC::GlobalTensor<ElementD> gmD;
    gmD.SetGlobalBuffer(params.ptrD);

    AscendC::DataCopy(ubBias, gmBias[blockNOffset + tileNOffset], tileN);
    AscendC::DataCopy(ubScale, gmScale[blockNOffset + tileNOffset], tileN);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

    for (uint32_t row = 0; row < actualBlockShape.m(); ++row) {
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
      AscendC::DataCopy(
          ubC, gmBlockC[row * L1TileShape::N + tileNOffset], tileN);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      AscendC::Add(ubC, ubC, ubBias, tileN);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(ubFloat, ubC, AscendC::RoundMode::CAST_RINT, tileN);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(ubFloat, ubFloat, ubScale, tileN);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
      AscendC::Cast(ubD, ubFloat, AscendC::RoundMode::CAST_RINT, tileN);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      AscendC::DataCopy(
          gmD[row * params.problemShape.n() + blockNOffset + tileNOffset],
          ubD,
          tileN);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
  }

  CATLASS_DEVICE
  void static_epilogue_two_dimensional(
      Params const& params,
      GemmCoord const& block_coord,
      GemmCoord const& actual_block_shape,
      AscendC::GlobalTensor<ElementC> const& gm_block_c) {
    constexpr uint32_t kEpilogueTileN = 256;
    constexpr uint32_t kMaxRows = 16;
    constexpr uint32_t kMaxElements = kMaxRows * kEpilogueTileN;
    constexpr uint32_t kDataBlockBytes = 32;
    constexpr uint32_t kVectorRepeatElements = 64;

    const uint32_t row_count = actual_block_shape.m();
    const uint32_t tile_n = actual_block_shape.n();
    const uint32_t element_count = row_count * tile_n;
    const uint32_t block_n_offset = block_coord.n() * L1TileShape::N;

    size_t ub_offset = 0;
    auto ub_c = resource.ubBuf.template GetBufferByByte<int32_t>(ub_offset);
    ub_offset += kMaxElements * sizeof(int32_t);
    auto ub_bias = resource.ubBuf.template GetBufferByByte<int32_t>(ub_offset);
    ub_offset += kEpilogueTileN * sizeof(int32_t);
    auto ub_scale = resource.ubBuf.template GetBufferByByte<float>(ub_offset);
    ub_offset += kEpilogueTileN * sizeof(float);
    auto ub_float = resource.ubBuf.template GetBufferByByte<float>(ub_offset);
    ub_offset += kMaxElements * sizeof(float);
    auto ub_d = resource.ubBuf.template GetBufferByByte<ElementD>(ub_offset);

    AscendC::GlobalTensor<int32_t> gm_bias;
    gm_bias.SetGlobalBuffer(params.ptrBias);
    AscendC::GlobalTensor<ElementScale> gm_scale;
    gm_scale.SetGlobalBuffer(params.ptrScale);
    AscendC::GlobalTensor<ElementD> gm_d;
    gm_d.SetGlobalBuffer(params.ptrD);

    const uint16_t accumulator_burst_length =
        static_cast<uint16_t>(tile_n * sizeof(int32_t) / kDataBlockBytes);
    const uint16_t accumulator_source_gap = static_cast<uint16_t>(
        (L1TileShape::N - tile_n) * sizeof(int32_t) / kDataBlockBytes);
    AscendC::DataCopy(ub_bias, gm_bias[block_n_offset], tile_n);
    AscendC::DataCopy(ub_scale, gm_scale[block_n_offset], tile_n);
    AscendC::DataCopy(ub_c,
                      gm_block_c,
                      AscendC::DataCopyParams{static_cast<uint16_t>(row_count),
                                              accumulator_burst_length,
                                              accumulator_source_gap,
                                              0});
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

    const uint8_t row_stride = static_cast<uint8_t>(accumulator_burst_length);
    const AscendC::BinaryRepeatParams broadcast_params{
        1, 1, 1, row_stride, row_stride, 0};
    for (uint32_t column_offset = 0; column_offset < tile_n;
         column_offset += kVectorRepeatElements) {
      const uint32_t vector_elements =
          tile_n - column_offset < kVectorRepeatElements
              ? tile_n - column_offset
              : kVectorRepeatElements;
      AscendC::Add(ub_c[column_offset],
                   ub_c[column_offset],
                   ub_bias[column_offset],
                   vector_elements,
                   static_cast<uint8_t>(row_count),
                   broadcast_params);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(ub_float, ub_c, AscendC::RoundMode::CAST_RINT, element_count);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t column_offset = 0; column_offset < tile_n;
         column_offset += kVectorRepeatElements) {
      const uint32_t vector_elements =
          tile_n - column_offset < kVectorRepeatElements
              ? tile_n - column_offset
              : kVectorRepeatElements;
      AscendC::Mul(ub_float[column_offset],
                   ub_float[column_offset],
                   ub_scale[column_offset],
                   vector_elements,
                   static_cast<uint8_t>(row_count),
                   broadcast_params);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(ub_d, ub_float, AscendC::RoundMode::CAST_RINT, element_count);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    const uint16_t output_burst_length =
        static_cast<uint16_t>(tile_n * sizeof(ElementD) / kDataBlockBytes);
    const uint16_t output_destination_gap =
        static_cast<uint16_t>((params.problemShape.n() - tile_n) *
                              sizeof(ElementD) / kDataBlockBytes);
    AscendC::DataCopy(gm_d[block_n_offset],
                      ub_d,
                      AscendC::DataCopyParams{static_cast<uint16_t>(row_count),
                                              output_burst_length,
                                              0,
                                              output_destination_gap});
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
  }

  friend struct AicWaitFunc;
  friend struct AicSetFunc;

  struct AicWaitFunc {
    using MatmulKernel = QuantMatmulNzWorkspace<BlockMmad,
                                                BlockEpilogue,
                                                BlockScheduler,
                                                WORKSPACE_STAGES>;

    CATLASS_DEVICE
    AicWaitFunc() = default;

    CATLASS_DEVICE
    void operator()() const {
      Arch::CrossCoreWaitFlag(ptr->flagAivFinishComputeList[stageId]);
    }

    MatmulKernel* ptr{nullptr};
    uint32_t stageId;
  };

  struct AicSetFunc {
    using MatmulKernel = QuantMatmulNzWorkspace<BlockMmad,
                                                BlockEpilogue,
                                                BlockScheduler,
                                                WORKSPACE_STAGES>;

    CATLASS_DEVICE
    AicSetFunc() = default;

    CATLASS_DEVICE
    void operator()() const {
      Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(
          ptr->flagAicFinishStoreList[stageId]);
    }

    MatmulKernel* ptr{nullptr};
    uint32_t stageId;
  };

  Arch::CrossCoreFlag flagAicFinishStoreList[WORKSPACE_STAGES];
  Arch::CrossCoreFlag flagAivFinishComputeList[WORKSPACE_STAGES];

  AicWaitFunc aicWaitFuncList[WORKSPACE_STAGES];
  AicSetFunc aicSetFuncList[WORKSPACE_STAGES];
  Arch::Resource<ArchTag> resource;
};

}  // namespace Catlass::Gemm::Kernel
