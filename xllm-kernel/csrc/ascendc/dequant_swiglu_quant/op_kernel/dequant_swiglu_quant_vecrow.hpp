/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file dequant_swiglu_quant_vecrow.hpp
 * \brief Vectorised-row fast path for dynamic per-token quant.
 *
 * The existing dynamic classes process one row at a time and, per row, pay a
 * ReduceMax + V_S sync + GetValue/SetValue + Muls(scalar) round-trip (see
 * DequantSwigluQuantDynamicBase::dynamicMultiColMax, which ends in
 * maxTempLocal.GetValue(rowId)). With rows in the tens of thousands that scalar
 * traffic dominates: three shapes moving identical bytes took 47/70/123us
 * purely because their row counts differ.
 *
 * Here the whole dequant -> SwiGLU -> abs-max -> quant chain runs in one
 * Register-API __VEC_SCOPE__ per tile. Each row uses two passes: the first
 * spills its computed FP32 SwiGLU values once while accumulating one amax
 * register; the second reloads those values and quantizes with the
 * register-resident inverse scale. Input tiles are double-buffered so MTE2
 * overlaps the VF work.
 *
 * Scope: dynamic quant, no bias / quant_offset / group_index. Everything else
 * keeps using the original classes.
 */

#ifndef CANN_DEQUANT_SWIGLU_QUANT_VECROW_HPP
#define CANN_DEQUANT_SWIGLU_QUANT_VECROW_HPP

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"

namespace XllmKernelDequantSwigluQuant {
using namespace AscendC;

namespace VecRow {

constexpr uint32_t UB_BLOCK_BYTES = 32;
// Row strides are padded to a whole vector register so every chunk load in the
// VF loops sits on a natural boundary.
constexpr uint32_t VL_FP32 = 64;
constexpr uint32_t MAX_TILE_ROWS = 64;

__aicore__ inline uint32_t CeilDivU(uint32_t a, uint32_t b) {
  return (b == 0) ? 0 : (a + b - 1) / b;
}

}  // namespace VecRow

/**
 * InType: int32_t (dequant path) or half/bfloat16_t (direct path).
 * hasWeightScale: int32 path multiplies by weight_scale[2H].
 */
template <typename InType, bool isInt32>
class DequantSwigluQuantVecRow {
 public:
  __aicore__ inline DequantSwigluQuantVecRow() {}
  __aicore__ inline ~DequantSwigluQuantVecRow() {}

  __aicore__ inline void Init(GM_ADDR x_gm,
                              GM_ADDR weight_scale_gm,
                              GM_ADDR activation_scale_gm,
                              GM_ADDR quant_scale_gm,
                              GM_ADDR y_gm,
                              GM_ADDR scale_gm,
                              const SwiGluTilingData* tilingData,
                              TPipe* pipe_) {
    using namespace VecRow;
    pipe = pipe_;
    curBlockIdx = GetBlockIdx();

    H = static_cast<uint32_t>(tilingData->colLen);
    rowNum = static_cast<uint32_t>(tilingData->rowLen);
    actLeft = (tilingData->activateLeft != 0);
    hasQs = (tilingData->quantScaleIsEmpty == 0);
    hasActScale = (tilingData->activateScaleIsEmpty == 0);

    useCoreNum = tilingData->usedCoreNum;
    if (rowNum < useCoreNum) {
      useCoreNum = rowNum;
    }

    // Same even split the other classes use, so core assignment matches.
    uint32_t perRoundCnt = useCoreNum == 0 ? 0 : rowNum / useCoreNum;
    uint32_t remainCnt = rowNum - useCoreNum * perRoundCnt;
    numRound = perRoundCnt;
    if (curBlockIdx < remainCnt) {
      numRound = perRoundCnt + 1;
      biasOffset = curBlockIdx * (perRoundCnt + 1);
    } else {
      biasOffset = (perRoundCnt + 1) * remainCnt +
                   (curBlockIdx - remainCnt) * perRoundCnt;
    }

    xGm.SetGlobalBuffer((__gm__ InType*)x_gm);
    yGm.SetGlobalBuffer((__gm__ int8_t*)y_gm);
    scaleGm.SetGlobalBuffer((__gm__ float*)scale_gm);
    if constexpr (isInt32) {
      weightScaleGm.SetGlobalBuffer((__gm__ float*)weight_scale_gm);
      if (hasActScale) {
        actScaleGm.SetGlobalBuffer((__gm__ float*)activation_scale_gm);
      }
    }
    if (hasQs) {
      quantScaleGm.SetGlobalBuffer((__gm__ float*)quant_scale_gm);
    }

    // The host is the sole UB-capacity authority. S is the exact Align64
    // row stride used by its Align32 peak equation.
    strideIn = CeilDivU(H, VL_FP32) * VL_FP32;
    strideF = strideIn;
    strideI8 = strideIn;
    tileRows = tilingData->tileRows;
    if (tileRows > numRound) tileRows = numRound;
    if (tileRows < 1) tileRows = 1;

    // A/B each have ping and pong halves. SwiGLU, y and scale are single
    // buffered; INT32 scales and optional quant scale are core-resident.
    pipe->InitBuffer(bufA, 2 * tileRows * strideIn * sizeof(InType));
    pipe->InitBuffer(bufB, 2 * tileRows * strideIn * sizeof(InType));
    pipe->InitBuffer(
        bufSwi, tileRows * strideF * sizeof(float) + VL_FP32 * sizeof(float));
    pipe->InitBuffer(bufY, tileRows * strideI8);
    pipe->InitBuffer(bufScale, tileRows * sizeof(float));
    if constexpr (isInt32) {
      pipe->InitBuffer(bufWsA, strideF * sizeof(float));
      pipe->InitBuffer(bufWsB, strideF * sizeof(float));
      // Ping and pong are individually aligned: tileRows * 4 is not
      // generally a legal MTE2/VEC UB base for the second half.
      const uint32_t actHalfRows =
          CeilDivU(tileRows * sizeof(float), UB_BLOCK_BYTES) * UB_BLOCK_BYTES /
          sizeof(float);
      pipe->InitBuffer(bufAct, 2 * actHalfRows * sizeof(float));
    }
    if (hasQs) {
      pipe->InitBuffer(bufQs, strideF * sizeof(float));
    }
  }

  __aicore__ inline void Process() {
    using namespace VecRow;
    if (curBlockIdx >= useCoreNum || numRound == 0) {
      return;
    }
    LocalTensor<InType> aLocal = bufA.template Get<InType>();
    LocalTensor<InType> bLocal = bufB.template Get<InType>();
    LocalTensor<float> swiLocal = bufSwi.Get<float>();
    LocalTensor<int8_t> yLocal = bufY.Get<int8_t>();
    LocalTensor<float> scaleLocal = bufScale.Get<float>();
    LocalTensor<float> wsALocal;
    LocalTensor<float> wsBLocal;
    LocalTensor<float> actLocal;

    // weight_scale / quant_scale are row-invariant: load once per core.
    if constexpr (isInt32) {
      wsALocal = bufWsA.Get<float>();
      wsBLocal = bufWsB.Get<float>();
      actLocal = bufAct.Get<float>();
      DataCopyExtParams cpWs{
          1, static_cast<uint32_t>(H * sizeof(float)), 0, 0, 0};
      DataCopyPadExtParams<float> ppWs{false, 0, 0, 0};
      DataCopyPad(wsALocal, weightScaleGm[0], cpWs, ppWs);
      DataCopyPad(wsBLocal, weightScaleGm[H], cpWs, ppWs);
    }
    if (hasQs) {
      DataCopyExtParams cpQs{
          1, static_cast<uint32_t>(H * sizeof(float)), 0, 0, 0};
      DataCopyPadExtParams<float> ppQs{false, 0, 0, 0};
      DataCopyPad(bufQs.Get<float>(), quantScaleGm[0], cpQs, ppQs);
    }
    if (isInt32 || hasQs) {
      event_t eid =
          static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
      SetFlag<HardEvent::MTE2_V>(eid);
      WaitFlag<HardEvent::MTE2_V>(eid);
    }

    __ubuf__ InType* aAddr = (__ubuf__ InType*)aLocal.GetPhyAddr();
    __ubuf__ InType* bAddr = (__ubuf__ InType*)bLocal.GetPhyAddr();
    __ubuf__ float* swiAddr = (__ubuf__ float*)swiLocal.GetPhyAddr();
    __ubuf__ int8_t* yAddr = (__ubuf__ int8_t*)yLocal.GetPhyAddr();
    __ubuf__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
    __ubuf__ float* wsAAddr = nullptr;
    __ubuf__ float* wsBAddr = nullptr;
    __ubuf__ float* actAddr = nullptr;
    if constexpr (isInt32) {
      wsAAddr = (__ubuf__ float*)wsALocal.GetPhyAddr();
      wsBAddr = (__ubuf__ float*)wsBLocal.GetPhyAddr();
      actAddr = (__ubuf__ float*)actLocal.GetPhyAddr();
    }
    __ubuf__ float* qsAddr =
        hasQs ? (__ubuf__ float*)bufQs.Get<float>().GetPhyAddr() : nullptr;

    // activate_left picks which half drives the sigmoid. Matches
    // DequantSwigluQuantDynamicBase::BaseProcess: activateLeft==0 -> gate is
    // the second half. Resolved here so the VF loops stay branch-free.
    __ubuf__ InType* gateAddr = actLeft ? aAddr : bAddr;
    __ubuf__ InType* upAddr = actLeft ? bAddr : aAddr;
    __ubuf__ float* wsGateAddr = actLeft ? wsAAddr : wsBAddr;
    __ubuf__ float* wsUpAddr = actLeft ? wsBAddr : wsAAddr;

    uint32_t dstStrideIn = (strideIn - H) * sizeof(InType) / UB_BLOCK_BYTES;
    uint32_t dstStrideY = (strideI8 - H) / UB_BLOCK_BYTES;
    uint32_t inHalf = tileRows * strideIn;
    uint32_t actHalf = CeilDivU(tileRows * sizeof(float), UB_BLOCK_BYTES) *
                       UB_BLOCK_BYTES / sizeof(float);

    DataCopyPadExtParams<InType> ppIn{false, 0, 0, 0};
    DataCopyPadExtParams<float> ppAct{false, 0, 0, 0};

    event_t eidM2V[2] = {
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V)),
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V))};
    event_t eidV2M2[2] = {
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2)),
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2))};
    bool pendingV2M2[2] = {false, false};
    event_t eidM3V =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    event_t eidV2M3 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

#define DSQ_VECROW_ISSUE_LOAD(rIdx, halfIdx)                                   \
  do {                                                                         \
    uint32_t _rows = tileRows;                                                 \
    if (_rows > numRound - (rIdx)) _rows = numRound - (rIdx);                  \
    uint32_t _start = biasOffset + (rIdx);                                     \
    DataCopyExtParams _cpIn{static_cast<uint16_t>(_rows),                      \
                            static_cast<uint32_t>(H * sizeof(InType)),         \
                            static_cast<uint32_t>(H * sizeof(InType)),         \
                            static_cast<uint32_t>(dstStrideIn),                \
                            0};                                                \
    DataCopyPad(aLocal[(halfIdx) * inHalf], xGm[_start * 2 * H], _cpIn, ppIn); \
    DataCopyPad(                                                               \
        bLocal[(halfIdx) * inHalf], xGm[_start * 2 * H + H], _cpIn, ppIn);     \
    if constexpr (isInt32) {                                                   \
      if (hasActScale) {                                                       \
        DataCopyExtParams _cpAct{                                              \
            1, static_cast<uint32_t>(_rows * sizeof(float)), 0, 0, 0};         \
        DataCopyPad(                                                           \
            actLocal[(halfIdx) * actHalf], actScaleGm[_start], _cpAct, ppAct); \
      }                                                                        \
    }                                                                          \
  } while (0)

    DSQ_VECROW_ISSUE_LOAD(0u, 0u);
    SetFlag<HardEvent::MTE2_V>(eidM2V[0]);

    uint32_t half = 0;
    for (uint32_t r = 0; r < numRound; r += tileRows) {
      uint32_t curRows = tileRows;
      if (curRows > numRound - r) curRows = numRound - r;
      uint32_t rowStart = biasOffset + r;
      uint32_t nextR = r + tileRows;
      uint32_t nextHalf = half ^ 1;

      // Start the next tile's load before touching this one, so MTE2 runs
      // underneath the VF work below. The other half is free: its compute
      // finished an iteration ago.
      if (nextR < numRound) {
        if (pendingV2M2[nextHalf]) {
          WaitFlag<HardEvent::V_MTE2>(eidV2M2[nextHalf]);
          pendingV2M2[nextHalf] = false;
        }
        DSQ_VECROW_ISSUE_LOAD(nextR, nextHalf);
        SetFlag<HardEvent::MTE2_V>(eidM2V[nextHalf]);
      }

      WaitFlag<HardEvent::MTE2_V>(eidM2V[half]);

      __ubuf__ float* tileActAddr = nullptr;
      if constexpr (isInt32) {
        tileActAddr = actAddr + half * actHalf;
      }
      ComputeRows(gateAddr + half * inHalf,
                  upAddr + half * inHalf,
                  wsGateAddr,
                  wsUpAddr,
                  qsAddr,
                  tileActAddr,
                  swiAddr,
                  scaleAddr,
                  yAddr,
                  curRows);
      SetFlag<HardEvent::V_MTE2>(eidV2M2[half]);
      pendingV2M2[half] = true;

      SetFlag<HardEvent::V_MTE3>(eidV2M3);
      WaitFlag<HardEvent::V_MTE3>(eidV2M3);

      DataCopyExtParams cpY{static_cast<uint16_t>(curRows),
                            static_cast<uint32_t>(H),
                            dstStrideY,
                            0,
                            0};
      DataCopyPad(yGm[rowStart * H], yLocal, cpY);

      DataCopyExtParams cpScale{
          1, static_cast<uint32_t>(curRows * sizeof(float)), 0, 0, 0};
      DataCopyPad(scaleGm[rowStart], scaleLocal, cpScale);

      // swi/y/scale are single-buffered: the next iteration's VF writes
      // must not start before these stores drain.
      SetFlag<HardEvent::MTE3_V>(eidM3V);
      WaitFlag<HardEvent::MTE3_V>(eidM3V);

      half = nextHalf;
    }
    if (pendingV2M2[0]) WaitFlag<HardEvent::V_MTE2>(eidV2M2[0]);
    if (pendingV2M2[1]) WaitFlag<HardEvent::V_MTE2>(eidV2M2[1]);
#undef DSQ_VECROW_ISSUE_LOAD
  }

 private:
  // One row, one VF scope, two passes. Pass1 spills only the already-computed
  // FP32 SwiGLU row and keeps its single amax accumulator in registers.
  // Pass2 consumes the register-resident scale/inverse and reloads SwiGLU;
  // Exp/Div/SwiGLU are never recomputed and no max buffer exists.
  __aicore__ inline void ComputeRows(__ubuf__ InType* gateIn,
                                     __ubuf__ InType* upIn,
                                     __ubuf__ float* wsGate,
                                     __ubuf__ float* wsUp,
                                     __ubuf__ float* qs,
                                     __ubuf__ float* actBuf,
                                     __ubuf__ float* swi,
                                     __ubuf__ float* scaleOut,
                                     __ubuf__ int8_t* yOut,
                                     uint32_t rows) {
    using namespace VecRow;
    static constexpr Reg::CastTrait castS32ToF32 = {Reg::RegLayout::UNKNOWN,
                                                    Reg::SatMode::UNKNOWN,
                                                    Reg::MaskMergeMode::ZEROING,
                                                    RoundMode::CAST_NONE};
    static constexpr Reg::CastTrait castB16ToF32 = {Reg::RegLayout::ZERO,
                                                    Reg::SatMode::UNKNOWN,
                                                    Reg::MaskMergeMode::ZEROING,
                                                    RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait castF32ToS16 = {Reg::RegLayout::ZERO,
                                                    Reg::SatMode::SAT,
                                                    Reg::MaskMergeMode::ZEROING,
                                                    RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait castS16ToF16 = {Reg::RegLayout::ZERO,
                                                    Reg::SatMode::UNKNOWN,
                                                    Reg::MaskMergeMode::ZEROING,
                                                    RoundMode::CAST_ROUND};
    static constexpr Reg::CastTrait castF16ToS8 = {Reg::RegLayout::ZERO,
                                                   Reg::SatMode::SAT,
                                                   Reg::MaskMergeMode::ZEROING,
                                                   RoundMode::CAST_TRUNC};

    const uint16_t chunks = static_cast<uint16_t>(CeilDivU(H, VL_FP32));
    const uint16_t rowCnt = static_cast<uint16_t>(rows);
    const uint32_t hLocal = H;
    const uint32_t strideInLocal = strideIn;
    const uint32_t strideFLocal = strideF;
    const uint32_t strideI8Local = strideI8;
    const bool hasQsLocal = hasQs;
    const bool hasActLocal = hasActScale;

    __VEC_SCOPE__ {
      Reg::RegTensor<float> vGate, vUp, vActScale, vTmp, vOne;
      Reg::RegTensor<float> vMax, vAbs, vScale, vScaleDup, vInv, vVal;
      Reg::RegTensor<InType> vRaw;
      Reg::RegTensor<int16_t> vS16;
      Reg::RegTensor<half> vHalf;
      Reg::RegTensor<int8_t> vQ;
      Reg::MaskReg m;
      Reg::MaskReg mAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
      Reg::MaskReg minusOne = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
      Reg::UnalignReg actLoadUreg;
      Reg::UnalignReg scaleStoreUreg;
      __ubuf__ float* actRead = actBuf;
      __ubuf__ float* scaleWrite = scaleOut;

      if constexpr (isInt32) {
        if (hasActLocal) {
          Reg::LoadUnAlignPre(actLoadUreg, actRead);
        }
      }

      for (uint16_t i = 0; i < rowCnt; i++) {
        uint32_t sreg = hLocal;
        Reg::Duplicate(vMax, 0.0f, mAll);
        Reg::Duplicate(vOne, 1.0f, mAll);
        if constexpr (isInt32) {
          if (hasActLocal) {
            Reg::LoadUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(
                vActScale, actLoadUreg, actRead, 1);
            Reg::Duplicate(vActScale, vActScale, mAll);
          } else {
            Reg::Duplicate(vActScale, 1.0f, mAll);
          }
        }

        // Pass 1: dequant -> SiLU -> up mul -> optional quant scale ->
        // spill/amax.
        for (uint16_t c = 0; c < chunks; c++) {
          m = Reg::UpdateMask<float>(sreg);
          const uint32_t inOff = i * strideInLocal + c * VL_FP32;
          const uint32_t fOff = i * strideFLocal + c * VL_FP32;
          if constexpr (isInt32) {
            Reg::LoadAlign(vRaw, gateIn + inOff);
            Reg::Cast<float, InType, castS32ToF32>(vGate, vRaw, m);
            Reg::LoadAlign(vTmp, wsGate + c * VL_FP32);
            Reg::Mul(vGate, vGate, vTmp, m);
            Reg::Mul(vGate, vGate, vActScale, m);
            Reg::LoadAlign(vRaw, upIn + inOff);
            Reg::Cast<float, InType, castS32ToF32>(vUp, vRaw, m);
            Reg::LoadAlign(vTmp, wsUp + c * VL_FP32);
            Reg::Mul(vUp, vUp, vTmp, m);
            Reg::Mul(vUp, vUp, vActScale, m);
          } else {
            Reg::LoadAlign<InType, Reg::LoadDist::DIST_UNPACK_B16>(
                vRaw, gateIn + inOff);
            Reg::Cast<float, InType, castB16ToF32>(vGate, vRaw, m);
            Reg::LoadAlign<InType, Reg::LoadDist::DIST_UNPACK_B16>(
                vRaw, upIn + inOff);
            Reg::Cast<float, InType, castB16ToF32>(vUp, vRaw, m);
          }

          Reg::Muls(vTmp, vGate, -1.0f, m);
          Reg::Exp(vTmp, vTmp, m);
          Reg::Adds(vTmp, vTmp, 1.0f, m);
          Reg::Div(vGate, vGate, vTmp, m);
          Reg::Mul(vGate, vGate, vUp, m);
          if (hasQsLocal) {
            Reg::LoadAlign(vTmp, qs + c * VL_FP32);
            Reg::Mul(vGate, vGate, vTmp, m);
          }
          Reg::StoreAlign(swi + fOff, vGate, m);
          Reg::Duplicate(vAbs, 0.0f, mAll);
          Reg::Abs(vAbs, vGate, m);
          // Tail lanes are zeroed before the full-mask running max.
          Reg::Max(vMax, vMax, vAbs, mAll);
        }

        Reg::ReduceMax(vMax, vMax, mAll);
        // Preserve the baseline/Golden FP32 arithmetic order exactly:
        // amax / 127 is not bit-equivalent to amax * (1 / 127), and
        // the one-ulp scale difference can move RINT boundary values.
        Reg::Duplicate(vTmp, 127.0f, minusOne);
        Reg::Div(vScale, vMax, vTmp, minusOne);
        Reg::Maxs(vScale, vScale, 1e-12f, minusOne);
        // Scale output is also compact (one FP32 per row); use the
        // target-SDK unaligned stream instead of issuing aligned VEC
        // stores at scaleOut + i.
        Reg::StoreUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(
            scaleWrite, vScale, scaleStoreUreg, 1);
        Reg::Duplicate(vScaleDup, vScale, mAll);
        Reg::Div(vInv, vOne, vScaleDup, mAll);

        // Pass 2: reload FP32 SwiGLU, multiply inverse, exact int8 cast/pack
        // chain.
        sreg = hLocal;
        for (uint16_t c = 0; c < chunks; c++) {
          m = Reg::UpdateMask<float>(sreg);
          const uint32_t fOff = i * strideFLocal + c * VL_FP32;
          Reg::LoadAlign(vVal, swi + fOff);
          Reg::Mul(vVal, vVal, vInv, m);
          Reg::Cast<int16_t, float, castF32ToS16>(vS16, vVal, m);
          Reg::Cast<half, int16_t, castS16ToF16>(vHalf, vS16, m);
          Reg::Cast<int8_t, half, castF16ToS8>(vQ, vHalf, m);
          Reg::StoreAlign<int8_t, Reg::StoreDist::DIST_PACK4_B32>(
              yOut + i * strideI8Local + c * VL_FP32, vQ, m);
        }
      }
      Reg::StoreUnAlignPost(scaleWrite, scaleStoreUreg, 0);
    }
  }

 private:
  TPipe* pipe = nullptr;
  uint32_t curBlockIdx = 0;
  uint32_t H = 0;
  uint32_t rowNum = 0;
  uint32_t useCoreNum = 0;
  uint32_t numRound = 0;
  uint32_t biasOffset = 0;
  uint32_t tileRows = 1;
  uint32_t strideIn = 0;
  uint32_t strideF = 0;
  uint32_t strideI8 = 0;
  bool actLeft = false;
  bool hasQs = false;
  bool hasActScale = false;

  GlobalTensor<InType> xGm;
  GlobalTensor<float> weightScaleGm;
  GlobalTensor<float> actScaleGm;
  GlobalTensor<float> quantScaleGm;
  GlobalTensor<int8_t> yGm;
  GlobalTensor<float> scaleGm;

  TBuf<TPosition::VECCALC> bufA, bufB, bufSwi, bufY;
  TBuf<TPosition::VECCALC> bufWsA, bufWsB, bufQs, bufAct, bufScale;
};

}  // namespace XllmKernelDequantSwigluQuant

#endif  // CANN_DEQUANT_SWIGLU_QUANT_VECROW_HPP
