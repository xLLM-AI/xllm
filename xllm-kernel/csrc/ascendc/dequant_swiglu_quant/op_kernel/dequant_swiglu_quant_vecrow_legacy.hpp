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
 * Here the whole dequant -> SwiGLU -> abs-max -> quant chain runs in MicroAPI
 * inside two __VEC_SCOPE__ blocks per tile. The per-row max never leaves a
 * vector register: ReduceMax lands it in lane 0 and a broadcast load feeds it
 * straight back into the divide. Input tiles are double-buffered so MTE2
 * overlaps the VF work.
 *
 * Scope: dynamic quant, no bias / quant_offset / group_index. Everything else
 * keeps using the original classes.
 */

#ifndef CANN_DEQUANT_SWIGLU_QUANT_VECROW_LEGACY_HPP
#define CANN_DEQUANT_SWIGLU_QUANT_VECROW_LEGACY_HPP

// Restored verbatim from direct_baseline (S1-R2-SPLIT legacy route).
// The guard isolates the file-level Reg declarations and __local_mem__
// signatures in the device compile region for the xasc_combined host
// pass; everything inside is byte-for-byte the original kernel.
#if defined(__NPU_ARCH__)

#include "kernel_operator.h"

namespace XllmKernelDequantSwigluQuant {
using namespace AscendC;

namespace VecRowLegacy {

constexpr uint32_t UB_BLOCK_BYTES = 32;
// Row strides are padded to a whole vector register so every chunk load in the
// VF loops sits on a natural boundary.
constexpr uint32_t VREG_BYTES = 256;
constexpr uint32_t VL_FP32 = 64;
constexpr uint32_t MAX_TILE_ROWS = 64;
constexpr uint32_t UB_BUDGET_BYTES = 184 * 1024;

// <float, int32_t> resolves to the round-only cast; CAST_NONE is the plain
// widening the original int32 path uses.
constexpr Reg::CastTrait castS32ToF32 = {Reg::RegLayout::UNKNOWN,
                                         Reg::SatMode::UNKNOWN,
                                         Reg::MaskMergeMode::ZEROING,
                                         RoundMode::CAST_NONE};
constexpr Reg::CastTrait castB16ToF32 = {Reg::RegLayout::ZERO,
                                         Reg::SatMode::UNKNOWN,
                                         Reg::MaskMergeMode::ZEROING,
                                         RoundMode::UNKNOWN};
// fp32 -> int8 goes fp32 -> s16 -> fp16 -> s8, the same chain adv_api's
// TransRegForS8 uses (quantize_impl.h:145). Casting straight to int8 rounds and
// packs lanes wrongly -- it yields a correct scale but a garbage y.
constexpr Reg::CastTrait castF32ToS16 = {Reg::RegLayout::ZERO,
                                         Reg::SatMode::SAT,
                                         Reg::MaskMergeMode::ZEROING,
                                         RoundMode::CAST_RINT};

__aicore__ inline uint32_t CeilDivU(uint32_t a, uint32_t b) {
  return (b == 0) ? 0 : (a + b - 1) / b;
}

// Pad `count` elements of `sz` bytes up to a whole `toBytes` boundary.
__aicore__ inline uint32_t AlignElTo(uint32_t count,
                                     uint32_t sz,
                                     uint32_t toBytes) {
  uint32_t bytes = count * sz;
  return (bytes + toBytes - 1) / toBytes * toBytes / sz;
}

}  // namespace VecRowLegacy

/**
 * InType: int32_t (dequant path) or half/bfloat16_t (direct path).
 * hasWeightScale: int32 path multiplies by weight_scale[2H].
 */
template <typename InType, bool isInt32>
class DequantSwigluQuantVecRowLegacy {
 public:
  __aicore__ inline DequantSwigluQuantVecRowLegacy() {}
  __aicore__ inline ~DequantSwigluQuantVecRowLegacy() {}

  __aicore__ inline void Init(GM_ADDR x_gm,
                              GM_ADDR weight_scale_gm,
                              GM_ADDR activation_scale_gm,
                              GM_ADDR quant_scale_gm,
                              GM_ADDR y_gm,
                              GM_ADDR scale_gm,
                              const SwiGluTilingData* tilingData,
                              TPipe* pipe_) {
    using namespace VecRowLegacy;
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

    strideIn = AlignElTo(H, sizeof(InType), VREG_BYTES);
    strideF = AlignElTo(H, sizeof(float), VREG_BYTES);
    strideI8 = AlignElTo(H, 1, VREG_BYTES);

    uint32_t perRowBytes =
        2 * strideIn * sizeof(InType) + strideF * sizeof(float) + strideI8;
    uint32_t sharedBytes = 2 * strideF * sizeof(float) +
                           (hasQs ? strideF * sizeof(float) : 0) +
                           3 * MAX_TILE_ROWS * sizeof(float);
    uint32_t avail =
        (UB_BUDGET_BYTES > sharedBytes) ? (UB_BUDGET_BYTES - sharedBytes) : 0;
    tileRows = (perRowBytes > 0) ? (avail / perRowBytes) : 1;
    if (tileRows > MAX_TILE_ROWS) tileRows = MAX_TILE_ROWS;
    if (tileRows > numRound) tileRows = numRound;
    if (tileRows < 1) tileRows = 1;
    // Two input sets so the next tile's MTE2 overlaps this tile's VEC work.
    if (tileRows > 1) tileRows /= 2;

    pipe->InitBuffer(bufA, 2 * tileRows * strideIn * sizeof(InType));
    pipe->InitBuffer(bufB, 2 * tileRows * strideIn * sizeof(InType));
    pipe->InitBuffer(bufSwi, tileRows * strideF * sizeof(float));
    pipe->InitBuffer(bufY, tileRows * strideI8);
    pipe->InitBuffer(bufMax, MAX_TILE_ROWS * sizeof(float));
    pipe->InitBuffer(bufScale, MAX_TILE_ROWS * sizeof(float));
    pipe->InitBuffer(bufWsA, strideF * sizeof(float));
    pipe->InitBuffer(bufWsB, strideF * sizeof(float));
    pipe->InitBuffer(bufAct, 2 * MAX_TILE_ROWS * sizeof(float));
    if (hasQs) {
      pipe->InitBuffer(bufQs, strideF * sizeof(float));
    }
  }

  __aicore__ inline void Process() {
    using namespace VecRowLegacy;
    if (curBlockIdx >= useCoreNum || numRound == 0) {
      return;
    }

    LocalTensor<InType> aLocal = bufA.template Get<InType>();
    LocalTensor<InType> bLocal = bufB.template Get<InType>();
    LocalTensor<float> swiLocal = bufSwi.Get<float>();
    LocalTensor<int8_t> yLocal = bufY.Get<int8_t>();
    LocalTensor<float> maxLocal = bufMax.Get<float>();
    LocalTensor<float> scaleLocal = bufScale.Get<float>();
    LocalTensor<float> wsALocal = bufWsA.Get<float>();
    LocalTensor<float> wsBLocal = bufWsB.Get<float>();
    LocalTensor<float> actLocal = bufAct.Get<float>();

    // weight_scale / quant_scale are row-invariant: load once per core.
    if constexpr (isInt32) {
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

    __local_mem__ InType* aAddr = (__local_mem__ InType*)aLocal.GetPhyAddr();
    __local_mem__ InType* bAddr = (__local_mem__ InType*)bLocal.GetPhyAddr();
    __local_mem__ float* swiAddr = (__local_mem__ float*)swiLocal.GetPhyAddr();
    __local_mem__ int8_t* yAddr = (__local_mem__ int8_t*)yLocal.GetPhyAddr();
    __local_mem__ float* maxAddr = (__local_mem__ float*)maxLocal.GetPhyAddr();
    __local_mem__ float* scaleAddr =
        (__local_mem__ float*)scaleLocal.GetPhyAddr();
    __local_mem__ float* wsAAddr = (__local_mem__ float*)wsALocal.GetPhyAddr();
    __local_mem__ float* wsBAddr = (__local_mem__ float*)wsBLocal.GetPhyAddr();
    __local_mem__ float* actAddr = (__local_mem__ float*)actLocal.GetPhyAddr();
    __local_mem__ float* qsAddr =
        hasQs ? (__local_mem__ float*)bufQs.Get<float>().GetPhyAddr() : nullptr;

    // activate_left picks which half drives the sigmoid. Matches
    // DequantSwigluQuantDynamicBase::BaseProcess: activateLeft==0 -> gate is
    // the second half. Resolved here so the VF loops stay branch-free.
    __local_mem__ InType* gateAddr = actLeft ? aAddr : bAddr;
    __local_mem__ InType* upAddr = actLeft ? bAddr : aAddr;
    __local_mem__ float* wsGateAddr = actLeft ? wsAAddr : wsBAddr;
    __local_mem__ float* wsUpAddr = actLeft ? wsBAddr : wsAAddr;

    uint32_t dstStrideIn = (strideIn - H) * sizeof(InType) / UB_BLOCK_BYTES;
    uint32_t dstStrideY = (strideI8 - H) / UB_BLOCK_BYTES;
    uint32_t inHalf = tileRows * strideIn;
    uint32_t actHalf = VecRowLegacy::MAX_TILE_ROWS;

    DataCopyPadExtParams<InType> ppIn{false, 0, 0, 0};
    DataCopyPadExtParams<float> ppAct{false, 0, 0, 0};

    event_t eidM2V[2] = {
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V)),
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V))};
    event_t eidM3M2 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
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
        DSQ_VECROW_ISSUE_LOAD(nextR, nextHalf);
        SetFlag<HardEvent::MTE2_V>(eidM2V[nextHalf]);
      }

      WaitFlag<HardEvent::MTE2_V>(eidM2V[half]);

      ComputeSwigluAndMax(gateAddr + half * inHalf,
                          upAddr + half * inHalf,
                          wsGateAddr,
                          wsUpAddr,
                          qsAddr,
                          actAddr + half * actHalf,
                          swiAddr,
                          maxAddr,
                          curRows);

      QuantizeRows(swiAddr, maxAddr, scaleAddr, yAddr, curRows);

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
      SetFlag<HardEvent::MTE3_MTE2>(eidM3M2);
      WaitFlag<HardEvent::MTE3_MTE2>(eidM3M2);

      half = nextHalf;
    }
#undef DSQ_VECROW_ISSUE_LOAD
  }

 private:
  /**
   * Pass 1: dequant + SwiGLU for every row in the tile, spilling the fp32
   * result to `swi` and leaving each row's abs-max in maxBuf[i].
   *
   * The abs-max accumulation deliberately uses a full mask while the Abs that
   * feeds it uses the tail mask: ZEROING zeroes the inactive lanes of the Abs
   * result, and 0 is the identity for a max over absolute values. Masking the
   * Max itself would instead zero the lanes already holding the running max
   * from earlier chunks.
   */
  __aicore__ inline void ComputeSwigluAndMax(__local_mem__ InType* gateIn,
                                             __local_mem__ InType* upIn,
                                             __local_mem__ float* wsGate,
                                             __local_mem__ float* wsUp,
                                             __local_mem__ float* qs,
                                             __local_mem__ float* actBuf,
                                             __local_mem__ float* swi,
                                             __local_mem__ float* maxBuf,
                                             uint32_t rows) {
    using namespace VecRowLegacy;
    const uint16_t chunks = static_cast<uint16_t>(CeilDivU(H, VL_FP32));
    const uint16_t rowCnt = static_cast<uint16_t>(rows);
    const uint32_t hLocal = H;
    const uint32_t strideInLocal = strideIn;
    const uint32_t strideFLocal = strideF;
    const bool hasQsLocal = hasQs;
    const bool hasActLocal = hasActScale;

    __VEC_SCOPE__ {
      MicroAPI::RegTensor<float> vGate, vUp, vScale, vTmp, vOne, vSig, vMax,
          vAbs;
      MicroAPI::RegTensor<InType> vRaw;
      MicroAPI::MaskReg m;
      MicroAPI::MaskReg mAll =
          MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
      MicroAPI::MaskReg minusOne =
          MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();

      for (uint16_t i = 0; i < rowCnt; i++) {
        uint32_t sreg = hLocal;
        MicroAPI::Duplicate<float>(vMax, 0.0f);
        MicroAPI::Duplicate<float>(vOne, 1.0f, mAll);

        if constexpr (isInt32) {
          // activation_scale is per row: one broadcast load feeds all
          // chunks, replacing a per-row Muls(scalar).
          if (hasActLocal) {
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(
                vScale, actBuf + i);
          } else {
            MicroAPI::Duplicate<float>(vScale, 1.0f, mAll);
          }
        }

        for (uint16_t c = 0; c < chunks; c++) {
          m = MicroAPI::UpdateMask<float>(sreg);
          uint32_t inOff = i * strideInLocal + c * VL_FP32;
          uint32_t fOff = i * strideFLocal + c * VL_FP32;

          if constexpr (isInt32) {
            MicroAPI::DataCopy<InType, MicroAPI::LoadDist::DIST_NORM>(
                vRaw, gateIn + inOff);
            MicroAPI::Cast<float, InType, castS32ToF32>(vGate, vRaw, m);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                vTmp, wsGate + c * VL_FP32);
            MicroAPI::Mul(vGate, vGate, vTmp, m);
            MicroAPI::Mul(vGate, vGate, vScale, m);

            MicroAPI::DataCopy<InType, MicroAPI::LoadDist::DIST_NORM>(
                vRaw, upIn + inOff);
            MicroAPI::Cast<float, InType, castS32ToF32>(vUp, vRaw, m);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                vTmp, wsUp + c * VL_FP32);
            MicroAPI::Mul(vUp, vUp, vTmp, m);
            MicroAPI::Mul(vUp, vUp, vScale, m);
          } else {
            MicroAPI::DataCopy<InType, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vRaw, gateIn + inOff);
            MicroAPI::Cast<float, InType, castB16ToF32>(vGate, vRaw, m);
            MicroAPI::DataCopy<InType, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vRaw, upIn + inOff);
            MicroAPI::Cast<float, InType, castB16ToF32>(vUp, vRaw, m);
          }

          // SiLU(gate) = gate / (1 + exp(-gate)) -- the same op
          // sequence adv_api's Sigmoid uses on this arch, so results
          // match the original path bit for bit.
          MicroAPI::Muls(vTmp, vGate, -1.0f, m);
          MicroAPI::Exp(vTmp, vTmp, m);
          MicroAPI::Adds(vTmp, vTmp, 1.0f, m);
          MicroAPI::Div(vSig, vOne, vTmp, m);
          MicroAPI::Mul(vGate, vGate, vSig, m);

          MicroAPI::Mul(vGate, vGate, vUp, m);

          if (hasQsLocal) {
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                vTmp, qs + c * VL_FP32);
            MicroAPI::Mul(vGate, vGate, vTmp, m);
          }

          MicroAPI::DataCopy(swi + fOff, vGate, m);

          MicroAPI::Abs(vAbs, vGate, m);
          MicroAPI::Max(vMax, vMax, vAbs, mAll);
        }

        MicroAPI::ReduceMax(vMax, vMax, mAll);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
            maxBuf + i, vMax, minusOne);
      }
    }
  }

  /**
   * Pass 2: turn each row's max into its scale and quantise. The max is pulled
   * back as a broadcast vector load, so scale and 1/scale are computed in-lane
   * and the divide never touches the scalar unit.
   */
  __aicore__ inline void QuantizeRows(__local_mem__ float* swi,
                                      __local_mem__ float* maxBuf,
                                      __local_mem__ float* scaleBuf,
                                      __local_mem__ int8_t* yOut,
                                      uint32_t rows) {
    using namespace VecRowLegacy;
    const uint16_t chunks = static_cast<uint16_t>(CeilDivU(H, VL_FP32));
    const uint16_t rowCnt = static_cast<uint16_t>(rows);
    const uint32_t hLocal = H;
    const uint32_t strideFLocal = strideF;
    const uint32_t strideI8Local = strideI8;

    __VEC_SCOPE__ {
      MicroAPI::RegTensor<float> vVal, vScale, vInv, vOne;
      MicroAPI::RegTensor<half> vHalf;
      MicroAPI::RegTensor<int8_t> vQ;
      MicroAPI::MaskReg m;
      MicroAPI::MaskReg mAll =
          MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
      MicroAPI::MaskReg minusOne =
          MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();

      for (uint16_t i = 0; i < rowCnt; i++) {
        uint32_t sreg = hLocal;

        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(vScale,
                                                                    maxBuf + i);
        MicroAPI::Muls(vScale, vScale, 1.0f / 127.0f, mAll);
        MicroAPI::Maxs(vScale, vScale, 1e-12f, mAll);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
            scaleBuf + i, vScale, minusOne);

        MicroAPI::Duplicate<float>(vOne, 1.0f, mAll);
        MicroAPI::Div(vInv, vOne, vScale, mAll);

        for (uint16_t c = 0; c < chunks; c++) {
          m = MicroAPI::UpdateMask<float>(sreg);
          MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
              vVal, swi + i * strideFLocal + c * VL_FP32);
          MicroAPI::Mul(vVal, vVal, vInv, m);
          // SAT on the s16 step clamps to [-128,127] once the value
          // lands in int8, so the explicit clamp is free.
          MicroAPI::Cast<int16_t, float, castF32ToS16>(
              (MicroAPI::RegTensor<int16_t>&)vHalf, vVal, m);
          MicroAPI::Cast<half, int16_t, LayoutZMrgZRndRSatS>(
              vHalf, (MicroAPI::RegTensor<int16_t>&)vHalf, m);
          MicroAPI::Cast<int8_t, half, LayoutZMrgZRndRSatS>(vQ, vHalf, m);
          MicroAPI::DataCopy<int8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
              yOut + i * strideI8Local + c * VL_FP32, vQ, m);
        }
      }
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
  TBuf<TPosition::VECCALC> bufWsA, bufWsB, bufQs, bufAct, bufMax, bufScale;
};

}  // namespace XllmKernelDequantSwigluQuant

#endif  // defined(__NPU_ARCH__)

#endif  // CANN_DEQUANT_SWIGLU_QUANT_VECROW_LEGACY_HPP
