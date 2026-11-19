/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file add_rms_norm_dynamic_quant_normal_kernel.h
 * \brief
 */

#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_NORMAL_KERNEL_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_NORMAL_KERNEL_H_

#include "rms_norm_dynamic_quant_base.h"

template <typename T, typename T_Y, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddRmsNormDynamicQuantNormal
    : public KernelAddRmsNormDynamicQuantBase<T, T_Y, TILING_KEY, BUFFER_NUM> {
 public:
  __aicore__ inline KernelAddRmsNormDynamicQuantNormal(TPipe* pipe) {
    Ppipe = pipe;
  }

  __aicore__ inline void Init(GM_ADDR x,
                              GM_ADDR gamma,
                              GM_ADDR smooth1,
                              GM_ADDR smooth2,
                              GM_ADDR beta,
                              GM_ADDR y1,
                              GM_ADDR y2,
                              GM_ADDR outScale1,
                              GM_ADDR outScale2,
                              GM_ADDR workspace,
                              const RmsNormDynamicQuantTilingData* tiling) {
    this->InitBaseParams(tiling);
    this->InitInGlobalTensors(x, gamma, smooth1, smooth2, beta);
    this->InitOutGlobalTensors(y1, y2, outScale1, outScale2);
    this->numRowsAligned = (this->rowStep + ELEM_PER_BLK_FP32 - 1) /
                           ELEM_PER_BLK_FP32 * ELEM_PER_BLK_FP32;
    this->ubAligned = static_cast<uint32_t>(
        (this->numLastDimAligned - this->numLastDim) / ELEM_PER_BLK_FP16);
    /*
      UB = 3 * this->rowStep * alignedCol * sizeof(T)
          + 2 * this->rowStep * alignedCol * sizeof(float)
          + Count(gamma,beta,bias) * alignedCol * sizeof(T)
          + 512Bytes(256 + reduceOut)
    */
    Ppipe->InitBuffer(
        inRowsQue,
        BUFFER_NUM,
        2 * this->rowStep * this->numLastDimAligned * sizeof(T));  // 2 * D * 2
    Ppipe->InitBuffer(
        outRowsQue,
        BUFFER_NUM,
        2 * this->rowStep * this->numLastDimAligned * sizeof(T));  // D * 2
    Ppipe->InitBuffer(
        xBufFp32,
        this->rowStep * this->numLastDimAligned * sizeof(float));  // D * 4
    Ppipe->InitBuffer(
        yBufFp32,
        this->rowStep * this->numLastDimAligned * sizeof(float));  // D * 4
    Ppipe->InitBuffer(weightBuf01,
                      this->numLastDimAligned * sizeof(T));  // D * 2
    Ppipe->InitBuffer(weightBuf02,
                      this->numLastDimAligned * sizeof(T));  // D * 2
    Ppipe->InitBuffer(weightBuf03,
                      this->numLastDimAligned * sizeof(T));  // D * 2
    if (this->betaFlag == 1) {
      Ppipe->InitBuffer(weightBuf04, this->numLastDimAligned * sizeof(T));
    }
    // 2 dynamic quant operator required 2 scale buffer.
    Ppipe->InitBuffer(scalesBuf, 2 * this->numRowsAligned * sizeof(float));
  }

  __aicore__ inline void Process() {
    int32_t rowMoveCnt = CEIL_DIV(this->rowWork, this->rowStep);
    CopyInWeights();

    LocalTensor<T> gammaLocal = weightBuf01.template Get<T>();

    int32_t gmOffset = 0;
    int32_t gmOffsetScale = 0;
    int32_t elementCount = this->numLastDimAligned * this->rowStep;

    for (int32_t rowIdx = 0; rowIdx < rowMoveCnt - 1; ++rowIdx) {
      CopyInX(gmOffset, this->rowStep, elementCount);
      ComputeRmsNorm(this->rowStep, elementCount, gammaLocal);
      ComputeDynamicQuant(this->rowStep, elementCount);
      CopyOut(gmOffset, gmOffsetScale, this->rowStep);
      gmOffset += this->rowStep * this->numLastDim;
      gmOffsetScale += this->rowStep;
    }
    {
      elementCount = this->numLastDimAligned * this->rowTail_;
      int32_t rowIdx = rowMoveCnt - 1;
      CopyInX(gmOffset, this->rowTail_, elementCount);
      ComputeRmsNorm(this->rowTail_, elementCount, gammaLocal);
      ComputeDynamicQuant(this->rowTail_, elementCount);
      CopyOut(gmOffset, gmOffsetScale, this->rowTail_);
    }
  }

 private:
  __aicore__ inline void CopyInX(int32_t gmOffset,
                                 int32_t rowCount,
                                 int32_t elementCount) {
    LocalTensor<T> xLocalIn = inRowsQue.template AllocTensor<T>();
    DataCopyExStride(xLocalIn,
                     this->xGm[gmOffset],
                     this->numLastDim,
                     rowCount,
                     this->ubAligned);
    inRowsQue.EnQue(xLocalIn);
  }

  __aicore__ inline void CopyOutY(int32_t gmOffset,
                                  int32_t rowCount,
                                  int32_t elementCount) {
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> yLocal = xBufFp32.Get<float>();
    LocalTensor<T> yOut = yBufFp32.Get<T>();
    PipeBarrier<PIPE_ALL>();
    if constexpr (is_same<T, half>::value) {
      Cast(yOut, yLocal, RoundMode::CAST_NONE, elementCount);
    } else {  // BF16
      Cast(yOut, yLocal, RoundMode::CAST_RINT, elementCount);
    }
    PipeBarrier<PIPE_ALL>();
    DataCopyExStride(
        this->xGm[gmOffset], yOut, this->numLastDim, rowCount, this->ubAligned);
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void CopyInWeights() {
    LocalTensor<T> gammaLocal = weightBuf01.template Get<T>();
    DataCopyEx(gammaLocal, this->gammaGm, this->numLastDim);
    if ((this->isOld && this->smooth1Exist) || this->newSingleFirst) {
      LocalTensor<T> smooth1Local = weightBuf02.template Get<T>();
      DataCopyEx(smooth1Local, this->smooth1Gm, this->numLastDim);
    }
    if (this->oldDouble || this->newSingleSecond) {
      LocalTensor<T> smooth2Local = weightBuf03.template Get<T>();
      DataCopyEx(smooth2Local, this->smooth2Gm, this->numLastDim);
    }
    if (this->betaFlag == 1) {
      LocalTensor<T> betaLocal = weightBuf04.template Get<T>();
      DataCopyEx(betaLocal, this->betaGm, this->numLastDim);
    }
  }

  __aicore__ inline void ComputeRmsNorm(int32_t nums,
                                        int32_t elementCount,
                                        LocalTensor<T>& gammaLocal) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    // registration-build adaptation: opc also instantiates the int4
    // quant-output variant (T_Y = integer_sub_type<4, true>); AscendC::Reg
    // TypeGet has no sub-byte specialization and the validated direct-invoke
    // acceptance covered T_Y = int8_t only -> keep the baseline implementation
    // for non-int8 instantiations
    if constexpr (is_same<T_Y, int8_t>::value) {
      ComputeRmsNormVf(nums, gammaLocal);
      return;
    }
#endif
    LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();  // xLocalFp32 <-- x
    LocalTensor<T> xInputLocal = inRowsQue.template DeQue<T>();
    LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
    Cast(xLocalFp32, xInputLocal, RoundMode::CAST_NONE, elementCount);
    PipeBarrier<PIPE_V>();

    Mul(yLocalFp32,
        xLocalFp32,
        xLocalFp32,
        elementCount);  // yLocalFp32 <- x ** 2
    PipeBarrier<PIPE_V>();

    // reduce#1 for mean
    for (int32_t rid = 0; rid < nums; ++rid) {
      auto roundOffset = rid * this->numLastDimAligned;
      float squareSumTemp =
          ReduceSumHalfInterval(yLocalFp32[roundOffset],
                                this->numLastDim);  // aveLocalTemp <-- E(x**2)
      float rstdLocalTemp = 1 / sqrt(squareSumTemp * this->aveNum + this->eps);
      event_t eventSV =
          static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
      SetFlag<HardEvent::S_V>(eventSV);
      WaitFlag<HardEvent::S_V>(eventSV);
      Muls(xLocalFp32[roundOffset],
           xLocalFp32[roundOffset],
           rstdLocalTemp,
           this->numLastDim);  // xLocalFp32 <- x * rstd
    }
    PipeBarrier<PIPE_V>();

    Cast(yLocalFp32,
         gammaLocal,
         RoundMode::CAST_NONE,
         this->numLastDim);  // yLocalFp32 <- gamma
    PipeBarrier<PIPE_V>();
    for (int32_t rid = 0; rid < nums; ++rid) {
      auto roundOffset = rid * this->numLastDimAligned;
      Mul(xLocalFp32[roundOffset],
          xLocalFp32[roundOffset],
          yLocalFp32,
          this->numLastDim);  // xLocalFp32 <- x * rstd * gamma
      PipeBarrier<PIPE_V>();
    }

    if (this->betaFlag == 1) {
      LocalTensor<T> betaLocal = weightBuf04.Get<T>();
      Cast(yLocalFp32,
           betaLocal,
           RoundMode::CAST_NONE,
           this->numLastDim);  // yLocalFp32 <- gamma
      for (int32_t rid = 0; rid < nums; ++rid) {
        auto roundOffset = rid * this->numLastDimAligned;
        PipeBarrier<PIPE_V>();
        Add(xLocalFp32[roundOffset],
            xLocalFp32[roundOffset],
            yLocalFp32,
            this->numLastDim);
        PipeBarrier<PIPE_V>();
      }
    }
    inRowsQue.FreeTensor(xInputLocal);
  }

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
  __aicore__ inline void ComputeRmsNormVf(int32_t nums,
                                          LocalTensor<T>& gammaLocal) {
    LocalTensor<T> xInputLocal = inRowsQue.template DeQue<T>();
    LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
    __ubuf__ T* xInBase = (__ubuf__ T*)xInputLocal.GetPhyAddr();
    __ubuf__ T* gammaBase = (__ubuf__ T*)gammaLocal.GetPhyAddr();
    __ubuf__ float* outBase = (__ubuf__ float*)xLocalFp32.GetPhyAddr();

    // beta branch resolved outside the __VEC_SCOPE__: runtime branches with
    // live vector values inside a vector function are not lowerable on this
    // backend
    if (this->betaFlag == 1) {
      LocalTensor<T> betaLocal = weightBuf04.Get<T>();
      ComputeRmsNormVfImpl<true>(nums,
                                 xInBase,
                                 gammaBase,
                                 (__ubuf__ T*)betaLocal.GetPhyAddr(),
                                 outBase);
    } else {
      ComputeRmsNormVfImpl<false>(nums, xInBase, gammaBase, nullptr, outBase);
    }
    inRowsQue.FreeTensor(xInputLocal);
  }

  template <bool HAS_BETA>
  __aicore__ inline void ComputeRmsNormVfImpl(int32_t nums,
                                              __ubuf__ T* xInBase,
                                              __ubuf__ T* gammaBase,
                                              __ubuf__ T* betaBase,
                                              __ubuf__ float* outBase) {
    // xasc_combined standard form: function-scope static constexpr CastTrait
    // (cast probe verified)
    static constexpr AscendC::Reg::CastTrait kTraitB16ToB32 = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};

    const uint32_t h = static_cast<uint32_t>(this->numLastDim);
    const uint32_t ha = static_cast<uint32_t>(this->numLastDimAligned);
    const uint16_t loopCnt =
        static_cast<uint16_t>((h + ELEM_PER_REP_FP32 - 1) / ELEM_PER_REP_FP32);
    const float aveNum = this->aveNum;
    const float eps = this->eps;

    __VEC_SCOPE__ {
      AscendC::Reg::RegTensor<T> xB16Reg;
      AscendC::Reg::RegTensor<T> wB16Reg;
      AscendC::Reg::RegTensor<float> xReg;
      AscendC::Reg::RegTensor<float> wReg;
      AscendC::Reg::RegTensor<float> sqReg;
      AscendC::Reg::RegTensor<float> accReg;
      AscendC::Reg::RegTensor<float> statReg;
      AscendC::Reg::RegTensor<float> rstdReg;
      AscendC::Reg::RegTensor<float> oneReg;
      AscendC::Reg::MaskReg pregAll =
          AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
      AscendC::Reg::MaskReg pregLoop;

      AscendC::Reg::Duplicate(oneReg, 1.0f, pregAll);

      for (uint16_t rid = 0; rid < static_cast<uint16_t>(nums); ++rid) {
        __ubuf__ T* rowX = xInBase + rid * ha;
        __ubuf__ float* rowOut = outBase + rid * ha;

        // pass 1: sum(x * x) accumulated in registers
        AscendC::Reg::Duplicate(accReg, 0.0f, pregAll);
        uint32_t remain = h;
        for (uint16_t j = 0; j < loopCnt; ++j) {
          pregLoop = AscendC::Reg::UpdateMask<float>(remain);
          AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
              xB16Reg, rowX + j * ELEM_PER_REP_FP32);
          AscendC::Reg::Cast<float, T, kTraitB16ToB32>(xReg, xB16Reg, pregLoop);
          AscendC::Reg::Mul(sqReg, xReg, xReg, pregLoop);
          // full-mask accumulate is safe: lanes outside pregLoop are zeroed by
          // the Cast
          AscendC::Reg::Add(accReg, accReg, sqReg, pregAll);
        }
        AscendC::Reg::ReduceSum(statReg, accReg, pregAll);
        // Reduce leaves the result in lane 0 only: broadcast before vector use
        AscendC::Reg::Duplicate(statReg, statReg, pregAll);
        AscendC::Reg::Muls(statReg, statReg, aveNum, pregAll);
        AscendC::Reg::Adds(statReg, statReg, eps, pregAll);
        AscendC::Reg::Sqrt(statReg, statReg, pregAll);
        AscendC::Reg::Div(rstdReg, oneReg, statReg, pregAll);

        // pass 2: out = (x * rstd) * gamma (+ beta), fp32 rows stay in UB for
        // the quant stage
        remain = h;
        for (uint16_t j = 0; j < loopCnt; ++j) {
          pregLoop = AscendC::Reg::UpdateMask<float>(remain);
          AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
              xB16Reg, rowX + j * ELEM_PER_REP_FP32);
          AscendC::Reg::Cast<float, T, kTraitB16ToB32>(xReg, xB16Reg, pregLoop);
          AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
              wB16Reg, gammaBase + j * ELEM_PER_REP_FP32);
          AscendC::Reg::Cast<float, T, kTraitB16ToB32>(wReg, wB16Reg, pregLoop);
          AscendC::Reg::Mul(xReg, xReg, rstdReg, pregLoop);
          AscendC::Reg::Mul(xReg, xReg, wReg, pregLoop);
          if constexpr (HAS_BETA) {
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                wB16Reg, betaBase + j * ELEM_PER_REP_FP32);
            AscendC::Reg::Cast<float, T, kTraitB16ToB32>(
                wReg, wB16Reg, pregLoop);
            AscendC::Reg::Add(xReg, xReg, wReg, pregLoop);
          }
          AscendC::Reg::StoreAlign<float>(
              rowOut + j * ELEM_PER_REP_FP32, xReg, pregLoop);
        }
      }
      // make the fp32 rows visible to the quant-stage loads
      AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE,
                                AscendC::Reg::MemType::VEC_LOAD>();
    }
  }
#endif

  __aicore__ inline void ComputeDynamicQuant(int32_t nums,
                                             int32_t elementCount) {
    LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();  // xLocalFp32 <-- y
    LocalTensor<float> scaleLocal = scalesBuf.Get<float>();
    LocalTensor<float> zLocalFp32 = outRowsQue.template AllocTensor<float>();
    LocalTensor<T_Y> outQuant01 = zLocalFp32.ReinterpretCast<T_Y>();
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    // registration-build adaptation: non-int8 quant-output instantiations keep
    // the baseline implementation (see ComputeRmsNorm)
    if constexpr (is_same<T_Y, int8_t>::value) {
      // compute gating aligned verbatim with the CopyOut write gating
      // (skip-unwritten-quant-path); both paths active -> fused dual two-pass,
      // otherwise single-path via the flag gates (which skip the never-written
      // second path)
      const bool quant1Active = this->isOld || (this->outQuant1Flag == 1);
      const bool quant2Active = this->oldDouble || (this->outQuant2Flag == 1);
      if (quant1Active && quant2Active) {
        doQuantFusedVf(scaleLocal, xLocalFp32, outQuant01, nums, elementCount);
      } else {
        doQuant1withFlag(
            scaleLocal, xLocalFp32, outQuant01, nums, elementCount);
        doQuant2withFlag(
            scaleLocal, xLocalFp32, outQuant01, nums, elementCount);
      }
      outRowsQue.EnQue(zLocalFp32);
      return;
    }
#endif
    doQuant1withFlag(scaleLocal, xLocalFp32, outQuant01, nums, elementCount);
    doQuant2withFlag(scaleLocal, xLocalFp32, outQuant01, nums, elementCount);
    outRowsQue.EnQue(zLocalFp32);
  }

  __aicore__ inline void doQuant1withFlag(LocalTensor<float> scaleLocal,
                                          LocalTensor<float> xLocalFp32,
                                          LocalTensor<T_Y> outQuant01,
                                          int32_t nums,
                                          int32_t elementCount) {
    // aligned verbatim with the CopyOut write gate `isOld || outQuant1Flag ==
    // 1` (skip-unwritten-quant-path)
    if (!(this->isOld || this->outQuant1Flag == 1)) {
      return;
    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    // registration-build adaptation: non-int8 quant-output instantiations keep
    // the baseline implementation (see ComputeRmsNorm)
    if constexpr (is_same<T_Y, int8_t>::value) {
      doQuantVf(scaleLocal, xLocalFp32, outQuant01, nums, elementCount, 0);
      return;
    }
#endif
    LocalTensor<float> tmpFp32 = inRowsQue.template AllocTensor<float>();
    LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
    LocalTensor<float> scale1Local = scaleLocal[0];
    if (this->smooth1Exist) {
      // compute smooth1
      LocalTensor<T> smooth1Local = weightBuf02.Get<T>();
      LocalTensor<float> smooth1Fp32 =
          yLocalFp32[(nums - 1) * this->numLastDimAligned];
      Cast(smooth1Fp32, smooth1Local, RoundMode::CAST_NONE, this->numLastDim);
      PipeBarrier<PIPE_V>();
      for (int32_t rid = 0; rid < nums; ++rid) {
        Mul(yLocalFp32[rid * this->numLastDimAligned],
            xLocalFp32[rid * this->numLastDimAligned],
            smooth1Fp32,
            this->numLastDim);  // yLocalFp32 <-- y * smooth1
      }
      PipeBarrier<PIPE_V>();
    } else {
      for (int32_t rid = 0; rid < nums; ++rid) {
        Muls(yLocalFp32[rid * this->numLastDimAligned],
             xLocalFp32[rid * this->numLastDimAligned],
             (float)(1.0),
             this->numLastDim);  // yLocalFp32 <-- y * 1
      }
      PipeBarrier<PIPE_V>();
    }
    ScaleTensor(yLocalFp32, tmpFp32, scale1Local, elementCount, nums);
    PipeBarrier<PIPE_V>();
    Cast(yLocalFp32.ReinterpretCast<int32_t>(),
         yLocalFp32,
         RoundMode::CAST_RINT,
         elementCount);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(yLocalFp32.ReinterpretCast<half>(),
         yLocalFp32.ReinterpretCast<int32_t>(),
         RoundMode::CAST_NONE,
         elementCount);
    PipeBarrier<PIPE_V>();
    Cast(outQuant01,
         yLocalFp32.ReinterpretCast<half>(),
         RoundMode::CAST_TRUNC,
         elementCount);
    PipeBarrier<PIPE_V>();
    inRowsQue.FreeTensor(tmpFp32);
  }

  __aicore__ inline void doQuant2withFlag(LocalTensor<float> scaleLocal,
                                          LocalTensor<float> xLocalFp32,
                                          LocalTensor<T_Y> outQuant01,
                                          int32_t nums,
                                          int32_t elementCount) {
    // aligned verbatim with the CopyOut write gate `oldDouble || outQuant2Flag
    // == 1` (skip-unwritten-quant-path)
    if (!(this->oldDouble || this->outQuant2Flag == 1)) {
      return;
    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    // registration-build adaptation: non-int8 quant-output instantiations keep
    // the baseline implementation (see ComputeRmsNorm)
    if constexpr (is_same<T_Y, int8_t>::value) {
      doQuantVf(scaleLocal, xLocalFp32, outQuant01, nums, elementCount, 1);
      return;
    }
#endif
    LocalTensor<float> tmpFp32 = inRowsQue.template AllocTensor<float>();
    LocalTensor<float> scale2Local = scaleLocal[this->numRowsAligned];
    LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
    LocalTensor<T_Y> outQuant02 = outQuant01[elementCount];
    if (this->smooth2Exist) {
      LocalTensor<T> smooth2Local = weightBuf03.Get<T>();
      Cast(tmpFp32, smooth2Local, RoundMode::CAST_NONE, this->numLastDim);
      PipeBarrier<PIPE_V>();
      for (int32_t rid = 0; rid < nums; ++rid) {
        Mul(xLocalFp32[rid * this->numLastDimAligned],
            xLocalFp32[rid * this->numLastDimAligned],
            tmpFp32,
            this->numLastDim);  // yLocalFp32 <-- y * smooth2
      }
      PipeBarrier<PIPE_V>();
    } else {
      for (int32_t rid = 0; rid < nums; ++rid) {
        Muls(xLocalFp32[rid * this->numLastDimAligned],
             xLocalFp32[rid * this->numLastDimAligned],
             (float)(1.0),
             this->numLastDim);  // yLocalFp32 <-- y * 1
      }
      PipeBarrier<PIPE_V>();
    }
    ScaleTensor(xLocalFp32, tmpFp32, scale2Local, elementCount, nums);
    PipeBarrier<PIPE_V>();
    Cast(xLocalFp32.ReinterpretCast<int32_t>(),
         xLocalFp32,
         RoundMode::CAST_RINT,
         elementCount);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(xLocalFp32.ReinterpretCast<half>(),
         xLocalFp32.ReinterpretCast<int32_t>(),
         RoundMode::CAST_NONE,
         elementCount);
    PipeBarrier<PIPE_V>();
    Cast(outQuant02,
         xLocalFp32.ReinterpretCast<half>(),
         RoundMode::CAST_TRUNC,
         elementCount);
    PipeBarrier<PIPE_V>();
    inRowsQue.FreeTensor(tmpFp32);
  }

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
  // quantPath 0: first dynamic-quant branch (smooth1 / y1 / scale1).
  // quantPath 1: second dynamic-quant branch (smooth2 / y2 / scale2, base
  // offset elementCount).
  __aicore__ inline void doQuantVf(LocalTensor<float>& scaleLocal,
                                   LocalTensor<float>& xLocalFp32,
                                   LocalTensor<T_Y>& outQuant01,
                                   int32_t nums,
                                   int32_t elementCount,
                                   int32_t quantPath) {
    __ubuf__ float* outBase = (__ubuf__ float*)xLocalFp32.GetPhyAddr();
    __ubuf__ T* smoothBase = nullptr;
    if (quantPath == 0) {
      if (this->smooth1Exist) {
        LocalTensor<T> smooth1Local = weightBuf02.Get<T>();
        smoothBase = (__ubuf__ T*)smooth1Local.GetPhyAddr();
      }
    } else {
      if (this->smooth2Exist) {
        LocalTensor<T> smooth2Local = weightBuf03.Get<T>();
        smoothBase = (__ubuf__ T*)smooth2Local.GetPhyAddr();
      }
    }
    __ubuf__ T_Y* yBase = (__ubuf__ T_Y*)outQuant01.GetPhyAddr();
    if (quantPath == 1) {
      yBase += elementCount;
    }
    __ubuf__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
    if (quantPath == 1) {
      scaleAddr += this->numRowsAligned;
    }

    // smooth branch resolved outside the __VEC_SCOPE__: runtime branches with
    // live vector values inside a vector function are not lowerable on this
    // backend
    if (smoothBase != nullptr) {
      DoQuantVfImpl<true>(outBase, smoothBase, yBase, scaleAddr, nums);
    } else {
      DoQuantVfImpl<false>(outBase, nullptr, yBase, scaleAddr, nums);
    }
  }

  template <bool HAS_SMOOTH>
  __aicore__ inline void DoQuantVfImpl(__ubuf__ float* outBase,
                                       __ubuf__ T* smoothBase,
                                       __ubuf__ T_Y* yBase,
                                       __ubuf__ float* scaleAddr,
                                       int32_t nums) {
    // xasc_combined standard form: function-scope static constexpr CastTrait
    // (cast probe verified)
    static constexpr AscendC::Reg::CastTrait kTraitB16ToB32 = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait kTraitF32ToS16Rint = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::NO_SAT,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
    static constexpr AscendC::Reg::CastTrait kTraitS16ToF16 = {
        AscendC::Reg::RegLayout::UNKNOWN,
        AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_NONE};
    static constexpr AscendC::Reg::CastTrait kTraitF16ToI8Trunc = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::NO_SAT,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_TRUNC};

    const uint32_t h = static_cast<uint32_t>(this->numLastDim);
    const uint32_t ha = static_cast<uint32_t>(this->numLastDimAligned);
    const uint16_t loopCnt =
        static_cast<uint16_t>((h + ELEM_PER_REP_FP32 - 1) / ELEM_PER_REP_FP32);
    const float quantMax = this->quantMaxVal;

    __VEC_SCOPE__ {
      AscendC::Reg::RegTensor<T> sB16Reg;
      AscendC::Reg::RegTensor<float> outReg;
      AscendC::Reg::RegTensor<float> smReg;
      AscendC::Reg::RegTensor<float> tReg;
      AscendC::Reg::RegTensor<float> absReg;
      AscendC::Reg::RegTensor<float> amaxReg;
      AscendC::Reg::RegTensor<float> redReg;
      AscendC::Reg::RegTensor<float> scTmpReg;
      AscendC::Reg::RegTensor<float> scaleReg;
      AscendC::Reg::RegTensor<float> oneReg;
      AscendC::Reg::RegTensor<float> cstReg;
      AscendC::Reg::RegTensor<float> qReg;
      AscendC::Reg::RegTensor<int16_t> qS16Reg;
      AscendC::Reg::RegTensor<half> qF16Reg;
      AscendC::Reg::RegTensor<T_Y> qI8Reg;
      AscendC::Reg::MaskReg pregAll =
          AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
      AscendC::Reg::MaskReg pregLoop;
      AscendC::Reg::UnalignRegForStore scaleCursor;

      AscendC::Reg::Duplicate(oneReg, 1.0f, pregAll);
      // loop-invariant constants kept in dedicated read-only registers
      AscendC::Reg::Duplicate(cstReg, quantMax, pregAll);

      for (uint16_t rid = 0; rid < static_cast<uint16_t>(nums); ++rid) {
        __ubuf__ float* rowOut = outBase + rid * ha;
        __ubuf__ T_Y* rowY = yBase + rid * ha;

        // pass 1: amax(|t|), t = out * smooth (smooth absent: t = out,
        // Muls-by-1.0 is an exact identity)
        AscendC::Reg::Duplicate(amaxReg, 0.0f, pregAll);
        uint32_t remain = h;
        for (uint16_t j = 0; j < loopCnt; ++j) {
          pregLoop = AscendC::Reg::UpdateMask<float>(remain);
          AscendC::Reg::LoadAlign<float>(outReg,
                                         rowOut + j * ELEM_PER_REP_FP32);
          if constexpr (HAS_SMOOTH) {
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                sB16Reg, smoothBase + j * ELEM_PER_REP_FP32);
            AscendC::Reg::Cast<float, T, kTraitB16ToB32>(
                smReg, sB16Reg, pregLoop);
            AscendC::Reg::Mul(tReg, outReg, smReg, pregLoop);
            AscendC::Reg::Abs(absReg, tReg, pregLoop);
          } else {
            AscendC::Reg::Abs(absReg, outReg, pregLoop);
          }
          // full-mask max is safe: lanes outside pregLoop are zeroed by the
          // load and abs >= 0
          AscendC::Reg::Max(amaxReg, amaxReg, absReg, pregAll);
        }
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(
            redReg, amaxReg, pregAll);
        // Reduce leaves the result in lane 0 only: broadcast before vector use
        AscendC::Reg::Duplicate(redReg, redReg, pregAll);

        // scaleTemp = quantMaxVal / amax; scale = 1 / scaleTemp (same
        // expression as baseline ScaleTensor)
        AscendC::Reg::Div(scTmpReg, cstReg, redReg, pregAll);
        AscendC::Reg::Div(scaleReg, oneReg, scTmpReg, pregAll);
        AscendC::Reg::StoreUnAlign<float,
                                   AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            scaleAddr, scaleReg, scaleCursor, 1);

        // pass 2: y = Cast(Cast(Cast(t * scaleTemp, RINT), NONE), TRUNC)
        remain = h;
        for (uint16_t j = 0; j < loopCnt; ++j) {
          pregLoop = AscendC::Reg::UpdateMask<float>(remain);
          AscendC::Reg::LoadAlign<float>(outReg,
                                         rowOut + j * ELEM_PER_REP_FP32);
          if constexpr (HAS_SMOOTH) {
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                sB16Reg, smoothBase + j * ELEM_PER_REP_FP32);
            AscendC::Reg::Cast<float, T, kTraitB16ToB32>(
                smReg, sB16Reg, pregLoop);
            AscendC::Reg::Mul(qReg, outReg, smReg, pregLoop);
            AscendC::Reg::Mul(qReg, qReg, scTmpReg, pregLoop);
          } else {
            AscendC::Reg::Mul(qReg, outReg, scTmpReg, pregLoop);
          }
          AscendC::Reg::Cast<int16_t, float, kTraitF32ToS16Rint>(
              qS16Reg, qReg, pregLoop);
          AscendC::Reg::Cast<half, int16_t, kTraitS16ToF16>(
              qF16Reg, qS16Reg, pregLoop);
          AscendC::Reg::Cast<T_Y, half, kTraitF16ToI8Trunc>(
              qI8Reg, qF16Reg, pregLoop);
          AscendC::Reg::StoreAlign<T_Y,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
              rowY + j * ELEM_PER_REP_FP32, qI8Reg, pregLoop);
        }
      }
      AscendC::Reg::StoreUnAlignPost<float>(scaleAddr, scaleCursor, 0);
    }
  }

  // fused dual-path dynamic quant: replaces the sequential doQuant1withFlag +
  // doQuant2withFlag pair when both CopyOut write gates are open; out rows are
  // loaded once per pass for both paths, per-element expressions are identical
  // to DoQuantVfImpl
  __aicore__ inline void doQuantFusedVf(LocalTensor<float>& scaleLocal,
                                        LocalTensor<float>& xLocalFp32,
                                        LocalTensor<T_Y>& outQuant01,
                                        int32_t nums,
                                        int32_t elementCount) {
    __ubuf__ float* outBase = (__ubuf__ float*)xLocalFp32.GetPhyAddr();
    __ubuf__ T* s1Base = nullptr;
    __ubuf__ T* s2Base = nullptr;
    if (this->smooth1Exist) {
      LocalTensor<T> smooth1Local = weightBuf02.Get<T>();
      s1Base = (__ubuf__ T*)smooth1Local.GetPhyAddr();
    }
    if (this->smooth2Exist) {
      LocalTensor<T> smooth2Local = weightBuf03.Get<T>();
      s2Base = (__ubuf__ T*)smooth2Local.GetPhyAddr();
    }
    __ubuf__ T_Y* y1Base = (__ubuf__ T_Y*)outQuant01.GetPhyAddr();
    __ubuf__ T_Y* y2Base = y1Base + elementCount;
    __ubuf__ float* scale1Addr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
    __ubuf__ float* scale2Addr = scale1Addr + this->numRowsAligned;

    // smooth branches resolved outside the __VEC_SCOPE__: runtime branches with
    // live vector values inside a vector function are not lowerable on this
    // backend
    if (s1Base != nullptr) {
      if (s2Base != nullptr) {
        DoQuantFusedVfImpl<true, true>(outBase,
                                       s1Base,
                                       s2Base,
                                       y1Base,
                                       y2Base,
                                       scale1Addr,
                                       scale2Addr,
                                       nums);
      } else {
        DoQuantFusedVfImpl<true, false>(outBase,
                                        s1Base,
                                        nullptr,
                                        y1Base,
                                        y2Base,
                                        scale1Addr,
                                        scale2Addr,
                                        nums);
      }
    } else {
      if (s2Base != nullptr) {
        DoQuantFusedVfImpl<false, true>(outBase,
                                        nullptr,
                                        s2Base,
                                        y1Base,
                                        y2Base,
                                        scale1Addr,
                                        scale2Addr,
                                        nums);
      } else {
        DoQuantFusedVfImpl<false, false>(outBase,
                                         nullptr,
                                         nullptr,
                                         y1Base,
                                         y2Base,
                                         scale1Addr,
                                         scale2Addr,
                                         nums);
      }
    }
  }

  // fused dual two-pass quant: pass 1 shares one out load per chunk to build
  // both amax accumulators, pass 2 shares one out load per chunk for both cast
  // chains; per-element expressions, scale computation and the 3-level cast
  // chain are identical to DoQuantVfImpl (bit-exact by construction)
  template <bool HAS_S1, bool HAS_S2>
  __aicore__ inline void DoQuantFusedVfImpl(__ubuf__ float* outBase,
                                            __ubuf__ T* s1Base,
                                            __ubuf__ T* s2Base,
                                            __ubuf__ T_Y* y1Base,
                                            __ubuf__ T_Y* y2Base,
                                            __ubuf__ float* scale1Addr,
                                            __ubuf__ float* scale2Addr,
                                            int32_t nums) {
    // xasc_combined standard form: function-scope static constexpr CastTrait
    // (cast probe verified)
    static constexpr AscendC::Reg::CastTrait kTraitB16ToB32 = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait kTraitF32ToS16Rint = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::NO_SAT,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
    static constexpr AscendC::Reg::CastTrait kTraitS16ToF16 = {
        AscendC::Reg::RegLayout::UNKNOWN,
        AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_NONE};
    static constexpr AscendC::Reg::CastTrait kTraitF16ToI8Trunc = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::NO_SAT,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_TRUNC};

    const uint32_t h = static_cast<uint32_t>(this->numLastDim);
    const uint32_t ha = static_cast<uint32_t>(this->numLastDimAligned);
    const uint16_t loopCnt =
        static_cast<uint16_t>((h + ELEM_PER_REP_FP32 - 1) / ELEM_PER_REP_FP32);
    const float quantMax = this->quantMaxVal;

    __VEC_SCOPE__ {
      AscendC::Reg::RegTensor<T> s1B16Reg;
      AscendC::Reg::RegTensor<T> s2B16Reg;
      AscendC::Reg::RegTensor<float> outReg;
      AscendC::Reg::RegTensor<float> sm1Reg;
      AscendC::Reg::RegTensor<float> sm2Reg;
      AscendC::Reg::RegTensor<float> t1Reg;
      AscendC::Reg::RegTensor<float> t2Reg;
      AscendC::Reg::RegTensor<float> abs1Reg;
      AscendC::Reg::RegTensor<float> abs2Reg;
      AscendC::Reg::RegTensor<float> amax1Reg;
      AscendC::Reg::RegTensor<float> amax2Reg;
      AscendC::Reg::RegTensor<float> red1Reg;
      AscendC::Reg::RegTensor<float> red2Reg;
      AscendC::Reg::RegTensor<float> scTmp1Reg;
      AscendC::Reg::RegTensor<float> scTmp2Reg;
      AscendC::Reg::RegTensor<float> scale1Reg;
      AscendC::Reg::RegTensor<float> scale2Reg;
      AscendC::Reg::RegTensor<float> oneReg;
      AscendC::Reg::RegTensor<float> cstReg;
      AscendC::Reg::RegTensor<float> q1Reg;
      AscendC::Reg::RegTensor<float> q2Reg;
      AscendC::Reg::RegTensor<int16_t> q1S16Reg;
      AscendC::Reg::RegTensor<int16_t> q2S16Reg;
      AscendC::Reg::RegTensor<half> q1F16Reg;
      AscendC::Reg::RegTensor<half> q2F16Reg;
      AscendC::Reg::RegTensor<T_Y> q1I8Reg;
      AscendC::Reg::RegTensor<T_Y> q2I8Reg;
      AscendC::Reg::MaskReg pregAll =
          AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
      AscendC::Reg::MaskReg pregLoop;
      AscendC::Reg::UnalignRegForStore scale1Cursor;
      AscendC::Reg::UnalignRegForStore scale2Cursor;

      AscendC::Reg::Duplicate(oneReg, 1.0f, pregAll);
      // loop-invariant constants kept in dedicated read-only registers
      AscendC::Reg::Duplicate(cstReg, quantMax, pregAll);

      for (uint16_t rid = 0; rid < static_cast<uint16_t>(nums); ++rid) {
        __ubuf__ float* rowOut = outBase + rid * ha;
        __ubuf__ T_Y* rowY1 = y1Base + rid * ha;
        __ubuf__ T_Y* rowY2 = y2Base + rid * ha;

        // pass 1: amax1(|t1|) and amax2(|t2|) share one out load per chunk;
        // t = out * smooth (smooth absent: t = out, Muls-by-1.0 is an exact
        // identity)
        AscendC::Reg::Duplicate(amax1Reg, 0.0f, pregAll);
        AscendC::Reg::Duplicate(amax2Reg, 0.0f, pregAll);
        uint32_t remain = h;
        for (uint16_t j = 0; j < loopCnt; ++j) {
          pregLoop = AscendC::Reg::UpdateMask<float>(remain);
          AscendC::Reg::LoadAlign<float>(outReg,
                                         rowOut + j * ELEM_PER_REP_FP32);
          if constexpr (HAS_S1) {
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                s1B16Reg, s1Base + j * ELEM_PER_REP_FP32);
            AscendC::Reg::Cast<float, T, kTraitB16ToB32>(
                sm1Reg, s1B16Reg, pregLoop);
            AscendC::Reg::Mul(t1Reg, outReg, sm1Reg, pregLoop);
            AscendC::Reg::Abs(abs1Reg, t1Reg, pregLoop);
          } else {
            AscendC::Reg::Abs(abs1Reg, outReg, pregLoop);
          }
          if constexpr (HAS_S2) {
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                s2B16Reg, s2Base + j * ELEM_PER_REP_FP32);
            AscendC::Reg::Cast<float, T, kTraitB16ToB32>(
                sm2Reg, s2B16Reg, pregLoop);
            AscendC::Reg::Mul(t2Reg, outReg, sm2Reg, pregLoop);
            AscendC::Reg::Abs(abs2Reg, t2Reg, pregLoop);
          } else {
            AscendC::Reg::Abs(abs2Reg, outReg, pregLoop);
          }
          // full-mask max is safe: lanes outside pregLoop are zeroed by the
          // masked ops
          AscendC::Reg::Max(amax1Reg, amax1Reg, abs1Reg, pregAll);
          AscendC::Reg::Max(amax2Reg, amax2Reg, abs2Reg, pregAll);
        }
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(
            red1Reg, amax1Reg, pregAll);
        // Reduce leaves the result in lane 0 only: broadcast before vector use
        AscendC::Reg::Duplicate(red1Reg, red1Reg, pregAll);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(
            red2Reg, amax2Reg, pregAll);
        AscendC::Reg::Duplicate(red2Reg, red2Reg, pregAll);

        // scaleTemp = quantMaxVal / amax; scale = 1 / scaleTemp (same
        // expression as baseline ScaleTensor)
        AscendC::Reg::Div(scTmp1Reg, cstReg, red1Reg, pregAll);
        AscendC::Reg::Div(scale1Reg, oneReg, scTmp1Reg, pregAll);
        AscendC::Reg::StoreUnAlign<float,
                                   AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            scale1Addr, scale1Reg, scale1Cursor, 1);
        AscendC::Reg::Div(scTmp2Reg, cstReg, red2Reg, pregAll);
        AscendC::Reg::Div(scale2Reg, oneReg, scTmp2Reg, pregAll);
        AscendC::Reg::StoreUnAlign<float,
                                   AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            scale2Addr, scale2Reg, scale2Cursor, 1);

        // pass 2: y = Cast(Cast(Cast(t * scaleTemp, RINT), NONE), TRUNC), one
        // out load per chunk for both paths
        remain = h;
        for (uint16_t j = 0; j < loopCnt; ++j) {
          pregLoop = AscendC::Reg::UpdateMask<float>(remain);
          AscendC::Reg::LoadAlign<float>(outReg,
                                         rowOut + j * ELEM_PER_REP_FP32);
          if constexpr (HAS_S1) {
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                s1B16Reg, s1Base + j * ELEM_PER_REP_FP32);
            AscendC::Reg::Cast<float, T, kTraitB16ToB32>(
                sm1Reg, s1B16Reg, pregLoop);
            AscendC::Reg::Mul(q1Reg, outReg, sm1Reg, pregLoop);
            AscendC::Reg::Mul(q1Reg, q1Reg, scTmp1Reg, pregLoop);
          } else {
            AscendC::Reg::Mul(q1Reg, outReg, scTmp1Reg, pregLoop);
          }
          if constexpr (HAS_S2) {
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                s2B16Reg, s2Base + j * ELEM_PER_REP_FP32);
            AscendC::Reg::Cast<float, T, kTraitB16ToB32>(
                sm2Reg, s2B16Reg, pregLoop);
            AscendC::Reg::Mul(q2Reg, outReg, sm2Reg, pregLoop);
            AscendC::Reg::Mul(q2Reg, q2Reg, scTmp2Reg, pregLoop);
          } else {
            AscendC::Reg::Mul(q2Reg, outReg, scTmp2Reg, pregLoop);
          }
          AscendC::Reg::Cast<int16_t, float, kTraitF32ToS16Rint>(
              q1S16Reg, q1Reg, pregLoop);
          AscendC::Reg::Cast<half, int16_t, kTraitS16ToF16>(
              q1F16Reg, q1S16Reg, pregLoop);
          AscendC::Reg::Cast<T_Y, half, kTraitF16ToI8Trunc>(
              q1I8Reg, q1F16Reg, pregLoop);
          AscendC::Reg::StoreAlign<T_Y,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
              rowY1 + j * ELEM_PER_REP_FP32, q1I8Reg, pregLoop);
          AscendC::Reg::Cast<int16_t, float, kTraitF32ToS16Rint>(
              q2S16Reg, q2Reg, pregLoop);
          AscendC::Reg::Cast<half, int16_t, kTraitS16ToF16>(
              q2F16Reg, q2S16Reg, pregLoop);
          AscendC::Reg::Cast<T_Y, half, kTraitF16ToI8Trunc>(
              q2I8Reg, q2F16Reg, pregLoop);
          AscendC::Reg::StoreAlign<T_Y,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
              rowY2 + j * ELEM_PER_REP_FP32, q2I8Reg, pregLoop);
        }
      }
      AscendC::Reg::StoreUnAlignPost<float>(scale1Addr, scale1Cursor, 0);
      AscendC::Reg::StoreUnAlignPost<float>(scale2Addr, scale2Cursor, 0);
    }
  }
#endif

  __aicore__ inline void CopyOut(int32_t gmOffset,
                                 int32_t gmOffsetScale,
                                 int32_t rowCount) {
    LocalTensor<T_Y> outY12 = outRowsQue.template DeQue<T_Y>();
    LocalTensor<float> scaleLocal = scalesBuf.Get<float>();
    SyncCopyOut<HardEvent::S_MTE3>();
    SyncCopyOut<HardEvent::V_MTE3>();
    if (this->isOld || (this->outQuant1Flag == 1)) {
      LocalTensor<T_Y> outQuant01 = outY12[0];
      LocalTensor<float> scale1Local = scaleLocal[0];
      DataCopyEx(this->y1Gm[gmOffset], outQuant01, this->numLastDim, rowCount);
      DataCopyEx(this->outScale1Gm[gmOffsetScale], scale1Local, rowCount);
    }
    if (this->oldDouble || (this->outQuant2Flag == 1)) {
      LocalTensor<T_Y> outQuant02 = outY12[rowCount * this->numLastDimAligned];
      LocalTensor<float> scale2Local = scaleLocal[this->numRowsAligned];
      DataCopyEx(this->y2Gm[gmOffset], outQuant02, this->numLastDim, rowCount);
      DataCopyEx(this->outScale2Gm[gmOffsetScale], scale2Local, rowCount);
    }
    SyncCopyOut<HardEvent::MTE3_S>();
    outRowsQue.FreeTensor(outY12);
  }

  // registration-build adaptation: compiled unconditionally again (as in the
  // original source); the int4 baseline fallback of doQuant{1,2}withFlag needs
  // it on 3510
  __aicore__ inline void ScaleTensor(LocalTensor<float>& srcTensor,
                                     LocalTensor<float>& tmpTensor,
                                     LocalTensor<float>& scaleTensor,
                                     int32_t size,
                                     int32_t nums) {
    float maxTemp;
    float scaleTemp;
    event_t eventVS;
    event_t eventSV;
    Abs(tmpTensor, srcTensor, size);  // tmpLocal <-- |y * smooth1|
    PipeBarrier<PIPE_V>();
    for (int32_t rid = 0; rid < nums; ++rid) {
      ReduceMaxInplace(tmpTensor[rid * this->numLastDimAligned],
                       this->numLastDim);
      eventVS =
          static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
      SetFlag<HardEvent::V_S>(eventVS);
      WaitFlag<HardEvent::V_S>(eventVS);
      maxTemp = tmpTensor[rid * this->numLastDimAligned].GetValue(0);  // Reduce
      scaleTemp = this->quantMaxVal / maxTemp;
      scaleTensor.SetValue(rid, 1 / scaleTemp);
      eventSV =
          static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
      SetFlag<HardEvent::S_V>(eventSV);
      WaitFlag<HardEvent::S_V>(eventSV);
      auto srcSlice = srcTensor[rid * this->numLastDimAligned];
      Muls(srcSlice, srcSlice, scaleTemp, this->numLastDim);
    }
  }

 private:
  TPipe* Ppipe = nullptr;
  TQue<QuePosition::VECIN, BUFFER_NUM> inRowsQue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outRowsQue;

  TBuf<TPosition::VECCALC> xBufFp32;
  TBuf<TPosition::VECCALC> yBufFp32;

  TBuf<TPosition::VECCALC> weightBuf01;
  TBuf<TPosition::VECCALC> weightBuf02;
  TBuf<TPosition::VECCALC> weightBuf03;
  TBuf<TPosition::VECCALC> weightBuf04;
  TBuf<TPosition::VECCALC> scalesBuf;

  uint32_t numRowsAligned;
  uint32_t ubAligned;
};

#endif  // __ADD_RMS_NORM_DYNAMIC_QUANT_NORMAL_KERNEL_H_
