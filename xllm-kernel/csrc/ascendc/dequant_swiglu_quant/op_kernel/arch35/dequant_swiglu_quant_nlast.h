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
 * \file dequant_swiglu_quant_nlast.h
 * \brief
 */

#ifndef DEQUANT_SWIGLU_QUANT_NLAST_H
#define DEQUANT_SWIGLU_QUANT_NLAST_H

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "dequant_swiglu_quant_common.h"

namespace DequantSwigluQuantV35Ops {
using namespace AscendC;

template <typename TActScale,
          typename TQuantScale,
          typename TGroup,
          typename TBias,
          typename TXtype,
          typename TYtype>
class DequantSwigluQuantNlast {
 public:
  static constexpr bool hasActScale_ = IsSameType<TActScale, float>::value;
  static constexpr bool hasQuantScale_ = IsSameType<TQuantScale, float>::value;
  static constexpr bool hasGroupIndex_ = IsSameType<TGroup, int64_t>::value;
  static constexpr bool hasBiasIndex_ =
      IsSameType<TBias, float>::value || IsSameType<TBias, half>::value ||
      IsSameType<TBias, bfloat16_t>::value || IsSameType<TBias, int32_t>::value;
  static constexpr bool ifBiasIntIndex_ = IsSameType<TBias, int32_t>::value;
  static constexpr bool ifBiasFloatIndex_ = IsSameType<TBias, float>::value;
  static constexpr bool ifBiasFloat16Index_ = IsSameType<TBias, half>::value;
  static constexpr bool ifBiasBfloat16Index_ =
      IsSameType<TBias, bfloat16_t>::value;
  static constexpr bool ifXIntIndex_ = IsSameType<TXtype, int32_t>::value;
  static constexpr bool ifXBf16Index_ = IsSameType<TXtype, bfloat16_t>::value;
  static constexpr bool ifXFloat16Index_ = IsSameType<TXtype, half>::value;
  static constexpr bool ifYInt8Index_ = IsSameType<TYtype, int8_t>::value;
  static constexpr bool ifYFloat8e4m3Index_ =
      IsSameType<TYtype, fp8_e4m3fn_t>::value;
  static constexpr bool ifYFloat8e5m2Index_ =
      IsSameType<TYtype, fp8_e5m2_t>::value;
  static constexpr bool ifYFloat4e2m1Index_ =
      IsSameType<TYtype, fp4x2_e2m1_t>::value;
  static constexpr bool ifYFloat4e1m2Index_ =
      IsSameType<TYtype, fp4x2_e1m2_t>::value;
  static constexpr bool ifYHiFloat8Index_ =
      IsSameType<TYtype, hifloat8_t>::value;

  __aicore__ inline DequantSwigluQuantNlast(TPipe* pipe) { pipe_ = pipe; };

  __aicore__ inline void Init(
      GM_ADDR x,
      GM_ADDR weightScale,
      GM_ADDR activationScale,
      GM_ADDR bias,
      GM_ADDR quantScale,
      GM_ADDR quantOffset,
      GM_ADDR groupIndex,
      GM_ADDR y,
      GM_ADDR scale,
      const DequantSwigluQuantV35NlastTilingData* tilingData);

  __aicore__ inline void Process();

 private:
  __aicore__ inline void ProcessPerLoop(int64_t ubIdx0,
                                        int64_t curSize0,
                                        int64_t ubIdx1,
                                        int64_t curSize1);

 protected:
  /* global memory address */
  // input global mem
  GlobalTensor<TXtype> xGm_;
  GlobalTensor<float> weightScaleGm_;
  GlobalTensor<TActScale> activationScaleGm_;
  GlobalTensor<TBias> biasGm_;
  GlobalTensor<TQuantScale> quantScaleGm_;
  GlobalTensor<float> quantOffsetGm_;
  GlobalTensor<TGroup> groupIndexGm_;

  // output global mem
  GlobalTensor<TYtype> yGm_;
  GlobalTensor<float> scaleGm_;

  LocalTensor<float> inScaleLocal;
  LocalTensor<TBias> biasLocal;
  __local_mem__ TBias* biasPtr;

  /* ascendc variable */
  TPipe* pipe_ = nullptr;
  TQue<QuePosition::VECIN, 1> xActQueue_;
  TQue<QuePosition::VECIN, 1> inScaleQueue_;
  TQue<QuePosition::VECIN, 1> biasQueue_;
  TQue<QuePosition::VECOUT, 1> yQueue_;
  TQue<QuePosition::VECOUT, 1> scaleQueue_;

  TBuf<> tmpBuffer;

  int64_t blockIdx_ = GetBlockIdx();
  int64_t blockIdx0_ = 0;
  int64_t blockIdx1_ = 0;
  int64_t gmOffset0_ = 0;
  int64_t xLastAlignB32_ = 0;
  int64_t xLastAlign_ = 0;
  int64_t yLastAlign_ = 0;
  int64_t yLastAlignB4_ = 0;
  int64_t aScaleAlignB32_ = 0;
  int64_t biasAlign_ = 0;
  int64_t roundMode_ = 0;
  float scalarMaxNum_ = 127.0;

  const DequantSwigluQuantV35NlastTilingData* tl_ = nullptr;
};

template <typename TActScale,
          typename TQuantScale,
          typename TGroup,
          typename TBias,
          typename TXtype,
          typename TYtype>
__aicore__ inline void
DequantSwigluQuantNlast<TActScale, TQuantScale, TGroup, TBias, TXtype, TYtype>::
    Init(GM_ADDR x,
         GM_ADDR weightScale,
         GM_ADDR activationScale,
         GM_ADDR bias,
         GM_ADDR quantScale,
         GM_ADDR quantOffset,
         GM_ADDR groupIndex,
         GM_ADDR y,
         GM_ADDR scale,
         const DequantSwigluQuantV35NlastTilingData* tilingData) {
  tl_ = tilingData;
  uint32_t blockElem = BLOCK_SIZE / sizeof(TXtype);
  xLastAlign_ = CeilDivision(tl_->inDim2, blockElem) * blockElem;

  xLastAlignB32_ = CeilDivision(tl_->inDim2, BLOCK_ELEM_B32) * BLOCK_ELEM_B32;

  yLastAlign_ = CeilDivision(tl_->inDim2, BLOCK_ELEM_B8) * BLOCK_ELEM_B8;
  yLastAlignB4_ =
      CeilDivision(tl_->inDim2 / FP4_WEIGHT, BLOCK_ELEM_B8) * BLOCK_ELEM_B8;
  aScaleAlignB32_ = BLOCK_ELEM_B32;

  blockElem = BLOCK_SIZE / sizeof(TBias);
  biasAlign_ = CeilDivision(tl_->inDim2, blockElem) * blockElem;

  roundMode_ = tl_->roundMode;
  if constexpr (ifYFloat8e4m3Index_) {
    scalarMaxNum_ = 448.0;
  }
  if constexpr (ifYFloat8e5m2Index_) {
    scalarMaxNum_ = 57344.0;
  }
  if constexpr (ifYFloat4e2m1Index_) {
    scalarMaxNum_ = 6.0;
  }
  if constexpr (ifYFloat4e1m2Index_) {
    scalarMaxNum_ = 1.75;
  }
  if constexpr (ifYHiFloat8Index_) {
    scalarMaxNum_ = 32768.0;
  }

  xGm_.SetGlobalBuffer((__gm__ TXtype*)x);
  weightScaleGm_.SetGlobalBuffer((__gm__ float*)weightScale);
  activationScaleGm_.SetGlobalBuffer((__gm__ TActScale*)activationScale);
  quantScaleGm_.SetGlobalBuffer((__gm__ TQuantScale*)quantScale);

  yGm_.SetGlobalBuffer((__gm__ TYtype*)y);
  scaleGm_.SetGlobalBuffer((__gm__ float*)scale);
  int64_t singleUbFormer = tl_->ubFormer0 * tl_->ubFormer1;
  // x + activation_scale
  pipe_->InitBuffer(
      xActQueue_,
      DOUBLE_BUFFER,
      (singleUbFormer * xLastAlign_ * SWI_FACTOR) * sizeof(TXtype) +
          (singleUbFormer * aScaleAlignB32_ * SWI_FACTOR) * sizeof(float));
  // weight_scale + quant_scale
  pipe_->InitBuffer(
      inScaleQueue_, 1, (xLastAlignB32_ + xLastAlignB32_) * sizeof(float));

  pipe_->InitBuffer(
      yQueue_, DOUBLE_BUFFER, singleUbFormer * yLastAlign_ * sizeof(TYtype));
  pipe_->InitBuffer(
      scaleQueue_, 1, singleUbFormer * aScaleAlignB32_ * sizeof(float));

  pipe_->InitBuffer(tmpBuffer, singleUbFormer * xLastAlignB32_ * sizeof(float));

  if constexpr (hasBiasIndex_) {
    biasGm_.SetGlobalBuffer((__gm__ TBias*)bias);
    pipe_->InitBuffer(biasQueue_, 1, biasAlign_ * sizeof(TBias));
  }
}

template <typename TActScale,
          typename TQuantScale,
          typename TGroup,
          typename TBias,
          typename TXtype,
          typename TYtype>
__aicore__ inline void
DequantSwigluQuantNlast<TActScale, TQuantScale, TGroup, TBias, TXtype, TYtype>::
    Process() {
  DataCopyPadParams padParams{false, 0, 0, 0};
  inScaleLocal = inScaleQueue_.AllocTensor<float>();
  // copy_in: weight_scale(1, H)
  if constexpr (ifXIntIndex_) {
    DataCopyParams dataCopyWeightScaleParams;
    dataCopyWeightScaleParams.blockCount = 1;
    dataCopyWeightScaleParams.blockLen = tl_->inDim2 * sizeof(float);
    dataCopyWeightScaleParams.srcStride = 0;
    dataCopyWeightScaleParams.dstStride = 0;
    DataCopyPad(
        inScaleLocal, weightScaleGm_, dataCopyWeightScaleParams, padParams);
  }

  // copy_in: quant_scale(1, H)
  if constexpr (hasQuantScale_) {
    DataCopyParams dataCopyQuantScaleParams;
    dataCopyQuantScaleParams.blockCount = 1;
    dataCopyQuantScaleParams.blockLen = tl_->inDim2 * sizeof(TQuantScale);
    dataCopyQuantScaleParams.srcStride = 0;
    dataCopyQuantScaleParams.dstStride = 0;
    DataCopyPad(inScaleLocal[xLastAlignB32_],
                quantScaleGm_,
                dataCopyQuantScaleParams,
                padParams);
  }
  inScaleQueue_.EnQue(inScaleLocal);
  inScaleLocal = inScaleQueue_.DeQue<float>();

  // copy_in: bias(1, H)
  if constexpr (hasBiasIndex_) {
    biasLocal = biasQueue_.AllocTensor<TBias>();
    DataCopyParams dataCopyBiasParams;
    dataCopyBiasParams.blockCount = 1;
    dataCopyBiasParams.blockLen = tl_->inDim2 * sizeof(TBias);
    dataCopyBiasParams.srcStride = 0;
    dataCopyBiasParams.dstStride = 0;
    DataCopyPad(biasLocal, biasGm_, dataCopyBiasParams, padParams);
    biasQueue_.EnQue(biasLocal);
    biasLocal = biasQueue_.DeQue<TBias>();
  }

  blockIdx0_ = blockIdx_ / tl_->blockNum1;
  blockIdx1_ = blockIdx_ - blockIdx0_ * tl_->blockNum1;
  int64_t loopNum0 = blockIdx0_ == tl_->blockNum0 - 1
                         ? tl_->ubLoopOfTailBlock0
                         : tl_->ubLoopOfFormerBlock0;
  int64_t loopNum1 = blockIdx1_ == tl_->blockNum1 - 1
                         ? tl_->ubLoopOfTailBlock1
                         : tl_->ubLoopOfFormerBlock1;
  int64_t tailSize0 = blockIdx0_ == tl_->blockNum0 - 1
                          ? tl_->ubTailOfTailBlock0
                          : tl_->ubTailOfFormerBlock0;
  int64_t tailSize1 = blockIdx1_ == tl_->blockNum1 - 1
                          ? tl_->ubTailOfTailBlock1
                          : tl_->ubTailOfFormerBlock1;
  for (int64_t ubIdx0 = 0; ubIdx0 < loopNum0; ubIdx0++) {
    int64_t curSize0 = ubIdx0 == loopNum0 - 1 ? tailSize0 : tl_->ubFormer0;
    gmOffset0_ = blockIdx0_ * tl_->blockFormer0 + ubIdx0 * tl_->ubFormer0;
    for (int64_t ubIdx1 = 0; ubIdx1 < loopNum1; ubIdx1++) {
      int64_t curSize1 = ubIdx1 == loopNum1 - 1 ? tailSize1 : tl_->ubFormer1;
      // act and gating are divided
      ProcessPerLoop(ubIdx0, curSize0, ubIdx1, curSize1);
    }
  }

  inScaleQueue_.FreeTensor(inScaleLocal);
  if constexpr (hasBiasIndex_) {
    biasQueue_.FreeTensor(biasLocal);
  }
}

template <typename TActScale,
          typename TQuantScale,
          typename TGroup,
          typename TBias,
          typename TXtype,
          typename TYtype>
__aicore__ inline void
DequantSwigluQuantNlast<TActScale, TQuantScale, TGroup, TBias, TXtype, TYtype>::
    ProcessPerLoop(int64_t ubIdx0,
                   int64_t curSize0,
                   int64_t ubIdx1,
                   int64_t curSize1) {
  int64_t actDim1Offset = tl_->actRight * tl_->outDim1;
  int64_t gateDim1Offset = tl_->outDim1 - actDim1Offset;
  int64_t gmOffset1 = blockIdx1_ * tl_->blockFormer1 + ubIdx1 * tl_->ubFormer1;
  int64_t curNlastSize = curSize0 * curSize1;

  int64_t xGmOffset = (gmOffset0_ * tl_->inDim1 + gmOffset1) * tl_->inDim2;
  DataCopyPadParams padParams{false, 0, 0, 0};
  LoopModeParams loopParams;
  LocalTensor<TXtype> xActLocal = xActQueue_.AllocTensor<TXtype>();
  loopParams.loop2Size = 1;
  loopParams.loop1Size = curSize0;
  loopParams.loop2SrcStride = 0;
  loopParams.loop2DstStride = 0;
  loopParams.loop1SrcStride = tl_->inDim1 * tl_->inDim2 * sizeof(TXtype);
  loopParams.loop1DstStride = curSize1 * xLastAlign_ * sizeof(TXtype);
  SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
  DataCopyParams dataCopyXParams;
  dataCopyXParams.blockCount = curSize1;
  dataCopyXParams.blockLen = tl_->inDim2 * sizeof(TXtype);
  dataCopyXParams.srcStride = 0;
  dataCopyXParams.dstStride = 0;
  DataCopyPad(xActLocal[0],
              xGm_[xGmOffset + actDim1Offset * tl_->inDim2],
              dataCopyXParams,
              padParams);
  DataCopyPad(xActLocal[xLastAlign_ * curNlastSize],
              xGm_[xGmOffset + gateDim1Offset * tl_->inDim2],
              dataCopyXParams,
              padParams);

  // copy_in: activation_scale(ubFormer, )
  LocalTensor<float> xActLocalFp32 =
      xActLocal.template ReinterpretCast<float>();
  int64_t actLocalOffset = xLastAlign_ * curNlastSize * SWI_FACTOR;
  if constexpr (hasActScale_) {
    loopParams.loop2Size = 1;
    loopParams.loop1Size = curSize0;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    loopParams.loop1SrcStride = tl_->inDim1 * sizeof(float);
    loopParams.loop1DstStride = curSize1 * aScaleAlignB32_ * sizeof(float);
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyParams dataCopyActScaleParams;
    dataCopyActScaleParams.blockCount = curSize1;
    dataCopyActScaleParams.blockLen = sizeof(float);
    dataCopyActScaleParams.srcStride = 0;
    dataCopyActScaleParams.dstStride = 0;
    // only support x int32
    DataCopyPad(xActLocalFp32[actLocalOffset],
                activationScaleGm_[gmOffset0_ * tl_->inDim1 + gmOffset1 +
                                   actDim1Offset],
                dataCopyActScaleParams,
                padParams);
    DataCopyPad(xActLocalFp32[actLocalOffset + curNlastSize * aScaleAlignB32_],
                activationScaleGm_[gmOffset0_ * tl_->inDim1 + gmOffset1 +
                                   gateDim1Offset],
                dataCopyActScaleParams,
                padParams);
  }
  ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
  xActQueue_.EnQue(xActLocal);
  xActLocal = xActQueue_.DeQue<TXtype>();

  LocalTensor<TYtype> yLocal = yQueue_.AllocTensor<TYtype>();
  LocalTensor<float> scaleLocal = scaleQueue_.AllocTensor<float>();
  LocalTensor<float> tmpXLocal = tmpBuffer.Get<float>();

  __local_mem__ float* tmpXPtr = (__local_mem__ float*)tmpXLocal.GetPhyAddr();
  __local_mem__ TYtype* yPtr = (__local_mem__ TYtype*)yLocal.GetPhyAddr();
  __local_mem__ uint8_t* yFp4Ptr = (__local_mem__ uint8_t*)yLocal.GetPhyAddr();
  __local_mem__ float* scalePtr = (__local_mem__ float*)scaleLocal.GetPhyAddr();
  __local_mem__ TXtype* x1Ptr = (__local_mem__ TXtype*)xActLocal.GetPhyAddr(0);
  __local_mem__ TXtype* x2Ptr =
      (__local_mem__ TXtype*)xActLocal.GetPhyAddr(xLastAlign_ * curNlastSize);
  __local_mem__ float* aScale1Ptr =
      (__local_mem__ float*)xActLocalFp32.GetPhyAddr(actLocalOffset);
  __local_mem__ float* aScale2Ptr =
      (__local_mem__ float*)xActLocalFp32.GetPhyAddr(
          actLocalOffset + curNlastSize * aScaleAlignB32_);
  __local_mem__ float* wScalePtr =
      (__local_mem__ float*)inScaleLocal.GetPhyAddr(0);
  __local_mem__ float* qScalePtr =
      (__local_mem__ float*)inScaleLocal.GetPhyAddr(xLastAlignB32_);

  if constexpr (hasBiasIndex_) {
    biasPtr = (__local_mem__ TBias*)biasLocal.GetPhyAddr(0);
  }

  __VEC_SCOPE__ {
    AscendC::MicroAPI::RegTensor<TXtype> vreg0, vreg10;
    AscendC::MicroAPI::RegTensor<float> vreg1, vreg2, vreg3, vreg4, vreg5,
        vreg6;
    AscendC::MicroAPI::RegTensor<float> vreg7, vreg8, vreg9, vreg11, vreg12;
    AscendC::MicroAPI::RegTensor<float> vreg13, vreg14, vreg15;
    AscendC::MicroAPI::RegTensor<int32_t> vreg16, verg18, vreg19;
    AscendC::MicroAPI::RegTensor<float> vreg20, vreg21;
    AscendC::MicroAPI::RegTensor<half> vreg24, vreg25;
    AscendC::MicroAPI::RegTensor<bfloat16_t> vreg26, vreg27;
    AscendC::MicroAPI::MaskReg mask;

    constexpr uint16_t sizePerRepeat = AscendC::GetVecLen() / sizeof(float);
    uint32_t width = tl_->inDim2;
    uint16_t repeatTimes = CeilDivision(tl_->inDim2, sizePerRepeat);
    const float scalarOne = 1.0;

    for (uint16_t j = 0; j < repeatTimes; j++) {
      mask = AscendC::MicroAPI::UpdateMask<uint32_t>(width);
      auto wScaleAddr = wScalePtr + j * sizePerRepeat;
      AscendC::MicroAPI::DataCopy(vreg2, wScaleAddr);
      for (uint16_t i = 0; i < static_cast<uint16_t>(curNlastSize); i++) {
        auto x1Addr = x1Ptr + i * xLastAlign_ + j * sizePerRepeat;
        auto x2Addr = x2Ptr + i * xLastAlign_ + j * sizePerRepeat;
        auto dstAddr = tmpXPtr + i * xLastAlign_ + j * sizePerRepeat;
        // vreg0 -> x1, vreg10 -> x2
        if constexpr (ifXIntIndex_) {
          AscendC::MicroAPI::DataCopy(vreg0, x1Addr);
          AscendC::MicroAPI::DataCopy(vreg10, x2Addr);
        } else {
          AscendC::MicroAPI::
              DataCopy<TXtype, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                  vreg0, x1Addr);
          AscendC::MicroAPI::
              DataCopy<TXtype, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                  vreg10, x2Addr);
        }

        if constexpr (ifXIntIndex_ && ifBiasIntIndex_) {
          auto biasAddr = biasPtr + j * sizePerRepeat;
          AscendC::MicroAPI::DataCopy(vreg16, biasAddr);
          // x + bias
          AscendC::MicroAPI::Add(vreg0, vreg0, vreg16, mask);
          AscendC::MicroAPI::Add(vreg10, vreg10, vreg16, mask);
        }

        // x:int32->float32
        if constexpr (ifXIntIndex_) {
          AscendC::MicroAPI::Cast<float, TXtype, CAST_INT32_TO_FP32>(
              vreg1, vreg0, mask);
          AscendC::MicroAPI::Cast<float, TXtype, CAST_INT32_TO_FP32>(
              vreg11, vreg10, mask);
          // x * weight
          AscendC::MicroAPI::Mul(vreg3, vreg1, vreg2, mask);
          AscendC::MicroAPI::Mul(vreg13, vreg11, vreg2, mask);
        } else {
          AscendC::MicroAPI::Cast<float, TXtype, CAST_BF16_FP16_TO_FP32>(
              vreg3, vreg0, mask);
          AscendC::MicroAPI::Cast<float, TXtype, CAST_BF16_FP16_TO_FP32>(
              vreg13, vreg10, mask);
        }

        // x * activation_scale
        if constexpr (hasActScale_) {
          auto aScale1Addr = aScale1Ptr + i * aScaleAlignB32_;
          AscendC::MicroAPI::
              DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                  vreg4, aScale1Addr);
          AscendC::MicroAPI::Mul(vreg3, vreg3, vreg4, mask);
          auto aScale2Addr = aScale2Ptr + i * aScaleAlignB32_;
          AscendC::MicroAPI::
              DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                  vreg12, aScale2Addr);
          AscendC::MicroAPI::Mul(vreg13, vreg13, vreg12, mask);
        }

        if constexpr (ifBiasFloatIndex_ || ifBiasFloat16Index_ ||
                      ifBiasBfloat16Index_) {
          auto biasAddr = biasPtr + j * sizePerRepeat;
          if constexpr (ifBiasFloatIndex_) {
            AscendC::MicroAPI::DataCopy(vreg20, biasAddr);
            AscendC::MicroAPI::DataCopy(vreg21, biasAddr);
          }
          if constexpr (ifBiasFloat16Index_) {
            AscendC::MicroAPI::
                DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vreg24, biasAddr);
            AscendC::MicroAPI::
                DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vreg25, biasAddr);
            AscendC::MicroAPI::Cast<float, half, CAST_BF16_FP16_TO_FP32>(
                vreg20, vreg24, mask);
            AscendC::MicroAPI::Cast<float, half, CAST_BF16_FP16_TO_FP32>(
                vreg21, vreg25, mask);
          }
          if constexpr (ifBiasBfloat16Index_) {
            AscendC::MicroAPI::DataCopy<
                bfloat16_t,
                AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg26, biasAddr);
            AscendC::MicroAPI::DataCopy<
                bfloat16_t,
                AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg27, biasAddr);
            AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(
                vreg20, vreg26, mask);
            AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(
                vreg21, vreg27, mask);
          }
          AscendC::MicroAPI::Add(vreg3, vreg3, vreg20, mask);
          AscendC::MicroAPI::Add(vreg13, vreg13, vreg21, mask);
        }
        // Swish
        AscendC::MicroAPI::Muls(vreg6, vreg3, -(scalarOne), mask);
        AscendC::MicroAPI::Exp(vreg7, vreg6, mask);
        AscendC::MicroAPI::Adds(vreg8, vreg7, scalarOne, mask);
        AscendC::MicroAPI::Div<float, &DIV_MODE>(vreg9, vreg3, vreg8, mask);

        // glu
        AscendC::MicroAPI::Mul(vreg15, vreg9, vreg13, mask);

        // store: reg->ub
        AscendC::MicroAPI::DataCopy(dstAddr, vreg15, mask);
      }
    }
  }  // __VEC_SCOPE__

  xActQueue_.FreeTensor(xActLocal);

  __VEC_SCOPE__ {
    AscendC::MicroAPI::RegTensor<float> vreg0, vreg1, vreg2, vreg3, vreg4,
        vreg5, vreg6, vreg7, vreg8, vregDiv;
    AscendC::MicroAPI::RegTensor<float> vregTmpX;
    AscendC::MicroAPI::RegTensor<float> vreg16;
    AscendC::MicroAPI::RegTensor<int16_t> vreg17;
    AscendC::MicroAPI::RegTensor<half> vreg18;
    AscendC::MicroAPI::RegTensor<int8_t> vreg19;
    AscendC::MicroAPI::MaskReg mask, maskTail, maskOne;

    constexpr uint16_t sizePerRepeat = AscendC::GetVecLen() / sizeof(float);
    uint16_t repeatTimes = CeilDivision(tl_->inDim2, sizePerRepeat);

    uint32_t block = static_cast<uint32_t>(sizePerRepeat);
    uint32_t tailBlock = tl_->inDim2 - sizePerRepeat * (repeatTimes - 1);
    uint32_t numOne = 1;
    mask = AscendC::MicroAPI::UpdateMask<uint32_t>(block);
    maskTail = AscendC::MicroAPI::UpdateMask<uint32_t>(tailBlock);
    maskOne = AscendC::MicroAPI::UpdateMask<uint32_t>(numOne);

    float scalarOne = 1.0;
    float scalarZero = 0;
    float scalarMaxNum = scalarMaxNum_;

    AscendC::MicroAPI::Duplicate(vregDiv, scalarMaxNum);

    for (uint16_t i = 0; i < static_cast<uint16_t>(curNlastSize); i++) {
      auto scaleAddr = scalePtr + i * aScaleAlignB32_;
      AscendC::MicroAPI::Duplicate(vregTmpX, scalarZero);

      uint16_t j = repeatTimes - 1;
      auto tmpXAddr = tmpXPtr + i * xLastAlign_ + j * sizePerRepeat;
      AscendC::MicroAPI::DataCopy(vreg0, tmpXAddr);

      // x * quant_scale
      if constexpr (hasQuantScale_) {
        auto qScaleAddr = qScalePtr + j * sizePerRepeat;
        AscendC::MicroAPI::DataCopy(vreg1, qScaleAddr);
        AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, maskTail);
        AscendC::MicroAPI::DataCopy(tmpXAddr, vreg0, maskTail);
      }

      AscendC::MicroAPI::Abs(vreg3, vreg0, maskTail);
      AscendC::MicroAPI::Max(vregTmpX, vregTmpX, vreg3, maskTail);

      for (uint16_t j = 0; j < static_cast<uint16_t>(repeatTimes - 1); j++) {
        auto tmpXAddr = tmpXPtr + i * xLastAlign_ + j * sizePerRepeat;
        AscendC::MicroAPI::DataCopy(vreg0, tmpXAddr);

        // x * quant_scale
        if constexpr (hasQuantScale_) {
          auto qScaleAddr = qScalePtr + j * sizePerRepeat;
          AscendC::MicroAPI::DataCopy(vreg1, qScaleAddr);
          AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
          AscendC::MicroAPI::DataCopy(tmpXAddr, vreg0, mask);
        }

        AscendC::MicroAPI::Abs(vreg3, vreg0, mask);
        AscendC::MicroAPI::Max(vregTmpX, vregTmpX, vreg3, mask);
      }

      AscendC::MicroAPI::ReduceMax(vreg4, vregTmpX, mask);
      AscendC::MicroAPI::Div(vreg5, vreg4, vregDiv, mask);
      AscendC::MicroAPI::DataCopy(scaleAddr, vreg5, maskOne);
    }
  }  // __VEC_SCOPE__

  scaleQueue_.EnQue<float>(scaleLocal);
  scaleLocal = scaleQueue_.DeQue<float>();

  int64_t outGmOffset = gmOffset0_ * tl_->outDim1 + gmOffset1;
  // copy_out: scale
  loopParams.loop2Size = 1;
  loopParams.loop1Size = curSize0;
  loopParams.loop2SrcStride = 0;
  loopParams.loop2DstStride = 0;
  loopParams.loop1SrcStride = curSize1 * aScaleAlignB32_ * sizeof(float);
  loopParams.loop1DstStride = tl_->outDim1 * sizeof(float);
  SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
  DataCopyParams dataCopyScaleParams;
  dataCopyScaleParams.blockCount = curSize1;
  dataCopyScaleParams.blockLen = sizeof(float);
  dataCopyScaleParams.srcStride = 0;
  dataCopyScaleParams.dstStride = 0;
  DataCopyPad(scaleGm_[outGmOffset], scaleLocal[0], dataCopyScaleParams);

  __VEC_SCOPE__ {
    constexpr uint16_t sizePerRepeat = AscendC::GetVecLen() / sizeof(float);
    uint16_t repeatTimes = CeilDivision(tl_->inDim2, sizePerRepeat);
    uint32_t fp4ElemNum = static_cast<uint32_t>(sizePerRepeat) / FP4_WEIGHT;

    AscendC::MicroAPI::RegTensor<float> vreg6, vreg7, vreg8;
    AscendC::MicroAPI::RegTensor<int16_t> vreg9;
    AscendC::MicroAPI::RegTensor<half> vreg10;
    AscendC::MicroAPI::RegTensor<int8_t> vreg11;
    AscendC::MicroAPI::RegTensor<fp8_e4m3fn_t> vreg12;
    AscendC::MicroAPI::RegTensor<fp8_e5m2_t> vreg13;
    AscendC::MicroAPI::RegTensor<bfloat16_t> vreg14, vreg15;
    AscendC::MicroAPI::RegTensor<fp4x2_e2m1_t> vreg16;
    AscendC::MicroAPI::RegTensor<fp4x2_e1m2_t> vreg17;
    AscendC::MicroAPI::RegTensor<hifloat8_t> vreg18;

    AscendC::MicroAPI::MaskReg mask;
    AscendC::MicroAPI::MaskReg maskFP4;
    maskFP4 = AscendC::MicroAPI::UpdateMask<uint32_t>(fp4ElemNum);
    for (uint16_t i = 0; i < static_cast<uint16_t>(curNlastSize); i++) {
      auto scaleAddr = scalePtr + i * aScaleAlignB32_;
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
          vreg6, scaleAddr);

      uint32_t width = static_cast<uint32_t>(tl_->inDim2);
      for (uint16_t j = 0; j < repeatTimes; j++) {
        mask = AscendC::MicroAPI::UpdateMask<uint32_t>(width);

        auto tmpXAddr = tmpXPtr + i * xLastAlign_ + j * sizePerRepeat;
        auto yAddr = yPtr + i * yLastAlign_ + j * sizePerRepeat;
        auto yFp4Addr =
            yFp4Ptr + i * yLastAlignB4_ + j * sizePerRepeat / FP4_WEIGHT;

        AscendC::MicroAPI::DataCopy(vreg7, tmpXAddr);
        AscendC::MicroAPI::Div(vreg8, vreg7, vreg6, mask);

        if constexpr (ifYFloat8e4m3Index_) {
          // float32 -> float8_e4m3
          AscendC::MicroAPI::Cast<fp8_e4m3fn_t, float, CAST_FP32_TO_FP8>(
              vreg12, vreg8, mask);
          AscendC::MicroAPI::DataCopy<
              fp8_e4m3fn_t,
              AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
              yAddr, vreg12, mask);
        } else if constexpr (ifYFloat8e5m2Index_) {
          // float32 -> float8_e5m2
          AscendC::MicroAPI::Cast<fp8_e5m2_t, float, CAST_FP32_TO_FP8>(
              vreg13, vreg8, mask);
          AscendC::MicroAPI::DataCopy<
              fp8_e5m2_t,
              AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
              yAddr, vreg13, mask);
        } else if constexpr (ifYFloat4e2m1Index_) {
          // float32 -> bfloat16 -> float4_e2m1
          AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(
              vreg14, vreg8, mask);
          AscendC::MicroAPI::Pack(
              (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg14,
              (AscendC::MicroAPI::RegTensor<uint32_t>&)vreg14);
          if (roundMode_ == 1) {
            AscendC::MicroAPI::
                Cast<fp4x2_e2m1_t, bfloat16_t, CAST_BF16_TO_FP4_ROUND>(
                    vreg16, vreg14, mask);
          } else if (roundMode_ == 2) {
            AscendC::MicroAPI::
                Cast<fp4x2_e2m1_t, bfloat16_t, CAST_BF16_TO_FP4_FLOOR>(
                    vreg16, vreg14, mask);
          } else if (roundMode_ == 3) {
            AscendC::MicroAPI::
                Cast<fp4x2_e2m1_t, bfloat16_t, CAST_BF16_TO_FP4_CEIL>(
                    vreg16, vreg14, mask);
          } else if (roundMode_ == 4) {
            AscendC::MicroAPI::
                Cast<fp4x2_e2m1_t, bfloat16_t, CAST_BF16_TO_FP4_TRUNC>(
                    vreg16, vreg14, mask);
          } else {
            AscendC::MicroAPI::
                Cast<fp4x2_e2m1_t, bfloat16_t, CAST_BF16_TO_FP4_RINT>(
                    vreg16, vreg14, mask);
          }
          AscendC::MicroAPI::
              DataCopy<uint8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                  yFp4Addr,
                  (AscendC::MicroAPI::RegTensor<uint8_t>&)vreg16,
                  maskFP4);
        } else if constexpr (ifYFloat4e1m2Index_) {
          // float32 -> bfloat16 -> float4_e1m2
          AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(
              vreg15, vreg8, mask);
          AscendC::MicroAPI::Pack(
              (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg15,
              (AscendC::MicroAPI::RegTensor<uint32_t>&)vreg15);
          if (roundMode_ == 1) {
            AscendC::MicroAPI::
                Cast<fp4x2_e1m2_t, bfloat16_t, CAST_BF16_TO_FP4_ROUND>(
                    vreg17, vreg15, mask);
          } else if (roundMode_ == 2) {
            AscendC::MicroAPI::
                Cast<fp4x2_e1m2_t, bfloat16_t, CAST_BF16_TO_FP4_FLOOR>(
                    vreg17, vreg15, mask);
          } else if (roundMode_ == 3) {
            AscendC::MicroAPI::
                Cast<fp4x2_e1m2_t, bfloat16_t, CAST_BF16_TO_FP4_CEIL>(
                    vreg17, vreg15, mask);
          } else if (roundMode_ == 4) {
            AscendC::MicroAPI::
                Cast<fp4x2_e1m2_t, bfloat16_t, CAST_BF16_TO_FP4_TRUNC>(
                    vreg17, vreg15, mask);
          } else {
            AscendC::MicroAPI::
                Cast<fp4x2_e1m2_t, bfloat16_t, CAST_BF16_TO_FP4_RINT>(
                    vreg17, vreg15, mask);
          }
          AscendC::MicroAPI::
              DataCopy<uint8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                  yFp4Addr,
                  (AscendC::MicroAPI::RegTensor<uint8_t>&)vreg17,
                  maskFP4);
        } else if constexpr (ifYInt8Index_) {
          // float32 -> int8
          AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(
              vreg9, vreg8, mask);
          AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(
              vreg10, vreg9, mask);
          AscendC::MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(
              vreg11, vreg10, mask);

          AscendC::MicroAPI::
              DataCopy<int8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                  yAddr, vreg11, mask);
        } else if constexpr (ifYHiFloat8Index_) {
          // float32 -> hifloat8
          AscendC::MicroAPI::Cast<hifloat8_t, float, CAST_FP32_TO_HI8>(
              vreg18, vreg8, mask);
          AscendC::MicroAPI::DataCopy<
              hifloat8_t,
              AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
              yAddr, vreg18, mask);
        }
      }
    }
  }  // __VEC_SCOPE__
  scaleQueue_.FreeTensor(scaleLocal);

  yQueue_.EnQue<TYtype>(yLocal);
  yLocal = yQueue_.DeQue<TYtype>();
  int64_t yGmOffset = outGmOffset * tl_->inDim2;
  // copy_out: y
  if constexpr (ifYFloat4e2m1Index_ || ifYFloat4e1m2Index_) {
    loopParams.loop2Size = 1;
    loopParams.loop1Size = curSize0;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    loopParams.loop1SrcStride = curSize1 * yLastAlignB4_ * sizeof(uint8_t);
    loopParams.loop1DstStride =
        tl_->outDim1 * tl_->inDim2 / FP4_WEIGHT * sizeof(uint8_t);
    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
    DataCopyParams dataCopyYParams;
    dataCopyYParams.blockCount = curSize1;
    dataCopyYParams.blockLen = tl_->inDim2 * sizeof(uint8_t) / FP4_WEIGHT;
    dataCopyYParams.srcStride = 0;
    dataCopyYParams.dstStride = 0;
    DataCopyPad(
        yGm_.template ReinterpretCast<uint8_t>()[yGmOffset / FP4_WEIGHT],
        yLocal.template ReinterpretCast<uint8_t>()[0],
        dataCopyYParams);
    yQueue_.FreeTensor(yLocal);
  } else {
    loopParams.loop2Size = 1;
    loopParams.loop1Size = curSize0;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    loopParams.loop1SrcStride = curSize1 * yLastAlign_ * sizeof(TYtype);
    loopParams.loop1DstStride = tl_->outDim1 * tl_->inDim2 * sizeof(TYtype);
    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
    DataCopyParams dataCopyYParams;
    dataCopyYParams.blockCount = curSize1;
    dataCopyYParams.blockLen = tl_->inDim2 * sizeof(TYtype);
    dataCopyYParams.srcStride = 0;
    dataCopyYParams.dstStride = 0;
    DataCopyPad(yGm_[yGmOffset], yLocal[0], dataCopyYParams);
    yQueue_.FreeTensor(yLocal);
  }
  ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
}

}  // namespace DequantSwigluQuantV35Ops
#endif  // DEQUANT_SWIGLU_QUANT_NLAST_H
