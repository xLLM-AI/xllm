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
 * \file dequant_swiglu_quant_common.h
 * \brief
 */

#ifndef DEQUANT_SWIGLU_QUANT_COMMON_H
#define DEQUANT_SWIGLU_QUANT_COMMON_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace DequantSwigluQuantV35Ops {
using namespace AscendC;

constexpr static uint32_t DOUBLE_BUFFER = 2;
constexpr static uint32_t BLOCK_SIZE = 32;  // 32B
constexpr static uint32_t BLOCK_ELEM_B32 = BLOCK_SIZE / sizeof(float);
constexpr static uint32_t BLOCK_ELEM_B8 = BLOCK_SIZE / sizeof(int8_t);
constexpr static uint32_t INT8SYMBOL = 2;         // dstType:int8
constexpr static uint32_t FLOAT8E5M2SYMBOL = 35;  // dstType:float8e5m2
constexpr static uint32_t FLOAT8E4M3SYMBOL = 36;  // dstType:float8e4m3
constexpr static uint32_t FLOAT4E2M1SYMBOL = 40;  // dstType:float4e2m1
constexpr static uint32_t FLOAT4E1M2SYMBOL = 41;  // dstType:float4e1m2
constexpr static int64_t SWI_FACTOR = 2;
constexpr static int64_t FP4_WEIGHT = 2;

static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_FP16_TO_FP32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_TO_FP4 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_TO_FP4_RINT = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_TO_FP4_ROUND = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};
static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_TO_FP4_FLOOR = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_FLOOR};
static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_TO_FP4_CEIL = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_CEIL};
static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_TO_FP4_TRUNC = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC};
static constexpr AscendC::MicroAPI::CastTrait CAST_INT32_TO_FP32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_INT16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_INT16_TO_FP16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};
static constexpr AscendC::MicroAPI::CastTrait CAST_FP16_TO_INT8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC};
constexpr static AscendC::MicroAPI::CastTrait CAST_FP32_TO_HI8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_ROUND};
static constexpr AscendC::MicroAPI::DivSpecificMode DIV_MODE = {
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    true,
};

template <typename TXtype, bool ifXFloat16Index_, bool ifXBf16Index_>
__aicore__ inline void FloatDequant(__local_mem__ TXtype* xPtr,
                                    __local_mem__ float* dstPtr,
                                    uint32_t repeatTimes,
                                    uint32_t sizePerRepeat,
                                    uint32_t count) {
  __local_mem__ TXtype* xAddr;
  __local_mem__ float* dstAddr;
  AscendC::MicroAPI::RegTensor<TXtype> vreg0;
  AscendC::MicroAPI::RegTensor<float> vreg1;
  AscendC::MicroAPI::MaskReg mask;

  for (uint16_t vfRepeat = 0; vfRepeat < static_cast<uint16_t>(repeatTimes);
       vfRepeat++) {
    mask = AscendC::MicroAPI::UpdateMask<uint32_t>(count);
    xAddr = xPtr + vfRepeat * sizePerRepeat;
    dstAddr = dstPtr + vfRepeat * sizePerRepeat;
    if constexpr (ifXFloat16Index_) {
      AscendC::MicroAPI::DataCopy<half,
                                  AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
          vreg0, xAddr);
    }
    if constexpr (ifXBf16Index_) {
      AscendC::MicroAPI::DataCopy<bfloat16_t,
                                  AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
          vreg0, xAddr);
    }
    AscendC::MicroAPI::Cast<float, TXtype, CAST_BF16_FP16_TO_FP32>(
        vreg1, vreg0, mask);
    AscendC::MicroAPI::DataCopy(dstAddr, vreg1, mask);
  }
}

template <typename TXtype,
          typename TBias,
          bool hasBiasIndex_,
          bool hasActScale_>
__aicore__ inline void Int32Dequant(__ubuf__ TXtype* xPtr,
                                    __ubuf__ float* dstPtr,
                                    __ubuf__ float* wScalePtr,
                                    __ubuf__ float* aScalePtr,
                                    __ubuf__ TBias* biasPtr,
                                    uint16_t repeatTimes,
                                    uint16_t sizePerRepeat,
                                    uint32_t count) {
  AscendC::MicroAPI::RegTensor<TXtype> vreg0, vreg10;
  AscendC::MicroAPI::RegTensor<float> vreg1, vreg2, vreg3, vreg4, vreg5, vreg6;
  AscendC::MicroAPI::RegTensor<float> vreg7, vreg8, vreg9, vreg11, vreg12;
  AscendC::MicroAPI::RegTensor<float> vreg13, vreg14, vreg15;
  AscendC::MicroAPI::RegTensor<int32_t> vreg16, vreg17, verg18, vreg19;
  AscendC::MicroAPI::RegTensor<float> vreg20, vreg21;
  AscendC::MicroAPI::RegTensor<half> vreg24, vreg25;
  AscendC::MicroAPI::RegTensor<bfloat16_t> vreg26, vreg27;
  AscendC::MicroAPI::MaskReg mask;

  for (uint16_t j = 0; j < repeatTimes; j++) {
    mask = AscendC::MicroAPI::UpdateMask<uint32_t>(count);

    auto wScaleAddr = wScalePtr + j * sizePerRepeat;
    AscendC::MicroAPI::DataCopy(vreg2, wScaleAddr);
    auto xAddr = xPtr + j * sizePerRepeat;
    auto dstAddr = dstPtr + j * sizePerRepeat;
    // vreg0 -> x1, vreg10 -> x2
    AscendC::MicroAPI::DataCopy(vreg0, xAddr);
    if constexpr (hasBiasIndex_) {
      if constexpr (IsSameType<TBias, int32_t>::value) {
        auto biasAddr = biasPtr + j * sizePerRepeat;
        AscendC::MicroAPI::DataCopy(vreg16, biasAddr);
        // x + bias
        AscendC::MicroAPI::Add(vreg0, vreg0, vreg16, mask);
      }
    }
    // x:int32->float32
    AscendC::MicroAPI::Cast<float, TXtype, CAST_INT32_TO_FP32>(
        vreg1, vreg0, mask);
    // x * weight
    AscendC::MicroAPI::Mul(vreg3, vreg1, vreg2, mask);
    // x * activation_scale
    if constexpr (hasActScale_) {
      auto aScaleAddr = aScalePtr;
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
          vreg4, aScaleAddr);
      AscendC::MicroAPI::Mul(vreg3, vreg3, vreg4, mask);
    }
    if constexpr (hasBiasIndex_ && !IsSameType<TBias, int32_t>::value) {
      auto biasAddr = biasPtr + j * sizePerRepeat;
      if constexpr (IsSameType<TBias, float>::value) {
        AscendC::MicroAPI::DataCopy(vreg20, biasAddr);
      }
      if constexpr (IsSameType<TBias, half>::value) {
        AscendC::MicroAPI::
            DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vreg24, biasAddr);
        AscendC::MicroAPI::Cast<float, half, CAST_BF16_FP16_TO_FP32>(
            vreg20, vreg24, mask);
      }
      if constexpr (IsSameType<TBias, bfloat16_t>::value) {
        AscendC::MicroAPI::
            DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vreg26, biasAddr);
        AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(
            vreg20, vreg26, mask);
      }
      AscendC::MicroAPI::Add(vreg3, vreg3, vreg20, mask);
    }
    // store: reg->ub
    AscendC::MicroAPI::DataCopy(dstAddr, vreg3, mask);
  }
}  // Int32Dequant end

template <bool hasQuantScale, bool quantIsOne>
__aicore__ inline void SwigluSingleYWithQuantScale(
    __local_mem__ float* xPtr,
    __local_mem__ float* qScalePtr,
    __local_mem__ float* dstPtr,
    uint32_t offset,
    uint32_t repeatTimes,
    uint32_t sizePerRepeat,
    uint32_t count,
    bool ifQuantIsOne) {
  float scalarOne = 1.0f;
  uint32_t maskScalarOne = 1;
  __local_mem__ float* xActAddr;
  __local_mem__ float* xGateAddr;
  __local_mem__ float* dstAddr;
  __local_mem__ float* qScaleAddr;
  AscendC::MicroAPI::RegTensor<float> vreg1;
  AscendC::MicroAPI::RegTensor<float> vreg2;
  AscendC::MicroAPI::RegTensor<float> vreg6;
  AscendC::MicroAPI::RegTensor<float> vreg7;
  AscendC::MicroAPI::RegTensor<float> vreg8;
  AscendC::MicroAPI::RegTensor<float> vreg9;
  AscendC::MicroAPI::RegTensor<float> vreg15;
  AscendC::MicroAPI::RegTensor<float> vreg16;
  AscendC::MicroAPI::RegTensor<float> vreg17;
  AscendC::MicroAPI::MaskReg mask;
  AscendC::MicroAPI::MaskReg maskForScale;

  for (uint16_t vfRepeat = 0; vfRepeat < static_cast<uint16_t>(repeatTimes);
       vfRepeat++) {
    mask = AscendC::MicroAPI::UpdateMask<uint32_t>(count);
    xActAddr = xPtr + vfRepeat * sizePerRepeat;
    xGateAddr = xActAddr + offset;
    dstAddr = dstPtr + vfRepeat * sizePerRepeat;
    AscendC::MicroAPI::DataCopy(vreg1, xActAddr);
    AscendC::MicroAPI::DataCopy(vreg2, xGateAddr);

    AscendC::MicroAPI::Muls(vreg6, vreg1, -(scalarOne), mask);
    AscendC::MicroAPI::Exp(vreg7, vreg6, mask);
    AscendC::MicroAPI::Adds(vreg8, vreg7, scalarOne, mask);
    AscendC::MicroAPI::Div(vreg9, vreg1, vreg8, mask);
    AscendC::MicroAPI::Mul(vreg15, vreg9, vreg2, mask);

    if constexpr (hasQuantScale) {
      if constexpr (quantIsOne) {
        AscendC::MicroAPI::DataCopy(vreg16, qScalePtr);
        AscendC::MicroAPI::Duplicate(vreg17, vreg16, mask);
        AscendC::MicroAPI::Mul(vreg15, vreg15, vreg17, mask);
      } else {
        qScaleAddr = qScalePtr + vfRepeat * sizePerRepeat;
        AscendC::MicroAPI::DataCopy(vreg16, qScaleAddr);
        AscendC::MicroAPI::Div(vreg15, vreg15, vreg16, mask);
      }
    }
    AscendC::MicroAPI::DataCopy(dstAddr, vreg15, mask);
  }
}

template <bool HasQuantScale, bool quantIsOne>
__aicore__ inline void SwigluV2SingleYWithQuantScale(
    __ubuf__ float* x1Ptr,
    __ubuf__ float* quantScalePtr,
    __ubuf__ float* dstPtr,
    uint32_t count,
    uint32_t offset,
    uint32_t xDimPerLoop,
    uint16_t repeatTimes,
    uint16_t sizePerRepeat,
    uint32_t xTypeUbAlignB32,
    bool ifQuantIsOne,
    float clampLit,
    float gluAlpha,
    float gluBias) {
  // x
  AscendC::MicroAPI::RegTensor<float> vregX0, vregX1;
  AscendC::MicroAPI::RegTensor<float> vreg0, vreg1, vreg2, vreg3, vreg4, vreg5,
      vreg6, vreg7;
  // quant_scale
  AscendC::MicroAPI::RegTensor<float> vregQuantScale;
  AscendC::MicroAPI::MaskReg mask, tailMask;

  const float scalarOne = 1.0;
  uint32_t tailCount = count - (repeatTimes - 1) * sizePerRepeat;
  uint32_t tailWidth = 0;
  uint32_t width = sizePerRepeat;
  uint32_t ifEven = 0;

  if ((repeatTimes & 1) == 0) {
    tailWidth = (64 + tailCount) / 2;
    ifEven = 1;
  } else {
    tailWidth = tailCount / 2;
  }

  mask = AscendC::MicroAPI::UpdateMask<uint32_t>(width);
  tailMask = AscendC::MicroAPI::UpdateMask<uint32_t>(tailWidth);
  repeatTimes = CeilDivision(repeatTimes, 2);

  for (uint16_t j = 0; j < static_cast<uint16_t>(xDimPerLoop); j++) {
    for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes - 1); i++) {
      auto x1Addr = x1Ptr + j * xTypeUbAlignB32 + i * sizePerRepeat * 2;
      auto x2Addr = x1Addr + sizePerRepeat;
      auto dstAddr = dstPtr + j * xTypeUbAlignB32 + i * sizePerRepeat;

      // ub -> reg
      AscendC::MicroAPI::DataCopy(vregX0, x1Addr);
      AscendC::MicroAPI::DataCopy(vregX1, x2Addr);

      AscendC::MicroAPI::DeInterleave(vreg0, vreg1, vregX0, vregX1);

      // swish x1:activate, x2:gate
      AscendC::MicroAPI::Mins(vreg0, vreg0, clampLit, mask);
      AscendC::MicroAPI::Muls(vreg2, vreg0, -gluAlpha, mask);
      AscendC::MicroAPI::Exp(vreg3, vreg2, mask);
      AscendC::MicroAPI::Adds(vreg4, vreg3, scalarOne, mask);
      AscendC::MicroAPI::Div(vreg5, vreg0, vreg4, mask);

      // glu
      AscendC::MicroAPI::Mins(vreg1, vreg1, clampLit, mask);
      AscendC::MicroAPI::Maxs(vreg1, vreg1, -clampLit, mask);
      AscendC::MicroAPI::Adds(vreg1, vreg1, gluBias, mask);

      AscendC::MicroAPI::Mul(vreg6, vreg5, vreg1, mask);

      if constexpr (HasQuantScale) {
        if constexpr (quantIsOne) {
          AscendC::MicroAPI::
              DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                  vregQuantScale, quantScalePtr);
          AscendC::MicroAPI::Mul(vreg6, vreg6, vregQuantScale, mask);
        } else {
          auto quantScaleAddr = quantScalePtr + i * sizePerRepeat;
          AscendC::MicroAPI::DataCopy(vregQuantScale, quantScaleAddr);
          AscendC::MicroAPI::Div(vreg6, vreg6, vregQuantScale, mask);
        }
      }

      // reg -> ub
      AscendC::MicroAPI::DataCopy(dstAddr, vreg6, mask);
    }

    uint16_t i = repeatTimes - 1;
    auto x1Addr = x1Ptr + j * xTypeUbAlignB32 + i * sizePerRepeat * 2;
    auto dstAddr = dstPtr + j * xTypeUbAlignB32 + i * sizePerRepeat;
    if (ifEven == 1) {
      // ub -> reg
      auto x2Addr = x1Addr + sizePerRepeat;
      AscendC::MicroAPI::DataCopy(vregX0, x1Addr);
      AscendC::MicroAPI::DataCopy(vregX1, x2Addr);

      AscendC::MicroAPI::DeInterleave(vreg0, vreg1, vregX0, vregX1);
    } else {
      // ub -> reg
      AscendC::MicroAPI::DataCopy(vregX0, x1Addr);
      AscendC::MicroAPI::DeInterleave(vreg0, vreg1, vregX0, vregX0);
    }

    // swish x1:activate, x2:gate
    AscendC::MicroAPI::Mins(vreg0, vreg0, clampLit, tailMask);
    AscendC::MicroAPI::Muls(vreg2, vreg0, -gluAlpha, tailMask);
    AscendC::MicroAPI::Exp(vreg3, vreg2, tailMask);
    AscendC::MicroAPI::Adds(vreg4, vreg3, scalarOne, tailMask);
    AscendC::MicroAPI::Div(vreg5, vreg0, vreg4, tailMask);

    // glu
    AscendC::MicroAPI::Mins(vreg1, vreg1, clampLit, tailMask);
    AscendC::MicroAPI::Maxs(vreg1, vreg1, -clampLit, tailMask);
    AscendC::MicroAPI::Adds(vreg1, vreg1, gluBias, tailMask);

    AscendC::MicroAPI::Mul(vreg6, vreg5, vreg1, tailMask);

    if constexpr (HasQuantScale) {
      if constexpr (quantIsOne) {
        AscendC::MicroAPI::DataCopy<float,
                                    AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
            vregQuantScale, quantScalePtr);
        AscendC::MicroAPI::Mul(vreg6, vreg6, vregQuantScale, tailMask);
      } else {
        auto quantScaleTailAddr = quantScalePtr + i * sizePerRepeat;
        AscendC::MicroAPI::DataCopy(vregQuantScale, quantScaleTailAddr);
        AscendC::MicroAPI::Div(vreg6, vreg6, vregQuantScale, tailMask);
      }
    }

    // reg -> ub
    AscendC::MicroAPI::DataCopy(dstAddr, vreg6, tailMask);
  }
}  // SwigluV2SingleYWithQuantScale end

__aicore__ inline void StaticQuant(__ubuf__ float* xPtr,
                                   __ubuf__ float* quantScalePtr,
                                   __ubuf__ float* quantOffsetPtr,
                                   __ubuf__ float* dstPtr,
                                   uint32_t count,
                                   uint32_t xDimPerLoop,
                                   uint16_t repeatTimes,
                                   uint16_t sizePerRepeat,
                                   uint32_t xTypeUbAlignB32,
                                   uint16_t quantIsOne,
                                   bool hasQuantOffset) {
  // swiglu(x) * quant_scale
  AscendC::MicroAPI::RegTensor<float> vreg0;
  // quant_scale,quant_offset
  AscendC::MicroAPI::RegTensor<float> vregQuantScale, vregQuantOffset;
  // mask
  AscendC::MicroAPI::MaskReg mask;

  for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); i++) {
    mask = AscendC::MicroAPI::UpdateMask<uint32_t>(count);
    auto quantScaleAddr = quantScalePtr + i * sizePerRepeat;

    AscendC::MicroAPI::DataCopy(vregQuantScale, quantScaleAddr);

    for (uint16_t j = 0; j < static_cast<uint16_t>(xDimPerLoop); j++) {
      auto xAddr = xPtr + j * xTypeUbAlignB32 + i * sizePerRepeat;
      auto dstAddr = dstPtr + j * xTypeUbAlignB32 + i * sizePerRepeat;

      // ub -> reg
      AscendC::MicroAPI::DataCopy(vreg0, xAddr);
      AscendC::MicroAPI::Div(vreg0, vreg0, vregQuantScale, mask);

      // reg -> ub
      AscendC::MicroAPI::DataCopy(dstAddr, vreg0, mask);
    }
  }

}  // StaticQuant end

__aicore__ inline void StaticQuantWithQuantOffset(
    __ubuf__ float* xPtr,
    __ubuf__ float* quantScalePtr,
    __ubuf__ float* quantOffsetPtr,
    __ubuf__ float* dstPtr,
    uint32_t count,
    uint32_t xDimPerLoop,
    uint16_t repeatTimes,
    uint16_t sizePerRepeat,
    uint32_t xTypeUbAlignB32,
    uint16_t quantIsOne,
    bool hasQuantOffset) {
  // swiglu(x) * quant_scale
  AscendC::MicroAPI::RegTensor<float> vreg0;
  // quant_scale,quant_offset
  AscendC::MicroAPI::RegTensor<float> vregQuantScale, vregQuantOffset;
  // mask
  AscendC::MicroAPI::MaskReg mask;

  for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); i++) {
    mask = AscendC::MicroAPI::UpdateMask<uint32_t>(count);
    auto quantScaleAddr = quantScalePtr + i * sizePerRepeat;
    auto quantOffsetAddr = quantOffsetPtr + i * sizePerRepeat;

    AscendC::MicroAPI::DataCopy(vregQuantScale, quantScaleAddr);
    AscendC::MicroAPI::DataCopy(vregQuantOffset, quantOffsetAddr);

    for (uint16_t j = 0; j < static_cast<uint16_t>(xDimPerLoop); j++) {
      auto xAddr = xPtr + j * xTypeUbAlignB32 + i * sizePerRepeat;
      auto dstAddr = dstPtr + j * xTypeUbAlignB32 + i * sizePerRepeat;

      // ub -> reg
      AscendC::MicroAPI::DataCopy(vreg0, xAddr);
      AscendC::MicroAPI::Div(vreg0, vreg0, vregQuantScale, mask);
      AscendC::MicroAPI::Add(vreg0, vreg0, vregQuantOffset, mask);

      // reg -> ub
      AscendC::MicroAPI::DataCopy(dstAddr, vreg0, mask);
    }
  }

}  // StaticQuantWithQuantOffset end

__aicore__ inline void StaticQuantWithOne(__ubuf__ float* xPtr,
                                          __ubuf__ float* quantScalePtr,
                                          __ubuf__ float* quantOffsetPtr,
                                          __ubuf__ float* dstPtr,
                                          uint32_t count,
                                          uint32_t xDimPerLoop,
                                          uint16_t repeatTimes,
                                          uint16_t sizePerRepeat,
                                          uint32_t xTypeUbAlignB32,
                                          uint16_t quantIsOne,
                                          bool hasQuantOffset) {
  // swiglu(x) * quant_scale
  AscendC::MicroAPI::RegTensor<float> vreg0;
  // quant_scale,quant_offset
  AscendC::MicroAPI::RegTensor<float> vregQuantScale, vregQuantOffset;
  // mask
  AscendC::MicroAPI::MaskReg mask;

  for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); i++) {
    mask = AscendC::MicroAPI::UpdateMask<uint32_t>(count);
    auto quantScaleAddr = quantScalePtr;

    AscendC::MicroAPI::DataCopy<float,
                                AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
        vregQuantScale, quantScaleAddr);

    for (uint16_t j = 0; j < static_cast<uint16_t>(xDimPerLoop); j++) {
      auto xAddr = xPtr + j * xTypeUbAlignB32 + i * sizePerRepeat;
      auto dstAddr = dstPtr + j * xTypeUbAlignB32 + i * sizePerRepeat;

      // ub -> reg
      AscendC::MicroAPI::DataCopy(vreg0, xAddr);
      AscendC::MicroAPI::Mul(vreg0, vreg0, vregQuantScale, mask);

      // reg -> ub
      AscendC::MicroAPI::DataCopy(dstAddr, vreg0, mask);
    }
  }

}  // StaticQuantWithOne end

__aicore__ inline void StaticQuantWithQuantOffsetOne(
    __ubuf__ float* xPtr,
    __ubuf__ float* quantScalePtr,
    __ubuf__ float* quantOffsetPtr,
    __ubuf__ float* dstPtr,
    uint32_t count,
    uint32_t xDimPerLoop,
    uint16_t repeatTimes,
    uint16_t sizePerRepeat,
    uint32_t xTypeUbAlignB32,
    uint16_t quantIsOne,
    bool hasQuantOffset) {
  // swiglu(x) * quant_scale
  AscendC::MicroAPI::RegTensor<float> vreg0;
  // quant_scale,quant_offset
  AscendC::MicroAPI::RegTensor<float> vregQuantScale, vregQuantOffset;
  // mask
  AscendC::MicroAPI::MaskReg mask;

  for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); i++) {
    mask = AscendC::MicroAPI::UpdateMask<uint32_t>(count);
    auto quantScaleAddr = quantScalePtr;
    auto quantOffsetAddr = quantOffsetPtr;

    AscendC::MicroAPI::DataCopy<float,
                                AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
        vregQuantScale, quantScaleAddr);
    AscendC::MicroAPI::DataCopy<float,
                                AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
        vregQuantOffset, quantOffsetAddr);

    for (uint16_t j = 0; j < static_cast<uint16_t>(xDimPerLoop); j++) {
      auto xAddr = xPtr + j * xTypeUbAlignB32 + i * sizePerRepeat;
      auto dstAddr = dstPtr + j * xTypeUbAlignB32 + i * sizePerRepeat;

      // ub -> reg
      AscendC::MicroAPI::DataCopy(vreg0, xAddr);
      AscendC::MicroAPI::Mul(vreg0, vreg0, vregQuantScale, mask);
      AscendC::MicroAPI::Add(vreg0, vreg0, vregQuantOffset, mask);

      // reg -> ub
      AscendC::MicroAPI::DataCopy(dstAddr, vreg0, mask);
    }
  }

}  // StaticQuantWithQuantoffsetOne end

template <bool hasQuantOffset, bool quantIsOne>
__aicore__ inline void StaticQuantSingleY(__local_mem__ float* xPtr,
                                          __local_mem__ float* qOffsetPtr,
                                          __local_mem__ float* dstPtr,
                                          uint32_t repeatTimes,
                                          uint32_t sizePerRepeat,
                                          uint32_t count,
                                          bool ifQuantIsOne) {
  uint32_t maskScalarOne = 1;
  __local_mem__ float* xAddr;
  __local_mem__ float* dstAddr;
  __local_mem__ float* qOffsetAddr;
  AscendC::MicroAPI::RegTensor<float> vreg0;
  AscendC::MicroAPI::RegTensor<float> vreg1;
  AscendC::MicroAPI::RegTensor<float> vreg2;
  AscendC::MicroAPI::RegTensor<float> vreg3;
  AscendC::MicroAPI::RegTensor<float> vreg4;
  AscendC::MicroAPI::MaskReg mask;
  AscendC::MicroAPI::MaskReg maskForScale;

  for (uint16_t vfRepeat = 0; vfRepeat < static_cast<uint16_t>(repeatTimes);
       vfRepeat++) {
    mask = AscendC::MicroAPI::UpdateMask<uint32_t>(count);
    xAddr = xPtr + vfRepeat * sizePerRepeat;
    dstAddr = dstPtr + vfRepeat * sizePerRepeat;
    AscendC::MicroAPI::DataCopy(vreg1, xAddr);
    if constexpr (hasQuantOffset) {
      if constexpr (quantIsOne) {
        AscendC::MicroAPI::DataCopy(vreg2, qOffsetPtr);
        // AscendC::MicroAPI::Duplicate(vreg3, vreg2, maskForScale);
        AscendC::MicroAPI::Duplicate(vreg3, vreg2, mask);
        AscendC::MicroAPI::Add(vreg1, vreg1, vreg3, mask);
      } else {
        qOffsetAddr = qOffsetPtr + vfRepeat * sizePerRepeat;
        AscendC::MicroAPI::DataCopy(vreg2, qOffsetAddr);
        AscendC::MicroAPI::Add(vreg1, vreg1, vreg2, mask);
      }
    }
    AscendC::MicroAPI::DataCopy(dstAddr, vreg1, mask);
  }
}

template <typename TXtype,
          typename TYtype,
          bool ifYFloat8e4m3Index_,
          bool ifYFloat8e5m2Index_,
          bool ifYFloat4e2m1Index_,
          bool ifYFloat4e1m2Index_,
          bool ifYHiFloat8Index_>
__aicore__ inline void CastY(__ubuf__ float* xPtr,
                             __ubuf__ TYtype* yPtr,
                             __ubuf__ uint8_t* yFp4Ptr,
                             uint32_t ubFactorDimy,
                             uint32_t xDimPerLoop,
                             uint16_t repeatTimes,
                             uint16_t sizePerRepeat,
                             int64_t roundMode,
                             uint32_t xTypeUbAlignB32,
                             uint32_t yUbAlignB8,
                             uint32_t yUbAlignB4) {
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
  AscendC::MicroAPI::RegTensor<uint16_t> yRegTensor;
  AscendC::MicroAPI::RegTensor<uint8_t> out;

  AscendC::MicroAPI::MaskReg maskFull8;
  AscendC::MicroAPI::MaskReg mask;
  uint32_t fp4Width = 32;

  maskFull8 = AscendC::MicroAPI::UpdateMask<uint32_t>(fp4Width);

  for (uint16_t i = 0; i < static_cast<uint16_t>(xDimPerLoop); i++) {
    uint32_t width = static_cast<uint32_t>(ubFactorDimy);
    for (uint16_t j = 0; j < repeatTimes; j++) {
      mask = AscendC::MicroAPI::UpdateMask<uint32_t>(width);

      auto tmpXAddr = xPtr + i * xTypeUbAlignB32 + j * sizePerRepeat;
      auto yAddr = yPtr + i * yUbAlignB8 + j * sizePerRepeat;
      auto yFp4Addr = yFp4Ptr + i * yUbAlignB4 + (j * sizePerRepeat / 2);

      AscendC::MicroAPI::DataCopy(vreg8, tmpXAddr);

      if constexpr (ifYFloat8e4m3Index_) {
        // float32 -> float8_e4m3
        AscendC::MicroAPI::Cast<fp8_e4m3fn_t, float, CAST_FP32_TO_FP8>(
            vreg12, vreg8, mask);
        AscendC::MicroAPI::DataCopy<
            fp8_e4m3fn_t,
            AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(yAddr, vreg12, mask);
      } else if constexpr (ifYFloat8e5m2Index_) {
        // float32 -> float8_e5m2
        AscendC::MicroAPI::Cast<fp8_e5m2_t, float, CAST_FP32_TO_FP8>(
            vreg13, vreg8, mask);
        AscendC::MicroAPI::
            DataCopy<fp8_e5m2_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                yAddr, vreg13, mask);
      } else if constexpr (ifYFloat4e2m1Index_) {
        // float32 -> bfloat16 -> float4_e2m1
        AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(
            vreg14, vreg8, mask);
        AscendC::MicroAPI::Pack(
            (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg14,
            (AscendC::MicroAPI::RegTensor<uint32_t>&)vreg14);
        if (roundMode == 1) {
          AscendC::MicroAPI::Cast<fp4x2_e2m1_t,
                                  bfloat16_t,
                                  CAST_BF16_TO_FP4_ROUND>(vreg16, vreg14, mask);
        } else if (roundMode == 2) {
          AscendC::MicroAPI::Cast<fp4x2_e2m1_t,
                                  bfloat16_t,
                                  CAST_BF16_TO_FP4_FLOOR>(vreg16, vreg14, mask);
        } else if (roundMode == 3) {
          AscendC::MicroAPI::Cast<fp4x2_e2m1_t,
                                  bfloat16_t,
                                  CAST_BF16_TO_FP4_CEIL>(vreg16, vreg14, mask);
        } else if (roundMode == 4) {
          AscendC::MicroAPI::Cast<fp4x2_e2m1_t,
                                  bfloat16_t,
                                  CAST_BF16_TO_FP4_TRUNC>(vreg16, vreg14, mask);
        } else {
          AscendC::MicroAPI::Cast<fp4x2_e2m1_t,
                                  bfloat16_t,
                                  CAST_BF16_TO_FP4_RINT>(vreg16, vreg14, mask);
        }
        AscendC::MicroAPI::
            DataCopy<uint8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                yFp4Addr,
                (AscendC::MicroAPI::RegTensor<uint8_t>&)vreg16,
                maskFull8);
      } else if constexpr (ifYFloat4e1m2Index_) {
        // float32 -> bfloat16 -> float4_e1m2
        AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(
            vreg15, vreg8, mask);
        AscendC::MicroAPI::Pack(
            (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg15,
            (AscendC::MicroAPI::RegTensor<uint32_t>&)vreg15);
        if (roundMode == 1) {
          AscendC::MicroAPI::Cast<fp4x2_e1m2_t,
                                  bfloat16_t,
                                  CAST_BF16_TO_FP4_ROUND>(vreg17, vreg15, mask);
        } else if (roundMode == 2) {
          AscendC::MicroAPI::Cast<fp4x2_e1m2_t,
                                  bfloat16_t,
                                  CAST_BF16_TO_FP4_FLOOR>(vreg17, vreg15, mask);
        } else if (roundMode == 3) {
          AscendC::MicroAPI::Cast<fp4x2_e1m2_t,
                                  bfloat16_t,
                                  CAST_BF16_TO_FP4_CEIL>(vreg17, vreg15, mask);
        } else if (roundMode == 4) {
          AscendC::MicroAPI::Cast<fp4x2_e1m2_t,
                                  bfloat16_t,
                                  CAST_BF16_TO_FP4_TRUNC>(vreg17, vreg15, mask);
        } else {
          AscendC::MicroAPI::Cast<fp4x2_e1m2_t,
                                  bfloat16_t,
                                  CAST_BF16_TO_FP4_RINT>(vreg17, vreg15, mask);
        }
        AscendC::MicroAPI::
            DataCopy<uint8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                yFp4Addr,
                (AscendC::MicroAPI::RegTensor<uint8_t>&)vreg17,
                maskFull8);
      } else if constexpr (ifYHiFloat8Index_) {
        AscendC::MicroAPI::Cast<hifloat8_t, float, CAST_FP32_TO_HI8>(
            vreg18, vreg8, mask);
        AscendC::MicroAPI::
            DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                yAddr, vreg18, mask);
      } else {
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
      }
    }
  }

}  // CastY end

template <typename TXtype,
          typename TBias,
          bool ifXIntIndex_,
          bool ifXFloat16Index_,
          bool ifXBf16Index_,
          bool hasBiasIndex_,
          bool hasActScale_,
          bool ifBiasIntIndex_,
          bool ifBiasFloatIndex_,
          bool ifBiasFloat16Index_,
          bool ifBiasBfloat16Index_>
__aicore__ inline void DequantSwigluV2(__ubuf__ TXtype* xPtr,
                                       __ubuf__ float* dstPtr,
                                       __ubuf__ float* wScalePtr,
                                       __ubuf__ float* aScalePtr,
                                       __ubuf__ TBias* biasPtr,
                                       uint32_t xDimPerLoop,
                                       uint32_t count,
                                       uint16_t repeatTimes,
                                       uint16_t sizePerRepeat,
                                       uint32_t xTypeUbAlignB32,
                                       uint32_t xTUbAlignB32,
                                       uint32_t dstTypeUbAlignB32,
                                       uint32_t aScaleUbAlignB32,
                                       uint32_t offset,
                                       float clampLit,
                                       float gluAlpha,
                                       float gluBias) {
  AscendC::MicroAPI::RegTensor<TXtype> vreg0, vreg10;
  AscendC::MicroAPI::RegTensor<float> vregX0, vregX1;
  AscendC::MicroAPI::RegTensor<float> vreg1, vreg2, vreg3, vreg4, vreg5, vreg6;
  AscendC::MicroAPI::RegTensor<float> vreg7, vreg8, vreg9, vreg11, vreg12;
  AscendC::MicroAPI::RegTensor<float> vreg13, vreg14, vreg15;
  AscendC::MicroAPI::RegTensor<int32_t> vreg16, vreg17, verg18, vreg19;
  AscendC::MicroAPI::RegTensor<float> vreg20, vreg21, vreg44, vregRes;
  AscendC::MicroAPI::RegTensor<half> vreg24, vreg25;
  AscendC::MicroAPI::RegTensor<bfloat16_t> vreg26, vreg27;
  AscendC::MicroAPI::MaskReg mask, tailMask;

  const float scalarOne = 1.0;
  uint32_t tailCount = count - (repeatTimes - 1) * sizePerRepeat;
  uint32_t tailWidth = 0;
  uint32_t width = sizePerRepeat;
  uint32_t ifEven = 0;

  if ((repeatTimes & 1) == 0) {
    tailWidth = (64 + tailCount) / 2;
    ifEven = 1;
  } else {
    tailWidth = tailCount / 2;
  }

  mask = AscendC::MicroAPI::UpdateMask<uint32_t>(width);
  tailMask = AscendC::MicroAPI::UpdateMask<uint32_t>(tailWidth);
  repeatTimes = CeilDivision(repeatTimes, 2);

  for (uint16_t j = 0; j < static_cast<uint16_t>(xDimPerLoop); j++) {
    for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes - 1); i++) {
      auto x1Addr = xPtr + j * xTypeUbAlignB32 + i * sizePerRepeat * 2;
      auto x2Addr = x1Addr + sizePerRepeat;
      auto dstAddr = dstPtr + j * dstTypeUbAlignB32 + i * sizePerRepeat;

      if constexpr (ifXIntIndex_) {
        auto wScale1Addr = wScalePtr + i * sizePerRepeat * 2;
        auto wScale2Addr = wScale1Addr + sizePerRepeat;

        AscendC::MicroAPI::DataCopy(vreg2, wScale1Addr);
        AscendC::MicroAPI::DataCopy(vreg12, wScale2Addr);
      }

      // vreg0 -> x1, vreg10 -> x2
      if constexpr (ifXFloat16Index_) {
        AscendC::MicroAPI::
            DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vreg0, x1Addr);
        AscendC::MicroAPI::
            DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vreg10, x2Addr);
      }
      if constexpr (ifXBf16Index_) {
        AscendC::MicroAPI::
            DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vreg0, x1Addr);
        AscendC::MicroAPI::
            DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vreg10, x2Addr);
      }
      if constexpr (ifXIntIndex_) {
        AscendC::MicroAPI::DataCopy(vreg0, x1Addr);
        AscendC::MicroAPI::DataCopy(vreg10, x2Addr);
      }

      if constexpr (hasBiasIndex_) {
        if constexpr (ifXIntIndex_ && ifBiasIntIndex_) {
          auto bias1Addr = biasPtr + i * sizePerRepeat * 2;
          auto bias2Addr = bias1Addr + sizePerRepeat;
          AscendC::MicroAPI::DataCopy(vreg16, bias1Addr);
          AscendC::MicroAPI::DataCopy(vreg17, bias2Addr);
          // x + bias
          AscendC::MicroAPI::Add(vreg0, vreg0, vreg16, mask);
          AscendC::MicroAPI::Add(vreg10, vreg10, vreg17, mask);
        }
      }

      // x:int32->float32
      if constexpr (ifXIntIndex_) {
        AscendC::MicroAPI::Cast<float, TXtype, CAST_INT32_TO_FP32>(
            vreg1, vreg0, mask);
        AscendC::MicroAPI::Cast<float, TXtype, CAST_INT32_TO_FP32>(
            vreg11, vreg10, mask);
        // x * weight
        AscendC::MicroAPI::Mul(vreg3, vreg1, vreg2, mask);
        AscendC::MicroAPI::Mul(vreg13, vreg11, vreg12, mask);
      }
      if constexpr (ifXBf16Index_ || ifXFloat16Index_) {
        AscendC::MicroAPI::Cast<float, TXtype, CAST_BF16_FP16_TO_FP32>(
            vreg3, vreg0, mask);
        AscendC::MicroAPI::Cast<float, TXtype, CAST_BF16_FP16_TO_FP32>(
            vreg13, vreg10, mask);
      }

      // x * activation_scale
      if constexpr (hasActScale_) {
        auto aScaleAddr = aScalePtr + j * aScaleUbAlignB32;
        AscendC::MicroAPI::DataCopy<float,
                                    AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
            vreg4, aScaleAddr);
        AscendC::MicroAPI::Mul(vreg3, vreg3, vreg4, mask);
        AscendC::MicroAPI::Mul(vreg13, vreg13, vreg4, mask);
      }

      if constexpr (ifBiasFloatIndex_ || ifBiasFloat16Index_ ||
                    ifBiasBfloat16Index_) {
        auto bias1Addr = biasPtr + i * sizePerRepeat * 2;
        auto bias2Addr = bias1Addr + sizePerRepeat;
        if constexpr (ifBiasFloatIndex_) {
          AscendC::MicroAPI::DataCopy(vreg20, bias1Addr);
          AscendC::MicroAPI::DataCopy(vreg21, bias2Addr);
        }
        if constexpr (ifBiasFloat16Index_) {
          AscendC::MicroAPI::
              DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                  vreg24, bias1Addr);
          AscendC::MicroAPI::
              DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                  vreg25, bias2Addr);
          AscendC::MicroAPI::Cast<float, half, CAST_BF16_FP16_TO_FP32>(
              vreg20, vreg24, mask);
          AscendC::MicroAPI::Cast<float, half, CAST_BF16_FP16_TO_FP32>(
              vreg21, vreg25, mask);
        }
        if constexpr (ifBiasBfloat16Index_) {
          AscendC::MicroAPI::DataCopy<
              bfloat16_t,
              AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg26, bias1Addr);
          AscendC::MicroAPI::DataCopy<
              bfloat16_t,
              AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg27, bias2Addr);
          AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(
              vreg20, vreg26, mask);
          AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(
              vreg21, vreg27, mask);
        }
        AscendC::MicroAPI::Add(vreg3, vreg3, vreg20, mask);
        AscendC::MicroAPI::Add(vreg13, vreg13, vreg21, mask);
      }

      AscendC::MicroAPI::DeInterleave(vregX0, vregX1, vreg3, vreg13);

      // swish x1:activate, x2:gate
      AscendC::MicroAPI::Mins(vregX0, vregX0, clampLit, mask);
      AscendC::MicroAPI::Muls(vreg20, vregX0, -gluAlpha, mask);
      AscendC::MicroAPI::Exp(vreg21, vreg20, mask);
      AscendC::MicroAPI::Adds(vreg21, vreg21, scalarOne, mask);
      AscendC::MicroAPI::Div(vregX0, vregX0, vreg21, mask);

      // glu
      AscendC::MicroAPI::Mins(vregX1, vregX1, clampLit, mask);
      AscendC::MicroAPI::Maxs(vregX1, vregX1, -clampLit, mask);
      AscendC::MicroAPI::Adds(vregX1, vregX1, gluBias, mask);

      AscendC::MicroAPI::Mul(vregRes, vregX0, vregX1, mask);

      // res -> ub
      AscendC::MicroAPI::DataCopy(dstAddr, vregRes, mask);
    }

    uint16_t i = repeatTimes - 1;
    auto x1Addr = xPtr + j * xTypeUbAlignB32 + i * sizePerRepeat * 2;
    auto x2Addr = x1Addr + sizePerRepeat;
    auto dstAddr = dstPtr + j * dstTypeUbAlignB32 + i * sizePerRepeat;

    if constexpr (ifXIntIndex_) {
      auto wScale1Addr = wScalePtr + i * sizePerRepeat * 2;
      auto wScale2Addr = wScale1Addr + sizePerRepeat;

      AscendC::MicroAPI::DataCopy(vreg2, wScale1Addr);
      AscendC::MicroAPI::DataCopy(vreg12, wScale2Addr);
    }

    // vreg0 -> x1, vreg10 -> x2
    if constexpr (ifXFloat16Index_) {
      AscendC::MicroAPI::DataCopy<half,
                                  AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
          vreg0, x1Addr);
      AscendC::MicroAPI::DataCopy<half,
                                  AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
          vreg10, x2Addr);
    }
    if constexpr (ifXBf16Index_) {
      AscendC::MicroAPI::DataCopy<bfloat16_t,
                                  AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
          vreg0, x1Addr);
      AscendC::MicroAPI::DataCopy<bfloat16_t,
                                  AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
          vreg10, x2Addr);
    }
    if constexpr (ifXIntIndex_) {
      AscendC::MicroAPI::DataCopy(vreg0, x1Addr);
      AscendC::MicroAPI::DataCopy(vreg10, x2Addr);
    }

    if constexpr (hasBiasIndex_) {
      if constexpr (ifXIntIndex_ && ifBiasIntIndex_) {
        auto bias1Addr = biasPtr + i * sizePerRepeat * 2;
        auto bias2Addr = bias1Addr + sizePerRepeat;
        AscendC::MicroAPI::DataCopy(vreg16, bias1Addr);
        AscendC::MicroAPI::DataCopy(vreg17, bias2Addr);
        // x + bias
        AscendC::MicroAPI::Add(vreg0, vreg0, vreg16, mask);
        AscendC::MicroAPI::Add(vreg10, vreg10, vreg17, mask);
      }
    }

    // x:int32->float32
    if constexpr (ifXIntIndex_) {
      AscendC::MicroAPI::Cast<float, TXtype, CAST_INT32_TO_FP32>(
          vreg1, vreg0, mask);
      AscendC::MicroAPI::Cast<float, TXtype, CAST_INT32_TO_FP32>(
          vreg11, vreg10, mask);
      // x * weight
      AscendC::MicroAPI::Mul(vreg3, vreg1, vreg2, mask);
      AscendC::MicroAPI::Mul(vreg13, vreg11, vreg12, mask);
    }
    if constexpr (ifXBf16Index_ || ifXFloat16Index_) {
      AscendC::MicroAPI::Cast<float, TXtype, CAST_BF16_FP16_TO_FP32>(
          vreg3, vreg0, mask);
      AscendC::MicroAPI::Cast<float, TXtype, CAST_BF16_FP16_TO_FP32>(
          vreg13, vreg10, mask);
    }

    // x * activation_scale
    if constexpr (hasActScale_) {
      auto aScaleAddr = aScalePtr + j * aScaleUbAlignB32;
      AscendC::MicroAPI::DataCopy<float,
                                  AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
          vreg4, aScaleAddr);
      AscendC::MicroAPI::Mul(vreg3, vreg3, vreg4, mask);
      AscendC::MicroAPI::Mul(vreg13, vreg13, vreg4, mask);
    }

    if constexpr (ifBiasFloatIndex_ || ifBiasFloat16Index_ ||
                  ifBiasBfloat16Index_) {
      auto bias1Addr = biasPtr + i * sizePerRepeat * 2;
      auto bias2Addr = bias1Addr + sizePerRepeat;
      if constexpr (ifBiasFloatIndex_) {
        AscendC::MicroAPI::DataCopy(vreg20, bias1Addr);
        AscendC::MicroAPI::DataCopy(vreg21, bias2Addr);
      }
      if constexpr (ifBiasFloat16Index_) {
        AscendC::MicroAPI::
            DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vreg24, bias1Addr);
        AscendC::MicroAPI::
            DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vreg25, bias2Addr);
        AscendC::MicroAPI::Cast<float, half, CAST_BF16_FP16_TO_FP32>(
            vreg20, vreg24, mask);
        AscendC::MicroAPI::Cast<float, half, CAST_BF16_FP16_TO_FP32>(
            vreg21, vreg25, mask);
      }
      if constexpr (ifBiasBfloat16Index_) {
        AscendC::MicroAPI::
            DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vreg26, bias1Addr);
        AscendC::MicroAPI::
            DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vreg27, bias2Addr);
        AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(
            vreg20, vreg26, mask);
        AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(
            vreg21, vreg27, mask);
      }
      AscendC::MicroAPI::Add(vreg3, vreg3, vreg20, mask);
      AscendC::MicroAPI::Add(vreg13, vreg13, vreg21, mask);
    }

    AscendC::MicroAPI::DeInterleave(vregX0, vregX1, vreg3, vreg13);

    // swish x1:activate, x2:gate
    AscendC::MicroAPI::Mins(vregX0, vregX0, clampLit, tailMask);
    AscendC::MicroAPI::Muls(vreg20, vregX0, -gluAlpha, tailMask);
    AscendC::MicroAPI::Exp(vreg21, vreg20, tailMask);
    AscendC::MicroAPI::Adds(vreg21, vreg21, scalarOne, tailMask);
    AscendC::MicroAPI::Div(vregX0, vregX0, vreg21, tailMask);

    // glu
    AscendC::MicroAPI::Mins(vregX1, vregX1, clampLit, tailMask);
    AscendC::MicroAPI::Maxs(vregX1, vregX1, -clampLit, tailMask);
    AscendC::MicroAPI::Adds(vregX1, vregX1, gluBias, tailMask);

    AscendC::MicroAPI::Mul(vregRes, vregX0, vregX1, tailMask);

    // res -> ub
    AscendC::MicroAPI::DataCopy(dstAddr, vregRes, tailMask);
  }

}  // end

template <typename TXtype,
          typename TBias,
          bool ifXIntIndex_,
          bool ifXFloat16Index_,
          bool ifXBf16Index_,
          bool hasBiasIndex_,
          bool hasActScale_,
          bool ifBiasIntIndex_,
          bool ifBiasFloatIndex_,
          bool ifBiasFloat16Index_,
          bool ifBiasBfloat16Index_>
__aicore__ inline void DequantSwigluV1(__ubuf__ TXtype* x1Ptr,
                                       __ubuf__ TXtype* x2Ptr,
                                       __ubuf__ float* dstPtr,
                                       __ubuf__ float* wScale1Ptr,
                                       __ubuf__ float* wScale2Ptr,
                                       __ubuf__ float* aScalePtr,
                                       __ubuf__ TBias* bias1Ptr,
                                       __ubuf__ TBias* bias2Ptr,
                                       uint32_t xDimPerLoop,
                                       uint32_t count,
                                       uint16_t repeatTimes,
                                       uint16_t sizePerRepeat,
                                       uint32_t xTypeUbAlignB32,
                                       uint32_t xTUbAlignB32,
                                       uint32_t dstTypeUbAlignB32,
                                       uint32_t aScaleUbAlignB32,
                                       uint32_t offset) {
  AscendC::MicroAPI::RegTensor<TXtype> vreg0, vreg10;
  AscendC::MicroAPI::RegTensor<float> vreg1, vreg2, vreg3, vreg4, vreg5, vreg6;
  AscendC::MicroAPI::RegTensor<float> vreg7, vreg8, vreg9, vreg11, vreg12;
  AscendC::MicroAPI::RegTensor<float> vreg13, vreg14, vreg15;
  AscendC::MicroAPI::RegTensor<int32_t> vreg16, vreg17, verg18, vreg19;
  AscendC::MicroAPI::RegTensor<float> vreg20, vreg21;
  AscendC::MicroAPI::RegTensor<half> vreg24, vreg25;
  AscendC::MicroAPI::RegTensor<bfloat16_t> vreg26, vreg27;
  AscendC::MicroAPI::MaskReg mask;

  const float scalarOne = 1.0;

  for (uint16_t j = 0; j < repeatTimes; j++) {
    mask = AscendC::MicroAPI::UpdateMask<uint32_t>(count);

    if constexpr (ifXIntIndex_) {
      auto wScale1Addr = wScale1Ptr + j * sizePerRepeat;
      auto wScale2Addr = wScale2Ptr + j * sizePerRepeat;

      AscendC::MicroAPI::DataCopy(vreg2, wScale1Addr);
      AscendC::MicroAPI::DataCopy(vreg12, wScale2Addr);
    }

    for (uint16_t i = 0; i < static_cast<uint16_t>(xDimPerLoop); i++) {
      auto x1Addr = x1Ptr + i * xTypeUbAlignB32 + j * sizePerRepeat;
      auto x2Addr = x2Ptr + i * xTypeUbAlignB32 + j * sizePerRepeat;
      auto dstAddr = dstPtr + i * dstTypeUbAlignB32 + j * sizePerRepeat;

      // vreg0 -> x1, vreg10 -> x2
      if constexpr (ifXFloat16Index_) {
        AscendC::MicroAPI::
            DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vreg0, x1Addr);
        AscendC::MicroAPI::
            DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vreg10, x2Addr);
      }
      if constexpr (ifXBf16Index_) {
        AscendC::MicroAPI::
            DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vreg0, x1Addr);
        AscendC::MicroAPI::
            DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vreg10, x2Addr);
      }
      if constexpr (ifXIntIndex_) {
        AscendC::MicroAPI::DataCopy(vreg0, x1Addr);
        AscendC::MicroAPI::DataCopy(vreg10, x2Addr);
      }

      if constexpr (hasBiasIndex_) {
        if constexpr (ifXIntIndex_ && ifBiasIntIndex_) {
          auto bias1Addr = bias1Ptr + j * sizePerRepeat;
          auto bias2Addr = bias2Ptr + j * sizePerRepeat;
          AscendC::MicroAPI::DataCopy(vreg16, bias1Addr);
          AscendC::MicroAPI::DataCopy(vreg17, bias2Addr);
          // x + bias
          AscendC::MicroAPI::Add(vreg0, vreg0, vreg16, mask);
          AscendC::MicroAPI::Add(vreg10, vreg10, vreg17, mask);
        }
      }

      // x:int32->float32
      if constexpr (ifXIntIndex_) {
        AscendC::MicroAPI::Cast<float, TXtype, CAST_INT32_TO_FP32>(
            vreg1, vreg0, mask);
        AscendC::MicroAPI::Cast<float, TXtype, CAST_INT32_TO_FP32>(
            vreg11, vreg10, mask);
        // x * weight
        AscendC::MicroAPI::Mul(vreg3, vreg1, vreg2, mask);
        AscendC::MicroAPI::Mul(vreg13, vreg11, vreg12, mask);
      }
      if constexpr (ifXBf16Index_ || ifXFloat16Index_) {
        AscendC::MicroAPI::Cast<float, TXtype, CAST_BF16_FP16_TO_FP32>(
            vreg3, vreg0, mask);
        AscendC::MicroAPI::Cast<float, TXtype, CAST_BF16_FP16_TO_FP32>(
            vreg13, vreg10, mask);
      }

      // x * activation_scale
      if constexpr (hasActScale_) {
        auto aScaleAddr = aScalePtr + i * aScaleUbAlignB32;
        AscendC::MicroAPI::DataCopy<float,
                                    AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
            vreg4, aScaleAddr);
        AscendC::MicroAPI::Mul(vreg3, vreg3, vreg4, mask);
        AscendC::MicroAPI::Mul(vreg13, vreg13, vreg4, mask);
      }

      if constexpr (ifBiasFloatIndex_ || ifBiasFloat16Index_ ||
                    ifBiasBfloat16Index_) {
        auto bias1Addr = bias1Ptr + j * sizePerRepeat;
        auto bias2Addr = bias2Ptr + j * sizePerRepeat;
        if constexpr (ifBiasFloatIndex_) {
          AscendC::MicroAPI::DataCopy(vreg20, bias1Addr);
          AscendC::MicroAPI::DataCopy(vreg21, bias2Addr);
        }
        if constexpr (ifBiasFloat16Index_) {
          AscendC::MicroAPI::
              DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                  vreg24, bias1Addr);
          AscendC::MicroAPI::
              DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                  vreg25, bias2Addr);
          AscendC::MicroAPI::Cast<float, half, CAST_BF16_FP16_TO_FP32>(
              vreg20, vreg24, mask);
          AscendC::MicroAPI::Cast<float, half, CAST_BF16_FP16_TO_FP32>(
              vreg21, vreg25, mask);
        }
        if constexpr (ifBiasBfloat16Index_) {
          AscendC::MicroAPI::DataCopy<
              bfloat16_t,
              AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg26, bias1Addr);
          AscendC::MicroAPI::DataCopy<
              bfloat16_t,
              AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg27, bias2Addr);
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
      AscendC::MicroAPI::Div(vreg9, vreg3, vreg8, mask);

      // glu
      AscendC::MicroAPI::Mul(vreg15, vreg9, vreg13, mask);

      // store: reg->ub
      AscendC::MicroAPI::DataCopy(dstAddr, vreg15, mask);
    }
  }

}  // end

}  // namespace DequantSwigluQuantV35Ops
#endif  // DEQUANT_SWIGLU_QUANT_COMMON_H
