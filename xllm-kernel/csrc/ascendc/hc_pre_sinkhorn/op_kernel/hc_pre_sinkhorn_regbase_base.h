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
 * \file hc_pre_sinkhorn_regbase_base.h
 * \brief
 */

#ifndef HC_PRE_SINKHORN_RGEBASE_BASE_H
#define HC_PRE_SINKHORN_RGEBASE_BASE_H

#include "kernel_operator.h"

namespace XllmKernelHcPreSinkhorn {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UnalignReg;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t VL_FP32 = 64;
// Number of comb-matrix rows held in registers by the four-unfold fast path.
constexpr int32_t COMB_UNFOLD_NUM = 4;
// Below this many rows per tile the plane-major transpose overhead outweighs
// the gain.
constexpr int32_t COMB_SOA_MIN_ROWS = 4;

__aicore__ inline int32_t CeilDiv(int32_t a, int b) {
  if (b == 0) {
    return a;
  }
  return (a + b - 1) / b;
}

__aicore__ inline int32_t CeilAlign(int32_t a, int b) {
  return CeilDiv(a, b) * b;
}

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t num) {
  int32_t elemNum = BLOCK_SIZE / sizeof(T);
  return CeilAlign(num, elemNum);
}

constexpr AscendC::MicroAPI::CastTrait castTraitB162B32Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitB322B16Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename T>
__aicore__ inline void LoadInputData(RegTensor<float>& dst,
                                     __local_mem__ T* src,
                                     MaskReg pregLoop,
                                     uint32_t srcOffset) {
  if constexpr (IsSameType<T, float>::value) {
    DataCopy(dst, src + srcOffset);
  } else if constexpr (IsSameType<T, half>::value ||
                       IsSameType<T, bfloat16_t>::value) {
    RegTensor<T> tmp;
    DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(tmp,
                                                              src + srcOffset);
    Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
  }
}

template <typename T>
__aicore__ inline void StoreOutputData(__local_mem__ T* dst,
                                       RegTensor<float>& src,
                                       MaskReg pregLoop,
                                       uint32_t dstOffset) {
  if constexpr (IsSameType<T, float>::value) {
    DataCopy(dst + dstOffset, src, pregLoop);
  } else if constexpr (IsSameType<T, half>::value ||
                       IsSameType<T, bfloat16_t>::value) {
    RegTensor<T> tmp;
    Cast<T, float, castTraitB322B16Even>(tmp, src, pregLoop);
    DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
        dst + dstOffset, tmp, pregLoop);
  }
}

template <typename T>
__aicore__ inline void LoadInputDataWithBrc(RegTensor<float>& dst,
                                            __local_mem__ T* src,
                                            MaskReg pregLoop,
                                            uint32_t srcOffset) {
  if constexpr (IsSameType<T, float>::value) {
    DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(dst,
                                                               src + srcOffset);
  } else if constexpr (IsSameType<T, half>::value ||
                       IsSameType<T, bfloat16_t>::value) {
    RegTensor<T> tmp;
    DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(tmp,
                                                           src + srcOffset);
    Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
  }
}

__aicore__ inline void VFSigmoid(RegTensor<float>& y,
                                 RegTensor<float>& x,
                                 RegTensor<float>& one,
                                 MaskReg pregLoop) {
  Muls(x, x, static_cast<float>(-1), pregLoop);
  Exp(x, x, pregLoop);
  Adds(x, x, static_cast<float>(1), pregLoop);
  Div(y, one, x, pregLoop);
}

__aicore__ inline void VFProcessPre(const LocalTensor<float>& preLocal,
                                    const LocalTensor<float>& mixLocal,
                                    const LocalTensor<float>& hcBaseLocal,
                                    const LocalTensor<float>& rsqrtLocal,
                                    float scale,
                                    float eps,
                                    uint16_t curRowNum,
                                    uint16_t curColNum) {
  __local_mem__ float* preLocalAddr =
      (__local_mem__ float*)preLocal.GetPhyAddr();
  __local_mem__ float* mixLocalAddr =
      (__local_mem__ float*)mixLocal.GetPhyAddr();
  __local_mem__ float* hcBaseLocalAddr =
      (__local_mem__ float*)hcBaseLocal.GetPhyAddr();
  __local_mem__ float* rsqrtLocalAddr =
      (__local_mem__ float*)rsqrtLocal.GetPhyAddr();
  uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
  uint32_t curColNumAlign = RoundUp<float>(curColNum);
  if (loopCount > 1) {
    __VEC_SCOPE__ {
      RegTensor<float> mix;
      RegTensor<float> base;
      RegTensor<float> rsqrt;
      RegTensor<float> one;
      MaskReg pregLoop = CreateMask<float>();
      uint32_t sreg = curColNum;
      Duplicate(one, static_cast<float>(1), pregLoop);
      for (uint16_t i = 0; i < loopCount; i++) {
        pregLoop = UpdateMask<float>(sreg);
        LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, i * VL_FP32);
        for (uint16_t j = 0; j < curRowNum; j++) {
          LoadInputDataWithBrc<float>(rsqrt, rsqrtLocalAddr, pregLoop, j);
          LoadInputData<float>(
              mix, mixLocalAddr, pregLoop, i * VL_FP32 + j * curColNumAlign);
          Mul(mix, mix, rsqrt, pregLoop);
          Muls(mix, mix, scale, pregLoop);
          Add(mix, mix, base, pregLoop);
          VFSigmoid(mix, mix, one, pregLoop);
          Adds(mix, mix, eps, pregLoop);
          StoreOutputData(
              preLocalAddr, mix, pregLoop, i * VL_FP32 + j * curColNumAlign);
        }
      }
    }
  } else {
    __VEC_SCOPE__ {
      RegTensor<float> mix;
      RegTensor<float> base;
      RegTensor<float> rsqrt;
      RegTensor<float> one;
      uint32_t sreg = curColNum;
      MaskReg pregLoop = UpdateMask<float>(sreg);
      Duplicate(one, static_cast<float>(1), pregLoop);
      LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, 0);
      for (uint16_t i = 0; i < curRowNum; i++) {
        LoadInputData<float>(mix, mixLocalAddr, pregLoop, i * curColNumAlign);
        LoadInputDataWithBrc<float>(rsqrt, rsqrtLocalAddr, pregLoop, i);
        Mul(mix, mix, rsqrt, pregLoop);
        Muls(mix, mix, scale, pregLoop);
        Add(mix, mix, base, pregLoop);
        VFSigmoid(mix, mix, one, pregLoop);
        Adds(mix, mix, eps, pregLoop);
        StoreOutputData(preLocalAddr, mix, pregLoop, i * curColNumAlign);
      }
    }
  }
}

__aicore__ inline void VFProcessPost(const LocalTensor<float>& postLocal,
                                     const LocalTensor<float>& mixLocal,
                                     const LocalTensor<float>& hcBaseLocal,
                                     const LocalTensor<float>& rsqrtLocal,
                                     float scale,
                                     float eps,
                                     uint16_t curRowNum,
                                     uint16_t curColNum) {
  __local_mem__ float* postLocalAddr =
      (__local_mem__ float*)postLocal.GetPhyAddr();
  __local_mem__ float* mixLocalAddr =
      (__local_mem__ float*)mixLocal.GetPhyAddr();
  __local_mem__ float* hcBaseLocalAddr =
      (__local_mem__ float*)hcBaseLocal.GetPhyAddr();
  __local_mem__ float* rsqrtLocalAddr =
      (__local_mem__ float*)rsqrtLocal.GetPhyAddr();
  uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
  uint32_t curColNumAlign = RoundUp<float>(curColNum);
  if (loopCount > 1) {
    __VEC_SCOPE__ {
      RegTensor<float> mix;
      RegTensor<float> base;
      RegTensor<float> rsqrt;
      RegTensor<float> one;
      MaskReg pregLoop = CreateMask<float>();
      uint32_t sreg = curColNum;
      Duplicate(one, static_cast<float>(1), pregLoop);
      for (uint16_t i = 0; i < loopCount; i++) {
        pregLoop = UpdateMask<float>(sreg);
        LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, i * VL_FP32);
        for (uint16_t j = 0; j < curRowNum; j++) {
          LoadInputData<float>(
              mix, mixLocalAddr, pregLoop, i * VL_FP32 + j * curColNumAlign);
          LoadInputDataWithBrc<float>(rsqrt, rsqrtLocalAddr, pregLoop, i);
          Mul(mix, mix, rsqrt, pregLoop);
          Muls(mix, mix, scale, pregLoop);
          Add(mix, mix, base, pregLoop);
          VFSigmoid(mix, mix, one, pregLoop);
          Muls(mix, mix, static_cast<float>(2.0), pregLoop);
          StoreOutputData(
              postLocalAddr, mix, pregLoop, i * VL_FP32 + j * curColNumAlign);
        }
      }
    }
  } else {
    __VEC_SCOPE__ {
      RegTensor<float> mix;
      RegTensor<float> base;
      RegTensor<float> rsqrt;
      RegTensor<float> one;
      uint32_t sreg = curColNum;
      MaskReg pregLoop = UpdateMask<float>(sreg);
      Duplicate(one, static_cast<float>(1), pregLoop);
      LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, 0);
      for (uint16_t i = 0; i < curRowNum; i++) {
        LoadInputData<float>(mix, mixLocalAddr, pregLoop, i * curColNumAlign);
        LoadInputDataWithBrc<float>(rsqrt, rsqrtLocalAddr, pregLoop, i);
        Mul(mix, mix, rsqrt, pregLoop);
        Muls(mix, mix, scale, pregLoop);
        Add(mix, mix, base, pregLoop);
        VFSigmoid(mix, mix, one, pregLoop);
        Muls(mix, mix, static_cast<float>(2.0), pregLoop);
        StoreOutputData(postLocalAddr, mix, pregLoop, i * curColNumAlign);
      }
    }
  }
}

// dim2是R轴，R轴小于64, 不需要回写UB
__aicore__ inline void VFProcessCombFragRLessVL(
    const LocalTensor<float>& combFragLocal,
    const LocalTensor<float>& mixLocal,
    const LocalTensor<float>& hcBaseLocal,
    const LocalTensor<float>& rsqrtLocal,
    float scale,
    float eps,
    uint16_t iters,
    uint16_t dim0,
    uint16_t dim1,
    uint16_t dim2) {
  __local_mem__ float* combFragLocalAddr =
      (__local_mem__ float*)combFragLocal.GetPhyAddr();
  __local_mem__ float* mixLocalAddr =
      (__local_mem__ float*)mixLocal.GetPhyAddr();
  __local_mem__ float* hcBaseLocalAddr =
      (__local_mem__ float*)hcBaseLocal.GetPhyAddr();
  __local_mem__ float* rsqrtLocalAddr =
      (__local_mem__ float*)rsqrtLocal.GetPhyAddr();
  uint32_t dim2Align = RoundUp<float>(dim2);
  __VEC_SCOPE__ {
    RegTensor<float> base;
    RegTensor<float> mix;
    RegTensor<float> rsqrt;
    RegTensor<float> max;
    RegTensor<float> sum;
    RegTensor<float> sum1;
    uint32_t sreg = dim2;
    MaskReg pregLoop = UpdateMask<float>(sreg);
    for (uint16_t i = 0; i < dim0; i++) {
      Duplicate(sum1, static_cast<float>(0), pregLoop);
      LoadInputDataWithBrc<float>(rsqrt, rsqrtLocalAddr, pregLoop, i);
      for (uint16_t j = 0; j < dim1; j++) {
        LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, j * dim2Align);
        LoadInputData<float>(
            mix, mixLocalAddr, pregLoop, i * dim1 * dim2Align + j * dim2Align);
        Mul(mix, mix, rsqrt, pregLoop);
        Muls(mix, mix, scale, pregLoop);
        Add(mix, mix, base, pregLoop);
        ReduceMax(max, mix, pregLoop);
        Duplicate(max, max, pregLoop);
        Sub(mix, mix, max, pregLoop);
        Exp(mix, mix, pregLoop);
        ReduceSum(sum, mix, pregLoop);
        Duplicate(sum, sum, pregLoop);
        Div(mix, mix, sum, pregLoop);
        Adds(mix, mix, eps, pregLoop);
        Add(sum1, sum1, mix, pregLoop);
        StoreOutputData(combFragLocalAddr,
                        mix,
                        pregLoop,
                        i * dim1 * dim2Align + j * dim2Align);
      }
      LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
      Adds(sum1, sum1, eps, pregLoop);
      for (uint16_t j = 0; j < dim1; j++) {
        LoadInputData<float>(mix,
                             combFragLocalAddr,
                             pregLoop,
                             i * dim1 * dim2Align + j * dim2Align);
        Div(mix, mix, sum1, pregLoop);
        StoreOutputData(combFragLocalAddr,
                        mix,
                        pregLoop,
                        i * dim1 * dim2Align + j * dim2Align);
      }
    }
    for (uint16_t i = 0; i < iters; i++) {
      LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
      for (uint16_t j = 0; j < dim0; j++) {
        Duplicate(sum1, static_cast<float>(0), pregLoop);
        for (uint16_t k = 0; k < dim1; k++) {
          LoadInputData<float>(mix,
                               combFragLocalAddr,
                               pregLoop,
                               j * dim1 * dim2Align + k * dim2Align);
          ReduceSum(sum, mix, pregLoop);
          Duplicate(sum, sum, pregLoop);
          Adds(sum, sum, eps, pregLoop);
          Div(mix, mix, sum, pregLoop);
          Add(sum1, sum1, mix, pregLoop);
          StoreOutputData(combFragLocalAddr,
                          mix,
                          pregLoop,
                          j * dim1 * dim2Align + k * dim2Align);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        Adds(sum1, sum1, eps, pregLoop);
        for (uint16_t k = 0; k < dim1; k++) {
          LoadInputData<float>(mix,
                               combFragLocalAddr,
                               pregLoop,
                               j * dim1 * dim2Align + k * dim2Align);
          Div(mix, mix, sum1, pregLoop);
          StoreOutputData(combFragLocalAddr,
                          mix,
                          pregLoop,
                          j * dim1 * dim2Align + k * dim2Align);
        }
      }
    }
  }
}

// [vec-05 + state_resident] Register-resident Sinkhorn iteration for generic M.
// Phase 1 (initial softmax + column norm) is identical to RLessVL (uses UB
// staging). Phase 2 (Sinkhorn iterations) loads M rows into RegTensor, iterates
// entirely in registers with no UB load/store and no LocalMemBar, then stores M
// rows back once. This eliminates iters * dim0 * M * 2 UB load/store + iters *
// dim0 LocalMemBar.
//
// Note: RegTensor arrays (RegTensor<float> mix[M]) are NOT supported by bisheng
// backend
// ("Unsupported Inst must be hoisted"). Use individual variables + if constexpr
// chains, consistent with the FourUnfold path (mix1..mix4). The compiler
// eliminates unused variables and dead branches for each template
// instantiation.

// Helper macros for compile-time unrolling over individual RegTensor variables
#define REG_LOAD_N(N)                                             \
  if constexpr (M > N) {                                          \
    LoadInputData<float>(mix##N,                                  \
                         combFragLocalAddr,                       \
                         pregLoop,                                \
                         i * dim1 * dim2Align + (N) * dim2Align); \
  }

#define REG_ROW_NORM_N(N)                \
  if constexpr (M > N) {                 \
    ReduceSum(sumR, mix##N, pregLoop);   \
    Duplicate(sumR, sumR, pregLoop);     \
    Adds(sumR, sumR, eps, pregLoop);     \
    Div(mix##N, mix##N, sumR, pregLoop); \
    Add(sumC, sumC, mix##N, pregLoop);   \
  }

#define REG_COL_NORM_N(N)                \
  if constexpr (M > N) {                 \
    Div(mix##N, mix##N, sumC, pregLoop); \
  }

#define REG_STORE_N(N)                                       \
  if constexpr (M > N) {                                     \
    StoreOutputData(combFragLocalAddr,                       \
                    mix##N,                                  \
                    pregLoop,                                \
                    i * dim1 * dim2Align + (N) * dim2Align); \
  }

// clang-format off
// Expand all 16 slots
#define REG_LOAD_ALL \
    REG_LOAD_N(0) REG_LOAD_N(1) REG_LOAD_N(2) REG_LOAD_N(3) \
    REG_LOAD_N(4) REG_LOAD_N(5) REG_LOAD_N(6) REG_LOAD_N(7) \
    REG_LOAD_N(8) REG_LOAD_N(9) REG_LOAD_N(10) REG_LOAD_N(11) \
    REG_LOAD_N(12) REG_LOAD_N(13) REG_LOAD_N(14) REG_LOAD_N(15)

#define REG_ROW_NORM_ALL \
    REG_ROW_NORM_N(0) REG_ROW_NORM_N(1) REG_ROW_NORM_N(2) REG_ROW_NORM_N(3) \
    REG_ROW_NORM_N(4) REG_ROW_NORM_N(5) REG_ROW_NORM_N(6) REG_ROW_NORM_N(7) \
    REG_ROW_NORM_N(8) REG_ROW_NORM_N(9) REG_ROW_NORM_N(10) REG_ROW_NORM_N(11) \
    REG_ROW_NORM_N(12) REG_ROW_NORM_N(13) REG_ROW_NORM_N(14) REG_ROW_NORM_N(15)

#define REG_COL_NORM_ALL \
    REG_COL_NORM_N(0) REG_COL_NORM_N(1) REG_COL_NORM_N(2) REG_COL_NORM_N(3) \
    REG_COL_NORM_N(4) REG_COL_NORM_N(5) REG_COL_NORM_N(6) REG_COL_NORM_N(7) \
    REG_COL_NORM_N(8) REG_COL_NORM_N(9) REG_COL_NORM_N(10) REG_COL_NORM_N(11) \
    REG_COL_NORM_N(12) REG_COL_NORM_N(13) REG_COL_NORM_N(14) REG_COL_NORM_N(15)

#define REG_STORE_ALL \
    REG_STORE_N(0) REG_STORE_N(1) REG_STORE_N(2) REG_STORE_N(3) \
    REG_STORE_N(4) REG_STORE_N(5) REG_STORE_N(6) REG_STORE_N(7) \
    REG_STORE_N(8) REG_STORE_N(9) REG_STORE_N(10) REG_STORE_N(11) \
    REG_STORE_N(12) REG_STORE_N(13) REG_STORE_N(14) REG_STORE_N(15)
// clang-format on

template <int M>
__aicore__ inline void VFProcessCombFragRegResident(
    const LocalTensor<float>& combFragLocal,
    const LocalTensor<float>& mixLocal,
    const LocalTensor<float>& hcBaseLocal,
    const LocalTensor<float>& rsqrtLocal,
    float scale,
    float eps,
    uint16_t iters,
    uint16_t dim0,
    uint16_t dim1,
    uint16_t dim2) {
  __local_mem__ float* combFragLocalAddr =
      (__local_mem__ float*)combFragLocal.GetPhyAddr();
  __local_mem__ float* mixLocalAddr =
      (__local_mem__ float*)mixLocal.GetPhyAddr();
  __local_mem__ float* hcBaseLocalAddr =
      (__local_mem__ float*)hcBaseLocal.GetPhyAddr();
  __local_mem__ float* rsqrtLocalAddr =
      (__local_mem__ float*)rsqrtLocal.GetPhyAddr();
  uint32_t dim2Align = RoundUp<float>(dim2);
  __VEC_SCOPE__ {
    RegTensor<float> base;
    RegTensor<float> mix;
    RegTensor<float> rsqrt;
    RegTensor<float> max;
    RegTensor<float> sum;
    RegTensor<float> sum1;
    uint32_t sreg = dim2;
    // [vec-09] mask creation done once, outside all loops
    MaskReg pregLoop = UpdateMask<float>(sreg);

    // Phase 1a: Initial softmax per row + accumulate column sum
    for (uint16_t i = 0; i < dim0; i++) {
      Duplicate(sum1, static_cast<float>(0), pregLoop);
      LoadInputDataWithBrc<float>(rsqrt, rsqrtLocalAddr, pregLoop, i);
      for (uint16_t j = 0; j < dim1; j++) {
        LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, j * dim2Align);
        LoadInputData<float>(
            mix, mixLocalAddr, pregLoop, i * dim1 * dim2Align + j * dim2Align);
        Mul(mix, mix, rsqrt, pregLoop);
        Muls(mix, mix, scale, pregLoop);
        Add(mix, mix, base, pregLoop);
        ReduceMax(max, mix, pregLoop);
        Duplicate(max, max, pregLoop);
        Sub(mix, mix, max, pregLoop);
        Exp(mix, mix, pregLoop);
        ReduceSum(sum, mix, pregLoop);
        Duplicate(sum, sum, pregLoop);
        Div(mix, mix, sum, pregLoop);
        Adds(mix, mix, eps, pregLoop);
        Add(sum1, sum1, mix, pregLoop);
        StoreOutputData(combFragLocalAddr,
                        mix,
                        pregLoop,
                        i * dim1 * dim2Align + j * dim2Align);
      }
      // Phase 1b: Column normalization
      LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
      Adds(sum1, sum1, eps, pregLoop);
      for (uint16_t j = 0; j < dim1; j++) {
        LoadInputData<float>(mix,
                             combFragLocalAddr,
                             pregLoop,
                             i * dim1 * dim2Align + j * dim2Align);
        Div(mix, mix, sum1, pregLoop);
        StoreOutputData(combFragLocalAddr,
                        mix,
                        pregLoop,
                        i * dim1 * dim2Align + j * dim2Align);
      }
    }

    // Phase 2: Register-resident Sinkhorn iterations
    // [vec-05] Load M rows into individual RegTensors, iterate without UB
    // staging. Declare up to 16 individual RegTensor variables; compiler
    // eliminates unused.
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    RegTensor<float> mix0, mix1, mix2, mix3, mix4, mix5, mix6, mix7;
    RegTensor<float> mix8, mix9, mix10, mix11, mix12, mix13, mix14, mix15;
    RegTensor<float> sumR;  // row sum (reused per row)
    RegTensor<float> sumC;  // column sum accumulator

    for (uint16_t i = 0; i < dim0; i++) {
      // Load M rows into registers (one-time UB load per dim0 iteration)
      REG_LOAD_ALL

      // Sinkhorn iterations: all in registers, no UB load/store, no LocalMemBar
      for (uint16_t iter = 0; iter < iters; iter++) {
        Duplicate(sumC, static_cast<float>(0), pregLoop);
        // Row normalization + accumulate column sum
        REG_ROW_NORM_ALL
        // Column normalization
        Adds(sumC, sumC, eps, pregLoop);
        REG_COL_NORM_ALL
      }

      // Store M rows back to UB (one-time UB store per dim0 iteration)
      REG_STORE_ALL
    }
  }
}

// ===================================================================================
// [round3 / vec-10] SoA (plane-major) comb_frag path for hcMult <= 4.
//
// Stores the M*M matrix entries as separate UB planes of length aAlign, so both
// Sinkhorn reductions become elementwise ops at full lane occupancy.
//   MTE2 : NDDMA MultiCopy does (A, M*M) -> (M*M, A) on the fly.
//   VEC  : register-resident planes, no UB traffic per iteration.
//   MTE3 : TransDataTo5HD (ldva address list) converts back to GM layout.
// ===================================================================================
constexpr int32_t COMB_SOA_MAX_MULT = 4;
constexpr int32_t TRANS_BLOCK_HALF = 8;
constexpr int32_t TRANS_BLOCK_FULL = 16;
// GM [A, hcMix] -> UB planes [planes, aAlign] via NDDMA MultiCopy.
// opc for this op is compiled with -Wno-constant-conversion so default
// NdDmaConfig::unsetPad survives -Werror.
__aicore__ inline void CopyInCombTransposed(const GlobalTensor<float>& mixesGm,
                                            const LocalTensor<float>& combT,
                                            uint32_t curA,
                                            uint32_t aAlign,
                                            uint32_t hcMix,
                                            uint32_t planes) {
  MultiCopyLoopInfo<2> loopInfo;
  loopInfo.loopSrcStride[0] = 1;
  loopInfo.loopDstStride[0] = aAlign;
  loopInfo.loopSize[0] = planes;
  loopInfo.loopSrcStride[1] = hcMix;
  loopInfo.loopDstStride[1] = 1;
  loopInfo.loopSize[1] = curA;
  MultiCopyParams<float, 2> params = {loopInfo, 0.0f};
  DataCopy<float, 2>(combT, mixesGm, params);
}

#define SOA_HAS(R, C) ((R) < M && (C) < M)
#define SOA_PLANE(R, C) (((R) * M + (C)) * aAlign + off)

#define SOA_AFFINE(R, C)                                             \
  if constexpr (SOA_HAS(R, C)) {                                     \
    DataCopy(c##R##C, combAddr + SOA_PLANE(R, C));                   \
    Mul(c##R##C, c##R##C, rsq, preg);                                \
    Muls(c##R##C, c##R##C, scale, preg);                             \
    LoadInputDataWithBrc<float>(bas, baseAddr, preg, (R) * M + (C)); \
    Add(c##R##C, c##R##C, bas, preg);                                \
  }

#define SOA_AFFINE_ROW(R) \
  SOA_AFFINE(R, 0) SOA_AFFINE(R, 1) SOA_AFFINE(R, 2) SOA_AFFINE(R, 3)

#define SOA_REDUCE_ROW(OP, R)      \
  OP(acc, c##R##0, c##R##1, preg); \
  if constexpr (M > 2) {           \
    OP(acc, acc, c##R##2, preg);   \
  }                                \
  if constexpr (M > 3) {           \
    OP(acc, acc, c##R##3, preg);   \
  }

#define SOA_APPLY_ROW(OP, R)         \
  OP(c##R##0, c##R##0, acc, preg);   \
  OP(c##R##1, c##R##1, acc, preg);   \
  if constexpr (M > 2) {             \
    OP(c##R##2, c##R##2, acc, preg); \
  }                                  \
  if constexpr (M > 3) {             \
    OP(c##R##3, c##R##3, acc, preg); \
  }

#define SOA_EXP_ROW(R)           \
  Exp(c##R##0, c##R##0, preg);   \
  Exp(c##R##1, c##R##1, preg);   \
  if constexpr (M > 2) {         \
    Exp(c##R##2, c##R##2, preg); \
  }                              \
  if constexpr (M > 3) {         \
    Exp(c##R##3, c##R##3, preg); \
  }

#define SOA_ADDS_ROW(R)                \
  Adds(c##R##0, c##R##0, eps, preg);   \
  Adds(c##R##1, c##R##1, eps, preg);   \
  if constexpr (M > 2) {               \
    Adds(c##R##2, c##R##2, eps, preg); \
  }                                    \
  if constexpr (M > 3) {               \
    Adds(c##R##3, c##R##3, eps, preg); \
  }

#define SOA_SOFTMAX_ROW(R) \
  if constexpr ((R) < M) { \
    SOA_REDUCE_ROW(Max, R) \
    SOA_APPLY_ROW(Sub, R)  \
    SOA_EXP_ROW(R)         \
    SOA_REDUCE_ROW(Add, R) \
    SOA_APPLY_ROW(Div, R)  \
    SOA_ADDS_ROW(R)        \
  }

#define SOA_ROW_NORM(R)        \
  if constexpr ((R) < M) {     \
    SOA_REDUCE_ROW(Add, R)     \
    Adds(acc, acc, eps, preg); \
    SOA_APPLY_ROW(Div, R)      \
  }

#define SOA_COL_NORM(C)             \
  if constexpr ((C) < M) {          \
    Add(acc, c0##C, c1##C, preg);   \
    if constexpr (M > 2) {          \
      Add(acc, acc, c2##C, preg);   \
    }                               \
    if constexpr (M > 3) {          \
      Add(acc, acc, c3##C, preg);   \
    }                               \
    Adds(acc, acc, eps, preg);      \
    Div(c0##C, c0##C, acc, preg);   \
    Div(c1##C, c1##C, acc, preg);   \
    if constexpr (M > 2) {          \
      Div(c2##C, c2##C, acc, preg); \
    }                               \
    if constexpr (M > 3) {          \
      Div(c3##C, c3##C, acc, preg); \
    }                               \
  }

#define SOA_STORE(R, C)                                  \
  if constexpr (SOA_HAS(R, C)) {                         \
    DataCopy(combAddr + SOA_PLANE(R, C), c##R##C, preg); \
  }

#define SOA_STORE_ROW(R) \
  SOA_STORE(R, 0) SOA_STORE(R, 1) SOA_STORE(R, 2) SOA_STORE(R, 3)

template <int M>
__aicore__ inline void VFProcessCombFragSoA(
    const LocalTensor<float>& combT,
    const LocalTensor<float>& hcBaseLocal,
    const LocalTensor<float>& rsqrtLocal,
    float scale,
    float eps,
    uint16_t iters,
    uint16_t curA,
    uint32_t aAlign) {
  __local_mem__ float* combAddr = (__local_mem__ float*)combT.GetPhyAddr();
  __local_mem__ float* baseAddr =
      (__local_mem__ float*)hcBaseLocal.GetPhyAddr();
  __local_mem__ float* rsqrtAddr =
      (__local_mem__ float*)rsqrtLocal.GetPhyAddr();
  uint16_t chunkCount = CeilDiv(curA, VL_FP32);
  __VEC_SCOPE__ {
    RegTensor<float> c00, c01, c02, c03;
    RegTensor<float> c10, c11, c12, c13;
    RegTensor<float> c20, c21, c22, c23;
    RegTensor<float> c30, c31, c32, c33;
    RegTensor<float> rsq;
    RegTensor<float> bas;
    RegTensor<float> acc;
    uint32_t remain = curA;
    MaskReg preg;
    for (uint16_t chunk = 0; chunk < chunkCount; chunk++) {
      preg = UpdateMask<float>(remain);
      uint32_t off = chunk * VL_FP32;
      DataCopy(rsq, rsqrtAddr + off);
      SOA_AFFINE_ROW(0)
      SOA_AFFINE_ROW(1)
      SOA_AFFINE_ROW(2)
      SOA_AFFINE_ROW(3)
      SOA_SOFTMAX_ROW(0)
      SOA_SOFTMAX_ROW(1)
      SOA_SOFTMAX_ROW(2)
      SOA_SOFTMAX_ROW(3)
      SOA_COL_NORM(0)
      SOA_COL_NORM(1)
      SOA_COL_NORM(2)
      SOA_COL_NORM(3)
      for (uint16_t iter = 0; iter < iters; iter++) {
        SOA_ROW_NORM(0)
        SOA_ROW_NORM(1)
        SOA_ROW_NORM(2)
        SOA_ROW_NORM(3)
        SOA_COL_NORM(0)
        SOA_COL_NORM(1)
        SOA_COL_NORM(2)
        SOA_COL_NORM(3)
      }
      SOA_STORE_ROW(0)
      SOA_STORE_ROW(1)
      SOA_STORE_ROW(2)
      SOA_STORE_ROW(3)
    }
  }
}

__aicore__ inline void TransposeCombSoAToAoS(
    const LocalTensor<float>& dst,
    const LocalTensor<float>& src,
    const LocalTensor<uint64_t>& vaAddr,
    uint16_t curA,
    uint32_t aAlign,
    int32_t planes) {
  int32_t groups = CeilDiv(planes, TRANS_BLOCK_HALF);
  uint8_t aRepeat = static_cast<uint8_t>(CeilDiv(curA, TRANS_BLOCK_FULL));
  TransDataTo5HDParams params;
  params.repeatTimes = aRepeat;
  params.srcRepStride = (aRepeat == 1) ? 0 : 2;
  params.dstRepStride =
      (aRepeat == 1) ? 0
                     : TRANS_BLOCK_FULL * (TRANS_BLOCK_FULL / TRANS_BLOCK_HALF);
  uint64_t dstBase = (uint64_t)(__ubuf__ float*)dst.GetPhyAddr();
  uint64_t srcBase = (uint64_t)(__ubuf__ float*)src.GetPhyAddr();
  LocalTensor<uint64_t> dstAddr = vaAddr;
  LocalTensor<uint64_t> srcAddr = vaAddr[TRANS_BLOCK_FULL];
  // xllm-ops compiles with --cce-auto-sync=off. SetValue is scalar-pipe UB
  // writes; TransDataTo5HD consumes the address list on the vector pipe.
  PipeBarrier<PIPE_V>();
  event_t evSV =
      static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
  event_t evVS =
      static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
  for (int32_t i = 0; i < groups; i++) {
    for (int32_t j = 0; j < TRANS_BLOCK_HALF; j++) {
      int32_t plane = i * TRANS_BLOCK_HALF + j;
      uint64_t srcOff =
          (uint64_t)(plane < planes ? plane : 0) * aAlign * sizeof(float);
      srcAddr.SetValue(j, srcBase + srcOff);
      srcAddr.SetValue(j + TRANS_BLOCK_HALF,
                       srcBase + srcOff + TRANS_BLOCK_HALF * sizeof(float));
      dstAddr.SetValue(
          j * 2,
          dstBase + (uint64_t)(i * TRANS_BLOCK_HALF + j * TRANS_BLOCK_FULL) *
                        sizeof(float));
      dstAddr.SetValue(
          j * 2 + 1,
          dstBase + (uint64_t)(i * TRANS_BLOCK_HALF +
                               (j + TRANS_BLOCK_HALF) * TRANS_BLOCK_FULL) *
                        sizeof(float));
    }
    SetFlag<HardEvent::S_V>(evSV);
    WaitFlag<HardEvent::S_V>(evSV);
    AscendC::TransDataTo5HD<float>(dstAddr, srcAddr, params);
    SetFlag<HardEvent::V_S>(evVS);
    WaitFlag<HardEvent::V_S>(evVS);
  }
}

__aicore__ inline void VFProcessIteration(RegTensor<float>& sum0,
                                          RegTensor<float>& sum1,
                                          RegTensor<float>& mix,
                                          float eps,
                                          MaskReg pregLoop) {
  ReduceSum(sum1, mix, pregLoop);
  Duplicate(sum1, sum1, pregLoop);
  Adds(sum1, sum1, eps, pregLoop);
  Div(mix, mix, sum1, pregLoop);
  Add(sum0, sum0, mix, pregLoop);
}

__aicore__ inline void VFProcessCombFragRLessVLUseFourUnfold(
    const LocalTensor<float>& combFragLocal,
    const LocalTensor<float>& mixLocal,
    const LocalTensor<float>& hcBaseLocal,
    const LocalTensor<float>& rsqrtLocal,
    float scale,
    float eps,
    uint16_t iters,
    uint16_t dim0,
    uint16_t dim1,
    uint16_t dim2) {
  __local_mem__ float* combFragLocalAddr =
      (__local_mem__ float*)combFragLocal.GetPhyAddr();
  __local_mem__ float* mixLocalAddr =
      (__local_mem__ float*)mixLocal.GetPhyAddr();
  __local_mem__ float* hcBaseLocalAddr =
      (__local_mem__ float*)hcBaseLocal.GetPhyAddr();
  __local_mem__ float* rsqrtLocalAddr =
      (__local_mem__ float*)rsqrtLocal.GetPhyAddr();
  uint32_t dim2Align = RoundUp<float>(dim2);
  __VEC_SCOPE__ {
    RegTensor<float> base;
    RegTensor<float> mix;
    RegTensor<float> mix1;
    RegTensor<float> mix2;
    RegTensor<float> mix3;
    RegTensor<float> mix4;
    RegTensor<float> rsqrt;
    RegTensor<float> max;
    RegTensor<float> sum;
    RegTensor<float> sum1;
    RegTensor<float> sum2;
    RegTensor<float> sum3;
    RegTensor<float> sum4;
    uint32_t sreg = dim2;
    MaskReg pregLoop = UpdateMask<float>(sreg);
    for (uint16_t i = 0; i < dim0; i++) {
      Duplicate(sum1, static_cast<float>(0), pregLoop);
      LoadInputDataWithBrc<float>(rsqrt, rsqrtLocalAddr, pregLoop, i);
      for (uint16_t j = 0; j < dim1; j++) {
        LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, j * dim2Align);
        LoadInputData<float>(
            mix, mixLocalAddr, pregLoop, i * dim1 * dim2Align + j * dim2Align);
        Mul(mix, mix, rsqrt, pregLoop);
        Muls(mix, mix, scale, pregLoop);
        Add(mix, mix, base, pregLoop);
        ReduceMax(max, mix, pregLoop);
        Duplicate(max, max, pregLoop);
        Sub(mix, mix, max, pregLoop);
        Exp(mix, mix, pregLoop);
        ReduceSum(sum, mix, pregLoop);
        Duplicate(sum, sum, pregLoop);
        Div(mix, mix, sum, pregLoop);
        Adds(mix, mix, eps, pregLoop);
        Add(sum1, sum1, mix, pregLoop);
        StoreOutputData(combFragLocalAddr,
                        mix,
                        pregLoop,
                        i * dim1 * dim2Align + j * dim2Align);
      }
      LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
      Adds(sum1, sum1, eps, pregLoop);
      for (uint16_t j = 0; j < dim1; j++) {
        LoadInputData<float>(mix,
                             combFragLocalAddr,
                             pregLoop,
                             i * dim1 * dim2Align + j * dim2Align);
        Div(mix, mix, sum1, pregLoop);
        StoreOutputData(combFragLocalAddr,
                        mix,
                        pregLoop,
                        i * dim1 * dim2Align + j * dim2Align);
      }
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    for (uint16_t i = 0; i < dim0; i++) {
      LoadInputData<float>(
          mix1, combFragLocalAddr, pregLoop, i * dim1 * dim2Align);
      LoadInputData<float>(mix2,
                           combFragLocalAddr,
                           pregLoop,
                           i * dim1 * dim2Align + 1 * dim2Align);
      LoadInputData<float>(mix3,
                           combFragLocalAddr,
                           pregLoop,
                           i * dim1 * dim2Align + 2 * dim2Align);
      LoadInputData<float>(mix4,
                           combFragLocalAddr,
                           pregLoop,
                           i * dim1 * dim2Align + 3 * dim2Align);
      for (uint16_t j = 0; j < iters; j++) {
        Duplicate(sum, static_cast<float>(0), pregLoop);
        VFProcessIteration(sum, sum1, mix1, eps, pregLoop);
        VFProcessIteration(sum, sum2, mix2, eps, pregLoop);
        VFProcessIteration(sum, sum3, mix3, eps, pregLoop);
        VFProcessIteration(sum, sum4, mix4, eps, pregLoop);
        Adds(sum, sum, eps, pregLoop);
        Div(mix1, mix1, sum, pregLoop);
        Div(mix2, mix2, sum, pregLoop);
        Div(mix3, mix3, sum, pregLoop);
        Div(mix4, mix4, sum, pregLoop);
      }
      StoreOutputData(combFragLocalAddr, mix1, pregLoop, i * dim1 * dim2Align);
      StoreOutputData(combFragLocalAddr,
                      mix2,
                      pregLoop,
                      i * dim1 * dim2Align + 1 * dim2Align);
      StoreOutputData(combFragLocalAddr,
                      mix3,
                      pregLoop,
                      i * dim1 * dim2Align + 2 * dim2Align);
      StoreOutputData(combFragLocalAddr,
                      mix4,
                      pregLoop,
                      i * dim1 * dim2Align + 3 * dim2Align);
    }
  }
}

template <typename T>
__aicore__ inline void VFProcessY(const LocalTensor<T>& yLocal,
                                  const LocalTensor<float>& mixLocal,
                                  const LocalTensor<T>& xLocal,
                                  uint16_t bs,
                                  uint16_t hcMult,
                                  uint16_t d) {
  __local_mem__ T* yLocalAddr = (__local_mem__ T*)yLocal.GetPhyAddr();
  __local_mem__ float* mixLocalAddr =
      (__local_mem__ float*)mixLocal.GetPhyAddr();
  __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
  uint32_t dAlign = RoundUp<T>(d);
  uint16_t loopCount = CeilDiv(d, VL_FP32);
  uint32_t hcMultAlign = RoundUp<float>(hcMult);
  if (loopCount > 1) {
    __VEC_SCOPE__ {
      RegTensor<float> x;
      RegTensor<float> mix;
      RegTensor<float> sum;
      MaskReg pregLoop;
      for (uint16_t i = 0; i < bs; i++) {
        uint32_t sreg = d;
        for (uint16_t j = 0; j < loopCount; j++) {
          pregLoop = UpdateMask<float>(sreg);
          Duplicate(sum, static_cast<float>(0), pregLoop);
          for (uint16_t k = 0; k < hcMult; k++) {
            LoadInputDataWithBrc<float>(
                mix, mixLocalAddr, pregLoop, i * hcMultAlign + k);
            LoadInputData<T>(x,
                             xLocalAddr,
                             pregLoop,
                             i * hcMult * dAlign + j * VL_FP32 + k * dAlign);
            Mul(x, mix, x, pregLoop);
            Add(sum, sum, x, pregLoop);
          }
          StoreOutputData(yLocalAddr, sum, pregLoop, i * dAlign + j * VL_FP32);
        }
      }
    }
  } else {
    __VEC_SCOPE__ {
      RegTensor<float> x;
      RegTensor<float> mix;
      RegTensor<float> sum;
      uint32_t sreg = d;
      MaskReg pregLoop = UpdateMask<float>(sreg);
      for (uint16_t i = 0; i < bs; i++) {
        Duplicate(sum, static_cast<float>(0), pregLoop);
        for (uint16_t j = 0; j < hcMult; j++) {
          LoadInputDataWithBrc<float>(
              mix, mixLocalAddr, pregLoop, i * hcMultAlign + j);
          LoadInputData<T>(
              x, xLocalAddr, pregLoop, i * hcMult * dAlign + j * dAlign);
          Mul(x, mix, x, pregLoop);
          Add(sum, sum, x, pregLoop);
        }
        StoreOutputData(yLocalAddr, sum, pregLoop, i * dAlign);
      }
    }
  }
}

template <typename T>
__aicore__ inline void CopyIn(const GlobalTensor<T>& inputGm,
                              const LocalTensor<T>& inputTensor,
                              const uint16_t nBurst,
                              const uint32_t copyLen,
                              uint32_t srcStride = 0) {
  DataCopyPadExtParams<T> dataCopyPadExtParams;
  dataCopyPadExtParams.isPad = false;
  dataCopyPadExtParams.leftPadding = 0;
  dataCopyPadExtParams.rightPadding = 0;
  dataCopyPadExtParams.paddingValue = 0;

  DataCopyExtParams dataCoptExtParams;
  dataCoptExtParams.blockCount = nBurst;
  dataCoptExtParams.blockLen = copyLen * sizeof(T);
  dataCoptExtParams.srcStride = srcStride * sizeof(T);
  dataCoptExtParams.dstStride = 0;
  DataCopyPad(inputTensor, inputGm, dataCoptExtParams, dataCopyPadExtParams);
}

template <typename T>
__aicore__ inline void CopyInWithLoopMode(const GlobalTensor<T>& inputGm,
                                          const LocalTensor<T>& inputTensor,
                                          const uint16_t outerLoop,
                                          const uint16_t nBurst,
                                          const uint32_t copyLen,
                                          const uint32_t gmLastDim,
                                          uint32_t srcStride = 0) {
  uint16_t copyLenAlign = RoundUp<T>(copyLen);
  LoopModeParams loopParams;
  loopParams.loop2Size = 1;
  loopParams.loop1Size = outerLoop;
  loopParams.loop2SrcStride = 0;
  loopParams.loop1SrcStride = gmLastDim * sizeof(T);
  loopParams.loop2DstStride = 0;
  loopParams.loop1DstStride = nBurst * copyLenAlign * sizeof(T);

  DataCopyPadExtParams<T> dataCopyPadExtParams;
  dataCopyPadExtParams.isPad = false;
  dataCopyPadExtParams.leftPadding = 0;
  dataCopyPadExtParams.rightPadding = 0;
  dataCopyPadExtParams.paddingValue = 0;

  DataCopyExtParams dataCoptExtParams;
  dataCoptExtParams.blockCount = nBurst;
  dataCoptExtParams.blockLen = copyLen * sizeof(T);
  dataCoptExtParams.srcStride = srcStride * sizeof(T);
  dataCoptExtParams.dstStride = 0;
  SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
  DataCopyPad(inputTensor, inputGm, dataCoptExtParams, dataCopyPadExtParams);
  ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
}

template <typename T>
__aicore__ inline void CopyOut(const LocalTensor<T>& outputTensor,
                               const GlobalTensor<T>& outputGm,
                               const uint16_t nBurst,
                               const uint32_t copyLen,
                               uint32_t dstStride = 0) {
  DataCopyExtParams dataCopyParams;
  dataCopyParams.blockCount = nBurst;
  dataCopyParams.blockLen = copyLen * sizeof(T);
  dataCopyParams.srcStride = 0;
  dataCopyParams.dstStride = dstStride * sizeof(T);
  DataCopyPad(outputGm, outputTensor, dataCopyParams);
}

__aicore__ inline void CopyOutCombSoA(const LocalTensor<float>& outputTensor,
                                      const GlobalTensor<float>& outputGm,
                                      uint32_t curA,
                                      int32_t planes) {
  if (planes == TRANS_BLOCK_FULL) {
    CopyOut(outputTensor, outputGm, 1, curA * TRANS_BLOCK_FULL);
    return;
  }
  constexpr int32_t rowBytes = TRANS_BLOCK_FULL * sizeof(float);
  int32_t burstBytes = planes * sizeof(float);
  DataCopyExtParams dataCopyParams;
  dataCopyParams.blockCount = static_cast<uint16_t>(curA);
  dataCopyParams.blockLen = burstBytes;
  dataCopyParams.srcStride =
      (rowBytes - CeilAlign(burstBytes, ONE_BLK_SIZE)) / ONE_BLK_SIZE;
  dataCopyParams.dstStride = 0;
  DataCopyPad(outputGm, outputTensor, dataCopyParams);
}

}  // namespace XllmKernelHcPreSinkhorn

#endif
