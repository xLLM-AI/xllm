/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Copyright 2026 The xLLM Authors. All Rights Reserved.
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
 * \file gamma_add_rms_norm_single_n.h
 * \brief add rms norm single n file
 */
#ifndef GAMMA_ADD_RMS_NORM_SINGLE_N_H_
#define GAMMA_ADD_RMS_NORM_SINGLE_N_H_
#include "gamma_add_rms_norm_base.h"

using namespace AscendC;
using namespace RmsNorm;

template <typename T, int32_t MODE>
class KernelGammaAddRmsNormSingleN {
  static constexpr int32_t MAXBUFFER = 195584;

 public:
  __aicore__ inline KernelGammaAddRmsNormSingleN(TPipe* pipe) { Ppipe = pipe; }
  __aicore__ inline void Init(GM_ADDR x1,
                              GM_ADDR x2,
                              GM_ADDR gamma,
                              GM_ADDR y,
                              GM_ADDR rstd,
                              GM_ADDR x,
                              GM_ADDR workspace,
                              const GammaAddRMSNormTilingData* tiling) {
    ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");

    this->numCol = tiling->num_col;
    this->blockFactor = 1;
    this->ubFactor = tiling->ub_factor;
    this->epsilon = tiling->epsilon;
    this->avgFactor = (this->numCol != 0) ? (float)1.0 / this->numCol : 0;
    this->addGammaOffset = tiling->add_gamma_offset;

    this->rowWork = 1;
    blockIdx_ = GetBlockIdx();
    // get start index for current core, core parallel
    x1Gm.SetGlobalBuffer((__gm__ T*)x1 + blockIdx_ * this->numCol,
                         this->numCol);
    x2Gm.SetGlobalBuffer((__gm__ T*)x2 + blockIdx_ * this->numCol,
                         this->numCol);
    gammaGm.SetGlobalBuffer((__gm__ T*)gamma, this->numCol);
    yGm.SetGlobalBuffer((__gm__ T*)y + blockIdx_ * this->numCol, this->numCol);

    if constexpr (MODE == GAMMA_ADD_RMS_NORM_MODE) {
      rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockIdx_, 1);
      xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx_ * this->numCol,
                          this->numCol);
    }
    if constexpr (MODE == PRE_RMS_NORM_MODE) {
      xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx_ * numCol, numCol);
    }

    Ppipe->InitBuffer(unitBuf, MAXBUFFER);  // (192 - 1) * 1024 byte
  }

  __aicore__ inline void Process() {
    if constexpr (is_same<T, half>::value) {
      ProcessFp16();
    } else if constexpr (is_same<T, float>::value) {
      ProcessFp32();
    } else {
      ProcessBf16();
    }
  }

 private:
  __aicore__ inline void ProcessFp16() {
    LocalTensor<float> ubLocal = unitBuf.Get<float>();
    LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();
    LocalTensor<T> x1Local = xLocal[0];
    LocalTensor<T> x2Local = xLocal[ubFactor];
    LocalTensor<float> xFp32Local = ubLocal[ubFactor];
    LocalTensor<float> sqxLocal = ubLocal[ubFactor * 2];
    LocalTensor<float> tmpLocal = ubLocal[ubFactor * 3];

    DataCopyCustom<T>(x1Local, x1Gm, numCol);
    event_t eventMTE2V1 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMTE2V1);
    DataCopyCustom<T>(x2Local, x2Gm, numCol);
    event_t eventMTE2V2 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V1);
    SetFlag<HardEvent::MTE2_V>(eventMTE2V2);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
    Add(x1Local, x1Local, x2Local, numCol);
    PipeBarrier<PIPE_V>();

    // copy gamma
    event_t eventVMTE2 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventVMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventVMTE2);

    DataCopyCustom<T>(x2Local, gammaGm, numCol);  // gammaLocal use x2Local
    SetFlag<HardEvent::MTE2_V>(eventMTE2V2);

    // copy x out
    event_t eventVMTE3 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
    if constexpr (MODE == GAMMA_ADD_RMS_NORM_MODE ||
                  MODE == PRE_RMS_NORM_MODE) {
      DataCopyCustom<T>(xGm, x1Local, numCol);
    }
    event_t eventMTE3V =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventMTE3V);

    Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, this->numCol);
    PipeBarrier<PIPE_V>();
    Mul(sqxLocal, xFp32Local, xFp32Local, this->numCol);
    PipeBarrier<PIPE_V>();
    Muls(sqxLocal, sqxLocal, this->avgFactor, this->numCol);
    PipeBarrier<PIPE_V>();
    ReduceSumCustom(sqxLocal, sqxLocal, tmpLocal, this->numCol);
    PipeBarrier<PIPE_V>();
    Adds(sqxLocal, sqxLocal, this->epsilon, 1);
    PipeBarrier<PIPE_V>();
    Sqrt(sqxLocal, sqxLocal, 1);
    Duplicate(tmpLocal, ONE, 1);
    PipeBarrier<PIPE_V>();
    Div(sqxLocal, tmpLocal, sqxLocal, 1);
    PipeBarrier<PIPE_V>();

    // copyout rstd
#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    SetFlag<HardEvent::V_MTE3>(eventVMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
    if constexpr (MODE == GAMMA_ADD_RMS_NORM_MODE) {
      DataCopyCustom<float>(rstdGm, sqxLocal, 1);
    }
#endif
    event_t eventVS_FP32 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVS_FP32);
    WaitFlag<HardEvent::V_S>(eventVS_FP32);
    float rstdValueFp32 = sqxLocal.GetValue(0);
    event_t eventSV_FP32 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventSV_FP32);
    WaitFlag<HardEvent::S_V>(eventSV_FP32);

    Muls(xFp32Local, xFp32Local, rstdValueFp32, this->numCol);
    PipeBarrier<PIPE_V>();
    WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
    Cast(x1Local, xFp32Local, RoundMode::CAST_NONE, this->numCol);
    PipeBarrier<PIPE_V>();
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
    if (addGammaOffset != 0U) {
      Adds(x2Local, x2Local, static_cast<half>(1.0), this->numCol);
      PipeBarrier<PIPE_V>();
    }
    Mul(x1Local, x1Local, x2Local, this->numCol);
    SetFlag<HardEvent::V_MTE3>(eventVMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
    DataCopyCustom<T>(yGm, x1Local, this->numCol);
  }

  __aicore__ inline void ProcessFp32() {
    LocalTensor<float> ubLocal = unitBuf.Get<float>();
    LocalTensor<T> x1Local = ubLocal[0];
    LocalTensor<T> x2Local = ubLocal[ubFactor];
    LocalTensor<float> sqxLocal = ubLocal[ubFactor * 2];
    LocalTensor<float> tmpLocal = ubLocal[ubFactor * 3];

    DataCopyCustom<T>(x1Local, x1Gm, numCol);
    event_t eventMTE2V1 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMTE2V1);
    DataCopyCustom<T>(x2Local, x2Gm, numCol);
    event_t eventMTE2V2 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMTE2V2);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V1);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
    Add(x1Local, x1Local, x2Local, numCol);
    PipeBarrier<PIPE_V>();

    // copy gamma
    event_t eventVMTE2 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventVMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventVMTE2);

    DataCopyCustom<T>(x2Local, gammaGm, numCol);  // gammaLocal use x2Local
    SetFlag<HardEvent::MTE2_V>(eventMTE2V2);

    // copy x out
    event_t eventVMTE3 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
    DataCopyCustom<T>(xGm, x1Local, numCol);
    event_t eventMTE3V =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventMTE3V);

    Mul(sqxLocal, x1Local, x1Local, numCol);
    PipeBarrier<PIPE_V>();
    Muls(sqxLocal, sqxLocal, this->avgFactor, this->numCol);
    PipeBarrier<PIPE_V>();
    ReduceSumCustom(sqxLocal, sqxLocal, tmpLocal, this->numCol);
    PipeBarrier<PIPE_V>();
    Adds(sqxLocal, sqxLocal, this->epsilon, 1);
    PipeBarrier<PIPE_V>();
    Sqrt(sqxLocal, sqxLocal, 1);
    Duplicate(tmpLocal, ONE, 1);
    PipeBarrier<PIPE_V>();
    Div(sqxLocal, tmpLocal, sqxLocal, 1);
    PipeBarrier<PIPE_V>();

    // copyout rstd
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    SetFlag<HardEvent::V_MTE3>(eventVMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
    DataCopyCustom<float>(rstdGm, sqxLocal, 1);
#endif
    event_t eventVS_FP16 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVS_FP16);
    WaitFlag<HardEvent::V_S>(eventVS_FP16);
    float rstdValue = sqxLocal.GetValue(0);
    event_t eventSV_FP16 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventSV_FP16);
    WaitFlag<HardEvent::S_V>(eventSV_FP16);
    WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
    Muls(x1Local, x1Local, rstdValue, numCol);
    PipeBarrier<PIPE_V>();
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
    if (addGammaOffset != 0U) {
      Adds(x2Local, x2Local, static_cast<float>(1.0), numCol);
      PipeBarrier<PIPE_V>();
    }
    Mul(x1Local, x1Local, x2Local, numCol);
    SetFlag<HardEvent::V_MTE3>(eventVMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
    DataCopyCustom<T>(yGm, x1Local, numCol);
  }

  __aicore__ inline void ProcessBf16() {
    LocalTensor<float> ubLocal = unitBuf.Get<float>();
    LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();
    LocalTensor<T> x1Local = xLocal[0];
    LocalTensor<T> x2Local = xLocal[ubFactor];
    LocalTensor<float> xFp32Local = ubLocal[ubFactor];
    LocalTensor<float> sqxLocal = ubLocal[ubFactor * 2];
    LocalTensor<float> tmpLocal = ubLocal[ubFactor * 3];

    DataCopyCustom<T>(x1Local, x1Gm, numCol);
    event_t eventMTE2V1_BF16_0 =
        static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    SetFlag<HardEvent::MTE2_V>(eventMTE2V1_BF16_0);
    DataCopyCustom<T>(x2Local, x2Gm, numCol);
    event_t eventMTE2V2_BF16_0 =
        static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    SetFlag<HardEvent::MTE2_V>(eventMTE2V2_BF16_0);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V1_BF16_0);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventMTE2V1_BF16_0);
    Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V2_BF16_0);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventMTE2V2_BF16_0);
    Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol);
    PipeBarrier<PIPE_V>();
    Add(xFp32Local, xFp32Local, sqxLocal, numCol);
    PipeBarrier<PIPE_V>();
    Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol);
    PipeBarrier<PIPE_V>();
    // copy gamma
    event_t eventVMTE2_BF16_0 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventVMTE2_BF16_0);
    WaitFlag<HardEvent::V_MTE2>(eventVMTE2_BF16_0);

    DataCopyCustom<T>(x2Local, gammaGm, numCol);  // gammaLocal use x2Local
    event_t eventMTE2V2_BF16_1 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMTE2V2_BF16_1);

    // copy x out
    event_t eventVMTE3_BF16_0 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVMTE3_BF16_0);
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3_BF16_0);
    if constexpr (MODE == GAMMA_ADD_RMS_NORM_MODE ||
                  MODE == PRE_RMS_NORM_MODE) {
      DataCopyCustom<T>(xGm, x1Local, numCol);
    }
    event_t eventMTE3V_BF16_0 =
        static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    SetFlag<HardEvent::MTE3_V>(eventMTE3V_BF16_0);

    Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
    PipeBarrier<PIPE_V>();
    Mul(sqxLocal, xFp32Local, xFp32Local, numCol);
    PipeBarrier<PIPE_V>();
    Muls(sqxLocal, sqxLocal, avgFactor, numCol);
    PipeBarrier<PIPE_V>();
    ReduceSumCustom(sqxLocal, sqxLocal, tmpLocal, numCol);
    PipeBarrier<PIPE_V>();
    Adds(sqxLocal, sqxLocal, epsilon, 1);
    PipeBarrier<PIPE_V>();
    Sqrt(sqxLocal, sqxLocal, 1);
    Duplicate(tmpLocal, ONE, 1);
    PipeBarrier<PIPE_V>();
    Div(sqxLocal, tmpLocal, sqxLocal, 1);
    PipeBarrier<PIPE_V>();
    event_t eventVS_BF16_0 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVS_BF16_0);
    WaitFlag<HardEvent::V_S>(eventVS_BF16_0);
    float rstdValue = sqxLocal.GetValue(0);
    event_t eventSV_BF16_0 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventSV_BF16_0);
    WaitFlag<HardEvent::S_V>(eventSV_BF16_0);
    // copyout rstd
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    event_t eventVMTE3_BF16_1 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVMTE3_BF16_1);
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3_BF16_1);
    if constexpr (MODE == GAMMA_ADD_RMS_NORM_MODE) {
      DataCopyCustom<float>(rstdGm, sqxLocal, 1);
    }
    event_t eventMTE3V2_BF16_0 =
        static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    SetFlag<HardEvent::MTE3_V>(eventMTE3V2_BF16_0);
#endif

    Muls(xFp32Local, xFp32Local, rstdValue, numCol);
    PipeBarrier<PIPE_V>();
    WaitFlag<HardEvent::MTE3_V>(eventMTE3V_BF16_0);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventMTE3V_BF16_0);
    Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol);
    PipeBarrier<PIPE_V>();
    Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
    PipeBarrier<PIPE_V>();
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V2_BF16_1);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    WaitFlag<HardEvent::MTE3_V>(eventMTE3V2_BF16_0);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventMTE3V2_BF16_0);
#endif
    Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol);
    PipeBarrier<PIPE_V>();
    if (addGammaOffset != 0U) {
      Adds(sqxLocal, sqxLocal, static_cast<float>(1.0), numCol);
      PipeBarrier<PIPE_V>();
      Cast(x2Local, sqxLocal, RoundMode::CAST_RINT, numCol);
      PipeBarrier<PIPE_V>();
      Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol);
      PipeBarrier<PIPE_V>();
    }
    Mul(xFp32Local, xFp32Local, sqxLocal, numCol);
    PipeBarrier<PIPE_V>();
    Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol);
    event_t eventVMTE3_BF16_2 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVMTE3_BF16_2);
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3_BF16_2);
    DataCopyCustom<T>(yGm, x1Local, numCol);
  }

 private:
  TPipe* Ppipe = nullptr;

  TBuf<TPosition::VECCALC> unitBuf;
  GlobalTensor<T> x1Gm;
  GlobalTensor<T> x2Gm;
  GlobalTensor<T> gammaGm;
  GlobalTensor<T> yGm;
  GlobalTensor<float> rstdGm;
  GlobalTensor<T> xGm;

  uint32_t numRow;
  uint32_t numCol;
  uint32_t blockFactor;  // number of calculations rows on each core
  uint32_t ubFactor;
  float epsilon;
  float avgFactor;
  int32_t blockIdx_;
  uint32_t rowWork = 1;
  uint32_t addGammaOffset = 0;
};
#endif  // _GAMMA_ADD_RMS_NORM_SINGLE_N_H_
