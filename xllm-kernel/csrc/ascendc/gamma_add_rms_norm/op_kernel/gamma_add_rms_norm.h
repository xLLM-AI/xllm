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
 * \file gamma_add_rms_norm.h
 * \brief add rms norm file
 */
#ifndef GAMMA_ADD_RMS_NORM_H_
#define GAMMA_ADD_RMS_NORM_H_
#include "gamma_add_rms_norm_base.h"

using namespace AscendC;
using namespace RmsNorm;

template <typename T, int32_t MODE>
class KernelGammaAddRmsNorm {
 public:
  __aicore__ inline KernelGammaAddRmsNorm(TPipe* pipe) { Ppipe = pipe; }
  __aicore__ inline void Init(GM_ADDR x1,
                              GM_ADDR x2,
                              GM_ADDR gamma,
                              GM_ADDR y,
                              GM_ADDR rstd,
                              GM_ADDR x,
                              GM_ADDR workspace,
                              const GammaAddRMSNormTilingData* tiling) {
    ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
    this->numRow = tiling->num_row;
    this->numCol = tiling->num_col;
    this->blockFactor = tiling->block_factor;
    this->rowFactor = tiling->row_factor;
    this->ubFactor = tiling->ub_factor;
    this->epsilon = tiling->epsilon;
    this->avgFactor = (this->numCol != 0) ? (float)1.0 / this->numCol : 0;
    this->addGammaOffset = tiling->add_gamma_offset;

    blockIdx_ = GetBlockIdx();
    if (blockIdx_ < GetBlockNum() - 1) {
      this->rowWork = this->blockFactor;
    } else if (blockIdx_ == GetBlockNum() - 1) {
      this->rowWork = this->numRow - (GetBlockNum() - 1) * this->blockFactor;
    }
    // get start index for current core, core parallel
    x1Gm.SetGlobalBuffer(
        (__gm__ T*)x1 + blockIdx_ * this->blockFactor * this->numCol,
        this->rowWork * this->numCol);
    x2Gm.SetGlobalBuffer(
        (__gm__ T*)x2 + blockIdx_ * this->blockFactor * this->numCol,
        this->rowWork * this->numCol);
    gammaGm.SetGlobalBuffer((__gm__ T*)gamma, this->numCol);
    yGm.SetGlobalBuffer(
        (__gm__ T*)y + blockIdx_ * this->blockFactor * this->numCol,
        this->rowWork * this->numCol);

    if constexpr (MODE == GAMMA_ADD_RMS_NORM_MODE) {
      rstdGm.SetGlobalBuffer(
          (__gm__ float*)rstd + blockIdx_ * this->blockFactor,
          this->blockFactor);
      xGm.SetGlobalBuffer(
          (__gm__ T*)x + blockIdx_ * this->blockFactor * this->numCol,
          this->rowWork * this->numCol);
    }
    if constexpr (MODE == PRE_RMS_NORM_MODE) {
      xGm.SetGlobalBuffer(
          (__gm__ T*)x + blockIdx_ * this->blockFactor * this->numCol,
          this->rowWork * this->numCol);
    }

    // pipe alloc memory to queue, the unit is Bytes
    Ppipe->InitBuffer(inQueueX, BUFFER_NUM, ubFactor * sizeof(T));
    Ppipe->InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(T));
    Ppipe->InitBuffer(outQueueY, BUFFER_NUM, ubFactor * sizeof(T));
    Ppipe->InitBuffer(outQueueRstd, BUFFER_NUM, rowFactor * sizeof(float));

    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
      Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
    }
    Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
    Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
  }

  __aicore__ inline void Process() {
    CopyInGamma();
    LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
    if (addGammaOffset != 0U) {
      AddGammaOffset(gammaLocal, numCol);
    }

    uint32_t i_o_max = RmsNorm::CeilDiv(this->rowWork, this->rowFactor);
    uint32_t row_tail = this->rowWork - (i_o_max - 1) * this->rowFactor;

    for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
      SubProcess(i_o, this->rowFactor, gammaLocal);
    }
    SubProcess(i_o_max - 1, row_tail, gammaLocal);
    inQueueGamma.FreeTensor(gammaLocal);
  }

  __aicore__ inline void SubProcess(uint32_t i_o,
                                    uint32_t calc_row_num,
                                    LocalTensor<T>& gammaLocal) {
    LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
    for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
      uint64_t gm_bias =
          (static_cast<uint64_t>(i_o) * static_cast<uint64_t>(this->rowFactor) +
           static_cast<uint64_t>(i_i)) *
          static_cast<uint64_t>(this->numCol);
      CopyIn(gm_bias);
      Compute(i_i, gammaLocal, rstdLocal);
      CopyOutY(gm_bias);
    }
    if constexpr (MODE == GAMMA_ADD_RMS_NORM_MODE) {
      outQueueRstd.EnQue<float>(rstdLocal);
      CopyOutRstd(i_o, calc_row_num);
    } else {
      outQueueRstd.FreeTensor(rstdLocal);
    }
  }

 private:
  __aicore__ inline void CopyIn(uint32_t gm_bias) {
    LocalTensor<T> x1Local_in = inQueueX.AllocTensor<T>();
    LocalTensor<T> x2Local = sqxBuf.Get<T>();
    LocalTensor<T> xLocal = outQueueY.AllocTensor<T>();

    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
      x2Local = x2Local[ubFactor];
    }

    DataCopyCustom<T>(x1Local_in, x1Gm[gm_bias], numCol);
    DataCopyCustom<T>(x2Local, x2Gm[gm_bias], numCol);
    inQueueX.EnQue(x1Local_in);
    auto x1Local = inQueueX.DeQue<T>();

    if constexpr (is_same<T, half>::value) {
      LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
      Add(xLocal, x1Local, x2Local, numCol);
      PipeBarrier<PIPE_V>();
      Cast(x1_fp32, xLocal, RoundMode::CAST_NONE, numCol);
      PipeBarrier<PIPE_V>();
    } else if constexpr (is_same<T, bfloat16_t>::value) {
      LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
      LocalTensor<float> x2_fp32 = sqxBuf.Get<float>();
      Cast(x1_fp32, x1Local, RoundMode::CAST_NONE, numCol);
      Cast(x2_fp32, x2Local, RoundMode::CAST_NONE, numCol);
      PipeBarrier<PIPE_V>();
      Add(x1_fp32, x1_fp32, x2_fp32, numCol);
      PipeBarrier<PIPE_V>();
      Cast(xLocal, x1_fp32, RoundMode::CAST_RINT, numCol);
      PipeBarrier<PIPE_V>();
    } else {
      Add(x1Local, x1Local, x2Local, numCol);
      PipeBarrier<PIPE_V>();
      Adds(xLocal, x1Local, (float)0, numCol);
    }
    inQueueX.FreeTensor(x1Local);

    // CopyOut x1 + x2
    outQueueY.EnQue(xLocal);
    auto x_out = outQueueY.DeQue<T>();
    if constexpr (MODE == GAMMA_ADD_RMS_NORM_MODE ||
                  MODE == PRE_RMS_NORM_MODE) {
      DataCopyCustom<T>(xGm[gm_bias], x_out, numCol);
    }
    outQueueY.FreeTensor(x_out);
  }

  __aicore__ inline void CopyInGamma() {
    LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
    DataCopyCustom<T>(gammaLocal, gammaGm, numCol);
    inQueueGamma.EnQue(gammaLocal);
  }

  __aicore__ inline void AddGammaOffset(LocalTensor<T>& gammaLocal,
                                        uint32_t elementNum) {
    if constexpr (is_same<T, bfloat16_t>::value) {
      LocalTensor<float> gammaFp32 = xFp32Buf.Get<float>();
      Cast(gammaFp32, gammaLocal, RoundMode::CAST_NONE, elementNum);
      PipeBarrier<PIPE_V>();
      Adds(gammaFp32, gammaFp32, static_cast<float>(1.0), elementNum);
      PipeBarrier<PIPE_V>();
      Cast(gammaLocal, gammaFp32, RoundMode::CAST_RINT, elementNum);
    } else {
      Adds(gammaLocal, gammaLocal, static_cast<T>(1.0), elementNum);
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Compute(uint32_t inner_progress,
                                 LocalTensor<float> gammaLocal,
                                 LocalTensor<float> rstdLocal) {
    LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
    Mul(sqx, xLocal, xLocal, numCol);
    PipeBarrier<PIPE_V>();

    Muls(sqx, sqx, avgFactor, numCol);
    PipeBarrier<PIPE_V>();

    ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
    PipeBarrier<PIPE_V>();
    Adds(sqx, sqx, epsilon, 1);
    PipeBarrier<PIPE_V>();

    Sqrt(sqx, sqx, 1);
    Duplicate(reduce_buf_local, ONE, 1);
    PipeBarrier<PIPE_V>();
    Div(sqx, reduce_buf_local, sqx, 1);
    PipeBarrier<PIPE_V>();
    event_t event_v_s =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event_v_s);
    WaitFlag<HardEvent::V_S>(event_v_s);
    float rstdValue = sqx.GetValue(0);
    event_t event_s_v =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(event_s_v);
    WaitFlag<HardEvent::S_V>(event_s_v);
    rstdLocal.SetValue(inner_progress, rstdValue);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    Muls(yLocal, xLocal, rstdValue, numCol);
    inQueueX.FreeTensor(xLocal);
    PipeBarrier<PIPE_V>();
    Mul(yLocal, gammaLocal, yLocal, numCol);
    PipeBarrier<PIPE_V>();
    outQueueY.EnQue<float>(yLocal);
  }

  __aicore__ inline void Compute(uint32_t inner_progress,
                                 LocalTensor<bfloat16_t> gammaLocal,
                                 LocalTensor<float> rstdLocal) {
    LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();

    Mul(sqx, x_fp32, x_fp32, numCol);
    PipeBarrier<PIPE_V>();

    Muls(sqx, sqx, this->avgFactor, this->numCol);
    PipeBarrier<PIPE_V>();
    ReduceSumCustom(sqx, sqx, reduce_buf_local, this->numCol);
    PipeBarrier<PIPE_V>();

    Adds(sqx, sqx, this->epsilon, 1);
    PipeBarrier<PIPE_V>();

    Sqrt(sqx, sqx, 1);
    Duplicate(reduce_buf_local, ONE, 1);
    PipeBarrier<PIPE_V>();
    Div(sqx, reduce_buf_local, sqx, 1);
    PipeBarrier<PIPE_V>();
    event_t event_v_s2 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event_v_s2);
    WaitFlag<HardEvent::V_S>(event_v_s2);
    float rstdValue2 = sqx.GetValue(0);
    event_t event_s_v2 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(event_s_v2);
    WaitFlag<HardEvent::S_V>(event_s_v2);
    rstdLocal.SetValue(inner_progress, rstdValue2);
    PipeBarrier<PIPE_V>();
    Muls(x_fp32, x_fp32, rstdValue2, this->numCol);
    PipeBarrier<PIPE_V>();
    LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();
    Cast(yLocal, x_fp32, RoundMode::CAST_RINT, this->numCol);
    PipeBarrier<PIPE_V>();
    Cast(x_fp32, yLocal, RoundMode::CAST_NONE, this->numCol);
    PipeBarrier<PIPE_V>();
    Cast(
        sqx, gammaLocal, RoundMode::CAST_NONE, numCol);  // gamma_fp32 reuse sqx
    PipeBarrier<PIPE_V>();
    Mul(x_fp32, x_fp32, sqx, numCol);
    PipeBarrier<PIPE_V>();
    Cast(yLocal, x_fp32, RoundMode::CAST_RINT, numCol);
    PipeBarrier<PIPE_V>();

    event_t event_v_mte =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(event_v_mte);
    WaitFlag<HardEvent::V_MTE2>(event_v_mte);

    outQueueY.EnQue<bfloat16_t>(yLocal);
  }

  __aicore__ inline void Compute(uint32_t inner_progress,
                                 LocalTensor<half> gammaLocal,
                                 LocalTensor<float> rstdLocal) {
    LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();

    Mul(sqx, x_fp32, x_fp32, this->numCol);
    PipeBarrier<PIPE_V>();

    Muls(sqx, sqx, this->avgFactor, this->numCol);
    PipeBarrier<PIPE_V>();

    ReduceSumCustom(sqx, sqx, reduce_buf_local, this->numCol);
    PipeBarrier<PIPE_V>();

    Adds(sqx, sqx, this->epsilon, 1);
    PipeBarrier<PIPE_V>();

    Sqrt(sqx, sqx, 1);
    Duplicate(reduce_buf_local, ONE, 1);
    PipeBarrier<PIPE_V>();
    Div(sqx, reduce_buf_local, sqx, 1);
    PipeBarrier<PIPE_V>();
    event_t event_v_s3 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event_v_s3);
    WaitFlag<HardEvent::V_S>(event_v_s3);
    float rstdValue3 = sqx.GetValue(0);
    event_t event_s_v3 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(event_s_v3);
    WaitFlag<HardEvent::S_V>(event_s_v3);
    rstdLocal.SetValue(inner_progress, rstdValue3);
    PipeBarrier<PIPE_V>();
    Muls(x_fp32, x_fp32, rstdValue3, this->numCol);
    PipeBarrier<PIPE_V>();
    LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
    Cast(yLocal, x_fp32, RoundMode::CAST_NONE, this->numCol);

    event_t event_v_mte =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(event_v_mte);
    WaitFlag<HardEvent::V_MTE2>(event_v_mte);

    PipeBarrier<PIPE_V>();
    Mul(yLocal, gammaLocal, yLocal, numCol);
    PipeBarrier<PIPE_V>();
    outQueueY.EnQue<half>(yLocal);
  }

  __aicore__ inline void CopyOutY(uint32_t progress) {
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    DataCopyCustom<T>(yGm[progress], yLocal, numCol);
    outQueueY.FreeTensor(yLocal);
  }

  __aicore__ inline void CopyOutRstd(uint32_t outer_progress, uint32_t num) {
    LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
#if __CCE_AICORE__ == 220 || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    DataCopyCustom<float>(
        rstdGm[outer_progress * this->rowFactor], rstdLocal, num);
#endif
    outQueueRstd.FreeTensor(rstdLocal);
  }

 private:
  TPipe* Ppipe = nullptr;
  // create queues for input, in this case depth is equal to buffer num
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGamma;
  // create queues for output, in this case depth is equal to buffer num
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueRstd;

  TBuf<TPosition::VECCALC> xFp32Buf;
  TBuf<TPosition::VECCALC> sqxBuf;
  TBuf<TPosition::VECCALC> reduceFp32Buf;
  GlobalTensor<T> x1Gm;
  GlobalTensor<T> x2Gm;
  GlobalTensor<T> gammaGm;
  GlobalTensor<T> yGm;
  GlobalTensor<float> rstdGm;
  GlobalTensor<T> xGm;

  uint32_t numRow;
  uint32_t numCol;
  uint32_t blockFactor;
  uint32_t rowFactor;
  uint32_t ubFactor;
  float epsilon;
  float avgFactor;
  int32_t blockIdx_;
  uint32_t rowWork = 1;
  uint32_t addGammaOffset = 0;
};
#endif  // GAMMA_ADD_RMS_NORM_H_
