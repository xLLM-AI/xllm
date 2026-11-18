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
 * \file gamma_add_rms_norm_split_d.h
 * \brief add rms norm split d file
 */
#ifndef GAMMA_ADD_RMS_NORM_SPLIT_D_H_
#define GAMMA_ADD_RMS_NORM_SPLIT_D_H_
#include "gamma_add_rms_norm_base.h"

using namespace AscendC;
using namespace RmsNorm;

template <typename T, int32_t MODE>
class KernelGammaAddRmsNormSplitD {
 public:
  __aicore__ inline KernelGammaAddRmsNormSplitD(TPipe* pipe) { Ppipe = pipe; }
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
    } else {
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
      xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx_ * blockFactor * numCol,
                          rowWork * numCol);
    }

    // pipe alloc memory to queue, the unit is Bytes.
    // We need 2 buffers here for both x1 and x2.
    Ppipe->InitBuffer(inQueueX, BUFFER_NUM, 2 * ubFactor * sizeof(T));
    Ppipe->InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(T));
    Ppipe->InitBuffer(outQueueY, BUFFER_NUM, ubFactor * sizeof(T));
    Ppipe->InitBuffer(outQueueRstd, BUFFER_NUM, rowFactor * sizeof(float));

    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
      Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
    }
    Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
    Ppipe->InitBuffer(sumBuf, rowFactor * NUM_PER_BLK_FP32 * sizeof(float));
    Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
  }

  __aicore__ inline void Process() {
    uint32_t i_o_max = RmsNorm::CeilDiv(rowWork, rowFactor);
    uint32_t row_tail = rowWork - (i_o_max - 1) * rowFactor;
    uint32_t j_max = RmsNorm::CeilDiv(numCol, ubFactor);
    uint32_t col_tail = numCol - (j_max - 1) * ubFactor;
    for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
      SubProcess(i_o, rowFactor, j_max, col_tail);
    }
    SubProcess(i_o_max - 1, row_tail, j_max, col_tail);
  }

  __aicore__ inline void SubProcess(uint32_t i_o,
                                    uint32_t calc_row_num,
                                    uint32_t j_max,
                                    uint32_t col_tail) {
    LocalTensor<float> sumLocal = sumBuf.Get<float>();

    LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
    Duplicate(rstdLocal, (float)0.0, calc_row_num);
    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < j_max - 1; j++) {
      ComputeFormer(i_o, calc_row_num, j, rstdLocal, sumLocal, ubFactor);
    }
    // do tail
    ComputeFormer(i_o, calc_row_num, j_max - 1, rstdLocal, sumLocal, col_tail);
    ComputeRstd(rstdLocal, calc_row_num);

    for (uint32_t j = 0; j < j_max - 1; j++) {
      ComputeLatter(i_o, calc_row_num, j, rstdLocal, ubFactor);
    }
    ComputeLatter(i_o, calc_row_num, j_max - 1, rstdLocal, col_tail);

    if constexpr (MODE == GAMMA_ADD_RMS_NORM_MODE) {
      outQueueRstd.EnQue<float>(rstdLocal);
      CopyOutRstd(i_o, calc_row_num);
    } else {
      outQueueRstd.FreeTensor(rstdLocal);
    }
  }

 private:
  __aicore__ inline void CopyInX1X2(uint32_t i_idx,
                                    uint32_t j_idx,
                                    uint32_t num) {
    LocalTensor<T> splitX1X2In = inQueueX.AllocTensor<T>();
    LocalTensor<T> splitX1In = splitX1X2In[0];
    LocalTensor<T> splitX2In = splitX1X2In[this->ubFactor];
    DataCopyCustom<T>(
        splitX1In, x1Gm[i_idx * this->numCol + j_idx * this->ubFactor], num);
    DataCopyCustom<T>(
        splitX2In, x2Gm[i_idx * this->numCol + j_idx * this->ubFactor], num);
    inQueueX.EnQue(splitX1X2In);
  }

  __aicore__ inline void AddX(uint32_t i_idx, uint32_t j_idx, uint32_t num) {
    CopyInX1X2(i_idx, j_idx, num);
    LocalTensor<T> splitX1X2Local = inQueueX.DeQue<T>();

    auto splitX1Local = splitX1X2Local[0];
    auto splitX2Local = splitX1X2Local[this->ubFactor];
    if constexpr (is_same<T, bfloat16_t>::value) {
      LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
      LocalTensor<float> x2_fp32 =
          splitX1X2Local.template ReinterpretCast<float>();
      Cast(x1_fp32, splitX1Local, RoundMode::CAST_NONE, num);
      PipeBarrier<PIPE_V>();
      Cast(x2_fp32, splitX2Local, RoundMode::CAST_NONE, num);
      PipeBarrier<PIPE_V>();
      Add(x1_fp32, x1_fp32, x2_fp32, num);
      PipeBarrier<PIPE_V>();
      Cast(splitX1X2Local, x1_fp32, RoundMode::CAST_RINT, num);
    } else {
      Add(splitX1X2Local, splitX1Local, splitX2Local, num);
    }
    PipeBarrier<PIPE_V>();
    inQueueX.EnQue(splitX1X2Local);
  }

  __aicore__ inline void CopyInAndAdd(uint32_t i_idx,
                                      uint32_t j_idx,
                                      uint32_t num) {
    CopyInX1X2(i_idx, j_idx, num);
    LocalTensor<T> splitX1X2Local = inQueueX.DeQue<T>();
    auto splitX1Local = splitX1X2Local[0];
    auto splitX2Local = splitX1X2Local[this->ubFactor];

    LocalTensor<T> splitXLocal = outQueueY.AllocTensor<T>();

    if constexpr (is_same<T, half>::value) {
      LocalTensor<float> splitX1Fp32 = xFp32Buf.Get<float>();

      Add(splitXLocal, splitX1Local, splitX2Local, num);
      PipeBarrier<PIPE_V>();
      Cast(splitX1Fp32, splitXLocal, RoundMode::CAST_NONE, num);
      PipeBarrier<PIPE_V>();
      // x1+x2 saved in x1_fp32
    } else if constexpr (is_same<T, bfloat16_t>::value) {
      LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
      LocalTensor<float> x2_fp32 =
          splitX1X2Local.template ReinterpretCast<float>();

      Cast(x1_fp32, splitX1Local, RoundMode::CAST_NONE, num);
      PipeBarrier<PIPE_V>();
      Cast(x2_fp32, splitX2Local, RoundMode::CAST_NONE, num);
      PipeBarrier<PIPE_V>();

      Add(x1_fp32, x1_fp32, x2_fp32, num);
      PipeBarrier<PIPE_V>();
      Cast(splitXLocal, x1_fp32, RoundMode::CAST_RINT, num);
      PipeBarrier<PIPE_V>();
      // x1+x2 saved in x1_fp32
    } else {
      Add(splitX1Local, splitX1Local, splitX2Local, num);
      PipeBarrier<PIPE_V>();
      Adds(splitXLocal, splitX1Local, (float)0.0, num);
      // x1+x2 saved in inQueueX
    }
    inQueueX.FreeTensor(splitX1X2Local);

    // copy out to workspace && x_out
    outQueueY.EnQue(splitXLocal);
    auto x_out = outQueueY.DeQue<T>();
    if constexpr (MODE == GAMMA_ADD_RMS_NORM_MODE ||
                  MODE == PRE_RMS_NORM_MODE) {
      DataCopyCustom<T>(xGm[i_idx * numCol + j_idx * ubFactor], x_out, num);
    }
    outQueueY.FreeTensor(x_out);
  }

  __aicore__ inline void ComputeFormer(uint32_t i_o_idx,
                                       uint32_t calc_row_num,
                                       uint32_t j_idx,
                                       LocalTensor<float>& rstdLocal,
                                       LocalTensor<float>& sumLocal,
                                       uint32_t num) {
    for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
      CopyInAndAdd(i_o_idx * rowFactor + i_i, j_idx, num);
      ComputeSum(i_i, sumLocal, num);
    }
    BlockReduceSumFP32(sumLocal, sumLocal, calc_row_num * NUM_PER_BLK_FP32);
    Add(rstdLocal, rstdLocal, sumLocal, calc_row_num);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void ComputeSum(uint32_t i_i_idx,
                                    LocalTensor<float>& sumLocal,
                                    uint32_t num) {
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
      LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
      PipeBarrier<PIPE_V>();
      Mul(sqx, x_fp32, x_fp32, num);
    } else {
      LocalTensor<T> xLocal = inQueueX.AllocTensor<float>();
      PipeBarrier<PIPE_V>();
      Mul(sqx, xLocal, xLocal, num);
      inQueueX.FreeTensor(xLocal);
    }
    PipeBarrier<PIPE_V>();
    Muls(sqx, sqx, avgFactor, num);
    PipeBarrier<PIPE_V>();
    // 8 means 8 fp32 pre block
    ReduceSumFP32ToBlock(sumLocal[i_i_idx * 8], sqx, reduce_buf_local, num);
  }

  __aicore__ inline void ComputeRstd(LocalTensor<float> rstdLocal,
                                     uint32_t num) {
    LocalTensor<float> splitReduceBufLocal = reduceFp32Buf.Get<float>();
    Adds(rstdLocal, rstdLocal, this->epsilon, num);
    PipeBarrier<PIPE_V>();
    Sqrt(rstdLocal, rstdLocal, num);
    Duplicate(splitReduceBufLocal, ONE, num);
    PipeBarrier<PIPE_V>();
    Div(rstdLocal, splitReduceBufLocal, rstdLocal, num);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void ComputeLatter(uint32_t i_o_idx,
                                       uint32_t calc_row_num,
                                       uint32_t j_idx,
                                       LocalTensor<float>& rstdLocal,
                                       uint32_t num) {
    CopyInGamma(j_idx, num);
    LocalTensor<T> splitGammaLocal = inQueueGamma.DeQue<T>();
    if (addGammaOffset != 0U) {
      AddGammaOffset(splitGammaLocal, num);
    }
    for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
      CopyInX(i_o_idx * rowFactor + i_i, j_idx, num);
      ComputeY(i_i, splitGammaLocal, rstdLocal, num);
      CopyOutY(i_o_idx * rowFactor + i_i, j_idx, num);
    }
    inQueueGamma.FreeTensor(splitGammaLocal);
  }

  __aicore__ inline void CopyInGamma(uint32_t j_idx, uint32_t num) {
    LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
    DataCopyCustom<T>(gammaLocal, gammaGm[j_idx * ubFactor], num);
    inQueueGamma.EnQue(gammaLocal);
  }

  __aicore__ inline void AddGammaOffset(LocalTensor<T>& gammaLocal,
                                        uint32_t num) {
    if constexpr (is_same<T, bfloat16_t>::value) {
      LocalTensor<float> gammaFp32 = xFp32Buf.Get<float>();
      Cast(gammaFp32, gammaLocal, RoundMode::CAST_NONE, num);
      PipeBarrier<PIPE_V>();
      Adds(gammaFp32, gammaFp32, static_cast<float>(1.0), num);
      PipeBarrier<PIPE_V>();
      Cast(gammaLocal, gammaFp32, RoundMode::CAST_RINT, num);
    } else {
      Adds(gammaLocal, gammaLocal, static_cast<T>(1.0), num);
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void CopyInX(uint32_t i_idx, uint32_t j_idx, uint32_t num) {
    if constexpr (MODE == GAMMA_ADD_RMS_NORM_MODE ||
                  MODE == PRE_RMS_NORM_MODE) {
      LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
      DataCopyCustom<T>(xLocal, xGm[i_idx * numCol + j_idx * ubFactor], num);
      inQueueX.EnQue<T>(xLocal);
    }
    if constexpr (MODE == POST_RMS_NORM_MODE) {
      AddX(i_idx, j_idx, num);
    }
    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
      LocalTensor<float> splitXFp32 = xFp32Buf.Get<float>();
      LocalTensor<T> splitXLocalDeq = inQueueX.DeQue<T>();
      Cast(splitXFp32, splitXLocalDeq, RoundMode::CAST_NONE, num);
      PipeBarrier<PIPE_V>();
      inQueueX.FreeTensor(splitXLocalDeq);
    }
  }

  __aicore__ inline void ComputeY(uint32_t i_i_idx,
                                  LocalTensor<half>& splitGammaLocal,
                                  LocalTensor<float>& splitRstdLocal,
                                  uint32_t num) {
    LocalTensor<float> splitXFp32 = xFp32Buf.Get<float>();
    LocalTensor<float> splitSqx = sqxBuf.Get<float>();
    event_t splitEventVS =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(splitEventVS);
    WaitFlag<HardEvent::V_S>(splitEventVS);
    float splitRstdValue = splitRstdLocal.GetValue(i_i_idx);
    event_t splitEventSV =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(splitEventSV);
    WaitFlag<HardEvent::S_V>(splitEventSV);
    PipeBarrier<PIPE_V>();
    Muls(splitXFp32, splitXFp32, splitRstdValue, num);
    PipeBarrier<PIPE_V>();
    LocalTensor<half> splitYLocal = outQueueY.AllocTensor<half>();
    Cast(splitYLocal, splitXFp32, RoundMode::CAST_NONE, num);
    PipeBarrier<PIPE_V>();
    Mul(splitYLocal, splitGammaLocal, splitYLocal, num);
    PipeBarrier<PIPE_V>();
    outQueueY.EnQue<half>(splitYLocal);
  }

  __aicore__ inline void ComputeY(uint32_t i_i_idx,
                                  LocalTensor<float>& gammaLocal,
                                  LocalTensor<float>& rstdLocal,
                                  uint32_t num) {
    LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    event_t event_v_s =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event_v_s);
    WaitFlag<HardEvent::V_S>(event_v_s);
    float rstdValue = rstdLocal.GetValue(i_i_idx);
    event_t event_s_v =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(event_s_v);
    WaitFlag<HardEvent::S_V>(event_s_v);
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    Muls(yLocal, xLocal, rstdValue, num);
    inQueueX.FreeTensor(xLocal);
    PipeBarrier<PIPE_V>();
    Mul(yLocal, gammaLocal, yLocal, num);
    PipeBarrier<PIPE_V>();
    outQueueY.EnQue<float>(yLocal);
  }

  __aicore__ inline void ComputeY(uint32_t i_i_idx,
                                  LocalTensor<bfloat16_t>& gammaLocal,
                                  LocalTensor<float>& rstdLocal,
                                  uint32_t num) {
    LocalTensor<float> splitXFp32Bf16 = xFp32Buf.Get<float>();
    LocalTensor<float> splitSqxBf16 = sqxBuf.Get<float>();
    event_t splitEventVSBf16 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(splitEventVSBf16);
    WaitFlag<HardEvent::V_S>(splitEventVSBf16);
    float splitRstdValueBf16 = rstdLocal.GetValue(i_i_idx);
    event_t splitEventSVBf16 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(splitEventSVBf16);
    WaitFlag<HardEvent::S_V>(splitEventSVBf16);
    PipeBarrier<PIPE_V>();
    Muls(splitXFp32Bf16, splitXFp32Bf16, splitRstdValueBf16, num);
    PipeBarrier<PIPE_V>();
    LocalTensor<bfloat16_t> splitYLocalBf16 =
        outQueueY.AllocTensor<bfloat16_t>();
    Cast(splitYLocalBf16, splitXFp32Bf16, RoundMode::CAST_RINT, num);
    PipeBarrier<PIPE_V>();
    Cast(splitXFp32Bf16, splitYLocalBf16, RoundMode::CAST_NONE, num);
    PipeBarrier<PIPE_V>();
    Cast(splitSqxBf16, gammaLocal, RoundMode::CAST_NONE, num);
    PipeBarrier<PIPE_V>();
    Mul(splitXFp32Bf16, splitXFp32Bf16, splitSqxBf16, num);
    PipeBarrier<PIPE_V>();
    Cast(splitYLocalBf16, splitXFp32Bf16, RoundMode::CAST_RINT, num);
    PipeBarrier<PIPE_V>();
    outQueueY.EnQue<bfloat16_t>(splitYLocalBf16);
  }

  __aicore__ inline void CopyOutY(uint32_t i_idx,
                                  uint32_t j_idx,
                                  uint32_t num) {
    LocalTensor<T> splitYLocalOut = outQueueY.DeQue<T>();
    DataCopyCustom<T>(yGm[i_idx * this->numCol + j_idx * this->ubFactor],
                      splitYLocalOut,
                      num);
    outQueueY.FreeTensor(splitYLocalOut);
  }

  __aicore__ inline void CopyOutRstd(uint32_t i_o_idx, uint32_t num) {
    LocalTensor<float> splitRstdLocal = outQueueRstd.DeQue<float>();
#if __CCE_AICORE__ == 220 || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    DataCopyCustom<float>(
        rstdGm[i_o_idx * this->rowFactor], splitRstdLocal, num);
#endif
    outQueueRstd.FreeTensor(splitRstdLocal);
  }

 private:
  TPipe* Ppipe = nullptr;
  // create input queues for split_d
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGamma;
  // create output queues for split_d
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueRstd;
  TBuf<TPosition::VECCALC> xFp32Buf;
  TBuf<TPosition::VECCALC> sqxBuf;
  TBuf<TPosition::VECCALC> sumBuf;
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

  int tempbufNum;
};
#endif  // _GAMMA_ADD_RMS_NORM_SPLIT_D_H_
