/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
the terms and conditions of  * CANN Open Software License Agreement Version 2.0
(the "License").  * Please refer to the License for details. You may not use
this file except in compliance with the License.  * THIS SOFTWARE IS PROVIDED ON
AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,  *
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE.  * See LICENSE in the root of the software repository for
the full text of the License.  */

/*!
 * \file mla_preprocess_bf16.h
 * \brief mla_preprocess_bf16.h
 */

#ifndef MLA_PREPROCESS_BF16_H
#define MLA_PREPROCESS_BF16_H

#include "lib/matmul_intf.h"
#include "mla_common.h"
#include "mla_iterator.h"
#include "mla_kernel_utils.h"
#include "mla_mem.h"
#include "mla_mma.h"
#include "mla_simd.h"
#include "mla_utils.h"
namespace MlaPreprocess {

// sync
constexpr int32_t RMSNORMQUANT1 = 1;
constexpr int32_t AIC_MM1_START = 2;
constexpr int32_t AIC_MM3_START = 3;
constexpr int32_t RMSNORMQUANT2 = 4;
constexpr int32_t MM2 = 5;
constexpr int32_t AIC_MM2_START = 6;
constexpr int32_t MMAIC = 7;
constexpr int32_t MMAIV = 8;
constexpr int32_t MM2OUT = 9;
constexpr int32_t EINSUMOUT = 11;
constexpr int32_t EINSUMQUANT = 12;

// ropeConcat
constexpr uint32_t ELE_NUM_FP16 = 16;         // 一个block fp16元素个数
constexpr uint32_t ELE_NUM_FP32 = 8;          // 一个block字节数 fp32元素个数
constexpr uint8_t DEFAULT_REPEAT_STRIDE = 8;  // 默认stride, 8 * 32 = 256
constexpr uint32_t NUM_BUFFER = 2;

// rmsNormQuant
constexpr int32_t NUM_PER_REP_FP32 =
    64;  // ONE_REPEAT_BYTE_SIZE / sizeof(float);
constexpr float ZERO = 0;
constexpr uint32_t BUF_FACTOR = 3;             // 1(g) + 1(sqx) + 1(sum) = 3
constexpr uint32_t OFFSET_GAMMA = 0;           // the offset of gamma is 0
constexpr uint32_t OFFSET_SQX = 1;             // the offset of sqx is 1
constexpr uint32_t OFFSET_SUM = 2;             // the offset of sum is 2
constexpr uint32_t OFFSET_ABS = 3;             // the offset of abs is 3
constexpr uint32_t OFFSET_WORKSPACE_BF16 = 4;  // the offset of workspace is 4
constexpr uint32_t OFFSET_Q_DOWN = 5;          // the offset of q_down is 5
constexpr uint32_t REPEAT_TIME_256 = 256;      // 128 default stride
constexpr uint32_t REPEAT_TIME_128 = 128;      // 128 default stride
constexpr uint32_t REPEAT_TIME_64 = 64;        // 64 default stride

constexpr uint8_t CACHE_MODE_KVCACHE = 0;     // 单入单出
constexpr uint8_t CACHE_MODE_KROPE_CTKV = 1;  // 双入双出
constexpr uint8_t CACHE_MODE_INT8_NZCACHE =
    2;  // 高性能cache（双入双出、NZ存储、CTKV使用I8）
constexpr uint8_t CACHE_MODE_NZCACHE = 3;  // 高性能cache（双入双出、NZ存储）

constexpr int32_t WORK_SIZE = 256;
constexpr float SCALE_FACTOR = 1536.0f;

// pp matmul
namespace {
constexpr uint32_t FLOAT_BLOCK_SIZE = 64;
constexpr uint32_t HALF_BLOCK_SIZE = 64;
constexpr uint32_t HALF_VECTOR_SIZE = 64;
constexpr uint32_t MM1_OUT_SIZE = 2112;
constexpr uint32_t SPLIT_SIZE_ONE = 576;
constexpr uint32_t SPLIT_SIZE_TWO = 1536;
constexpr uint32_t SPLIT_RMSNRORM_SIZE_ONE = 512;
constexpr uint32_t SPLIT_RMSNRORM_SIZE_TWO = 64;
constexpr uint32_t ROPE_SPLIT_SIZE_ONE = 64;
constexpr uint32_t ROPE_SPLIT_SIZE_TWO = 128;

constexpr uint32_t MMSIZE1 = 128 * 192;  // 24576
constexpr uint32_t MMSIZE2 = 64 * 128;   // 8192

constexpr uint64_t L0_PINGPONG_BUFFER_LEN = 32768;   // 32 KB
constexpr uint64_t L1_PINGPONG_BUFFER_LEN = 262144;  // 256 KB
constexpr uint64_t BLOCK_SIZE_16 = 16;
constexpr uint64_t BLOCK_SIZE_32 = 32;
constexpr uint64_t CUBE_MATRIX_SIZE_512 = 16 * 32;  // 16 * 23
constexpr uint64_t FB_BUFF_SIZE = 1024 * 7;
constexpr uint64_t SCALE_L1_LEN = 4096;
constexpr uint64_t BIAS_L1_LEN = 2048;
constexpr uint64_t CONST_0 = 0;
constexpr uint64_t CONST_4 = 4;
constexpr uint64_t CONST_64 = 64;
constexpr uint64_t CONST_128 = 128;

constexpr uint64_t BLOCK_SIZE_INT8 = 32;

constexpr uint32_t L0AB_PINGPONG_BUFFER_LEN_INT8 = 32768;  // 32 KB
constexpr uint64_t L1_PINGPONG_BUFFER_LEN_FP16 = 131072;   // 256 KB
constexpr uint64_t L1_PINGPONG_BUFFER_LEN_INT8 = 262144;   // 256 KB
constexpr uint64_t CONST_8 = 8;
constexpr uint64_t CONST_32 = 32;
constexpr uint64_t ND2NZ_STRIDE_LIMIT = 65536;

constexpr uint32_t AIC_FINISH_FLAG_ID = 1;
constexpr uint32_t AIV_FINISH_FLAG_ID = 2;
constexpr uint32_t MAX_HW_SYNC_COUNTER = 5;
constexpr uint32_t SYNC_MODE = 2;
}  // namespace
enum class QuantMode : int32_t {
  PER_TENSOR_ASYMM_QUANT = 0,
  PER_TOKEN_SYMM_QUANT,
  PER_TOKEN_ASYMM_QUANT,
  NO_QUANT,
};

template <typename T,
          bool WITH_BETA,
          bool FastComputeMode = false,
          QuantMode quantMode = QuantMode::PER_TENSOR_ASYMM_QUANT,
          bool NEED_DEQUANT = false>
class Quant {
 public:
  __aicore__ inline Quant() {}

  __aicore__ inline void Init(
      AscendC::GlobalTensor<T>& quantScaleGmTensor,
      AscendC::GlobalTensor<int8_t>& quantOffsetGmTensor,
      GM_ADDR perTokenDescaleGm,
      GM_ADDR perChannelDescaleGm,
      GM_ADDR gmInput,
      GM_ADDR gmOutput,
      uint32_t stride,
      uint32_t num_col,
      uint64_t gm_offset,
      uint64_t gm_out_offset,
      uint32_t row_work_,
      const MlaTilingData& mlaParams_) {
    this->quantScaleGmTensor = quantScaleGmTensor;
    this->quantOffsetGmTensor = quantOffsetGmTensor;
    this->perTokenDescaleGmTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(perTokenDescaleGm));
    this->perChannelDescaleGmTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(perChannelDescaleGm));
    if constexpr (!NEED_DEQUANT) {
      inputGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gmInput));
    } else {
      mmGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(gmInput));
    }
    outputGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(gmOutput));

    num_col_ = num_col;
    quantMin_ = INT8_MIN;
    this->num_row_ = mlaParams_.n;
    this->row_work = row_work;
    this->row_work_ = row_work_;
    gm_offset_ = gm_offset;
    gm_out_offset_ = gm_out_offset;
    num_col_align_int8 =
        (num_col_ + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
    num_col_align_f16 =
        (num_col_ + REPEAT_TIME_128 - 1) / REPEAT_TIME_128 * REPEAT_TIME_128;
    num_col_align_f32 =
        (num_col_ + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
    input_stride_ = stride;

    num_col_align_withStride_int8 =
        (num_col_ - input_stride_ + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 *
        REPEAT_TIME_256;
    num_col_align_withStride_fp16 =
        (num_col_ - input_stride_ + REPEAT_TIME_128 - 1) / REPEAT_TIME_128 *
        REPEAT_TIME_128;
    num_col_align_withStride_fp32 =
        (num_col_ - input_stride_ + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 *
        REPEAT_TIME_64;
  }

  __aicore__ inline void Launch(
      const AscendC::LocalTensor<int8_t>& dstTensor,
      const AscendC::LocalTensor<T>& srcTensor,
      const AscendC::LocalTensor<T>& quantScaleTensor,
      const AscendC::LocalTensor<int8_t>& quantOffsetTensor,
      const AscendC::LocalTensor<float>& res1Tensor,
      const AscendC::LocalTensor<float>& res3Tensor) {
    this->dstTensor = dstTensor;
    this->srcTensor = srcTensor;
    this->fp32_xy = res1Tensor;
    this->buf = res3Tensor;

    AscendC::LocalTensor<float> g =
        buf[OFFSET_GAMMA * num_col_align_withStride_fp32];  // 0
    AscendC::LocalTensor<float> sqx =
        buf[OFFSET_SQX * num_col_align_withStride_fp32];  // 1
    AscendC::LocalTensor<float> work =
        buf[OFFSET_SUM * num_col_align_withStride_fp32];  // 2
    AscendC::LocalTensor<float> abs =
        buf[OFFSET_ABS * num_col_align_withStride_fp32];  // 3
    AscendC::LocalTensor<float> sum =
        buf[OFFSET_WORKSPACE_BF16 * num_col_align_withStride_fp32];  // 4
    AscendC::LocalTensor<float> max =
        buf[OFFSET_WORKSPACE_BF16 * num_col_align_withStride_fp32 + 8];  // 5
    AscendC::LocalTensor<float> perTokenDescaleTensor =
        buf[OFFSET_WORKSPACE_BF16 * num_col_align_withStride_fp32 +
            ELE_NUM_FP16];  // 6

    SET_FLAG(MTE2, V, EVENT_ID1);
    if constexpr (quantMode == QuantMode::PER_TENSOR_ASYMM_QUANT) {
      AscendC::DataCopy(quantScaleTensor,
                        quantScaleGmTensor,
                        AscendC::DataCopyParams(1, 1, 0, 0));
      AscendC::DataCopy(quantOffsetTensor,
                        quantOffsetGmTensor,
                        AscendC::DataCopyParams(1, 1, 0, 0));
    }

    if constexpr (NEED_DEQUANT) {
      mmTensor =
          buf.ReinterpretCast<int32_t>()[OFFSET_WORKSPACE_BF16 *
                                             num_col_align_withStride_fp32 +
                                         ELE_NUM_FP16];
      deScaleTensor =
          buf[OFFSET_WORKSPACE_BF16 * num_col_align_withStride_fp32 +
              ELE_NUM_FP16 + MM1_OUT_SIZE];
      perTokenDescaleTensor =
          buf[OFFSET_WORKSPACE_BF16 * num_col_align_withStride_fp32 +
              ELE_NUM_FP16 + MM1_OUT_SIZE * CONST_2];
      AscendC::DataCopy(deScaleTensor,
                        perChannelDescaleGmTensor,
                        AscendC::DataCopyParams(
                            1, num_col_ * sizeof(float) / BLOCK_SIZE_32, 0, 0));
    }

    if constexpr (quantMode == QuantMode::PER_TENSOR_ASYMM_QUANT) {
      if (std::is_same<T, __bf16>::value) {
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        Cast(g, quantScaleTensor, AscendC::RoundMode::CAST_NONE, 1);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        input_scale_ = 1 / (float)(g.GetValue(0));
        input_offset_ = (float)(quantOffsetTensor.GetValue(0));
      } else {
        SET_FLAG(MTE2, S, EVENT_ID0);
        WAIT_FLAG(MTE2, S, EVENT_ID0);
        input_scale_ = 1 / (float)(quantScaleTensor.GetValue(0));
        input_offset_ = (float)(quantOffsetTensor.GetValue(0));
      }
      AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
      AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }
    WAIT_FLAG(MTE2, V, EVENT_ID1);
    uint64_t pid = 0;
    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    while (pid < row_work_) {
      uint64_t offset = pid * num_col_;
      uint64_t outOffset = pid * (num_col_ - input_stride_);
      WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
      if constexpr (!NEED_DEQUANT) {
        AscendC::DataCopy(
            srcTensor,
            inputGmTensor[gm_offset_ + offset],
            AscendC::DataCopyParams(1, num_col_ / BLOCK_SIZE_16, 0, 0));
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
      } else {
        /* Dequant start */
        AscendC::DataCopy(
            mmTensor,
            mmGmTensor[gm_offset_ + offset],
            AscendC::DataCopyParams(
                1, num_col_ * sizeof(int32_t) / BLOCK_SIZE_32, 0, 0));  // 2112
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        AscendC::Cast(mmTensor.ReinterpretCast<float>(),
                      mmTensor,
                      AscendC::RoundMode::CAST_NONE,
                      num_col_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(mmTensor.ReinterpretCast<float>(),
                     mmTensor.ReinterpretCast<float>(),
                     deScaleTensor,
                     num_col_);
        SET_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        gm_to_ub_align<Architecture::ASCEND_V220, float>(
            perTokenDescaleTensor,
            perTokenDescaleGmTensor[pid],
            0,              // sid
            1,              // nBurst
            sizeof(float),  // lenBurst
            0,              // leftPaddingNum
            0,              // rightPaddingNum
            0,              // srcGap
            0               // dstGap
        );
        SET_FLAG(MTE2, S, EVENT_ID0);
        WAIT_FLAG(MTE2, S, EVENT_ID0);
        float perTokenDescale = perTokenDescaleTensor.GetValue(0);
        SET_FLAG(S, V, EVENT_ID0);
        WAIT_FLAG(S, V, EVENT_ID0);
        AscendC::Muls(mmTensor.ReinterpretCast<float>(),
                      mmTensor.ReinterpretCast<float>(),
                      perTokenDescale,
                      num_col_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(srcTensor,
                      mmTensor.ReinterpretCast<float>(),
                      AscendC::RoundMode::CAST_RINT,
                      num_col_);
        AscendC::PipeBarrier<PIPE_V>();
      }

      Cast(fp32_xy,
           srcTensor[input_stride_],
           AscendC::RoundMode::CAST_NONE,
           REPEAT_TIME_64,
           num_col_align_withStride_fp32 / REPEAT_TIME_64,
           {1,
            1,
            AscendC::DEFAULT_REPEAT_STRIDE,
            AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
      AscendC::PipeBarrier<PIPE_V>();

      /* Quant start */
      if constexpr (quantMode == QuantMode::PER_TENSOR_ASYMM_QUANT) {
        Muls(fp32_xy,
             fp32_xy,
             input_scale_,
             REPEAT_TIME_64,
             num_col_align_withStride_fp32 / REPEAT_TIME_64,
             {1,
              1,
              AscendC::DEFAULT_REPEAT_STRIDE,
              AscendC::DEFAULT_REPEAT_STRIDE});
        AscendC::PipeBarrier<PIPE_V>();
        Adds(fp32_xy,
             fp32_xy,
             input_offset_,
             REPEAT_TIME_64,
             num_col_align_withStride_fp32 / REPEAT_TIME_64,
             {1,
              1,
              AscendC::DEFAULT_REPEAT_STRIDE,
              AscendC::DEFAULT_REPEAT_STRIDE});
        AscendC::PipeBarrier<PIPE_V>();
      } else if constexpr (quantMode == QuantMode::PER_TOKEN_SYMM_QUANT) {
        Abs(abs,
            fp32_xy,
            REPEAT_TIME_64,
            num_col_align_withStride_fp32 / REPEAT_TIME_64,
            {1,
             1,
             AscendC::DEFAULT_REPEAT_STRIDE,
             AscendC::DEFAULT_REPEAT_STRIDE});
        AscendC::PipeBarrier<PIPE_V>();
        ReduceMax(max, abs, work, num_col_ - input_stride_);
        AscendC::PipeBarrier<PIPE_V>();
        float scaleOut = max.GetValue(0) / 127;
        SET_FLAG(S, V, EVENT_ID0);
        WAIT_FLAG(S, V, EVENT_ID0);
        Muls(fp32_xy,
             fp32_xy,
             (float)(1 / scaleOut),
             REPEAT_TIME_64,
             num_col_align_withStride_fp32 / REPEAT_TIME_64,
             {1,
              1,
              AscendC::DEFAULT_REPEAT_STRIDE,
              AscendC::DEFAULT_REPEAT_STRIDE});
        AscendC::PipeBarrier<PIPE_V>();
        perTokenDescaleTensor.SetValue(0, scaleOut);
        SET_FLAG(S, MTE3, EVENT_ID0);
        WAIT_FLAG(S, MTE3, EVENT_ID0);
        if constexpr (!NEED_DEQUANT) {
          ub_to_gm_align<Architecture::ASCEND_V220, float>(
              perTokenDescaleGmTensor[pid],
              perTokenDescaleTensor,
              0,
              1,                  // nBurst
              1 * sizeof(float),  // lenBurst
              0,                  // leftPaddingNum
              0,                  // rightPaddingNum
              0,                  // srcGap
              0                   // dstGap
          );
        } else {
          ub_to_gm_align<Architecture::ASCEND_V220, float>(
              perTokenDescaleGmTensor[num_row_ + pid],
              perTokenDescaleTensor,
              0,
              1,                  // nBurst
              1 * sizeof(float),  // lenBurst
              0,                  // leftPaddingNum
              0,                  // rightPaddingNum
              0,                  // srcGap
              0                   // dstGap
          );
        }
        SET_FLAG(MTE3, V, EVENT_ID0);
        WAIT_FLAG(MTE3, V, EVENT_ID0);
      }

      AscendC::LocalTensor<half> tmpfp16 =
          buf.ReinterpretCast<half>()[OFFSET_GAMMA *
                                      num_col_align_withStride_fp32];
      CastFrom32To16(tmpfp16, fp32_xy, num_col_align_withStride_fp32);
      AscendC::PipeBarrier<PIPE_V>();
      CastFromF16ToI8(
          dstTensor, tmpfp16, quantMin_, num_col_align_withStride_fp16);
      AscendC::PipeBarrier<PIPE_V>();
      SET_FLAG(V, MTE3, EVENT_ID0);
      WAIT_FLAG(V, MTE3, EVENT_ID0);
      AscendC::DataCopy(
          outputGmTensor[gm_out_offset_ + outOffset],
          dstTensor,
          AscendC::DataCopyParams(
              1, (num_col_ - input_stride_) / BLOCK_SIZE_32, 0, 0));
      SET_FLAG(MTE3, MTE2, EVENT_ID0);
      ++pid;
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
  }

 private:
  AscendC::LocalTensor<int8_t> dstTensor;
  AscendC::LocalTensor<T> srcTensor;
  AscendC::LocalTensor<float> fp32_xy;
  AscendC::LocalTensor<float> buf;
  AscendC::LocalTensor<int32_t> mmTensor;
  AscendC::LocalTensor<float> deScaleTensor;

  AscendC::GlobalTensor<T> quantScaleGmTensor;
  AscendC::GlobalTensor<int8_t> quantOffsetGmTensor;
  AscendC::GlobalTensor<T> inputGmTensor;
  AscendC::GlobalTensor<int8_t> outputGmTensor;
  AscendC::GlobalTensor<float> perTokenDescaleGmTensor;
  AscendC::GlobalTensor<float> perChannelDescaleGmTensor;
  AscendC::GlobalTensor<int32_t> mmGmTensor;

  uint32_t num_col_{0};        // input columns
  uint32_t num_row_{0};        // input rows
  uint32_t row_work_{0};       // rows need process
  uint32_t row_work{0};        // rows need process
  uint32_t row_step_{0};       // rows move in once
  uint32_t row_tail_{0};       // rows move in last time
  uint64_t gm_offset_{0};      // GM data offset
  uint64_t gm_out_offset_{0};  // GM data offset
  float avg_factor_{1.0};      // 1/num_col_
  float input_scale_{1.0};
  float input_offset_{0};
  int32_t input_stride_{0};
  float epsilon_{1e-12f};
  uint32_t num_col_align_int8{0};
  uint32_t num_col_align_f16{0};
  uint32_t num_col_align_f32{0};
  uint32_t num_col_align_f32_long{0};
  uint32_t num_col_align_withStride_int8{0};
  uint32_t num_col_align_withStride_fp16{0};
  uint32_t num_col_align_withStride_fp32{0};
  uint32_t num_col_temp;
  half quantMin_{-128};
  uint32_t num_slice_{0};
  uint32_t tail_size_{0};
  uint32_t tail_copy_{0};
};

template <typename QkDtype,
          typename CosDtype,
          typename QOutDtype,
          int8_t CacheMode>
class RopeFp16 {
 public:
  __aicore__ inline RopeFp16() : blockIdx_(AscendC::GetBlockIdx()) {}

  __aicore__ inline void RopeInit(
      GM_ADDR qGm,
      AscendC::GlobalTensor<CosDtype>& cosGm,
      AscendC::GlobalTensor<CosDtype>& sinGm,
      AscendC::GlobalTensor<QOutDtype>& outRopeConcatGm,
      AscendC::GlobalTensor<QkDtype>& outRopeConcatGm2,
      MlaTilingData& ropeConcatParams) {
    qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ QkDtype*>(qGm));
    this->cosGm_ = cosGm;
    this->sinGm_ = sinGm;
    this->outRopeConcatGm_ = outRopeConcatGm;
    this->outRopeConcatGm2_ = outRopeConcatGm2;

    headDim = ropeConcatParams.headDim;
    headNumQ = ropeConcatParams.headNumQ;
    rotaryCoeff = ropeConcatParams.rotaryCoeff;
    ntokens = ropeConcatParams.ntokens;
    realCore = ropeConcatParams.realCore;
    nlCoreRun = ropeConcatParams.nlCoreRun;
    lCoreRun = ropeConcatParams.lCoreRun;
    maxNPerLoopForUb = ropeConcatParams.maxNPerLoopForUb;
    preCoreLoopTime = ropeConcatParams.preCoreLoopTime;
    preCoreLoopNLast = ropeConcatParams.preCoreLoopNLast;
    lastCoreLoopTime = ropeConcatParams.lastCoreLoopTime;
    lastCoreLoopNLast = ropeConcatParams.lastCoreLoopNLast;
    concatSize = ropeConcatParams.concatSize;
    loopTime = (blockIdx_ == realCore - 1) ? lastCoreLoopTime : preCoreLoopTime;
    lastLoopN =
        (blockIdx_ == realCore - 1) ? lastCoreLoopNLast : preCoreLoopNLast;
    this->repeatSize_ = 64;  // 64 = 256B / sizeof(fp32)
    this->rotateStride_ = this->headDim / this->rotaryCoeff;
    headBlockLen = static_cast<uint16_t>(this->headDim / ELE_NUM_FP16);
    headBlockLenFP32 = static_cast<uint16_t>(this->headDim / ELE_NUM_FP32);
    rotaryLen = static_cast<uint16_t>(this->rotateStride_ / ELE_NUM_FP32);
    concatBlockLen = static_cast<uint16_t>(this->concatSize / ELE_NUM_FP16);
    outLineOffset = this->headDim + this->concatSize;
    uint32_t dataNum = this->headDim * this->maxNPerLoopForUb;
    dataSizeFp16 = dataNum * sizeof(QkDtype);
    dataSizeFp32 = dataNum * sizeof(float);
    uint32_t concatDataSize =
        this->concatSize * sizeof(QkDtype) * this->maxNPerLoopForUb;
  }

  __aicore__ inline void Process() {
    if (blockIdx_ >= realCore) {
      return;
    }
    uint64_t startCoreLineIndex =
        this->blockIdx_ * this->nlCoreRun;  // 当前核处理head起始位置
    // 生成 [maxNPerLoopForUb,head_dim] 的 neg
    AscendC::LocalTensor<float> negLocal =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(dataSizeFp32 * 4 +
                                                    dataSizeFp16 * 3);
    ExpandNeg(negLocal, this->maxNPerLoopForUb);
    // 遍历处理每轮数据
    SET_FLAG(MTE3, MTE2, EVENT_ID1);
    for (uint32_t zz = 0; zz < this->loopTime; ++zz) {
      uint16_t loopN =
          (zz == this->loopTime - 1) ? this->lastLoopN : this->maxNPerLoopForUb;
      uint64_t startHead = startCoreLineIndex + zz * this->maxNPerLoopForUb;
      uint64_t endHead = startHead + loopN;

      // 搬入数据Q
      AscendC::LocalTensor<QkDtype> inputQ =
          buf.GetBuffer<BufferType::ASCEND_UB, QkDtype>(0);
      AscendC::LocalTensor<float> inputQCastFP32 =
          buf.GetBuffer<BufferType::ASCEND_UB, float>(dataSizeFp16);
      AscendC::LocalTensor<float> reverseQ =
          buf.GetBuffer<BufferType::ASCEND_UB, float>(dataSizeFp32 +
                                                      dataSizeFp16);
      uint64_t qOffset = startHead * 192 + 128;
      CopyQGenReverseQ(inputQ, inputQCastFP32, reverseQ, qOffset, loopN);

      // 搬入数据cos/sin
      AscendC::LocalTensor<QkDtype> inputCos =
          buf.GetBuffer<BufferType::ASCEND_UB, QkDtype>(dataSizeFp32 * 2 +
                                                        dataSizeFp16);
      AscendC::LocalTensor<QkDtype> inputSin =
          buf.GetBuffer<BufferType::ASCEND_UB, QkDtype>(dataSizeFp32 * 2 +
                                                        dataSizeFp16 * 2);
      uint64_t startSinCosHeadIndex = startHead;
      uint64_t headRemain = startHead % this->headNumQ;
      uint64_t localStartAddr = 0;
      if (headRemain != 0) {  // 需要前处理
        uint64_t preProcessHeadNum = this->headNumQ - headRemain;
        uint64_t needToProcesHead =
            preProcessHeadNum > loopN ? loopN : preProcessHeadNum;
        CopyCosSin(inputCos,
                   inputSin,
                   localStartAddr,
                   (startSinCosHeadIndex / this->headNumQ) * this->headDim,
                   needToProcesHead);
        startSinCosHeadIndex += needToProcesHead;
        localStartAddr += needToProcesHead * this->headDim;
      }
      // 循环迭代处理剩余数据
      if (startSinCosHeadIndex < endHead) {
        uint64_t startSinCosIndex = startSinCosHeadIndex / this->headNumQ;
        uint64_t endSinCosIndex =
            (endHead + this->headNumQ - 1) / this->headNumQ;
        for (uint32_t index = startSinCosIndex; index < endSinCosIndex;
             ++index) {
          // 尾数处理
          uint32_t repeatNum = index == endSinCosIndex - 1
                                   ? endHead - index * this->headNumQ
                                   : this->headNumQ;
          CopyCosSin(inputCos,
                     inputSin,
                     localStartAddr,
                     index * this->headDim,
                     repeatNum);
          localStartAddr += this->headDim * this->headNumQ;
        }
      }
      AscendC::LocalTensor<float> inputCosCastFP32 =
          buf.GetBuffer<BufferType::ASCEND_UB, float>(dataSizeFp32 * 2 +
                                                      dataSizeFp16 * 3);
      AscendC::LocalTensor<float> inputSinCastFP32 =
          buf.GetBuffer<BufferType::ASCEND_UB, float>(dataSizeFp32 * 3 +
                                                      dataSizeFp16 * 3);
      AscendC::Cast(inputCosCastFP32,
                    inputCos,
                    AscendC::RoundMode::CAST_NONE,
                    loopN * this->headDim);
      AscendC::Cast(inputSinCastFP32,
                    inputSin,
                    AscendC::RoundMode::CAST_NONE,
                    loopN * this->headDim);
      AscendC::PipeBarrier<PIPE_V>();

      // 计算rope结果
      uint32_t repeatTime = this->headDim * loopN;
      AscendC::Mul(
          inputQCastFP32, inputCosCastFP32, inputQCastFP32, repeatTime);
      AscendC::Mul(reverseQ, negLocal, reverseQ, repeatTime);
      AscendC::PipeBarrier<PIPE_V>();

      AscendC::Mul(reverseQ, inputSinCastFP32, reverseQ, repeatTime);
      AscendC::PipeBarrier<PIPE_V>();

      AscendC::Add(inputQCastFP32, reverseQ, inputQCastFP32, repeatTime);
      AscendC::PipeBarrier<PIPE_V>();

      // // 搬出rope结果
      // // cast fp16/bf16
      AscendC::Cast(inputQ,
                    inputQCastFP32,
                    AscendC::RoundMode::CAST_RINT,
                    loopN * this->headDim);
      AscendC::PipeBarrier<PIPE_V>();
      uint64_t outQOffset = startHead * outLineOffset + this->concatSize;
      uint64_t outQOffset2 = startHead * this->headDim;
      SET_FLAG(V, MTE3, EVENT_ID1);
      WAIT_FLAG(V, MTE3, EVENT_ID1);
      if constexpr (CacheMode == CACHE_MODE_KVCACHE) {
        AscendC::DataCopy(this->outRopeConcatGm_[outQOffset],
                          inputQ,
                          {loopN, headBlockLen, 0, concatBlockLen});
      } else {
        AscendC::DataCopy(this->outRopeConcatGm2_[outQOffset2],
                          inputQ,
                          loopN * this->headDim);
      }
      SET_FLAG(MTE3, MTE2, EVENT_ID1);
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
  }
  // 构建tensor -1 -1 -1 1 1 1
  template <typename BUF_TYPE>
  __aicore__ inline void ExpandNeg(
      const AscendC::LocalTensor<BUF_TYPE>& tempBuf,
      uint32_t headNumTemp) {
    for (uint32_t i = 0; i < this->rotateStride_; ++i) {
      tempBuf.SetValue(i, (BUF_TYPE)-1);
      tempBuf.SetValue(i + this->rotateStride_, (BUF_TYPE)1);
    }
    AscendC::SetFlag<HardEvent::S_V>(EVENT_ID1);
    AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID1);
    AscendC::Copy(tempBuf[this->headDim],
                  tempBuf,
                  this->headDim,
                  headNumTemp - 1,
                  {1, 1, headBlockLenFP32, 0});
    AscendC::PipeBarrier<PIPE_V>();
  }

  template <typename BUF_TYPE>
  __aicore__ inline void CopyQGenReverseQ(
      const AscendC::LocalTensor<BUF_TYPE>& tempBufQ,
      const AscendC::LocalTensor<float>& tempBufQCast,
      const AscendC::LocalTensor<float>& tempBufRverseQ,
      uint64_t qOffset,
      uint16_t loopN) {
    // 搬入数据Q
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
    AscendC::DataCopy(
        tempBufQ, this->qGm_[qOffset], {loopN, headBlockLen, 128 / 16, 0});
    SET_FLAG(MTE2, V, EVENT_ID1);
    WAIT_FLAG(MTE2, V, EVENT_ID1);
    // cast fp32
    AscendC::Cast(tempBufQCast,
                  tempBufQ,
                  AscendC::RoundMode::CAST_NONE,
                  loopN * this->headDim);
    AscendC::PipeBarrier<PIPE_V>();
    // 搬入数据reverseQ
    AscendC::DataCopy(tempBufRverseQ,
                      tempBufQCast[this->rotateStride_],
                      {loopN, rotaryLen, rotaryLen, rotaryLen});
    AscendC::DataCopy(tempBufRverseQ[this->rotateStride_],
                      tempBufQCast,
                      {loopN, rotaryLen, rotaryLen, rotaryLen});
    AscendC::PipeBarrier<PIPE_V>();
  }

  template <typename BUF_TYPE>
  __aicore__ inline void CopyCosSin(
      const AscendC::LocalTensor<BUF_TYPE>& tempBufCos,
      const AscendC::LocalTensor<BUF_TYPE>& tempBufSin,
      uint64_t localStartAddr,
      uint64_t gmStartAddr,
      uint64_t repeatNum) {
    AscendC::DataCopy(tempBufCos[localStartAddr],
                      this->cosGm_[gmStartAddr],
                      {1, headBlockLen, 0, 0});
    AscendC::DataCopy(tempBufSin[localStartAddr],
                      this->sinGm_[gmStartAddr],
                      {1, headBlockLen, 0, 0});
    SET_FLAG(MTE2, V, EVENT_ID1);
    WAIT_FLAG(MTE2, V, EVENT_ID1);
    AscendC::Copy(tempBufCos[localStartAddr + this->headDim],
                  tempBufCos[localStartAddr],
                  this->headDim,
                  repeatNum - 1,
                  {1, 1, headBlockLen, 0});
    AscendC::Copy(tempBufSin[localStartAddr + this->headDim],
                  tempBufSin[localStartAddr],
                  this->headDim,
                  repeatNum - 1,
                  {1, 1, headBlockLen, 0});
    AscendC::PipeBarrier<PIPE_V>();
  }

 private:
  AsdopsBuffer<Architecture::ASCEND_V220> buf;

  AscendC::GlobalTensor<QkDtype> qGm_;
  AscendC::GlobalTensor<CosDtype> cosGm_;
  AscendC::GlobalTensor<CosDtype> sinGm_;
  AscendC::GlobalTensor<QOutDtype> outRopeConcatGm_;
  AscendC::GlobalTensor<QkDtype> outRopeConcatGm2_;

  uint32_t repeatSize_{0};    // 一拍做几个元素
  uint32_t rotateStride_{0};  // this->headDim / 旋转系数
  uint32_t headDim;
  uint32_t headNumQ;
  uint32_t rotaryCoeff;
  uint32_t ntokens;
  uint32_t realCore;
  uint32_t nlCoreRun;
  uint32_t lCoreRun;
  uint32_t maxNPerLoopForUb;
  uint32_t preCoreLoopTime;
  uint32_t preCoreLoopNLast;
  uint32_t lastCoreLoopTime;
  uint32_t lastCoreLoopNLast;
  uint32_t concatSize;
  uint32_t blockIdx_;
  uint32_t loopTime{0};   // 当前核批处理数据轮数
  uint32_t lastLoopN{0};  // 当前核尾处理行数

  uint32_t dataSizeFp32;
  uint32_t dataSizeFp16;
  uint16_t headBlockLen{0};
  uint16_t headBlockLenFP32{0};
  uint16_t rotaryLen{0};
  uint16_t concatBlockLen{0};
  uint64_t outLineOffset{0};
};

__aicore__ inline void ReduceSumCustom(
    const AscendC::LocalTensor<float>& dst_local,
    const AscendC::LocalTensor<float>& src_local,
    const AscendC::LocalTensor<float>& work_local,
    int32_t count) {
#ifdef __DAV_C220_VEC__
  uint64_t mask = NUM_PER_REP_FP32;
  int32_t repeatTimes = count / NUM_PER_REP_FP32;
  int32_t tailCount = count % NUM_PER_REP_FP32;
  int32_t bodyCount = repeatTimes * NUM_PER_REP_FP32;
  AscendC::BinaryRepeatParams repeatParams;
  repeatParams.src0RepStride =
      AscendC::ONE_REPEAT_BYTE_SIZE / AscendC::ONE_BLK_SIZE;
  repeatParams.src0BlkStride = 1;
  repeatParams.src1RepStride = 0;
  repeatParams.src1BlkStride = 1;
  repeatParams.dstRepStride = 0;
  repeatParams.dstBlkStride = 1;
  Duplicate(work_local, ZERO, NUM_PER_REP_FP32);
  AscendC::PipeBarrier<PIPE_V>();
  if (likely(repeatTimes > 0)) {
    Add(work_local, src_local, work_local, mask, repeatTimes, repeatParams);
    AscendC::PipeBarrier<PIPE_V>();
  }
  if (unlikely(tailCount != 0)) {
    Add(work_local,
        src_local[bodyCount],
        work_local,
        tailCount,
        1,
        repeatParams);
    AscendC::PipeBarrier<PIPE_V>();
  }
  AscendC::AscendCUtils::SetMask<float>(NUM_PER_REP_FP32);
  cadd_v<Architecture::ASCEND_V220, float>(dst_local,   // dst
                                           work_local,  // src
                                           1,           // repeat
                                           0,           // dstRepeatStride
                                           1,           // srcBlockStride
                                           0);          // srcRepeatStride
  AscendC::PipeBarrier<PIPE_V>();
#endif
}

template <typename T,
          bool WITH_BETA,
          bool FastComputeMode = false,
          QuantMode quantMode = QuantMode::PER_TENSOR_ASYMM_QUANT,
          bool NEED_DEQUANT = false,
          bool NEED_Q_DOWN = false>
class RmsNormQuant {
 public:
  __aicore__ inline RmsNormQuant() {}

  __aicore__ inline void Init(
      AscendC::GlobalTensor<T>& gammaGmTensor,
      AscendC::GlobalTensor<T>& betaGmTensor,
      AscendC::GlobalTensor<T>& quantScaleGmTensor,
      AscendC::GlobalTensor<int8_t>& quantOffsetGmTensor,
      GM_ADDR perTokenDescaleGm,
      GM_ADDR perChannelDescaleGm,
      GM_ADDR gmInput,
      GM_ADDR gmOutput,
      uint32_t stride,
      uint32_t num_col,
      float avg_factor,
      uint64_t gm_offset,
      uint64_t gm_out_offset,
      uint32_t row_work_,
      const MlaTilingData& mlaParams_,
      AscendC::GlobalTensor<T>& qDownGmTensor) {
    this->gammaGmTensor = gammaGmTensor;
    this->betaGmTensor = betaGmTensor;
    this->quantScaleGmTensor = quantScaleGmTensor;
    this->quantOffsetGmTensor = quantOffsetGmTensor;
    this->perTokenDescaleGmTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(perTokenDescaleGm));
    this->perChannelDescaleGmTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(perChannelDescaleGm));
    if constexpr (!NEED_DEQUANT) {
      inputGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gmInput));
    } else {
      mmGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(gmInput));
    }
    outputGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(gmOutput));
    this->qDownGmTensor = qDownGmTensor;

    num_col_ = num_col;
    avg_factor_ = avg_factor;
    epsilon_ = 1e-6;
    quantMin_ = INT8_MIN;
    this->num_row_ = mlaParams_.n;
    this->row_work = row_work;
    this->row_work_ = row_work_;
    gm_offset_ = gm_offset;
    gm_out_offset_ = gm_out_offset;
    num_col_align_int8 =
        (num_col_ + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
    num_col_align_f16 =
        (num_col_ + REPEAT_TIME_128 - 1) / REPEAT_TIME_128 * REPEAT_TIME_128;
    num_col_align_f32 =
        (num_col_ + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
    input_stride_ = stride;

    num_col_align_withStride_int8 =
        (num_col_ - input_stride_ + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 *
        REPEAT_TIME_256;
    num_col_align_withStride_fp16 =
        (num_col_ - input_stride_ + REPEAT_TIME_128 - 1) / REPEAT_TIME_128 *
        REPEAT_TIME_128;
    num_col_align_withStride_fp32 =
        (num_col_ - input_stride_ + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 *
        REPEAT_TIME_64;
  }

  __aicore__ inline void Init(
      AscendC::GlobalTensor<T>& gammaGmTensor,
      AscendC::GlobalTensor<T>& betaGmTensor,
      AscendC::GlobalTensor<T>& quantScaleGmTensor,
      AscendC::GlobalTensor<int8_t>& quantOffsetGmTensor,
      GM_ADDR perTokenDescaleGm,
      GM_ADDR perChannelDescaleGm,
      GM_ADDR gmInput,
      GM_ADDR gmOutput,
      uint32_t stride,
      uint32_t num_col,
      float avg_factor,
      uint64_t gm_offset,
      uint64_t gm_out_offset,
      uint32_t row_work_,
      const MlaTilingData& mlaParams_) {
    this->gammaGmTensor = gammaGmTensor;
    this->betaGmTensor = betaGmTensor;
    this->quantScaleGmTensor = quantScaleGmTensor;
    this->quantOffsetGmTensor = quantOffsetGmTensor;
    this->perTokenDescaleGmTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(perTokenDescaleGm));
    this->perChannelDescaleGmTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(perChannelDescaleGm));
    if constexpr (!NEED_DEQUANT) {
      inputGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gmInput));
    } else {
      mmGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(gmInput));
    }
    outputGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(gmOutput));

    num_col_ = num_col;
    avg_factor_ = avg_factor;
    epsilon_ = 1e-6;
    quantMin_ = INT8_MIN;
    this->num_row_ = mlaParams_.n;
    this->row_work = row_work;
    this->row_work_ = row_work_;
    gm_offset_ = gm_offset;
    gm_out_offset_ = gm_out_offset;
    num_col_align_int8 =
        (num_col_ + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
    num_col_align_f16 =
        (num_col_ + REPEAT_TIME_128 - 1) / REPEAT_TIME_128 * REPEAT_TIME_128;
    num_col_align_f32 =
        (num_col_ + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
    input_stride_ = stride;

    num_col_align_withStride_int8 =
        (num_col_ - input_stride_ + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 *
        REPEAT_TIME_256;
    num_col_align_withStride_fp16 =
        (num_col_ - input_stride_ + REPEAT_TIME_128 - 1) / REPEAT_TIME_128 *
        REPEAT_TIME_128;
    num_col_align_withStride_fp32 =
        (num_col_ - input_stride_ + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 *
        REPEAT_TIME_64;
  }

  __aicore__ inline void Launch(
      const AscendC::LocalTensor<int8_t>& dstTensor,
      const AscendC::LocalTensor<T>& srcTensor,
      const AscendC::LocalTensor<T>& gammaTensor,
      const AscendC::LocalTensor<T>& betaTensor,
      const AscendC::LocalTensor<T>& quantScaleTensor,
      const AscendC::LocalTensor<int8_t>& quantOffsetTensor,
      const AscendC::LocalTensor<float>& res1Tensor,
      const AscendC::LocalTensor<float>& res3Tensor) {
    this->dstTensor = dstTensor;
    this->srcTensor = srcTensor;
    this->gammaTensor = gammaTensor;
    this->betaTensor = betaTensor;
    this->fp32_xy = res1Tensor;
    this->buf = res3Tensor;

    AscendC::LocalTensor<float> g =
        buf[OFFSET_GAMMA * num_col_align_withStride_fp32];  // 0
    AscendC::LocalTensor<float> sqx =
        buf[OFFSET_GAMMA * num_col_align_withStride_fp32];  // 1
    AscendC::LocalTensor<float> abs =
        buf[OFFSET_GAMMA * num_col_align_withStride_fp32];  // 1
    AscendC::LocalTensor<float> sum =
        buf[OFFSET_GAMMA * num_col_align_withStride_fp32];  // 1
    AscendC::LocalTensor<float> work =
        buf[OFFSET_SQX * num_col_align_withStride_fp32];  // 1
    AscendC::LocalTensor<float> max_cal =
        buf[OFFSET_SQX * num_col_align_withStride_fp32 + WORK_SIZE];  // 2
    AscendC::LocalTensor<float> perTokenDescaleTensor =
        buf[OFFSET_SQX * num_col_align_withStride_fp32 + WORK_SIZE +
            ELE_NUM_FP16];  // 3
    AscendC::DataCopy(gammaTensor,
                      gammaGmTensor,
                      AscendC::DataCopyParams(
                          1, (num_col_ - input_stride_) / BLOCK_SIZE_16, 0, 0));
    AscendC::DataCopy(betaTensor,
                      betaGmTensor,
                      AscendC::DataCopyParams(
                          1, (num_col_ - input_stride_) / BLOCK_SIZE_16, 0, 0));
    SET_FLAG(MTE2, V, EVENT_ID1);
    WAIT_FLAG(MTE2, V, EVENT_ID1);
    if constexpr (quantMode == QuantMode::PER_TENSOR_ASYMM_QUANT) {
      AscendC::DataCopy(quantScaleTensor,
                        quantScaleGmTensor,
                        AscendC::DataCopyParams(1, 1, 0, 0));
      AscendC::DataCopy(quantOffsetTensor,
                        quantOffsetGmTensor,
                        AscendC::DataCopyParams(1, 1, 0, 0));
    }

    if constexpr (NEED_DEQUANT) {
      mmTensor = buf.ReinterpretCast<
          int32_t>()[OFFSET_SQX * num_col_align_withStride_fp32 + WORK_SIZE +
                     ELE_NUM_FP16];  // 256 + 16:work和max_cal的空间使用
      deScaleTensor = buf[OFFSET_SQX * num_col_align_withStride_fp32 +
                          WORK_SIZE + ELE_NUM_FP16 + MM1_OUT_SIZE];
      perTokenDescaleTensor =
          buf[OFFSET_SQX * num_col_align_withStride_fp32 + WORK_SIZE +
              ELE_NUM_FP16 +
              MM1_OUT_SIZE *
                  CONST_2];  // MM1_OUT_SIZE * 2 是deScaleTensor的空间使用
      AscendC::DataCopy(
          deScaleTensor,
          perChannelDescaleGmTensor,
          AscendC::DataCopyParams(
              1, num_col_ * sizeof(int32_t) / BLOCK_SIZE_32, 0, 0));
    }

    if constexpr (quantMode == QuantMode::PER_TENSOR_ASYMM_QUANT) {
      if (std::is_same<T, bfloat16_t>::value) {
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        Cast(g, quantScaleTensor, AscendC::RoundMode::CAST_NONE, 1);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        input_scale_ = 1 / (float)(g.GetValue(0));
        input_offset_ = (float)(quantOffsetTensor.GetValue(0));
      } else {
        SET_FLAG(MTE2, S, EVENT_ID0);
        WAIT_FLAG(MTE2, S, EVENT_ID0);
        input_scale_ = 1 / (float)(quantScaleTensor.GetValue(0));
        input_offset_ = (float)(quantOffsetTensor.GetValue(0));
      }
      AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
      AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }

    uint64_t pid = 0;
    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    while (pid < row_work_) {
      uint64_t offset = pid * num_col_;
      uint64_t outOffset = pid * (num_col_ - input_stride_);
      WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
      if constexpr (!NEED_DEQUANT) {
        AscendC::DataCopy(
            srcTensor,
            inputGmTensor[gm_offset_ + offset],
            AscendC::DataCopyParams(1, num_col_ / BLOCK_SIZE_16, 0, 0));
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
      } else {
        /* Dequant start */
        AscendC::DataCopy(
            mmTensor,
            mmGmTensor[gm_offset_ + offset],
            AscendC::DataCopyParams(1, num_col_ / 8, 0, 0));  // 2112
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        AscendC::Cast(mmTensor.ReinterpretCast<float>(),
                      mmTensor,
                      AscendC::RoundMode::CAST_NONE,
                      num_col_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(mmTensor.ReinterpretCast<float>(),
                     mmTensor.ReinterpretCast<float>(),
                     deScaleTensor,
                     num_col_);
        SET_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        gm_to_ub_align<Architecture::ASCEND_V220, float>(
            perTokenDescaleTensor,
            perTokenDescaleGmTensor[pid],
            0,              // sid
            1,              // nBurst
            sizeof(float),  // lenBurst
            0,              // leftPaddingNum
            0,              // rightPaddingNum
            0,              // srcGap
            0               // dstGap
        );
        SET_FLAG(MTE2, S, EVENT_ID0);
        WAIT_FLAG(MTE2, S, EVENT_ID0);
        float perTokenDescale = perTokenDescaleTensor.GetValue(0);
        SET_FLAG(S, V, EVENT_ID0);
        WAIT_FLAG(S, V, EVENT_ID0);
        AscendC::Muls(mmTensor.ReinterpretCast<float>(),
                      mmTensor.ReinterpretCast<float>(),
                      perTokenDescale,
                      num_col_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(srcTensor,
                      mmTensor.ReinterpretCast<float>(),
                      AscendC::RoundMode::CAST_RINT,
                      num_col_);
        AscendC::PipeBarrier<PIPE_V>();
      }

      Cast(fp32_xy,
           srcTensor[input_stride_],
           AscendC::RoundMode::CAST_NONE,
           REPEAT_TIME_64,
           num_col_align_withStride_fp32 / REPEAT_TIME_64,
           {1,
            1,
            AscendC::DEFAULT_REPEAT_STRIDE,
            AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
      AscendC::PipeBarrier<PIPE_V>();
      Mul(sqx,
          fp32_xy,
          fp32_xy,
          REPEAT_TIME_64,
          num_col_align_withStride_fp32 / REPEAT_TIME_64,
          {1,
           1,
           1,
           AscendC::DEFAULT_REPEAT_STRIDE,
           AscendC::DEFAULT_REPEAT_STRIDE,
           AscendC::DEFAULT_REPEAT_STRIDE});
      AscendC::PipeBarrier<PIPE_V>();
      Muls(sqx, sqx, avg_factor_, num_col_ - input_stride_);
      AscendC::PipeBarrier<PIPE_V>();
      ReduceSumCustom(sum, sqx, work, num_col_ - input_stride_);
      AscendC::PipeBarrier<PIPE_V>();
      Adds(sum, sum, epsilon_, 1);
      AscendC::PipeBarrier<PIPE_V>();
      Sqrt(sum, sum, 1);
      SET_FLAG(V, S, EVENT_ID0);
      WAIT_FLAG(V, S, EVENT_ID0);
      float factor = 1 / sum.GetValue(0);
      SET_FLAG(S, V, EVENT_ID0);
      WAIT_FLAG(S, V, EVENT_ID0);
      Muls(fp32_xy,
           fp32_xy,
           factor,
           REPEAT_TIME_64,
           num_col_align_withStride_fp32 / REPEAT_TIME_64,
           {1,
            1,
            AscendC::DEFAULT_REPEAT_STRIDE,
            AscendC::DEFAULT_REPEAT_STRIDE});
      AscendC::PipeBarrier<PIPE_V>();
      Cast(buf[OFFSET_GAMMA * num_col_align_withStride_fp32],
           gammaTensor,
           AscendC::RoundMode::CAST_NONE,
           REPEAT_TIME_64,
           num_col_align_withStride_fp32 / REPEAT_TIME_64,
           {1,
            1,
            AscendC::DEFAULT_REPEAT_STRIDE,
            AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
      AscendC::PipeBarrier<PIPE_V>();
      Mul(fp32_xy,
          fp32_xy,
          g,
          REPEAT_TIME_64,
          num_col_align_withStride_fp32 / REPEAT_TIME_64,
          {1,
           1,
           1,
           AscendC::DEFAULT_REPEAT_STRIDE,
           AscendC::DEFAULT_REPEAT_STRIDE,
           AscendC::DEFAULT_REPEAT_STRIDE});
      AscendC::PipeBarrier<PIPE_V>();
      if constexpr (WITH_BETA) {
        AscendC::LocalTensor<T> b = this->betaTensor;
        Cast(g,
             b,
             AscendC::RoundMode::CAST_NONE,
             REPEAT_TIME_64,
             num_col_align_withStride_fp32 / REPEAT_TIME_64,
             {1,
              1,
              AscendC::DEFAULT_REPEAT_STRIDE,
              AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
        AscendC::PipeBarrier<PIPE_V>();
        Add(fp32_xy,
            fp32_xy,
            g,
            REPEAT_TIME_64,
            num_col_align_withStride_fp32 / REPEAT_TIME_64,
            {1,
             1,
             1,
             AscendC::DEFAULT_REPEAT_STRIDE,
             AscendC::DEFAULT_REPEAT_STRIDE,
             AscendC::DEFAULT_REPEAT_STRIDE});
        AscendC::PipeBarrier<PIPE_V>();
      }

      if constexpr (NEED_Q_DOWN) {
        /* Output q_down start */
        // cast
        AscendC::LocalTensor<T> q_down =
            buf[OFFSET_Q_DOWN * num_col_align_withStride_fp32]
                .ReinterpretCast<T>();
        AscendC::Cast(q_down,
                      fp32_xy,
                      AscendC::RoundMode::CAST_RINT,
                      num_col_align_withStride_fp32);
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        // copy out
        AscendC::DataCopy(
            qDownGmTensor[gm_out_offset_ + outOffset],
            q_down,
            AscendC::DataCopyParams(
                1, (num_col_ - input_stride_) / BLOCK_SIZE_16, 0, 0));
        SET_FLAG(MTE3, V, EVENT_ID0);
        WAIT_FLAG(MTE3, V, EVENT_ID0);
        AscendC::PipeBarrier<PIPE_V>();
      }

      /* Quant start */
      if constexpr (quantMode == QuantMode::PER_TENSOR_ASYMM_QUANT) {
        Muls(fp32_xy,
             fp32_xy,
             input_scale_,
             REPEAT_TIME_64,
             num_col_align_withStride_fp32 / REPEAT_TIME_64,
             {1,
              1,
              AscendC::DEFAULT_REPEAT_STRIDE,
              AscendC::DEFAULT_REPEAT_STRIDE});
        AscendC::PipeBarrier<PIPE_V>();
        Adds(fp32_xy,
             fp32_xy,
             input_offset_,
             REPEAT_TIME_64,
             num_col_align_withStride_fp32 / REPEAT_TIME_64,
             {1,
              1,
              AscendC::DEFAULT_REPEAT_STRIDE,
              AscendC::DEFAULT_REPEAT_STRIDE});
        AscendC::PipeBarrier<PIPE_V>();
      } else if constexpr (quantMode == QuantMode::PER_TOKEN_SYMM_QUANT) {
        Abs(abs,
            fp32_xy,
            REPEAT_TIME_64,
            num_col_align_withStride_fp32 / REPEAT_TIME_64,
            {1,
             1,
             AscendC::DEFAULT_REPEAT_STRIDE,
             AscendC::DEFAULT_REPEAT_STRIDE});
        AscendC::PipeBarrier<PIPE_V>();
        ReduceMax(max_cal, abs, work, num_col_ - input_stride_);
        AscendC::PipeBarrier<PIPE_V>();
        float scaleOut = max_cal.GetValue(0) / 127;
        SET_FLAG(S, V, EVENT_ID0);
        WAIT_FLAG(S, V, EVENT_ID0);
        Muls(fp32_xy,
             fp32_xy,
             (float)(1 / scaleOut),
             REPEAT_TIME_64,
             num_col_align_withStride_fp32 / REPEAT_TIME_64,
             {1,
              1,
              AscendC::DEFAULT_REPEAT_STRIDE,
              AscendC::DEFAULT_REPEAT_STRIDE});
        AscendC::PipeBarrier<PIPE_V>();
        perTokenDescaleTensor.SetValue(0, scaleOut);
        SET_FLAG(S, MTE3, EVENT_ID0);
        WAIT_FLAG(S, MTE3, EVENT_ID0);
        if constexpr (!NEED_DEQUANT) {
          ub_to_gm_align<Architecture::ASCEND_V220, float>(
              perTokenDescaleGmTensor[pid],
              perTokenDescaleTensor,
              0,
              1,                  // nBurst
              1 * sizeof(float),  // lenBurst
              0,                  // leftPaddingNum
              0,                  // rightPaddingNum
              0,                  // srcGap
              0                   // dstGap
          );
        } else {
          ub_to_gm_align<Architecture::ASCEND_V220, float>(
              perTokenDescaleGmTensor[num_row_ + pid],
              perTokenDescaleTensor,
              0,
              1,                  // nBurst
              1 * sizeof(float),  // lenBurst
              0,                  // leftPaddingNum
              0,                  // rightPaddingNum
              0,                  // srcGap
              0                   // dstGap
          );
        }
        SET_FLAG(MTE3, V, EVENT_ID0);
        WAIT_FLAG(MTE3, V, EVENT_ID0);
      }

      AscendC::LocalTensor<half> tmpfp16 =
          buf.ReinterpretCast<half>()[OFFSET_GAMMA *
                                      num_col_align_withStride_fp32 *
                                      2];  // 2: half类型每个元素2字节
      CastFrom32To16(tmpfp16, fp32_xy, num_col_align_withStride_fp32);
      AscendC::PipeBarrier<PIPE_V>();
      CastFromF16ToI8(
          dstTensor, tmpfp16, quantMin_, num_col_align_withStride_fp16);
      AscendC::PipeBarrier<PIPE_V>();
      SET_FLAG(V, MTE3, EVENT_ID0);
      WAIT_FLAG(V, MTE3, EVENT_ID0);
      AscendC::DataCopy(
          outputGmTensor[gm_out_offset_ + outOffset],
          dstTensor,
          AscendC::DataCopyParams(
              1, (num_col_ - input_stride_) / BLOCK_SIZE_32, 0, 0));
      SET_FLAG(MTE3, MTE2, EVENT_ID0);
      ++pid;
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
  }

 private:
  AscendC::LocalTensor<int8_t> dstTensor;
  AscendC::LocalTensor<T> srcTensor;
  AscendC::LocalTensor<T> gammaTensor;
  AscendC::LocalTensor<T> betaTensor;
  AscendC::LocalTensor<float> fp32_xy;
  AscendC::LocalTensor<float> buf;
  AscendC::LocalTensor<int32_t> mmTensor;
  AscendC::LocalTensor<float> deScaleTensor;

  AscendC::GlobalTensor<T> gammaGmTensor;
  AscendC::GlobalTensor<T> betaGmTensor;
  AscendC::GlobalTensor<T> quantScaleGmTensor;
  AscendC::GlobalTensor<int8_t> quantOffsetGmTensor;
  AscendC::GlobalTensor<T> inputGmTensor;
  AscendC::GlobalTensor<int8_t> outputGmTensor;
  AscendC::GlobalTensor<float> perTokenDescaleGmTensor;
  AscendC::GlobalTensor<float> perChannelDescaleGmTensor;
  AscendC::GlobalTensor<int32_t> mmGmTensor;
  AscendC::GlobalTensor<T> qDownGmTensor;

  uint32_t num_col_{0};        // 输入的列数
  uint32_t num_row_{0};        // 输入的行数
  uint32_t row_work_{0};       // 需要计算多少行
  uint32_t row_work{0};        // 需要计算多少行
  uint32_t row_step_{0};       // 除最后一次，每次搬入多少行
  uint32_t row_tail_{0};       // 最后一次搬入多少行数据
  uint64_t gm_offset_{0};      // GM数据起始位置偏移量
  uint64_t gm_out_offset_{0};  // GM数据起始位置偏移量
  float avg_factor_{1.0};      // num_col_的倒数
  float input_scale_{1.0};     // 非对称量化系数
  float input_offset_{0};      // 非对称量化偏移适配高精度
  int32_t input_stride_{0};
  float epsilon_{1e-12f};  // norm平滑参数
  uint32_t num_col_align_int8{0};
  uint32_t num_col_align_f16{0};
  uint32_t num_col_align_f32{0};
  uint32_t num_col_align_f32_long{0};
  uint32_t num_col_align_withStride_int8{0};
  uint32_t num_col_align_withStride_fp16{0};
  uint32_t num_col_align_withStride_fp32{0};
  uint32_t num_col_temp;
  half quantMin_{-128};
  uint32_t num_slice_{0};
  uint32_t tail_size_{0};
  uint32_t tail_copy_{0};
};

template <typename InDtype, typename ScaleDtype>
class EinSumQuant {
 public:
  __aicore__ explicit EinSumQuant() {}

  __aicore__ inline void Init(GM_ADDR einSumOutGm,
                              GM_ADDR scaleGm,
                              GM_ADDR quantOutGm,
                              const MlaTilingData& tilingData) {
    einSumOutGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ InDtype*>(einSumOutGm));
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(scaleGm));
    quantOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(quantOutGm));

    headNum = tilingData.esqHeadNum;
    colNum = tilingData.esqColNum;
    ubHeadLoop = tilingData.esqUbHeadLoop;
    headPerLoop = tilingData.esqHeadPerLoop;
    headTail = tilingData.esqHeadTail;
    colLoop = tilingData.esqColLoop;
    colTail = tilingData.esqColTail;

    currentIdx = (AscendC::GetBlockIdx() / 2) * 2 + GetSubBlockidx();
    if (currentIdx < tilingData.esqFrontCore) {
      batchNum = tilingData.esqFrontCoreBatch;
      currentCoreStartOffset =
          currentIdx * tilingData.esqFrontCoreBatch * headNum * colNum;
    } else {
      batchNum = tilingData.esqTailCoreBatch;
      currentCoreStartOffset =
          (tilingData.esqFrontCore * tilingData.esqFrontCoreBatch +
           (currentIdx - tilingData.esqFrontCore) *
               tilingData.esqTailCoreBatch) *
          headNum * colNum;
    }
    calcRepeatStride = static_cast<uint8_t>(colNum / ELE_NUM_FP32);
    padLen = RoundUp(headNum, ELE_NUM_FP16);
    calcLength = headPerLoop * colNum;

    // calc tensors' data size(bytes) and block
    scaleBrcbFp32DataSize = padLen * ELE_NUM_FP32 * sizeof(float);
    inputDataSize = calcLength * sizeof(InDtype);
    inputDataBlock = calcLength * sizeof(InDtype) / BLOCK_SIZE_32;
    inputFp32DataSize = calcLength * sizeof(float);
    int8OutDataBlcok = calcLength / BLOCK_SIZE_32;
    headTailDataBlock = headTail * colNum * sizeof(InDtype) / BLOCK_SIZE_32;
    int8TailOutDataBlock = headTail * colNum / BLOCK_SIZE_32;
    if (padLen > headNum) {
      scaleCopyParams = AscendC::DataCopyExtParams(
          1, static_cast<uint32_t>(headNum * sizeof(InDtype)), 0, 0, 0);
      scalePadParams = AscendC::DataCopyPadExtParams<InDtype>(
          true, 0, static_cast<uint8_t>(padLen - headNum), 0);
    }
  }

  __aicore__ inline void Process() {
    if (batchNum == 0) {
      return;
    }
    // init local tensor
    scaleBrcbFp32_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    inputTensor_ =
        buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(scaleBrcbFp32DataSize);
    inputFp32_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(
        scaleBrcbFp32DataSize + inputDataSize * NUM_BUFFER);
    int8OutTensor_ = buf.GetBuffer<BufferType::ASCEND_UB, int8_t>(
        scaleBrcbFp32DataSize +
        (inputDataSize + inputFp32DataSize) * NUM_BUFFER);

    // scale copyin, cast, brcb[H, 1] --> [H, 8], use input ub space
    if (headNum == padLen) {
      AscendC::DataCopy(inputTensor_, scaleGm_, headNum);
    } else {
      AscendC::DataCopyPad(
          inputTensor_, scaleGm_, scaleCopyParams, scalePadParams);
    }
    SET_FLAG(MTE2, V, EVENT_ID0);
    WAIT_FLAG(MTE2, V, EVENT_ID0);
    AscendC::Cast(
        inputFp32_, inputTensor_, AscendC::RoundMode::CAST_NONE, padLen);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Brcb(scaleBrcbFp32_, inputFp32_, padLen / ELE_NUM_FP32, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();

    uint8_t pingFlag = 0;
    // batch Loop
    SET_FLAG(V, MTE2, EVENT_ID0);  // input copyin wait vector release ub
    SET_FLAG(V, MTE2, EVENT_ID1);
    SET_FLAG(MTE3, V, EVENT_ID0);  // quant calc wait last result copyout
    SET_FLAG(MTE3, V, EVENT_ID1);
    for (uint32_t batchIdx = 0; batchIdx < batchNum; batchIdx++) {
      batchOffset = batchIdx * headNum * colNum;
      // ub Loop
      for (uint32_t ubLoopIdx = 0; ubLoopIdx < ubHeadLoop; ubLoopIdx++) {
        scaleBrcbOffset = ubLoopIdx * headPerLoop * ELE_NUM_FP32;
        inputLoopOffset = ubLoopIdx * headPerLoop * colNum;
        calcStartOffset =
            currentCoreStartOffset + batchOffset + inputLoopOffset;
        calcTmpOffset = pingFlag * calcLength;

        // input CopyIn and Cast
        WAIT_FLAG(V, MTE2, pingFlag);
        AscendC::DataCopy(inputTensor_[calcTmpOffset],
                          einSumOutGm_[calcStartOffset],
                          {1, inputDataBlock, 0, 0});
        SET_FLAG(MTE2, V, pingFlag);
        WAIT_FLAG(MTE2, V, pingFlag);
        AscendC::Cast(inputFp32_[calcTmpOffset],
                      inputTensor_[calcTmpOffset],
                      AscendC::RoundMode::CAST_NONE,
                      calcLength);
        AscendC::PipeBarrier<PIPE_V>();
        SET_FLAG(V, MTE2, pingFlag);
        // quant calc
        for (uint32_t colIdx = 0; colIdx < colLoop; colIdx++) {
          colOffset = colIdx * CONST_64;
          AscendC::Mul(inputFp32_[calcTmpOffset + colOffset],
                       inputFp32_[calcTmpOffset + colOffset],
                       scaleBrcbFp32_[scaleBrcbOffset],
                       CONST_64,
                       headPerLoop,
                       {1, 1, 0, calcRepeatStride, calcRepeatStride, 1});
        }
        AscendC::PipeBarrier<PIPE_V>();
        // quant fp32 --> fp16 --> int8
        CastFrom32To16(
            inputFp32_[calcTmpOffset].template ReinterpretCast<half>(),
            inputFp32_[calcTmpOffset],
            calcLength);
        AscendC::PipeBarrier<PIPE_V>();
        WAIT_FLAG(MTE3, V, pingFlag);  // wait last result copy out
        CastFromF16ToI8(
            int8OutTensor_[calcTmpOffset],
            inputFp32_[calcTmpOffset].template ReinterpretCast<half>(),
            quantMin_,
            calcLength);
        AscendC::PipeBarrier<PIPE_V>();
        SET_FLAG(V, MTE3, pingFlag);
        WAIT_FLAG(V, MTE3, pingFlag);
        // int8 CopyOut
        AscendC::DataCopy(quantOutGm_[calcStartOffset],
                          int8OutTensor_[calcTmpOffset],
                          {1, int8OutDataBlcok, 0, 0});
        SET_FLAG(MTE3, V, pingFlag);
        pingFlag = 1 - pingFlag;
      }

      // deal with head tail
      if (headTail > 0) {
        scaleBrcbOffset = ubHeadLoop * headPerLoop * ELE_NUM_FP32;
        inputLoopOffset = ubHeadLoop * headPerLoop * colNum;
        calcStartOffset =
            currentCoreStartOffset + batchOffset + inputLoopOffset;
        calcTmpOffset = pingFlag * calcLength;

        // input CopyIn and Cast
        WAIT_FLAG(V, MTE2, pingFlag);
        AscendC::DataCopy(inputTensor_[calcTmpOffset],
                          einSumOutGm_[calcStartOffset],
                          {1, headTailDataBlock, 0, 0});
        SET_FLAG(MTE2, V, pingFlag);
        WAIT_FLAG(MTE2, V, pingFlag);
        AscendC::Cast(inputFp32_[calcTmpOffset],
                      inputTensor_[calcTmpOffset],
                      AscendC::RoundMode::CAST_NONE,
                      headTail * colNum);
        AscendC::PipeBarrier<PIPE_V>();
        SET_FLAG(V, MTE2, pingFlag);
        // quant calc
        for (uint32_t colIdx = 0; colIdx < colLoop; colIdx++) {
          colOffset = colIdx * CONST_64;
          AscendC::Mul(inputFp32_[calcTmpOffset + colOffset],
                       inputFp32_[calcTmpOffset + colOffset],
                       scaleBrcbFp32_[scaleBrcbOffset],
                       CONST_64,
                       headTail,
                       {1, 1, 0, calcRepeatStride, calcRepeatStride, 1});
        }
        AscendC::PipeBarrier<PIPE_V>();
        // quant fp32 --> fp16 --> int8
        CastFrom32To16(
            inputFp32_[calcTmpOffset].template ReinterpretCast<half>(),
            inputFp32_[calcTmpOffset],
            headTail * colNum);
        AscendC::PipeBarrier<PIPE_V>();
        WAIT_FLAG(MTE3, V, pingFlag);  // wait last result copy out
        CastFromF16ToI8(
            int8OutTensor_[calcTmpOffset],
            inputFp32_[calcTmpOffset].template ReinterpretCast<half>(),
            quantMin_,
            headTail * colNum);
        AscendC::PipeBarrier<PIPE_V>();
        SET_FLAG(V, MTE3, pingFlag);
        WAIT_FLAG(V, MTE3, pingFlag);
        // int8 CopyOut
        AscendC::DataCopy(quantOutGm_[calcStartOffset],
                          int8OutTensor_[calcTmpOffset],
                          {1, int8TailOutDataBlock, 0, 0});
        SET_FLAG(MTE3, V, pingFlag);
        pingFlag = 1 - pingFlag;
      }
    }
    WAIT_FLAG(V, MTE2, EVENT_ID0);
    WAIT_FLAG(V, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID1);
  }

 private:
  AsdopsBuffer<Architecture::ASCEND_V220> buf;

  AscendC::GlobalTensor<InDtype> einSumOutGm_;
  AscendC::GlobalTensor<ScaleDtype> scaleGm_;
  AscendC::GlobalTensor<int8_t> quantOutGm_;

  AscendC::LocalTensor<float> scaleBrcbFp32_;
  AscendC::LocalTensor<InDtype> inputTensor_;
  AscendC::LocalTensor<float> inputFp32_;
  AscendC::LocalTensor<int8_t> int8OutTensor_;

  AscendC::DataCopyExtParams scaleCopyParams;
  AscendC::DataCopyPadExtParams<InDtype> scalePadParams;

  // 单核处理数据量[batchNum, headNum, colNum]
  uint32_t batchNum;  // 每个核处理的batch数量
  uint32_t headNum;   // head数量
  uint32_t colNum;    // 每行的列数
  // ub loop
  uint32_t ubHeadLoop;   // ub循环处理head的次数
  uint32_t headPerLoop;  // 每次ub循环处理的head数量
  uint32_t headTail;     // 最后一次处理的head数量
  // col loop
  uint32_t colLoop;  // col方向循环计算次数
  uint32_t colTail;  // 最后一次处理的col数量

  uint32_t currentIdx;
  uint64_t currentCoreStartOffset;
  uint32_t inputDataSize;  // 每次搬运输入的大小，bytes
  uint32_t inputFp32DataSize;
  uint32_t scaleBrcbFp32DataSize;
  uint16_t inputDataBlock;  // 每次搬运输入的block数，bytes
  uint16_t int8OutDataBlcok;
  uint16_t headTailDataBlock;
  uint16_t int8TailOutDataBlock;
  // gm offset
  uint64_t inputLoopOffset{0};
  uint64_t batchOffset{0};
  uint64_t calcStartOffset{0};
  // double buffer tmp tensor length
  uint32_t scaleBrcbOffset{0};
  uint32_t calcLength{0};
  uint32_t calcTmpOffset{0};

  half quantMin_{-128};
  uint32_t colOffset{0};
  uint32_t padLen;
  uint8_t calcRepeatStride;
};

#ifdef __DAV_C220_CUBE__
struct MatCoord {
  uint64_t m{0};
  uint64_t k{0};
  uint64_t n{0};
};

template <typename InDtype,
          typename OutDtype,
          DataFormat formatB,
          bool transB,
          uint32_t swizzleDirect,
          uint64_t splitGapA,
          uint64_t splitGapC>
class PpMatmulEinSum {
  using AccumDtype = float;

  template <DataFormat srcFormat, DataFormat dstFormat>
  using CopyGmToCbuf =
      gm_to_l1<Architecture::ASCEND_V220, InDtype, srcFormat, dstFormat>;
  using LoadCbufToCa = l1_to_l0_a<Architecture::ASCEND_V220,
                                  InDtype,
                                  false,
                                  DataFormat::ZN,
                                  DataFormat::ZZ>;
  using LoadCbufToCb = l1_to_l0_b<Architecture::ASCEND_V220,
                                  InDtype,
                                  transB,
                                  DataFormat::ZN,
                                  DataFormat::NZ>;
  using Mad =
      mmad<Architecture::ASCEND_V220, InDtype, InDtype, AccumDtype, false>;
  using CopyCcToGm = l0c_to_gm<Architecture::ASCEND_V220,
                               DataFormat::ND,
                               OutDtype,
                               AccumDtype>;

  static constexpr uint32_t L0_PINGPONG_BUFFER_LEN = 16384;
  static constexpr uint32_t L1_PINGPONG_BUFFER_LEN = 131072;
  static constexpr uint32_t CONST_16 = 16;
  static constexpr uint32_t CONST_256 = 256;

 public:
  __aicore__ explicit PpMatmulEinSum(){};

  __aicore__ inline void Init(GM_ADDR gmA,
                              GM_ADDR gmB,
                              GM_ADDR gmC,
                              const MlaTilingData& mlaParams) {
#ifdef __DAV_C220_CUBE__
    batch_size = mlaParams.mm3.numBatch;
    m = mlaParams.mm3.m;
    k = mlaParams.mm3.k;
    n = mlaParams.mm3.n;
    m0 = mlaParams.mm3.m0;
    k0 = mlaParams.mm3.k0;
    n0 = mlaParams.mm3.n0;
    tdim.m = mlaParams.mm3.mLoop;
    tdim.k = mlaParams.mm3.kLoop;
    tdim.n = mlaParams.mm3.nLoop;
    core_loop = mlaParams.mm3.coreLoop;
    swizzle_cnt = mlaParams.mm3.swizzleCount;
    num_core = mlaParams.mm3.blockDim;
    core_idx = AscendC::GetBlockIdx();
    ping_flag = 1;

    gm_a.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(gmA));
    gm_b.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(gmB));
    gm_c.SetGlobalBuffer(reinterpret_cast<__gm__ OutDtype*>(gmC));

    AsdopsBuffer<Architecture::ASCEND_V220> buf;
    l1_base_a = buf.GetBuffer<BufferType::ASCEND_CB, InDtype>(0);
    l1_base_b = buf.GetBuffer<BufferType::ASCEND_CB, InDtype>(
        RoundUp<CONST_256>(m0 * k0 * sizeof(InDtype)));
    l0a_base = buf.GetBuffer<BufferType::ASCEND_L0A, InDtype>(0);
    l0b_base = buf.GetBuffer<BufferType::ASCEND_L0B, InDtype>(0);
#endif
    return;
  }

  __aicore__ inline void Process() {
#ifdef __DAV_C220_CUBE__
    if (core_idx >= num_core) {
      WaitFlagDev(AIC_MM3_START);
      return;
    }
    using LocalTensor = AscendC::LocalTensor<InDtype>;

    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);
    SET_FLAG(FIX, M, EVENT_ID0);
    SET_FLAG(M, MTE1, EVENT_ID0);
    SET_FLAG(M, MTE1, EVENT_ID1);

    for (uint64_t loop_idx = core_idx; loop_idx < core_loop;
         loop_idx += num_core) {
      uint64_t batch_idx = loop_idx / tdim.n / tdim.m;
      MatCoord tidx{0};
      GetBaseBlockIdx(loop_idx, tidx);
      uint64_t offset_c = tidx.m * m0 * batch_size * (n + splitGapC) +
                          batch_idx * (n + splitGapC) + tidx.n * n0;
      uint64_t m_actual = (tidx.m == (tdim.m - 1)) ? (m - tidx.m * m0) : m0;
      uint64_t n_actual = (tidx.n == (tdim.n - 1)) ? (n - tidx.n * n0) : n0;
      uint64_t m_round = RoundUp<CONST_16>(m_actual);
      uint64_t n_round = RoundUp<CONST_16>(n_actual);
      uint64_t mn_max = m_round > n_round ? m_round : n_round;
      uint64_t k_part_len =
          L0_PINGPONG_BUFFER_LEN / mn_max / CONST_16 * CONST_16;
      uint64_t shuffle_k = en_shuffle_k ? (core_idx % tdim.k) : 0;

      uint64_t k_actual = (shuffle_k == tdim.k - 1) ? k - shuffle_k * k0 : k0;
      uint64_t k_round = (k_actual + CONST_16 - 1) / CONST_16 * CONST_16;

      LocalTensor l1_buf_a =
          ping_flag ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN];
      LocalTensor l1_buf_b =
          ping_flag ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_LEN];
      event_t event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

      if (loop_idx == core_idx) {
        WaitFlagDev(AIC_MM3_START);

        // Copy A from gm to l1 buffer
        uint64_t offset_a = GetOffsetA(batch_idx, tidx.m, shuffle_k);
        WAIT_FLAG(MTE1, MTE2, event_id);
        CopyTileA(
            l1_buf_a, gm_a[offset_a], m_actual, m_round, k_actual, k_round);
        SET_FLAG(MTE2, MTE1, event_id);

        // Copy B from gm to l1 buffer
        uint64_t offset_b = GetOffsetB(batch_idx, shuffle_k, tidx.n);
        WAIT_FLAG(MTE1, MTE2, event_id + 2);
        CopyTileB(
            l1_buf_b, gm_b[offset_b], k_actual, k_round, n_actual, n_round);
        SET_FLAG(MTE2, MTE1, event_id + 2);
      }

      for (tidx.k = 0; tidx.k < tdim.k; ++tidx.k) {
        shuffle_k = en_shuffle_k ? (tidx.k + core_idx) % tdim.k : tidx.k;
        uint64_t k_actual =
            (shuffle_k == (tdim.k - 1)) ? (k - shuffle_k * k0) : k0;
        uint64_t k_round = (k_actual + CONST_16 - 1) / CONST_16 * CONST_16;
        fdim.k = (k_actual + k_part_len - 1) / k_part_len;

        LocalTensor l1_buf_a =
            ping_flag ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN];
        LocalTensor l1_buf_b =
            ping_flag ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_LEN];
        auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

        if (tidx.k < tdim.k - 1) {
          uint64_t shuffle_k_next =
              en_shuffle_k ? (core_idx + tidx.k + 1) % tdim.k : (tidx.k + 1);
          uint64_t offset_a_next =
              GetOffsetA(batch_idx, tidx.m, shuffle_k_next);
          uint64_t offset_b_next =
              GetOffsetB(batch_idx, shuffle_k_next, tidx.n);

          uint64_t k_actual_next =
              (shuffle_k_next == (tdim.k - 1)) ? (k - shuffle_k_next * k0) : k0;
          uint64_t k_round_next =
              (k_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;

          LocalTensor l1_buf_a_next =
              (1 - ping_flag) ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN];
          LocalTensor l1_buf_b_next =
              (1 - ping_flag) ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_LEN];
          event_t event_id_next = (1 - ping_flag) ? EVENT_ID0 : EVENT_ID1;

          // Preload A from gm to l1 buffer.
          WAIT_FLAG(MTE1, MTE2, event_id_next);
          CopyTileA(l1_buf_a_next,
                    gm_a[offset_a_next],
                    m_actual,
                    m_round,
                    k_actual_next,
                    k_round_next);
          SET_FLAG(MTE2, MTE1, event_id_next);

          // Preload B from gm to l1 buffer.
          WAIT_FLAG(MTE1, MTE2, event_id_next + 2);
          CopyTileB(l1_buf_b_next,
                    gm_b[offset_b_next],
                    k_actual_next,
                    k_round_next,
                    n_actual,
                    n_round);
          SET_FLAG(MTE2, MTE1, event_id_next + 2);
        }

        if (tidx.k == tdim.k - 1 && loop_idx + num_core < core_loop) {
          uint64_t b_idx_next = (loop_idx + num_core) / tdim.n / tdim.m;
          MatCoord tidx{0};
          GetBaseBlockIdx(loop_idx + num_core, tidx);
          uint64_t shuffle_k_next = en_shuffle_k ? (core_idx % tdim.k) : 0;
          uint64_t m_actual_next =
              (tidx.m == (tdim.m - 1)) ? (m - tidx.m * m0) : m0;
          uint64_t n_actual_next =
              (tidx.n == (tdim.n - 1)) ? (n - tidx.n * n0) : n0;
          uint64_t m_round_next =
              (m_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;
          uint64_t n_round_next =
              (n_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;
          uint64_t k_actual_next =
              (shuffle_k_next == (tdim.k - 1)) ? (k - shuffle_k_next * k0) : k0;
          uint64_t k_round_next =
              (k_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;
          uint64_t offset_a_next =
              GetOffsetA(b_idx_next, tidx.m, shuffle_k_next);
          uint64_t offset_b_next =
              GetOffsetB(b_idx_next, shuffle_k_next, tidx.n);

          LocalTensor l1_buf_a_next =
              (1 - ping_flag) ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN];
          LocalTensor l1_buf_b_next =
              (1 - ping_flag) ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_LEN];
          event_t event_id_next = (1 - ping_flag) ? EVENT_ID0 : EVENT_ID1;

          // Preload A from gm to l1 buffer.
          WAIT_FLAG(MTE1, MTE2, event_id_next);
          CopyTileA(l1_buf_a_next,
                    gm_a[offset_a_next],
                    m_actual_next,
                    m_round_next,
                    k_actual_next,
                    k_round_next);
          SET_FLAG(MTE2, MTE1, event_id_next);

          // Preload B from gm to l1 buffer.
          WAIT_FLAG(MTE1, MTE2, event_id_next + 2);
          CopyTileB(l1_buf_b_next,
                    gm_b[offset_b_next],
                    k_actual_next,
                    k_round_next,
                    n_actual_next,
                    n_round_next);
          SET_FLAG(MTE2, MTE1, event_id_next + 2);
        }

        MatCoord fidx{0};
        for (fidx.k = 0; fidx.k < fdim.k; ++fidx.k) {
          uint32_t k0_round = (fidx.k < fdim.k - 1)
                                  ? k_part_len
                                  : k_round - fidx.k * k_part_len;
          uint32_t k0_actual = (fidx.k < fdim.k - 1)
                                   ? k_part_len
                                   : k_actual - fidx.k * k_part_len;

          auto mte1_mad_ping_flag = 1 - fidx.k % 2;
          auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
          LocalTensor l0a_buf =
              l0a_base[(fidx.k & 0b1) * L0_PINGPONG_BUFFER_LEN];
          LocalTensor l0b_buf =
              l0b_base[(fidx.k & 0b1) * L0_PINGPONG_BUFFER_LEN];

          // *** load matrix A from L1 to L0A
          if (fidx.k == 0) {
            WAIT_FLAG(MTE2, MTE1, event_id);
          }
          WAIT_FLAG(M, MTE1, mte1_mad_event_id);
          if ((m == 1) || (m_actual == 1)) {
            l1_to_l0_a<Architecture::ASCEND_V220,
                       InDtype,
                       false,
                       DataFormat::VECTOR,
                       DataFormat::VECTOR>(
                l0a_buf,                        // dst
                l1_buf_a[fidx.k * k_part_len],  // src
                0,                              // mTileCeil
                CeilDiv<CONST_256>(k0_round),   // kPartCeil
                0,                              // mSrcStride
                1,                              // kSrcStride
                0,                              // mDstStride
                0);                             // kDstStride
          } else {
            LoadCbufToCa(l0a_buf,                                  // l0Tensor
                         l1_buf_a[fidx.k * k_part_len * m_round],  // l1Tensor
                         m_round,                                  // mTileCeil
                         k0_round,                                 // kPartCeil
                         1,                                        // mSrcStride
                         m_round / CONST_16,                       // kSrcStride
                         k0_round / CONST_16,                      // mDstStride
                         1);                                       // kDstStride
          }
          if (fidx.k == fdim.k - 1) {
            SET_FLAG(MTE1, MTE2, event_id);
          }

          // *** load matrix B from L1 to L0B
          if (fidx.k == 0) {
            WAIT_FLAG(MTE2, MTE1, event_id + 2);
          }
          if constexpr (transB) {
            LoadCbufToCb(l0b_buf,                                  // l0Tensor
                         l1_buf_b[fidx.k * k_part_len * n_round],  // l1Tensor
                         n_round,                                  // nTileCeil
                         k0_round,                                 // kPartCeil
                         1,                                        // nSrcStride
                         n_round / CONST_16,                       // kSrcStride
                         1,                                        // nDstStride
                         k0_round / CONST_16);                     // kDstStride
          } else {
            LoadCbufToCb(l0b_buf,                                   // l0Tensor
                         l1_buf_b[fidx.k * k_part_len * CONST_16],  // l1Tensor
                         n_round,                                   // nTileCeil
                         k0_round,                                  // kPartCeil
                         k_round / CONST_16,   // nSrcStride
                         1,                    // kSrcStride
                         1,                    // nDstStride
                         n_round / CONST_16);  // kDstStride
          }
          if (fidx.k == fdim.k - 1) {
            SET_FLAG(MTE1, MTE2, event_id + 2);
          }

          SET_FLAG(MTE1, M, mte1_mad_event_id);
          WAIT_FLAG(MTE1, M, mte1_mad_event_id);

          bool init_c = (tidx.k == 0 && fidx.k == 0);
          if (init_c) {
            WAIT_FLAG(FIX, M, EVENT_ID0);
          }

          Mad(l0c_buf,    // c
              l0a_buf,    // a
              l0b_buf,    // b
              m_actual,   // mTileActual
              n_actual,   // nTileActual
              k0_actual,  // kTileActual
              init_c);    // initC

          PIPE_BARRIER(M);
          SET_FLAG(M, MTE1, mte1_mad_event_id);
        }

        ping_flag = 1 - ping_flag;
      }

      SET_FLAG(M, FIX, EVENT_ID0);
      WAIT_FLAG(M, FIX, EVENT_ID0);

      // copy from L0C to gm
      CopyCcToGm(gm_c[offset_c],                 // dst
                 l0c_buf,                        // src
                 m_actual,                       // mTileActual
                 n_actual,                       // nTileActual
                 m_round,                        // mTileCeil
                 (n + splitGapC) * batch_size);  // nActual
      SET_FLAG(FIX, M, EVENT_ID0);
    }

    WAIT_FLAG(M, MTE1, EVENT_ID0);
    WAIT_FLAG(M, MTE1, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
    WAIT_FLAG(FIX, M, EVENT_ID0);
#endif
  }

 private:
  __aicore__ inline void GetBaseBlockIdx(uint64_t index, MatCoord& tidx) {
    uint64_t in_batch_idx = index % (tdim.m * tdim.n);
    if constexpr (swizzleDirect == 0) {  // Zn
      uint64_t tile_block_loop = (tdim.m + swizzle_cnt - 1) / swizzle_cnt;
      uint64_t tile_block_idx = in_batch_idx / (swizzle_cnt * tdim.n);
      uint64_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * tdim.n);

      uint64_t n_row = swizzle_cnt;
      if (tile_block_idx == tile_block_loop - 1) {
        n_row = tdim.m - swizzle_cnt * tile_block_idx;
      }
      tidx.m = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_row;
      tidx.n = in_tile_block_idx / n_row;
      if (tile_block_idx % 2 != 0) {
        tidx.n = tdim.n - tidx.n - 1;
      }
    } else if constexpr (swizzleDirect == 1) {  // Nz
      uint64_t tile_block_loop = (tdim.n + swizzle_cnt - 1) / swizzle_cnt;
      uint64_t tile_block_idx = in_batch_idx / (swizzle_cnt * tdim.m);
      uint64_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * tdim.m);

      uint64_t n_col = swizzle_cnt;
      if (tile_block_idx == tile_block_loop - 1) {
        n_col = tdim.n - swizzle_cnt * tile_block_idx;
      }
      tidx.m = in_tile_block_idx / n_col;
      tidx.n = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_col;
      if (tile_block_idx % 2 != 0) {
        tidx.m = tdim.m - tidx.m - 1;
      }
    }
    return;
  }

  __aicore__ inline uint64_t GetOffsetA(const uint64_t bIdx,
                                        const uint64_t mIdx,
                                        const uint64_t kIdx) {
    return mIdx * m0 * batch_size * (k + splitGapA) + bIdx * (k + splitGapA) +
           kIdx * k0;
  }

  __aicore__ inline uint64_t GetOffsetB(const uint64_t bIdx,
                                        const uint64_t kIdx,
                                        const uint64_t nIdx) {
    if constexpr (formatB == DataFormat::ND) {
      if constexpr (transB) {
        return bIdx * k * n + nIdx * n0 * k + kIdx * k0;
      } else {
        return bIdx * k * n + kIdx * k0 * n + nIdx * n0;
      }
    } else {
      if constexpr (transB) {
        return bIdx * RoundUp<CONST_16>(n) * RoundUp<CONST_16>(k) +
               kIdx * k0 * RoundUp<CONST_16>(n) + nIdx * n0 * CONST_16;
      } else {
        return bIdx * RoundUp<CONST_16>(k) * RoundUp<CONST_16>(n) +
               nIdx * n0 * RoundUp<CONST_16>(k) + kIdx * k0 * CONST_16;
      }
    }
  }

  __aicore__ inline void CopyTileA(
      AscendC::LocalTensor<InDtype>& dstTensor,
      const AscendC::GlobalTensor<InDtype>& srcTensor,
      const uint64_t m_actual,
      const uint64_t m_round,
      const uint64_t k_actual,
      const uint64_t k_round) {
    if ((m == 1) || (m_actual == 1)) {
      CopyGmToCbuf<DataFormat::ND, DataFormat::ND>(dstTensor,  // dst
                                                   srcTensor,  // src
                                                   1,          // nTileActual
                                                   CONST_16,   // nTileCeil
                                                   1,          // nVal
                                                   k_actual,   // kTileActual
                                                   k_round,    // kTileCeil
                                                   k);         // dVal
    } else {
      CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(
          dstTensor,                      // dst
          srcTensor,                      // src
          m_actual,                       // nTileActual
          m_round,                        // nTileCeil
          m,                              // nVal
          k_actual,                       // dTileActual
          k_round,                        // dTileCeil
          (k + splitGapA) * batch_size);  // dVal
    }
  }

  __aicore__ inline void CopyTileB(
      AscendC::LocalTensor<InDtype>& dstTensor,
      const AscendC::GlobalTensor<InDtype>& srcTensor,
      const uint64_t k_actual,
      const uint64_t k_round,
      const uint64_t n_actual,
      const uint64_t n_round) {
    if constexpr (formatB == DataFormat::ND) {
      if constexpr (transB) {
        CopyGmToCbuf<formatB, DataFormat::NZ>(dstTensor,  // dst
                                              srcTensor,  // src
                                              n_actual,   // nTileActual
                                              n_round,    // nTileCeil
                                              n,          // nVal
                                              k_actual,   // dTileActual
                                              k_round,    // dTileCeil
                                              k);         // dVal
      } else {
        CopyGmToCbuf<formatB, DataFormat::NZ>(dstTensor,  // dst
                                              srcTensor,  // src
                                              k_actual,   // nTileActual
                                              k_round,    // nTileCeil
                                              k,          // nVal
                                              n_actual,   // dTileActual
                                              n_round,    // dTileCeil
                                              n);         // dVal
      }
    } else {
      if constexpr (transB) {
        CopyGmToCbuf<formatB, DataFormat::NZ>(dstTensor,  // dst
                                              srcTensor,  // src
                                              n_actual,   // nTileActual
                                              n_round,    // nTileCeil
                                              RoundUp<CONST_16>(n),  // nVal
                                              k_actual,  // dTileActual
                                              k_round,   // dTileCeil
                                              RoundUp<CONST_16>(k));  // dVal
      } else {
        CopyGmToCbuf<formatB, DataFormat::NZ>(dstTensor,  // dst
                                              srcTensor,  // src
                                              k_actual,   // nTileActual
                                              k_round,    // nTileCeil
                                              RoundUp<CONST_16>(k),  // nVal
                                              n_actual,  // dTileActual
                                              n_round,   // dTileCeil
                                              RoundUp<CONST_16>(n));  // dVal
      }
    }
  }

 private:
  AscendC::GlobalTensor<InDtype> gm_a;
  AscendC::GlobalTensor<InDtype> gm_b;
  AscendC::GlobalTensor<OutDtype> gm_c;
  AscendC::LocalTensor<InDtype> l1_base_a;
  AscendC::LocalTensor<InDtype> l1_base_b;
  AscendC::LocalTensor<InDtype> l0a_base;
  AscendC::LocalTensor<InDtype> l0b_base;
  AscendC::LocalTensor<float> l0c_buf;

  uint32_t num_core{0};
  uint32_t batch_size{0};
  uint32_t m{0};
  uint32_t k{0};
  uint32_t n{0};
  uint32_t m0{0};
  uint32_t k0{0};
  uint32_t n0{0};
  MatCoord tdim{0};
  MatCoord fdim{0};
  uint32_t core_loop{0};
  uint32_t swizzle_cnt{1};
  uint32_t core_idx{0};
  uint32_t en_shuffle_k{0};
  uint32_t ping_flag{0};
};

template <bool withSyncAll,
          uint32_t swizzleDir,
          DataFormat formatA = DataFormat::ND,
          DataFormat formatB = DataFormat::NZ>
class PpMatmulW8a8Aic {
  using InDtype = int8_t;
  using OutDtype = int32_t;
  using AccumDtype = int32_t;

  template <DataFormat srcFormat, DataFormat dstFormat>
  using CopyGmToCbuf =
      gm_to_l1<Architecture::ASCEND_V220, InDtype, srcFormat, dstFormat>;
  using LoadCbufToCa = l1_to_l0_a<Architecture::ASCEND_V220,
                                  InDtype,
                                  false,
                                  DataFormat::ZN,
                                  DataFormat::ZZ>;
  using LoadCbufToCb = l1_to_l0_b<Architecture::ASCEND_V220,
                                  InDtype,
                                  true,
                                  DataFormat::ZN,
                                  DataFormat::NZ>;
  using Mmad =
      mmad<Architecture::ASCEND_V220, InDtype, InDtype, AccumDtype, false>;
  using CopyCcToGm = l0c_to_gm<Architecture::ASCEND_V220,
                               DataFormat::ND,
                               OutDtype,
                               AccumDtype>;

  static constexpr uint64_t L0_PINGPONG_BUFFER_LEN = 32768;
  static constexpr uint64_t L1_PINGPONG_BUFFER_LEN = 262144;
  static constexpr uint64_t BLOCK_SIZE_16 = 16;
  static constexpr uint64_t BLOCK_SIZE_32 = 32;
  static constexpr uint64_t CUBE_MATRIX_SIZE_512 = 512;
  static constexpr uint64_t CONST_4 = 4;
  static constexpr uint64_t CONST_8 = 8;
  static constexpr uint64_t CONST_32 = 32;
  static constexpr uint64_t CONST_64 = 64;
  static constexpr uint64_t CONST_128 = 128;

 public:
  __aicore__ PpMatmulW8a8Aic() {};

  __aicore__ inline void Init(GM_ADDR gmA,
                              GM_ADDR gmB,
                              GM_ADDR gmC,
                              MlaPpMatmulTilingData& tilingdata,
                              uint32_t mode) {
    gm_a.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(gmA));
    gm_b.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(gmB));
    gm_c.SetGlobalBuffer(reinterpret_cast<__gm__ OutDtype*>(gmC));

    batch_size = tilingdata.numBatch;
    m = tilingdata.m;
    k = tilingdata.k;
    n = tilingdata.n;
    m0 = tilingdata.m0;
    k0 = tilingdata.k0;
    n0 = tilingdata.n0;
    m_loop = tilingdata.mLoop;
    k_loop = tilingdata.kLoop;
    n_loop = tilingdata.nLoop;
    core_loop = tilingdata.coreLoop;
    swizzle_cnt = tilingdata.swizzleCount;
    en_shuffle_k = tilingdata.enShuffleK;
    core_num = tilingdata.blockDim;
    load_all_Amat_flag = tilingdata.enLoadAllAmat;
    b0mat_pingpong_buffer_len = tilingdata.b0matPingPongBufferLen;

    core_idx = AscendC::GetBlockIdx();
    ping_flag = 1;
    MM1_MM2_mode = mode;  // 区分MM1 or MM2

    InitBuffer();
    return;
  }

  __aicore__ inline uint64_t GetOffsetA(const uint64_t batchIdx,
                                        const uint64_t mIdx,
                                        uint64_t kIdx) {
    return batchIdx * m * k + mIdx * m0 * k + kIdx * k0;
  }

  __aicore__ inline uint64_t GetOffsetB(const uint64_t batchIdx,
                                        const uint64_t kIdx,
                                        uint64_t nIdx) {
    if constexpr (formatB == DataFormat::ND) {
      return batchIdx * k * n + nIdx * n0 * k + kIdx * k0;
    } else {
      return batchIdx * RoundUp<BLOCK_SIZE_16>(n) * RoundUp<BLOCK_SIZE_32>(k) +
             kIdx * k0 * RoundUp<BLOCK_SIZE_16>(n) + nIdx * n0 * CONST_32;
    }
  }

  __aicore__ inline void CopyTileA(
      AscendC::LocalTensor<InDtype>& dstTensor,
      const AscendC::GlobalTensor<InDtype>& srcTensor,
      const uint64_t m_actual,
      const uint64_t m_round,
      const uint64_t k_actual,
      const uint64_t k_round) {
    if ((m == 1) || (m_actual == 1)) {
      CopyGmToCbuf<formatA, DataFormat::ND>(dstTensor,  // dst
                                            srcTensor,  // src
                                            1,
                                            BLOCK_SIZE_16,
                                            1,
                                            k_actual,
                                            k_round,
                                            k);
    } else {
      CopyGmToCbuf<formatA, DataFormat::NZ>(dstTensor,  // dst
                                            srcTensor,  // src
                                            m_actual,   // nTileActual
                                            m_round,    // nTileCeil
                                            n,          // nVal
                                            k_actual,   // dTileActual
                                            k_round,    // dTileCeil
                                            k);         // dVal
    }
  }

  __aicore__ inline void CopyTileB(
      const AscendC::LocalTensor<InDtype>& dstTensor,
      const AscendC::GlobalTensor<InDtype>& srcTensor,
      const uint64_t k_actual,
      const uint64_t k_round,
      const uint64_t n_actual,
      const uint64_t n_round) {
    if constexpr (formatB == DataFormat::ND) {
      CopyGmToCbuf<formatB, DataFormat::NZ>(dstTensor,  // dst
                                            srcTensor,  // src
                                            n_actual,   // nTileActual
                                            n_round,    // nTileCeil
                                            n,          // nVal
                                            k_actual,   // dTileActual
                                            k_round,    // dTileCeil
                                            k);         // dVal
    } else {
      CopyGmToCbuf<formatB, DataFormat::NZ>(dstTensor,  // dst
                                            srcTensor,  // src
                                            n_actual,   // nTileActual
                                            n_round,    // nTileCeil
                                            RoundUp<BLOCK_SIZE_16>(n),  // nVal
                                            k_actual,  // dTileActual
                                            k_round,   // dTileCeil
                                            RoundUp<BLOCK_SIZE_32>(k));  // dVal
    }
  }

  __aicore__ inline void PreloadWeight() {
    if (core_idx < core_num) {
      uint64_t m_idx = 0;
      uint64_t n_idx = 0;
      GetBaseBlockIdx(core_idx, m_idx, n_idx);
      uint64_t shuffle_k = en_shuffle_k ? core_idx % k_loop : 0;
      uint64_t offset_b = GetOffsetB(0, shuffle_k, n_idx);
      uint64_t k_actual = (shuffle_k == k_loop - 1) ? k - shuffle_k * k0 : k0;
      uint64_t k_round = RoundUp<BLOCK_SIZE_32>(k_actual);
      uint64_t n_actual = (n_idx == (n_loop - 1)) ? (n - n_idx * n0) : n0;
      uint64_t n_round = RoundUp<BLOCK_SIZE_16>(n_actual);
      CopyTileB(
          l1_base_b, gm_b[offset_b], k_actual, k_round, n_actual, n_round);
    }
    if (core_idx < core_num && k_loop > 1) {
      uint64_t m_idx = 0;
      uint64_t n_idx = 0;
      GetBaseBlockIdx(core_idx, m_idx, n_idx);
      uint64_t shuffle_k = en_shuffle_k ? (core_idx + 1) % k_loop : 1;
      uint64_t offset_b = GetOffsetB(0, shuffle_k, n_idx);
      uint64_t k_actual = (shuffle_k == k_loop - 1) ? k - shuffle_k * k0 : k0;
      uint64_t k_round = RoundUp<BLOCK_SIZE_32>(k_actual);
      uint64_t n_actual = (n_idx == (n_loop - 1)) ? (n - n_idx * n0) : n0;
      uint64_t n_round = RoundUp<BLOCK_SIZE_16>(n_actual);
      CopyTileB(l1_base_b[b0mat_pingpong_buffer_len],
                gm_b[offset_b],
                k_actual,
                k_round,
                n_actual,
                n_round);
    }
  }

  __aicore__ inline void Process();

 private:
  __aicore__ inline void InitBuffer() {
    AsdopsBuffer<Architecture::ASCEND_V220> buf;
    l1_base_a = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(0);

    // try load all A matrix
    uint32_t a_l1_size = RoundUp<BLOCK_SIZE_16>(m) * RoundUp<BLOCK_SIZE_32>(k);
    if (!load_all_Amat_flag) {
      a_l1_size = RoundUp<CUBE_MATRIX_SIZE_512>(m0 * k0);
    }

    l1_base_b = l1_base_a[a_l1_size];
    l0a_base = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(0);
    l0b_base = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(0);
    l0c_buf = buf.template GetBuffer<BufferType::ASCEND_L0C, AccumDtype>(0);
  }

  __aicore__ inline void GetBaseBlockIdx(uint64_t index,
                                         uint64_t& m_idx,
                                         uint64_t& n_idx) {
    uint64_t in_batch_idx = index % (m_loop * n_loop);
    if constexpr (swizzleDir == 0) {  // Zn
      uint64_t tile_block_loop = (m_loop + swizzle_cnt - 1) / swizzle_cnt;
      uint64_t tile_block_idx = in_batch_idx / (swizzle_cnt * n_loop);
      uint64_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * n_loop);

      uint64_t n_row = swizzle_cnt;
      if (tile_block_idx == tile_block_loop - 1) {
        n_row = m_loop - swizzle_cnt * tile_block_idx;
      }
      m_idx = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_row;
      n_idx = in_tile_block_idx / n_row;
      if ((tile_block_idx & 0b1) != 0) {
        n_idx = n_loop - n_idx - 1;
      }
    } else {  // Nz
      uint64_t tile_block_loop = (n_loop + swizzle_cnt - 1) / swizzle_cnt;
      uint64_t tile_block_idx = in_batch_idx / (swizzle_cnt * m_loop);
      uint64_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * m_loop);

      uint64_t n_col = swizzle_cnt;
      if (tile_block_idx == tile_block_loop - 1) {
        n_col = n_loop - swizzle_cnt * tile_block_idx;
      }
      m_idx = in_tile_block_idx / n_col;
      n_idx = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_col;
      if ((tile_block_idx & 0b1) != 0) {
        m_idx = m_loop - m_idx - 1;
      }
    }
    return;
  }

 private:
  AscendC::GlobalTensor<InDtype> gm_a;
  AscendC::GlobalTensor<InDtype> gm_b;
  AscendC::GlobalTensor<OutDtype> gm_c;

  AscendC::LocalTensor<InDtype> l1_base_a;
  AscendC::LocalTensor<InDtype> l1_base_b;
  AscendC::LocalTensor<InDtype> l0a_base;
  AscendC::LocalTensor<InDtype> l0b_base;
  AscendC::LocalTensor<AccumDtype> l0c_buf;

  uint64_t bias_bt{0};
  uint32_t core_num{0};
  uint32_t batch_size{0};
  uint32_t m{0};
  uint32_t k{0};
  uint32_t n{0};
  uint32_t m0{0};
  uint32_t k0{0};
  uint32_t n0{0};
  uint32_t m_loop{0};
  uint32_t n_loop{0};
  uint32_t k_loop{0};
  uint32_t core_loop{0};
  uint32_t core_idx{0};
  uint32_t ping_flag{0};
  uint32_t swizzle_cnt{1};
  uint32_t en_shuffle_k{0};
  uint32_t MM1_MM2_mode{0};
  uint64_t b0mat_pingpong_buffer_len{0};
  bool load_all_Amat_flag{false};
};

template <bool withSyncAll,
          uint32_t swizzleDir,
          DataFormat formatA,
          DataFormat formatB>
__aicore__ inline void
PpMatmulW8a8Aic<withSyncAll, swizzleDir, formatA, formatB>::Process() {
  using LocalTensor = AscendC::LocalTensor<InDtype>;
  if (core_idx >= core_num) {
    if (MM1_MM2_mode == 0) {
      WaitFlagDev(AIC_MM1_START);
    } else if (MM1_MM2_mode == 1) {
      WaitFlagDev(AIC_MM2_START);
    }
    return;
  }
  SET_FLAG(MTE1, MTE2, EVENT_ID0);
  SET_FLAG(MTE1, MTE2, EVENT_ID1);
  SET_FLAG(MTE1, MTE2, EVENT_ID2);
  SET_FLAG(MTE1, MTE2, EVENT_ID3);
  SET_FLAG(M, MTE1, EVENT_ID0);
  SET_FLAG(M, MTE1, EVENT_ID1);
  SET_FLAG(FIX, M, EVENT_ID0);
  SET_FLAG(FIX, MTE2, EVENT_ID0);
  SET_FLAG(MTE1, MTE2, EVENT_ID7);
  for (uint64_t loop_idx = core_idx; loop_idx < core_loop;
       loop_idx += core_num) {
    uint64_t batch_idx = loop_idx / n_loop / m_loop;
    uint64_t m_idx = 0;
    uint64_t n_idx = 0;
    GetBaseBlockIdx(loop_idx, m_idx, n_idx);
    uint64_t offset_a;
    uint64_t offset_b;
    uint64_t offset_bias;
    uint64_t offset_a_next;
    uint64_t offset_b_next;
    uint64_t offset_c = batch_idx * m * n + m_idx * m0 * n + n_idx * n0;
    uint64_t m_actual = (m_idx == (m_loop - 1)) ? (m - m_idx * m0) : m0;
    uint64_t n_actual = (n_idx == (n_loop - 1)) ? (n - n_idx * n0) : n0;
    uint64_t m_round = 0;
    uint64_t n_round = 0;
    uint64_t shuffle_k = en_shuffle_k ? core_idx % k_loop : 0;
    uint64_t m_round_16 = RoundUp<BLOCK_SIZE_16>(m_actual);
    uint64_t m_round_32 = RoundUp<BLOCK_SIZE_32>(m_actual);
    m_round = m_round_16;
    n_round = RoundUp<BLOCK_SIZE_16>(n_actual);

    uint64_t mn_max = m_round > n_round ? m_round : n_round;
    uint64_t k_part_len = 0;
    k_part_len =
        L0_PINGPONG_BUFFER_LEN / mn_max / BLOCK_SIZE_32 * BLOCK_SIZE_32;

    offset_b = GetOffsetB(batch_idx, shuffle_k, n_idx);
    offset_bias = batch_idx * n + n_idx * n0;

    uint64_t k_actual = (shuffle_k == k_loop - 1) ? k - shuffle_k * k0 : k0;
    uint64_t k_round = RoundUp<BLOCK_SIZE_32>(k_actual);

    auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

    // Wait after Scalar
    if (loop_idx == core_idx) {
      if (MM1_MM2_mode == 0) {
        WaitFlagDev(AIC_MM1_START);
      } else if (MM1_MM2_mode == 1) {
        WaitFlagDev(AIC_MM2_START);
      }
    }

    WAIT_FLAG(MTE1, MTE2, event_id);
    LocalTensor l1_buf_a =
        load_all_Amat_flag
            ? l1_base_a
            : (ping_flag ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN]);
    LocalTensor l1_buf_b =
        ping_flag ? l1_base_b : l1_base_b[b0mat_pingpong_buffer_len];
    if (load_all_Amat_flag) {
      if (loop_idx == core_idx) {
        offset_a = GetOffsetA(batch_idx, m_idx, 0);
        uint64_t k_actual_first = k;
        uint64_t k_round_first = RoundUp<BLOCK_SIZE_32>(k_actual_first);
        CopyTileA(l1_buf_a,
                  gm_a[offset_a],
                  m_actual,
                  m_round,
                  k_actual_first,
                  k_round_first);
      }
    } else {
      offset_a = GetOffsetA(batch_idx, m_idx, shuffle_k);
      CopyTileA(l1_buf_a, gm_a[offset_a], m_actual, m_round, k_actual, k_round);
    }
    SET_FLAG(MTE2, MTE1, event_id);

    WAIT_FLAG(MTE1, MTE2, event_id + CONST_2);
    // 首个权重矩阵块提前加载
    if (loop_idx != core_idx) {
      CopyTileB(l1_buf_b, gm_b[offset_b], k_actual, k_round, n_actual, n_round);
    }
    SET_FLAG(MTE2, MTE1, event_id + CONST_2);

    for (uint64_t k_idx = 0; k_idx < k_loop; k_idx++) {
      shuffle_k = en_shuffle_k ? (k_idx + core_idx) % k_loop : k_idx;
      uint32_t k_actual =
          (shuffle_k == (k_loop - 1)) ? (k - shuffle_k * k0) : k0;
      uint32_t k_round = RoundUp<BLOCK_SIZE_32>(k_actual);
      uint32_t k_part_loop = (k_actual + k_part_len - 1) / k_part_len;

      // --------- load whole A in l1a addr change -------------
      LocalTensor l1_buf_a =
          load_all_Amat_flag
              ? (l1_base_a[k_idx * m0 * k0 * sizeof(int8_t)])
              : (ping_flag ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN]);
      LocalTensor l1_buf_b =
          ping_flag ? l1_base_b : l1_base_b[b0mat_pingpong_buffer_len];
      auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

      if (k_idx < k_loop - 1) {
        uint64_t shuffle_k_next =
            en_shuffle_k ? (core_idx + k_idx + 1) % k_loop : k_idx + 1;

        offset_b_next = GetOffsetB(batch_idx, shuffle_k_next, n_idx);
        uint32_t k_actual_next =
            (shuffle_k_next == (k_loop - 1)) ? (k - shuffle_k_next * k0) : k0;
        uint32_t k_round_next = RoundUp<BLOCK_SIZE_32>(k_actual_next);

        LocalTensor l1_buf_a_next =
            load_all_Amat_flag
                ? l1_base_a
                : ((1 - ping_flag) ? l1_base_a
                                   : l1_base_a[L1_PINGPONG_BUFFER_LEN]);
        LocalTensor l1_buf_b_next =
            (1 - ping_flag) ? l1_base_b : l1_base_b[b0mat_pingpong_buffer_len];
        auto event_id_next = (1 - ping_flag) ? EVENT_ID0 : EVENT_ID1;

        WAIT_FLAG(MTE1, MTE2, event_id_next);
        if (!load_all_Amat_flag) {
          offset_a_next = GetOffsetA(batch_idx, m_idx, shuffle_k_next);
          CopyTileA(l1_buf_a_next,
                    gm_a[offset_a_next],
                    m_actual,
                    m_round,
                    k_actual_next,
                    k_round_next);
        }
        SET_FLAG(MTE2, MTE1, event_id_next);

        WAIT_FLAG(MTE1, MTE2, event_id_next + CONST_2);
        // 第二个权重矩阵预加载
        if (loop_idx != core_idx || k_idx != 0) {
          CopyTileB(l1_buf_b_next,
                    gm_b[offset_b_next],
                    k_actual_next,
                    k_round_next,
                    n_actual,
                    n_round);
        }
        SET_FLAG(MTE2, MTE1, event_id_next + CONST_2);
      }

      for (int k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
        uint32_t k0_round = (k_part_idx < k_part_loop - 1)
                                ? k_part_len
                                : k_round - k_part_idx * k_part_len;
        uint32_t k0_actual = (k_part_idx < k_part_loop - 1)
                                 ? k_part_len
                                 : k_actual - k_part_idx * k_part_len;

        auto mte1_mad_ping_flag = 1 - k_part_idx % 2;
        auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<InDtype> l0a_buf =
            l0a_base[(k_part_idx % 2) * L0_PINGPONG_BUFFER_LEN];
        AscendC::LocalTensor<InDtype> l0b_buf =
            l0b_base[(k_part_idx % 2) * L0_PINGPONG_BUFFER_LEN];

        // *** load matrix A from L1 to L0A
        if (k_part_idx == 0) {
          WAIT_FLAG(MTE2, MTE1, event_id);
        }
        WAIT_FLAG(M, MTE1, mte1_mad_event_id);
        if ((m == 1) || (m_actual == 1)) {
          l1_to_l0_a<Architecture::ASCEND_V220,
                     InDtype,
                     false,
                     DataFormat::VECTOR,
                     DataFormat::VECTOR>(
              l0a_buf,
              l1_buf_a[k_part_idx * k_part_len],
              0,                                        // mTileCeil
              CeilDiv<CUBE_MATRIX_SIZE_512>(k0_round),  // kPartCeil
              0,                                        // mSrcStride
              1,                                        // kSrcStride
              0,                                        // mDstStride
              0);                                       // kDstStride
        } else {
          LoadCbufToCa(l0a_buf,                                      // l0Tensor
                       l1_buf_a[k_part_idx * k_part_len * m_round],  // l1Tensor
                       m_round,                   // mTileCeil
                       k0_round,                  // kPartCeil
                       1,                         // mSrcStride
                       m_round / BLOCK_SIZE_16,   // kSrcStride
                       k0_round / BLOCK_SIZE_32,  // mDstStride
                       1);                        // kDstStride
        }
        if (k_part_idx == k_part_loop - 1) {
          SET_FLAG(MTE1, MTE2, event_id);
        }

        // *** load matrix B from L1 to L0B
        if (k_part_idx == 0) {
          WAIT_FLAG(MTE2, MTE1, event_id + CONST_2);
        }
        LoadCbufToCb(l0b_buf,                                      // l0Tensor
                     l1_buf_b[k_part_idx * k_part_len * n_round],  // l1Tensor
                     n_round,                                      // nTileCeil
                     k0_round,                                     // kPartCeil
                     1,                                            // nSrcStride
                     n_round / BLOCK_SIZE_16,                      // kSrcStride
                     1,                                            // nDstStride
                     k0_round / BLOCK_SIZE_32);                    // kDstStride
        if (k_part_idx == k_part_loop - 1) {
          SET_FLAG(MTE1, MTE2, event_id + CONST_2);
        }

        SET_FLAG(MTE1, M, mte1_mad_event_id);
        WAIT_FLAG(MTE1, M, mte1_mad_event_id);

        bool init_c = (k_idx == 0 && k_part_idx == 0);
        if (init_c) {
          WAIT_FLAG(FIX, M, EVENT_ID0);
        }
        Mmad(l0c_buf,
             l0a_buf,
             l0b_buf,
             m_actual,   // m
             n_actual,   // n
             k0_actual,  // k
             init_c);    // cmatrixInitVal
        PIPE_BARRIER(M);
        SET_FLAG(M, MTE1, mte1_mad_event_id);
      }

      ping_flag = 1 - ping_flag;
    }
    SET_FLAG(M, FIX, EVENT_ID0);
    WAIT_FLAG(M, FIX, EVENT_ID0);
    // copy from L0C to gm
    CopyCcToGm(gm_c[offset_c],  // dst
               l0c_buf,         // src
               m_actual,        // MSize
               n_actual,        // NSize
               m_round_16,      // srcStride
               n);              // dstStride_dst_D
    SET_FLAG(FIX, M, EVENT_ID0);
    if constexpr (!withSyncAll) {
      FftsCrossCoreSync<PIPE_FIX, SYNC_MODE>(MMAIC);
      if ((loop_idx / core_num + 1) % MAX_HW_SYNC_COUNTER == 0) {
        WaitFlagDev(MMAIV);
      }
    }
  }

  WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
  WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
  WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
  WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
  WAIT_FLAG(M, MTE1, EVENT_ID0);
  WAIT_FLAG(M, MTE1, EVENT_ID1);
  WAIT_FLAG(FIX, M, EVENT_ID0);
  WAIT_FLAG(FIX, MTE2, EVENT_ID0);
  WAIT_FLAG(MTE1, MTE2, EVENT_ID7);
}

#endif

#if defined(__DAV_C220_VEC__)

template <typename OutDtype, bool withSyncAll, QuantMode quantMode>
class PpMatmulW8a8Aiv {
  using InDtype = int32_t;
  using ScaleDtype = float;
  using BiasDtype = int32_t;

 public:
  __aicore__ PpMatmulW8a8Aiv() {};

  __aicore__ inline void Init(GM_ADDR gmInput,
                              GM_ADDR gmOutput,
                              GM_ADDR gmDescale,
                              GM_ADDR gmPerTensorBias,
                              GM_ADDR gmPertokenDescale,
                              const MlaPpMatmulTilingData& gmTilingData) {
    gmInput_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(gmInput));
    gmOutput_.SetGlobalBuffer(reinterpret_cast<__gm__ OutDtype*>(gmOutput));
    gmPerTensorScale_.SetGlobalBuffer(
        reinterpret_cast<__gm__ ScaleDtype*>(gmDescale));
    gmPerTensorBias_.SetGlobalBuffer(
        reinterpret_cast<__gm__ BiasDtype*>(gmPerTensorBias));
    gmPerTokenScale_.SetGlobalBuffer(
        reinterpret_cast<__gm__ ScaleDtype*>(gmPertokenDescale));

    batch_size = gmTilingData.numBatch;
    m = gmTilingData.m;
    k = gmTilingData.k;
    n = gmTilingData.n;
    m0 = gmTilingData.m0;
    k0 = gmTilingData.k0;
    n0 = gmTilingData.n0;
    m_loop = gmTilingData.mLoop;
    k_loop = gmTilingData.kLoop;
    n_loop = gmTilingData.nLoop;
    core_loop = gmTilingData.coreLoop;
    swizzle_cnt = gmTilingData.swizzleCount;
    swizzlDirect = gmTilingData.swizzleDirect;
    en_shuffle_k = gmTilingData.enShuffleK;

    AsdopsBuffer<Architecture::ASCEND_V220> buf;
    ubInput_ = buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(0);
    ubTempFp32_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(94 * 1024);
    ubOutput_ = buf.GetBuffer<BufferType::ASCEND_UB, OutDtype>(0);
    ubPerTensorScale_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(188 * 1024);
    block_size = BLOCK_SIZE_32;
    core_num = AscendC::GetBlockNum();
    core_idx = AscendC::GetBlockIdx() / 2;
    ping_flag = 1;
  }

  __aicore__ inline void GetBlockIdx(uint32_t index,
                                     uint32_t& m_idx,
                                     uint32_t& n_idx) {
    uint32_t in_batch_idx = index % (m_loop * n_loop);
    if (swizzlDirect == 0) {  // Zn
      uint32_t tile_block_loop = (m_loop + swizzle_cnt - 1) / swizzle_cnt;
      uint32_t tile_block_idx = in_batch_idx / (swizzle_cnt * n_loop);
      uint32_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * n_loop);

      uint32_t n_row = swizzle_cnt;
      if (tile_block_idx == tile_block_loop - 1) {
        n_row = m_loop - swizzle_cnt * tile_block_idx;
      }
      m_idx = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_row;
      n_idx = in_tile_block_idx / n_row;
      if (tile_block_idx % 2 != 0) {
        n_idx = n_loop - n_idx - 1;
      }
    } else {  // Nz
      uint32_t tile_block_loop = (n_loop + swizzle_cnt - 1) / swizzle_cnt;
      uint32_t tile_block_idx = in_batch_idx / (swizzle_cnt * m_loop);
      uint32_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * m_loop);

      uint32_t n_col = swizzle_cnt;
      if (tile_block_idx == tile_block_loop - 1) {
        n_col = n_loop - swizzle_cnt * tile_block_idx;
      }
      m_idx = in_tile_block_idx / n_col;
      n_idx = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_col;
      if (tile_block_idx % 2 != 0) {
        m_idx = m_loop - m_idx - 1;
      }
    }
  }

  __aicore__ inline void Process();

 private:
  AscendC::GlobalTensor<ScaleDtype> gmPerTensorScale_;
  AscendC::GlobalTensor<BiasDtype> gmPerTensorBias_;
  AscendC::GlobalTensor<ScaleDtype> gmPerTokenScale_;
  AscendC::GlobalTensor<InDtype> gmInput_;
  AscendC::GlobalTensor<OutDtype> gmOutput_;

  AscendC::LocalTensor<int32_t> ubInput_;
  AscendC::LocalTensor<float> ubTempFp32_;
  AscendC::LocalTensor<OutDtype> ubOutput_;
  AscendC::LocalTensor<float> ubPerTensorScale_;

  uint32_t core_num{0};
  uint32_t batch_size{0};
  uint32_t m{0};
  uint32_t k{0};
  uint32_t n{0};
  uint32_t m0{0};
  uint32_t k0{0};
  uint32_t n0{0};
  uint32_t m_loop{0};
  uint32_t n_loop{0};
  uint32_t k_loop{0};
  uint32_t core_loop{0};
  uint32_t core_idx{0};
  uint32_t ping_flag{0};
  uint32_t block_size{0};
  uint32_t cube_matrix_size{0};
  uint32_t swizzle_cnt{1};
  uint32_t en_shuffle_k{0};
  uint32_t swizzlDirect{0};
  uint64_t L1_PINGPONG_BUFFER_LEN{0};
  uint32_t L0AB_PINGPONG_BUFFER_LEN{0};
};

template <typename OutDtype, bool withSyncAll, QuantMode quantMode>
__aicore__ inline void
PpMatmulW8a8Aiv<OutDtype, withSyncAll, quantMode>::Process() {
  uint32_t m_idx = 0;
  uint32_t n_idx = 0;
  SET_FLAG(V, MTE2, EVENT_ID0);
  SET_FLAG(MTE3, V, EVENT_ID0);
  SET_FLAG(MTE3, MTE2, EVENT_ID0);
  for (uint32_t loop_idx = core_idx; loop_idx < core_loop;
       loop_idx += core_num) {
    GetBlockIdx(loop_idx, m_idx, n_idx);
    uint64_t batch_idx = loop_idx / n_loop / m_loop;
    uint64_t offsetC = batch_idx * m * n + m_idx * m0 * n + n_idx * n0;
    uint32_t m_actual = (m_idx == (m_loop - 1)) ? (m - m_idx * m0) : m0;
    uint32_t n_actual = (n_idx == (n_loop - 1)) ? (n - n_idx * n0) : n0;
    uint32_t m_round = RoundUp<CONST_8>(m_actual);
    uint32_t n_round = RoundUp<CONST_8>(n_actual);
    uint32_t n_round_16 = RoundUp<BLOCK_SIZE_16>(n_actual);
    uint32_t m_actual_per_vec = m_actual / AscendC::GetTaskRation();
    uint32_t m_offset = m + m_idx * m0;
    if (GetSubBlockidx() != 0) {
      offsetC += m_actual_per_vec * n;
      m_offset += m_actual_per_vec;
      m_actual_per_vec = m_actual - m_actual_per_vec;
    }

    if constexpr (!withSyncAll) {
      if (m_actual_per_vec == 0) {
        WaitFlagDev(MMAIC);
        if ((loop_idx / core_num + 1) % MAX_HW_SYNC_COUNTER == 1) {
          FftsCrossCoreSync<PIPE_MTE3, SYNC_MODE>(MMAIV);
        }
        continue;
      }
    }

    uint64_t offsetScale = batch_idx * n + n_idx * n0;
    bool aligned_s32 = ((n & 0b111) == 0);   // 32B aligned
    bool aligned_f16 = ((n & 0b1111) == 0);  // 32B aligned
    WAIT_FLAG(V, MTE2, EVENT_ID0);
    if constexpr (quantMode == QuantMode::PER_TENSOR_ASYMM_QUANT) {
      if (aligned_s32) {
        gm_to_ub<Architecture::ASCEND_V220, BiasDtype>(
            ubPerTensorScale_.ReinterpretCast<BiasDtype>(),
            gmPerTensorBias_[offsetScale],
            0,                                            // sid
            1,                                            // nBurst
            n_round * sizeof(BiasDtype) / BLOCK_SIZE_32,  // lenBurst
            0,                                            // srcStride
            0);                                           // dstStride
      } else {
        gm_to_ub_align<Architecture::ASCEND_V220, BiasDtype>(
            ubPerTensorScale_.ReinterpretCast<BiasDtype>(),
            gmPerTensorBias_[offsetScale],
            0,                         // sid
            1,                         // nBurst
            n_actual * sizeof(float),  // lenBurst
            0,                         // leftPaddingNum
            0,                         // rightPaddingNum
            0,                         // srcGap
            0);                        // dstGap
      }
    } else {
      if (aligned_s32) {
        gm_to_ub<Architecture::ASCEND_V220, float>(
            ubPerTensorScale_,
            gmPerTensorScale_[offsetScale],
            0,                            // sid
            1,                            // nBurst
            n_round * 4 / BLOCK_SIZE_32,  // lenBurst
            0,                            // srcStride
            0);                           // dstStride
      } else {
        gm_to_ub_align<Architecture::ASCEND_V220, float>(
            ubPerTensorScale_,
            gmPerTensorScale_[offsetScale],
            0,                         // sid
            1,                         // nBurst
            n_actual * sizeof(float),  // lenBurst
            0,                         // leftPaddingNum
            0,                         // rightPaddingNum
            0,                         // srcGap
            0);                        // dstGap
      }
    }

    if constexpr (!withSyncAll) {
      WaitFlagDev(MMAIC);
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    if (aligned_s32) {
      gm_to_ub<Architecture::ASCEND_V220, int32_t>(
          ubInput_,
          gmInput_[offsetC],
          0,                                                // sid
          m_actual_per_vec,                                 // nBurst
          n_round * sizeof(int32_t) / BLOCK_SIZE_32,        // lenBurst
          (n - n_round) * sizeof(int32_t) / BLOCK_SIZE_32,  // srcStride
          0                                                 // dstStride
      );
    } else {
      gm_to_ub_align<Architecture::ASCEND_V220, int32_t>(
          ubInput_,
          gmInput_[offsetC],
          0,                                 // sid
          m_actual_per_vec,                  // nBurst
          n_actual * sizeof(int32_t),        // lenBurst
          0,                                 // leftPaddingNum
          0,                                 // rightPaddingNum
          (n - n_actual) * sizeof(int32_t),  // srcGap
          0                                  // dstGap
      );
    }
    SET_FLAG(MTE2, V, EVENT_ID0);
    WAIT_FLAG(MTE2, V, EVENT_ID0);

    WAIT_FLAG(MTE3, V, EVENT_ID0);
    uint32_t nRepeatCnt = CeilDiv<CONST_64>(n_round);
    if constexpr (quantMode == QuantMode::PER_TENSOR_ASYMM_QUANT) {
      uint32_t rptStride = CeilDiv<CONST_32>(n_round * sizeof(BiasDtype));
      for (uint32_t i = 0; i < nRepeatCnt; ++i) {
        uint64_t mask =
            (i == nRepeatCnt - 1 ? (n_round - i * CONST_64) : CONST_64);
        AscendC::Add<BiasDtype, true>(
            ubInput_[i * CONST_64],  // dst
            ubInput_[i * CONST_64],  // src0
            ubPerTensorScale_
                .ReinterpretCast<BiasDtype>()[i * CONST_64],  // src1
            mask,                                             // mask
            m_actual_per_vec,                                 // repeat
            AscendC::BinaryRepeatParams(1,                    // dstBlockStride
                                        1,                    // src0BlockStride
                                        1,                    // src1BlockStride
                                        rptStride,            // dstRepeatStride
                                        rptStride,  // src0RepeatStride
                                        0));        // src1RepeatStride
        AscendC::PipeBarrier<PIPE_V>();
      }
      AscendC::SetVectorMask<uint8_t>((uint64_t)-1, (uint64_t)-1);
      SET_FLAG(V, MTE2, EVENT_ID0);
      WAIT_FLAG(V, MTE2, EVENT_ID0);
      if (aligned_s32) {
        gm_to_ub<Architecture::ASCEND_V220, ScaleDtype>(
            ubPerTensorScale_,
            gmPerTensorScale_[offsetScale],
            0,                                             // sid
            1,                                             // nBurst
            n_round * sizeof(ScaleDtype) / BLOCK_SIZE_32,  // lenBurst
            0,                                             // srcStride
            0                                              // dstStride
        );
      } else {
        gm_to_ub_align<Architecture::ASCEND_V220, ScaleDtype>(
            ubPerTensorScale_,
            gmPerTensorScale_[offsetScale],
            0,                              // sid
            1,                              // nBurst
            n_actual * sizeof(ScaleDtype),  // lenBurst
            0,                              // leftPaddingNum
            0,                              // rightPaddingNum
            0,                              // srcGap
            0                               // dstGap
        );
      }
      SET_FLAG(MTE2, V, EVENT_ID0);
      WAIT_FLAG(MTE2, V, EVENT_ID0);
    }

    // //  CASTF32 * f32 tf16
    constexpr uint32_t maxRepeat = 255;
    constexpr uint32_t perRepeatNum = maxRepeat * 64;
    uint32_t loopCnt =
        (m_actual_per_vec * n_actual + perRepeatNum - 1) / perRepeatNum;
    for (uint32_t i = 0; i < loopCnt; i++) {
      conv_v<Architecture::ASCEND_V220, int32_t, float>(
          ubInput_.ReinterpretCast<float>()[perRepeatNum * i],
          ubInput_[perRepeatNum * i],
          (uint8_t)maxRepeat,  // repeat
          (uint16_t)1,         // dstBlockStride
          (uint16_t)1,         // srcBlockStride
          (uint16_t)8,         // dstRepeatStride
          (uint16_t)8          // srcRepeatStride
      );
    }
    AscendC::PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < m_actual_per_vec; ++i) {
      mul_v<Architecture::ASCEND_V220, float>(
          ubTempFp32_[i * n_round],
          ubInput_.ReinterpretCast<float>()[i * n_round],
          ubPerTensorScale_.ReinterpretCast<float>(),
          (uint8_t)(nRepeatCnt),  // repeat
          (uint8_t)1,             // dstBlockStride
          (uint8_t)1,             // src0BlockStride
          (uint8_t)1,             // src1BlockStride
          (uint8_t)8,             // dstRepeatStride
          (uint8_t)8,             // src0RepeatStride
          (uint8_t)8              // src1RepeatStride
      );
      if constexpr (quantMode == QuantMode::PER_TOKEN_SYMM_QUANT) {
        AscendC::PipeBarrier<PIPE_V>();
        float perTokenDescale = gmPerTokenScale_.GetValue(m_offset + i);
        SET_FLAG(S, V, EVENT_ID0);
        WAIT_FLAG(S, V, EVENT_ID0);
        AscendC::Muls(ubTempFp32_[i * n_round],
                      ubTempFp32_[i * n_round],
                      perTokenDescale,
                      n_round);
      }
      AscendC::PipeBarrier<PIPE_V>();
    }
    SET_FLAG(V, MTE2, EVENT_ID0);
    AscendC::PipeBarrier<PIPE_V>();
    if (n_actual % 16 > 8) {
      for (uint32_t i = 0; i < loopCnt; i++) {
        if constexpr (std::is_same_v<OutDtype, bfloat16_t>) {
          convr_v<Architecture::ASCEND_V220, float, OutDtype>(
              ubOutput_[perRepeatNum * i],
              ubTempFp32_[perRepeatNum * i],
              (uint8_t)maxRepeat,  // repeat
              (uint16_t)1,         // dstBlockStride
              (uint16_t)1,         // srcBlockStride
              (uint16_t)4,         // dstRepeatStride
              (uint16_t)8);        // srcRepeatStride
        } else {
          conv_v<Architecture::ASCEND_V220, float, OutDtype>(
              ubOutput_[perRepeatNum * i],
              ubTempFp32_[perRepeatNum * i],
              (uint8_t)maxRepeat,  // repeat
              (uint16_t)1,         // dstBlockStride
              (uint16_t)1,         // srcBlockStride
              (uint16_t)4,         // dstRepeatStride
              (uint16_t)8);        // srcRepeatStride
        }
      }
    } else {
      if constexpr (std::is_same_v<OutDtype, __bf16>) {
        AscendC::Cast<OutDtype, float>(
            ubOutput_,                      // dst
            ubTempFp32_,                    // src
            AscendC::RoundMode::CAST_RINT,  // roundMode
            m_actual_per_vec * n_round);    // count
      } else {
        AscendC::Cast<OutDtype, float>(
            ubOutput_,                      // dst
            ubTempFp32_,                    // src
            AscendC::RoundMode::CAST_NONE,  // roundMode
            m_actual_per_vec * n_round);    // count
      }
    }
    SET_FLAG(V, MTE3, EVENT_ID0);
    WAIT_FLAG(V, MTE3, EVENT_ID0);
    if (aligned_f16) {
      ub_to_gm<Architecture::ASCEND_V220, OutDtype>(
          gmOutput_[offsetC],
          ubOutput_,
          0,
          m_actual_per_vec,                                 // nBurst
          n_round * sizeof(uint16_t) / BLOCK_SIZE_32,       // lenBurst
          0,                                                // srcStride
          (n - n_round) * sizeof(uint16_t) / BLOCK_SIZE_32  // dstStride
      );
    } else {
      ub_to_gm_align<Architecture::ASCEND_V220, OutDtype>(
          gmOutput_[offsetC],
          ubOutput_,
          0,
          m_actual_per_vec,                  // nBurst
          n_actual * sizeof(OutDtype),       // lenBurst
          0,                                 // leftPaddingNum
          0,                                 // rightPaddingNum
          0,                                 // srcGap
          (n - n_actual) * sizeof(OutDtype)  // dstGap
      );
    }
    SET_FLAG(MTE3, V, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    if constexpr (!withSyncAll) {
      if ((loop_idx / core_num + 1) % MAX_HW_SYNC_COUNTER == 1) {
        FftsCrossCoreSync<PIPE_MTE3, SYNC_MODE>(MMAIV);
      }
    }
  }
  WAIT_FLAG(V, MTE2, EVENT_ID0);
  WAIT_FLAG(MTE3, V, EVENT_ID0);
  WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
}
#endif

template <typename InDtype,
          int8_t CACHE_MODE,
          DataFormat weightFormat1 = DataFormat::NZ,
          DataFormat weightFormat2 = DataFormat::NZ,
          DataFormat weightFormat3 = DataFormat::ND,
          QuantMode quantMode = QuantMode::PER_TENSOR_ASYMM_QUANT>
class MLAOperation {
  static constexpr bool mm1WithSyncAll =
      (quantMode == QuantMode::PER_TOKEN_SYMM_QUANT);
  static constexpr uint64_t splitGapC =
      CACHE_MODE == CACHE_MODE_KVCACHE ? CONST_64 : CONST_0;
  using Q_OUT_DTYPE = typename std::
      conditional_t<CACHE_MODE == CACHE_MODE_INT8_NZCACHE, int8_t, InDtype>;
  using K_NOPE_DTYPE = typename std::
      conditional_t<CACHE_MODE == CACHE_MODE_INT8_NZCACHE, int8_t, InDtype>;

 public:
  __aicore__ inline MLAOperation(const MlaTilingData& mlaParams_) {
    blockIdx = AscendC::GetBlockIdx();
#ifdef __DAV_C220_VEC__
    sub_block_idx = static_cast<uint64_t>(GetSubBlockidx());
#endif
    vectorBlockIdx = (blockIdx / 2) * 2 + sub_block_idx;
    this->n = mlaParams_.n;
    this->num_core_ = mlaParams_.rmsNumCore1;
    this->num_col_1 = mlaParams_.rmsNumCol1;
    this->num_col_2 = mlaParams_.rmsNumCol2;
    this->num_row = mlaParams_.n;
    this->epsilon_ = 1e-6;
    this->hiddten_state = mlaParams_.hiddtenState;
    this->q_down_out_flag = mlaParams_.qDownOutFlag;
    this->mlaParams = mlaParams_;
  }

  __aicore__ inline void Init(GM_ADDR hiddenStateGm,
                              GM_ADDR gamma1Gm,
                              GM_ADDR beta1Gm,
                              GM_ADDR quantScale1Gm,
                              GM_ADDR quantOffset1Gm,
                              GM_ADDR wdqkvGm,
                              GM_ADDR bias1Gm,
                              GM_ADDR gamma2Gm,
                              GM_ADDR beta2Gm,
                              GM_ADDR quantScale2Gm,
                              GM_ADDR quantOffset2Gm,
                              GM_ADDR gamma3Gm,
                              GM_ADDR sin1Gm,
                              GM_ADDR cos1Gm,
                              GM_ADDR sin2Gm,
                              GM_ADDR cos2Gm,
                              GM_ADDR keycacheGm,
                              GM_ADDR slotMappingGm,
                              GM_ADDR wuqGm,
                              GM_ADDR bias2Gm,
                              GM_ADDR wukGm,
                              GM_ADDR descale1Gm,
                              GM_ADDR descale2Gm,
                              GM_ADDR gmCtkvScale,
                              GM_ADDR gmQnopeScale,
                              GM_ADDR qGm,
                              GM_ADDR keycacheOutGm,
                              GM_ADDR qGm2,
                              GM_ADDR keycacheOutGm2,
                              GM_ADDR s1Gm,
                              GM_ADDR s2Gm,
                              GM_ADDR s3Gm,
                              GM_ADDR s4Gm,
                              GM_ADDR s5Gm,
                              GM_ADDR qDownGm) {
    quantScale3GmTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ InDtype*>(gmCtkvScale));
    gamma3GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(gamma3Gm));
    sin1GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(sin1Gm));
    cos1GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(cos1Gm));
    keycacheGmTensor1.SetGlobalBuffer(
        reinterpret_cast<__gm__ K_NOPE_DTYPE*>(keycacheOutGm));
    keycacheGmTensor2.SetGlobalBuffer(
        reinterpret_cast<__gm__ InDtype*>(keycacheOutGm2));
    slotMappingGmTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(slotMappingGm));
    descale1gmTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(descale1Gm));
    s2GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(s2Gm));
    s3GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(s3Gm));
    s5GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(s5Gm));
    this->q_down_out_flag &= (qDownGm != nullptr);

#ifdef __DAV_C220_CUBE__
    mm_w8a8_aic_1.Init(s1Gm, wdqkvGm, s2Gm, mlaParams.mm1, 0);
    mm_w8a8_aic_1.PreloadWeight();
    if constexpr (quantMode == QuantMode::PER_TENSOR_ASYMM_QUANT) {
      mm_w8a8_aic_2.Init(s1Gm, wuqGm, s2Gm, mlaParams.mm2, 1);
    } else {
      // quantMode == QuantMode::PER_TOKEN_SYMM_QUANT
      mm_w8a8_aic_2.Init(s1Gm, wuqGm, s3Gm, mlaParams.mm2, 1);
    }
    if constexpr (CACHE_MODE == CACHE_MODE_INT8_NZCACHE) {
      mm_ein_sum.Init(s4Gm, wukGm, s1Gm, mlaParams);
    } else {
      mm_ein_sum.Init(s4Gm, wukGm, qGm, mlaParams);
    }
#endif

    hiddenStateGmTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ InDtype*>(hiddenStateGm));
    gamma1GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(gamma1Gm));
    quantScale1GmTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ InDtype*>(quantScale1Gm));
    quantOffset1GmTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int8_t*>(quantOffset1Gm));
    wdqkvGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(wdqkvGm));
    gamma2GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(gamma2Gm));
    quantScale2GmTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ InDtype*>(quantScale2Gm));
    quantOffset2GmTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int8_t*>(quantOffset2Gm));
    sin2GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(sin2Gm));
    cos2GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(cos2Gm));
    wuqGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(wuqGm));
    wukGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(wukGm));
    descale2gmTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(descale2Gm));
    s1GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(s1Gm));
    s4GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(s4Gm));
    qGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ Q_OUT_DTYPE*>(qGm));
    qGmTensor2.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(qGm2));
    bias1gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(bias1Gm));
    bias2gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(bias2Gm));
    beta1GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(beta1Gm));
    beta2GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(beta2Gm));
    if (q_down_out_flag) {
      qDownGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(qDownGm));
    }

#ifdef __DAV_C220_VEC__
    if constexpr (quantMode == QuantMode::PER_TENSOR_ASYMM_QUANT) {
      mm_w8a8_aiv_1.Init(s2Gm, s3Gm, descale1Gm, bias1Gm, s5Gm, mlaParams.mm1);
      mm_w8a8_aiv_2.Init(s2Gm, s4Gm, descale2Gm, bias2Gm, s5Gm, mlaParams.mm2);
    } else {
      // quantMode == QuantMode::PER_TOKEN_SYMM_QUANT
      mm_w8a8_aiv_2.Init(s3Gm, s4Gm, descale2Gm, bias2Gm, s5Gm, mlaParams.mm2);
    }
    row_work = (num_row + num_core_ - 1) / num_core_;
    row_work_ = 0;
    uint32_t need_core = (num_row + row_work - 1) / row_work;
    if (vectorBlockIdx < need_core - 1) {
      row_work_ = row_work;
    } else if (vectorBlockIdx == need_core - 1) {
      row_work_ = num_row - (need_core - 1) * row_work;
    } else {
      row_work_ = 0;
    }

    float avg_factor = float(1.0) / num_col_1;
    if (mlaParams.doRmsNorm) {
      rmsNormQuant1.Init(
          gamma1GmTensor,
          beta1GmTensor,
          quantScale1GmTensor,
          quantOffset1GmTensor,
          s5Gm + row_work * vectorBlockIdx * sizeof(float),
          descale1Gm,
          hiddenStateGm,
          s1Gm,
          0,
          num_col_1,
          avg_factor,
          vectorBlockIdx * static_cast<uint64_t>(row_work) * num_col_1,
          vectorBlockIdx * static_cast<uint64_t>(row_work) * num_col_1,
          row_work_,
          mlaParams);
    } else {
      quant.Init(quantScale1GmTensor,
                 quantOffset1GmTensor,
                 s5Gm + row_work * vectorBlockIdx * sizeof(float),
                 descale1Gm,
                 hiddenStateGm,
                 s1Gm,
                 0,
                 num_col_1,
                 vectorBlockIdx * static_cast<uint64_t>(row_work) * num_col_1,
                 vectorBlockIdx * static_cast<uint64_t>(row_work) * num_col_1,
                 row_work_,
                 mlaParams);
    }
    if constexpr (quantMode == QuantMode::PER_TENSOR_ASYMM_QUANT) {
      if (q_down_out_flag) {
        rmsNormQuant2QDownOut.Init(
            gamma2GmTensor,
            beta2GmTensor,
            quantScale2GmTensor,
            quantOffset2GmTensor,
            s5Gm + row_work * vectorBlockIdx * sizeof(float),
            descale1Gm,
            s3Gm,
            s1Gm,
            SPLIT_SIZE_ONE,
            num_col_2,
            1 / SCALE_FACTOR,
            vectorBlockIdx * static_cast<uint64_t>(row_work) * num_col_2,
            vectorBlockIdx * static_cast<uint64_t>(row_work) * SPLIT_SIZE_TWO,
            row_work_,
            mlaParams,
            qDownGmTensor);
      } else {
        rmsNormQuant2.Init(
            gamma2GmTensor,
            beta2GmTensor,
            quantScale2GmTensor,
            quantOffset2GmTensor,
            s5Gm + row_work * vectorBlockIdx * sizeof(float),
            descale1Gm,
            s3Gm,
            s1Gm,
            SPLIT_SIZE_ONE,
            num_col_2,
            1 / SCALE_FACTOR,
            vectorBlockIdx * static_cast<uint64_t>(row_work) * num_col_2,
            vectorBlockIdx * static_cast<uint64_t>(row_work) * SPLIT_SIZE_TWO,
            row_work_,
            mlaParams);
      }
    } else {
      // quantMode == QuantMode::PER_TOKEN_SYMM_QUANT
      if (q_down_out_flag) {
        rmsNormQuant2QDownOut.Init(
            gamma2GmTensor,
            beta2GmTensor,
            quantScale2GmTensor,
            quantOffset2GmTensor,
            s5Gm + row_work * vectorBlockIdx * sizeof(float),
            descale1Gm,
            s3Gm,
            s1Gm,
            SPLIT_SIZE_ONE,
            num_col_2,
            1 / SCALE_FACTOR,
            vectorBlockIdx * static_cast<uint64_t>(row_work) * num_col_2,
            vectorBlockIdx * static_cast<uint64_t>(row_work) * SPLIT_SIZE_TWO,
            row_work_,
            mlaParams,
            qDownGmTensor);
      } else {
        rmsNormQuant2.Init(
            gamma2GmTensor,
            beta2GmTensor,
            quantScale2GmTensor,
            quantOffset2GmTensor,
            s5Gm + row_work * vectorBlockIdx * sizeof(float),
            descale1Gm,
            s2Gm,
            s1Gm,
            SPLIT_SIZE_ONE,
            num_col_2,
            1 / SCALE_FACTOR,
            vectorBlockIdx * static_cast<uint64_t>(row_work) * num_col_2,
            vectorBlockIdx * static_cast<uint64_t>(row_work) * SPLIT_SIZE_TWO,
            row_work_,
            mlaParams);
      }
    }
    ropeFp16.RopeInit(
        s4Gm, cos2GmTensor, sin2GmTensor, qGmTensor, qGmTensor2, mlaParams);
    einSumQuant.Init(s1Gm, gmQnopeScale, qGm, mlaParams);
#endif
  }

  __aicore__ inline void ProcessCube();

  __aicore__ inline void ProcessVector();

 private:
  constexpr static uint32_t C0_SIZE = 16;
  constexpr static uint32_t I8_C0_SIZE = 32;

  template <class T1>
  __aicore__ inline void RmsNormAndRopeConvergence1(
      const AscendC::LocalTensor<T1>& srcTensor,
      const AscendC::LocalTensor<T1>& gammaTensor,
      const AscendC::LocalTensor<T1>& sinTensor,
      const AscendC::LocalTensor<T1>& cosTensor,
      const AscendC::LocalTensor<int32_t>& slotMappingTensor,
      const uint32_t sN,
      const AscendC::LocalTensor<float>& rmsNormTensor,
      const AscendC::LocalTensor<float>& gammaFp32,
      const AscendC::LocalTensor<float>& ropeKTensor,
      const AscendC::LocalTensor<float>& ropeKRevertTensor,
      const AscendC::LocalTensor<float>& calTensor,
      const AscendC::LocalTensor<T1>& outTmpTensor,
      AscendC::LocalTensor<half>& tmpfp16,
      AscendC::LocalTensor<int8_t>& int8OutTensor,
      float quantScale3) {
    int64_t slotMapGmOffset = vectorBlockIdx * row_work;
    AscendC::DataCopy(gammaTensor, gamma3GmTensor, SPLIT_RMSNRORM_SIZE_ONE);
    SET_FLAG(MTE2, V, EVENT_ID1);
    WAIT_FLAG(MTE2, V, EVENT_ID1);
    Cast(gammaFp32,
         gammaTensor,
         AscendC::RoundMode::CAST_NONE,
         SPLIT_RMSNRORM_SIZE_ONE);
    AscendC::DataCopyPad(
        slotMappingTensor,
        slotMappingGmTensor[slotMapGmOffset],
        AscendC::DataCopyExtParams(1, sN * sizeof(int32_t), 0, 0, 0),
        AscendC::DataCopyPadExtParams<int32_t>(false, 0, 8 - sN % 8, 0));
    if constexpr (quantMode == QuantMode::PER_TOKEN_SYMM_QUANT) {
      mmTensor = calTensor.ReinterpretCast<int32_t>()[SPLIT_SIZE_ONE];
      deScaleTensor = calTensor.ReinterpretCast<float>()[SPLIT_SIZE_ONE * 2];
      AscendC::DataCopy(
          deScaleTensor,
          descale1gmTensor,
          AscendC::DataCopyParams(
              1, SPLIT_SIZE_ONE * sizeof(int32_t) / BLOCK_SIZE_32, 0, 0));
    }
    SET_FLAG(MTE2, V, EVENT_ID2);
    WAIT_FLAG(MTE2, V, EVENT_ID2);
    SET_FLAG(MTE2, S, EVENT_ID2);
    WAIT_FLAG(MTE2, S, EVENT_ID2);
    for (uint64_t loop = 0; loop < sN; ++loop) {
      uint64_t offset =
          vectorBlockIdx * static_cast<uint64_t>(row_work) * num_col_2 +
          loop * MM1_OUT_SIZE;
      int64_t slotValue =
          static_cast<int64_t>(slotMappingTensor.GetValue(loop));
      if (slotValue == -1) {
        continue;
      }
      if constexpr (quantMode == QuantMode::PER_TENSOR_ASYMM_QUANT) {
        AscendC::DataCopy(
            srcTensor,
            s3GmTensor[offset],
            AscendC::DataCopyParams(1, MM1_OUT_SIZE / BLOCK_SIZE_16, 0, 0));
      } else {
        // quantMode == QuantMode::PER_TOKEN_SYMM_QUANT
        AscendC::DataCopy(
            mmTensor,
            s2GmTensor[offset],
            AscendC::DataCopyParams(
                1, SPLIT_SIZE_ONE * sizeof(int32_t) / BLOCK_SIZE_32, 0, 0));
      }
      AscendC::DataCopy(sinTensor,
                        sin1GmTensor[(row_work * vectorBlockIdx + loop) *
                                     SPLIT_RMSNRORM_SIZE_TWO],
                        SPLIT_RMSNRORM_SIZE_TWO);
      AscendC::DataCopy(cosTensor,
                        cos1GmTensor[(row_work * vectorBlockIdx + loop) *
                                     SPLIT_RMSNRORM_SIZE_TWO],
                        SPLIT_RMSNRORM_SIZE_TWO);
      SET_FLAG(MTE2, V, EVENT_ID0);
      // ND
      uint64_t cacheStart = static_cast<uint64_t>(slotValue) *
                            static_cast<uint64_t>(SPLIT_SIZE_ONE);
      uint64_t cacheStart1 = static_cast<uint64_t>(slotValue) *
                             static_cast<uint64_t>(SPLIT_RMSNRORM_SIZE_ONE);
      uint64_t cacheStart2 = static_cast<uint64_t>(slotValue) *
                             static_cast<uint64_t>(SPLIT_RMSNRORM_SIZE_TWO);
      // NZ
      uint32_t outer_idx = slotValue / 128;
      uint32_t inner_idx = slotValue % 128;

      SET_FLAG(S, MTE3, EVENT_ID0);
      /* RmsNorm start */
      WAIT_FLAG(MTE2, V, EVENT_ID0);
      if constexpr (quantMode == QuantMode::PER_TOKEN_SYMM_QUANT) {
        /* DeQuant */
        AscendC::Cast(mmTensor.ReinterpretCast<float>(),
                      mmTensor,
                      AscendC::RoundMode::CAST_NONE,
                      SPLIT_SIZE_ONE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(mmTensor.ReinterpretCast<float>(),
                     mmTensor.ReinterpretCast<float>(),
                     deScaleTensor,
                     SPLIT_SIZE_ONE);
        AscendC::PipeBarrier<PIPE_V>();
        float perTokenDescale =
            s5GmTensor.GetValue(row_work * vectorBlockIdx + loop);
        SET_FLAG(S, V, EVENT_ID0);
        WAIT_FLAG(S, V, EVENT_ID0);
        AscendC::Muls(mmTensor.ReinterpretCast<float>(),
                      mmTensor.ReinterpretCast<float>(),
                      perTokenDescale,
                      SPLIT_SIZE_ONE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(srcTensor,
                      mmTensor.ReinterpretCast<float>(),
                      AscendC::RoundMode::CAST_RINT,
                      SPLIT_SIZE_ONE);
        AscendC::PipeBarrier<PIPE_V>();
      }
      Cast(rmsNormTensor,
           srcTensor,
           AscendC::RoundMode::CAST_NONE,
           SPLIT_RMSNRORM_SIZE_ONE);
      AscendC::PipeBarrier<PIPE_V>();
      Mul(calTensor, rmsNormTensor, rmsNormTensor, SPLIT_RMSNRORM_SIZE_ONE);
      AscendC::PipeBarrier<PIPE_V>();
      ReduceSumCustom(calTensor[SPLIT_RMSNRORM_SIZE_ONE],
                      calTensor,
                      calTensor[SPLIT_RMSNRORM_SIZE_ONE * 2],
                      SPLIT_RMSNRORM_SIZE_ONE);
      SET_FLAG(V, S, EVENT_ID1);
      WAIT_FLAG(V, S, EVENT_ID1);
      float rms = sqrt(calTensor.GetValue(SPLIT_RMSNRORM_SIZE_ONE) /
                           SPLIT_RMSNRORM_SIZE_ONE +
                       epsilon_);
      SET_FLAG(S, V, EVENT_ID1);
      WAIT_FLAG(S, V, EVENT_ID1);
      AscendC::PipeBarrier<PIPE_V>();
      Duplicate(calTensor, rms, SPLIT_RMSNRORM_SIZE_ONE);
      AscendC::PipeBarrier<PIPE_V>();
      Div(calTensor, rmsNormTensor, calTensor, SPLIT_RMSNRORM_SIZE_ONE);
      AscendC::PipeBarrier<PIPE_V>();
      Mul(rmsNormTensor, gammaFp32, calTensor, SPLIT_RMSNRORM_SIZE_ONE);
      AscendC::PipeBarrier<PIPE_V>();
      if constexpr (CACHE_MODE == CACHE_MODE_INT8_NZCACHE) {
        // quant
        Muls(
            rmsNormTensor, rmsNormTensor, quantScale3, SPLIT_RMSNRORM_SIZE_ONE);
        AscendC::PipeBarrier<PIPE_V>();
        CastFrom32To16(tmpfp16, rmsNormTensor, SPLIT_RMSNRORM_SIZE_ONE);
        AscendC::PipeBarrier<PIPE_V>();
        CastFromF16ToI8(int8OutTensor, tmpfp16, -128, SPLIT_RMSNRORM_SIZE_ONE);
        AscendC::PipeBarrier<PIPE_V>();
      } else {
        AscendC::PipeBarrier<PIPE_V>();
        if (std::is_same<T1, bfloat16_t>::value) {
          Cast(outTmpTensor,
               rmsNormTensor,
               AscendC::RoundMode::CAST_RINT,
               SPLIT_RMSNRORM_SIZE_ONE);
        } else {
          Cast(outTmpTensor,
               rmsNormTensor,
               AscendC::RoundMode::CAST_NONE,
               SPLIT_RMSNRORM_SIZE_ONE);
        }
      }
      /* RmsNorm end */
      /* Rope K start */
      uint64_t revertOffset = SPLIT_RMSNRORM_SIZE_TWO / 2;
      Cast(ropeKTensor,
           srcTensor[SPLIT_RMSNRORM_SIZE_ONE],
           AscendC::RoundMode::CAST_NONE,
           SPLIT_RMSNRORM_SIZE_TWO);
      Cast(ropeKRevertTensor[revertOffset],
           srcTensor[SPLIT_RMSNRORM_SIZE_ONE],
           AscendC::RoundMode::CAST_NONE,
           revertOffset);
      Cast(ropeKRevertTensor,
           srcTensor[SPLIT_RMSNRORM_SIZE_ONE + revertOffset],
           AscendC::RoundMode::CAST_NONE,
           revertOffset);
      Duplicate(calTensor, static_cast<float>(-1), revertOffset);
      Duplicate(calTensor[revertOffset], static_cast<float>(1), revertOffset);
      AscendC::PipeBarrier<PIPE_V>();
      Cast(calTensor[SPLIT_RMSNRORM_SIZE_TWO],
           cosTensor,
           AscendC::RoundMode::CAST_NONE,
           SPLIT_RMSNRORM_SIZE_TWO);
      Cast(calTensor[SPLIT_RMSNRORM_SIZE_TWO * 2],
           sinTensor,
           AscendC::RoundMode::CAST_NONE,
           SPLIT_RMSNRORM_SIZE_TWO);
      AscendC::PipeBarrier<PIPE_V>();
      Mul(ropeKTensor,
          calTensor[SPLIT_RMSNRORM_SIZE_TWO],
          ropeKTensor,
          SPLIT_RMSNRORM_SIZE_TWO);
      Mul(ropeKRevertTensor,
          calTensor[SPLIT_RMSNRORM_SIZE_TWO * 2],
          ropeKRevertTensor,
          SPLIT_RMSNRORM_SIZE_TWO);
      AscendC::PipeBarrier<PIPE_V>();
      Mul(ropeKRevertTensor,
          calTensor,
          ropeKRevertTensor,
          SPLIT_RMSNRORM_SIZE_TWO);
      AscendC::PipeBarrier<PIPE_V>();
      Add(ropeKRevertTensor,
          ropeKTensor,
          ropeKRevertTensor,
          SPLIT_RMSNRORM_SIZE_TWO);
      AscendC::PipeBarrier<PIPE_V>();
      if (std::is_same<T1, bfloat16_t>::value) {
        Cast(outTmpTensor[SPLIT_RMSNRORM_SIZE_ONE],
             ropeKRevertTensor,
             AscendC::RoundMode::CAST_RINT,
             SPLIT_RMSNRORM_SIZE_TWO);
      } else {
        Cast(outTmpTensor[SPLIT_RMSNRORM_SIZE_ONE],
             ropeKRevertTensor,
             AscendC::RoundMode::CAST_NONE,
             SPLIT_RMSNRORM_SIZE_TWO);
      }
      AscendC::PipeBarrier<PIPE_V>();
      /* Rope K end */
      SET_FLAG(V, MTE3, EVENT_ID0);
      WAIT_FLAG(V, MTE3, EVENT_ID0);
      WAIT_FLAG(S, MTE3, EVENT_ID0);
      if constexpr (CACHE_MODE == CACHE_MODE_KVCACHE) {
        DataCopy(keycacheGmTensor1[cacheStart], outTmpTensor, SPLIT_SIZE_ONE);
      } else if constexpr (CACHE_MODE == CACHE_MODE_INT8_NZCACHE) {
        uint64_t cacheSatartI8Nz1 =
            outer_idx * 128 * 512 + inner_idx * I8_C0_SIZE;
        uint64_t cacheSatartNz2 = outer_idx * 128 * 64 + inner_idx * C0_SIZE;
        // nope:int8 nz
        AscendC::DataCopyExtParams outExt;
        outExt.blockCount = SPLIT_RMSNRORM_SIZE_ONE / I8_C0_SIZE;
        outExt.blockLen = I8_C0_SIZE * sizeof(int8_t);
        outExt.srcStride = 0;
        outExt.dstStride = (128 * I8_C0_SIZE - I8_C0_SIZE) * sizeof(int8_t);
        DataCopyPad(keycacheGmTensor1[cacheSatartI8Nz1], int8OutTensor, outExt);
        // rope:T1 nz
        outExt.blockCount = SPLIT_RMSNRORM_SIZE_TWO / C0_SIZE;
        outExt.blockLen = C0_SIZE * sizeof(T1);
        outExt.srcStride = 0;
        outExt.dstStride = (128 * C0_SIZE - C0_SIZE) * sizeof(T1);
        DataCopyPad(keycacheGmTensor2[cacheSatartNz2],
                    outTmpTensor[SPLIT_RMSNRORM_SIZE_ONE],
                    outExt);
      } else if constexpr (CACHE_MODE == CACHE_MODE_NZCACHE) {
        uint64_t cacheSatartNz1 = outer_idx * 128 * 512 + inner_idx * C0_SIZE;
        uint64_t cacheSatartNz2 = outer_idx * 128 * 64 + inner_idx * C0_SIZE;
        // nope:T1 nz
        AscendC::DataCopyExtParams outExt;
        outExt.blockCount = SPLIT_RMSNRORM_SIZE_ONE / C0_SIZE;
        outExt.blockLen = C0_SIZE * sizeof(T1);
        outExt.srcStride = 0;
        outExt.dstStride = (128 * C0_SIZE - C0_SIZE) * sizeof(T1);
        DataCopyPad(keycacheGmTensor1[cacheSatartNz1], outTmpTensor, outExt);
        // rope:T1 nz
        outExt.blockCount = SPLIT_RMSNRORM_SIZE_TWO / C0_SIZE;
        outExt.blockLen = C0_SIZE * sizeof(T1);
        outExt.srcStride = 0;
        outExt.dstStride = (128 * C0_SIZE - C0_SIZE) * sizeof(T1);
        DataCopyPad(keycacheGmTensor2[cacheSatartNz2],
                    outTmpTensor[SPLIT_RMSNRORM_SIZE_ONE],
                    outExt);
      } else {
        // keycache1
        DataCopy(keycacheGmTensor1[cacheStart1],
                 outTmpTensor,
                 SPLIT_RMSNRORM_SIZE_ONE);
        // keycache2
        DataCopy(keycacheGmTensor2[cacheStart2],
                 outTmpTensor[SPLIT_RMSNRORM_SIZE_ONE],
                 SPLIT_RMSNRORM_SIZE_TWO);
      }
      SET_FLAG(MTE3, MTE2, EVENT_ID1);
      WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
    }
  }

 private:
  uint32_t n;
  uint32_t rotaryCoeff;
  uint32_t blockIdx;
  uint32_t sub_block_idx;
  uint32_t vectorBlockIdx;
  uint32_t blockOffset;
  uint32_t perTaskNum;
  uint32_t resTaskNum;
  MlaTilingData mlaParams;

  uint32_t num_core_;
  uint32_t num_col_1;
  uint32_t num_col_2;
  float epsilon_;
  uint32_t num_row;
  uint32_t quantMin_;
  uint32_t row_work;
  uint32_t row_work_;
  uint32_t hiddten_state;
  bool q_down_out_flag;

  AsdopsBuffer<Architecture::ASCEND_V220> buf;
  AscendC::LocalTensor<int32_t> mmTensor;
  AscendC::LocalTensor<float> deScaleTensor;

  AscendC::GlobalTensor<InDtype> hiddenStateGmTensor;

  AscendC::GlobalTensor<InDtype> gamma1GmTensor;
  AscendC::GlobalTensor<InDtype> quantScale1GmTensor;
  AscendC::GlobalTensor<int8_t> quantOffset1GmTensor;

  AscendC::GlobalTensor<int8_t> wdqkvGmTensor;
  AscendC::GlobalTensor<InDtype> gamma2GmTensor;
  AscendC::GlobalTensor<InDtype> quantScale2GmTensor;
  AscendC::GlobalTensor<InDtype> quantScale3GmTensor;
  AscendC::GlobalTensor<int8_t> quantOffset2GmTensor;
  AscendC::GlobalTensor<InDtype> gamma3GmTensor;
  AscendC::GlobalTensor<InDtype> sin1GmTensor;
  AscendC::GlobalTensor<InDtype> cos1GmTensor;
  AscendC::GlobalTensor<InDtype> sin2GmTensor;
  AscendC::GlobalTensor<InDtype> cos2GmTensor;
  AscendC::GlobalTensor<K_NOPE_DTYPE> keycacheGmTensor1;
  AscendC::GlobalTensor<InDtype> keycacheGmTensor2;
  AscendC::GlobalTensor<int32_t> slotMappingGmTensor;
  AscendC::GlobalTensor<int8_t> wuqGmTensor;
  AscendC::GlobalTensor<InDtype> wukGmTensor;
  AscendC::GlobalTensor<InDtype> qDownGmTensor;

  // cachemode2-->int8; else bf16
  AscendC::GlobalTensor<Q_OUT_DTYPE> qGmTensor;
  AscendC::GlobalTensor<InDtype> qGmTensor2;
  AscendC::GlobalTensor<int8_t> s1GmTensor;
  AscendC::GlobalTensor<int32_t> s2GmTensor;
  AscendC::GlobalTensor<InDtype> s3GmTensor;
  AscendC::GlobalTensor<int32_t> s4GmTensor;
  AscendC::GlobalTensor<float> s5GmTensor;
  AscendC::GlobalTensor<float> descale1gmTensor;
  AscendC::GlobalTensor<float> descale2gmTensor;
  AscendC::GlobalTensor<InDtype> beta1GmTensor;
  AscendC::GlobalTensor<InDtype> beta2GmTensor;

  AscendC::GlobalTensor<int32_t> bias1gmTensor;
  AscendC::GlobalTensor<int32_t> bias2gmTensor;

#ifdef __DAV_C220_CUBE__
  PpMatmulW8a8Aic<mm1WithSyncAll, 0, DataFormat::ND, weightFormat1>
      mm_w8a8_aic_1;
  PpMatmulW8a8Aic<false, 0, DataFormat::ND, weightFormat2> mm_w8a8_aic_2;
  PpMatmulEinSum<InDtype, InDtype, weightFormat3, false, 0, CONST_64, splitGapC>
      mm_ein_sum;
#endif

#ifdef __DAV_C220_VEC__
  PpMatmulW8a8Aiv<InDtype, mm1WithSyncAll, quantMode> mm_w8a8_aiv_1;
  PpMatmulW8a8Aiv<InDtype, false, quantMode> mm_w8a8_aiv_2;
  Quant<InDtype, true, false, quantMode, false> quant;
  RmsNormQuant<InDtype, true, false, quantMode, false, false> rmsNormQuant1;
  RmsNormQuant<InDtype,
               true,
               false,
               quantMode,
               quantMode == QuantMode::PER_TOKEN_SYMM_QUANT,
               false>
      rmsNormQuant2;
  RmsNormQuant<InDtype,
               true,
               false,
               quantMode,
               quantMode == QuantMode::PER_TOKEN_SYMM_QUANT,
               true>
      rmsNormQuant2QDownOut;
  RopeFp16<InDtype, InDtype, Q_OUT_DTYPE, CACHE_MODE> ropeFp16;
  EinSumQuant<InDtype, InDtype> einSumQuant;
#endif
};

template <typename InDtype,
          int8_t CACHE_MODE,
          DataFormat weightFormat1,
          DataFormat weightFormat2,
          DataFormat weightFormat3,
          QuantMode quantMode>
__aicore__ inline void MLAOperation<InDtype,
                                    CACHE_MODE,
                                    weightFormat1,
                                    weightFormat2,
                                    weightFormat3,
                                    quantMode>::ProcessCube() {
#ifdef __DAV_C220_CUBE__
  mm_w8a8_aic_1.Process();
  if constexpr (quantMode == QuantMode::PER_TOKEN_SYMM_QUANT) {
    FftsCrossCoreSync<PIPE_FIX, 0>(MMAIC);
    WaitFlagDev(MMAIC);
    FftsCrossCoreSync<PIPE_FIX, 2>(MMAIV);
  }

  mm_w8a8_aic_2.PreloadWeight();
  mm_w8a8_aic_2.Process();
  mm_ein_sum.Process();
  if constexpr (CACHE_MODE == CACHE_MODE_INT8_NZCACHE) {
    FftsCrossCoreSync<PIPE_FIX, 0>(EINSUMOUT);
    WaitFlagDev(EINSUMOUT);
    FftsCrossCoreSync<PIPE_FIX, 0x2>(EINSUMQUANT);
  }
#endif
}

template <typename InDtype,
          int8_t CACHE_MODE,
          DataFormat weightFormat1,
          DataFormat weightFormat2,
          DataFormat weightFormat3,
          QuantMode quantMode>
__aicore__ inline void MLAOperation<InDtype,
                                    CACHE_MODE,
                                    weightFormat1,
                                    weightFormat2,
                                    weightFormat3,
                                    quantMode>::ProcessVector() {
#ifdef __DAV_C220_VEC__
  if (row_work_ != 0) {
    uint32_t num_col_align_int8 =
        (num_col_1 + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
    uint32_t num_col_align_f16 =
        (num_col_1 + REPEAT_TIME_128 - 1) / REPEAT_TIME_128 * REPEAT_TIME_128;
    uint32_t num_col_align_f32 =
        (num_col_1 + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
    AscendC::LocalTensor<InDtype> input_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(0);
    AscendC::LocalTensor<InDtype> gamma_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(hiddten_state * 2);
    AscendC::LocalTensor<InDtype> beta_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(hiddten_state * 2 +
                                                      hiddten_state * 2);
    AscendC::LocalTensor<InDtype> scale_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(
            hiddten_state * 2 + hiddten_state * 2 + hiddten_state * 2);
    AscendC::LocalTensor<int8_t> offset_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, int8_t>(
            hiddten_state * 2 + hiddten_state * 2 + hiddten_state * 2 + 32);
    AscendC::LocalTensor<float> res1_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            hiddten_state * 2 + hiddten_state * 2 + hiddten_state * 2 + 64);
    AscendC::LocalTensor<float> res3_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            hiddten_state * 2 + hiddten_state * 2 + hiddten_state * 2 + 64 +
            num_col_align_f32 * 4);
    AscendC::LocalTensor<int8_t> output_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, int8_t>(0);
    if (mlaParams.doRmsNorm) {
      rmsNormQuant1.Launch(output_tensor,
                           input_tensor,
                           gamma_tensor,
                           beta_tensor,
                           scale_tensor,
                           offset_tensor,
                           res1_tensor,
                           res3_tensor);
    } else {
      quant.Launch(output_tensor,
                   input_tensor,
                   scale_tensor,
                   offset_tensor,
                   res1_tensor,
                   res3_tensor);
    }
  }
  FftsCrossCoreSync<PIPE_MTE3, 0>(RMSNORMQUANT1);
  WaitFlagDev(RMSNORMQUANT1);
  AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(AIC_MM1_START);
  if constexpr (quantMode == QuantMode::PER_TENSOR_ASYMM_QUANT) {
    mm_w8a8_aiv_1.Process();
    FftsCrossCoreSync<PIPE_MTE3, 0>(RMSNORMQUANT2);
    WaitFlagDev(RMSNORMQUANT2);
  } else {  // quantMode == QuantMode::PER_TOKEN_SYMM_QUANT
    WaitFlagDev(MMAIV);
  }
  if (row_work_ != 0) {
    uint32_t num_col_align_int8 =
        (num_col_2 + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
    uint32_t num_col_align_f16 =
        (num_col_2 + REPEAT_TIME_128 - 1) / REPEAT_TIME_128 * REPEAT_TIME_128;
    uint32_t num_col_align_f32 =
        (num_col_2 + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
    AscendC::LocalTensor<InDtype> input_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(0);
    AscendC::LocalTensor<InDtype> gamma_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(MM1_OUT_SIZE * 2);
    AscendC::LocalTensor<InDtype> beta_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(MM1_OUT_SIZE * 2 +
                                                      SPLIT_SIZE_TWO * 2);
    AscendC::LocalTensor<InDtype> scale_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(
            MM1_OUT_SIZE * 2 + SPLIT_SIZE_TWO * 2 + SPLIT_SIZE_TWO * 2);
    AscendC::LocalTensor<int8_t> offset_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, int8_t>(
            MM1_OUT_SIZE * 2 + SPLIT_SIZE_TWO * 2 + SPLIT_SIZE_TWO * 2 + 32);
    AscendC::LocalTensor<float> res1_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            MM1_OUT_SIZE * 2 + SPLIT_SIZE_TWO * 2 + SPLIT_SIZE_TWO * 2 + 64);
    AscendC::LocalTensor<float> res3_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            MM1_OUT_SIZE * 2 + SPLIT_SIZE_TWO * 2 + SPLIT_SIZE_TWO * 2 + 64 +
            num_col_align_f32 * 4);
    AscendC::LocalTensor<int8_t> output_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, int8_t>(
            MM1_OUT_SIZE * 2 + SPLIT_SIZE_TWO * 2 + SPLIT_SIZE_TWO * 2 + 64 +
            num_col_align_f32 * 4 + BUF_FACTOR * num_col_align_f32 * 4 + 64 +
            MM1_OUT_SIZE * 4 * 2 + 32);
    if (q_down_out_flag) {
      rmsNormQuant2QDownOut.Launch(output_tensor,
                                   input_tensor,
                                   gamma_tensor,
                                   beta_tensor,
                                   scale_tensor,
                                   offset_tensor,
                                   res1_tensor,
                                   res3_tensor);
    } else {
      rmsNormQuant2.Launch(output_tensor,
                           input_tensor,
                           gamma_tensor,
                           beta_tensor,
                           scale_tensor,
                           offset_tensor,
                           res1_tensor,
                           res3_tensor);
    }
  }
  FftsCrossCoreSync<PIPE_MTE3, 0>(MM2);
  WaitFlagDev(MM2);
  AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(AIC_MM2_START);

  if (row_work_ != 0) {
    AscendC::LocalTensor<InDtype> input_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(0);
    AscendC::LocalTensor<InDtype> gamma_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(MM1_OUT_SIZE * 2);
    AscendC::LocalTensor<InDtype> sin_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(
            MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2);
    AscendC::LocalTensor<InDtype> cos_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(
            MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2 +
            SPLIT_RMSNRORM_SIZE_TWO * 2);
    AscendC::LocalTensor<int32_t> slotMapping_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, int32_t>(
            MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2 +
            SPLIT_RMSNRORM_SIZE_TWO * 4);
    int32_t rms3_ub_offset = MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2 +
                             SPLIT_RMSNRORM_SIZE_TWO * 4 + 4096 * 32;
    AscendC::LocalTensor<float> tmp32_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(rms3_ub_offset);

    int32_t out_ub_offset = MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2 +
                            SPLIT_RMSNRORM_SIZE_TWO * 4 + 4096 * 32 +
                            SPLIT_RMSNRORM_SIZE_ONE * 3 * 4 +
                            SPLIT_RMSNRORM_SIZE_TWO * 2 * 4 +
                            MM1_OUT_SIZE * 4 * 2 + 32;
    AscendC::LocalTensor<InDtype> temp_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(out_ub_offset);

    AscendC::LocalTensor<half> tmpfp16;
    AscendC::LocalTensor<int8_t> int8OutTensor;
    float scale3 = 0;
    if constexpr (CACHE_MODE == CACHE_MODE_INT8_NZCACHE) {
      // quantScale3
      AscendC::LocalTensor<InDtype> quantScaleTensor =
          buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(rms3_ub_offset);
      AscendC::LocalTensor<float> floatQuantScaleTensor =
          buf.GetBuffer<BufferType::ASCEND_UB, float>(rms3_ub_offset + 32);
      // int8out
      tmpfp16 = buf.GetBuffer<BufferType::ASCEND_UB, half>(
          rms3_ub_offset + SPLIT_RMSNRORM_SIZE_ONE * sizeof(float) * 2);
      int8OutTensor =
          buf.GetBuffer<BufferType::ASCEND_UB, int8_t>(out_ub_offset);
      AscendC::DataCopy(quantScaleTensor,
                        quantScale3GmTensor,
                        AscendC::DataCopyParams(1, 1, 0, 0));
      SET_FLAG(MTE2, V, EVENT_ID1);
      WAIT_FLAG(MTE2, V, EVENT_ID1);
      Cast(floatQuantScaleTensor,
           quantScaleTensor,
           AscendC::RoundMode::CAST_NONE,
           1);
      AscendC::SetFlag<HardEvent::V_S>(EVENT_ID1);
      AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID1);
      scale3 = 1 / (float)(floatQuantScaleTensor.GetValue(0));
    }

    RmsNormAndRopeConvergence1<InDtype>(
        input_tensor,        // n * 576
        gamma_tensor,        // gamma
        sin_tensor,          // sin
        cos_tensor,          // cons
        slotMapping_tensor,  // slotMapping
        row_work_,
        tmp32_tensor,
        tmp32_tensor[SPLIT_RMSNRORM_SIZE_ONE],
        tmp32_tensor[SPLIT_RMSNRORM_SIZE_ONE + SPLIT_RMSNRORM_SIZE_ONE],
        tmp32_tensor[SPLIT_RMSNRORM_SIZE_ONE + SPLIT_RMSNRORM_SIZE_ONE +
                     SPLIT_RMSNRORM_SIZE_TWO],
        tmp32_tensor[SPLIT_RMSNRORM_SIZE_ONE + SPLIT_RMSNRORM_SIZE_ONE +
                     SPLIT_RMSNRORM_SIZE_TWO + SPLIT_RMSNRORM_SIZE_TWO],
        temp_tensor,
        tmpfp16,
        int8OutTensor,
        scale3);
  }
  mm_w8a8_aiv_2.Process();
  FftsCrossCoreSync<PIPE_MTE3, 0>(MM2OUT);
  WaitFlagDev(MM2OUT);
  AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(AIC_MM3_START);
  ropeFp16.Process();

  if constexpr (CACHE_MODE == CACHE_MODE_INT8_NZCACHE) {
    WaitFlagDev(EINSUMQUANT);
    einSumQuant.Process();
  }
#endif
}
}  // namespace MlaPreprocess
#endif  // MLA_PREPROCESS_BF16_H