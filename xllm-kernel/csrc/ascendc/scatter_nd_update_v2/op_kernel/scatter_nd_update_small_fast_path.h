/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
 * \file scatter_nd_update_small_fast_path.h
 * \brief Small-scale single-stage fast path kernel (tilingKey 40/41)
 *
 * Active when indexRow <= 32 and updates fit in UB. Single kernel, single core:
 * no sort stage, no workspace round-trip, no SyncAll, no TPipe rebuild.
 * Duplicate indices keep the original last-write (later row wins) semantics via
 * strictly sequential per-row writes.
 */
#ifndef SCATTER_ND_UPDATE_SMALL_FAST_PATH_H
#define SCATTER_ND_UPDATE_SMALL_FAST_PATH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_nd_update_common.h"

namespace XllmKernelScatterNdUpdateV2 {

template <typename T, bool IS_INT64_INDICES>
class SmallFastPathKernel {
 public:
  __aicore__ inline SmallFastPathKernel() = delete;
  __aicore__ inline SmallFastPathKernel(
      GM_ADDR indices,
      GM_ADDR updates,
      GM_ADDR output,
      const ScatterNdUpdateV2TilingData& tiling,
      TPipe& pipe) {
    InitParams(tiling);
    InitBuffers(pipe);
    SetGmAddr(indices, updates, output);
  }

  __aicore__ inline void InitParams(const ScatterNdUpdateV2TilingData& tiling) {
    // fast path is only enabled when blockNum_ == 0, so blockRemainLength_ ==
    // indexRow
    indexRow_ = tiling.linearIndexTiling.blockRemainLength;
    rowAlloc_ = (indexRow_ + ALIGN_NUM - 1) & ~(ALIGN_NUM - 1);
    indexDim_ = tiling.linearIndexTiling.indexDim;
    indicesMask_ = tiling.linearIndexTiling.indicesMask;
    scatterLength_ = tiling.scatterTiling.scatterLength;
    scatterAlignLength_ = tiling.scatterTiling.scatterAlignLength;
  }

  __aicore__ inline void InitBuffers(TPipe& pipe) {
    if constexpr (IS_INT64_INDICES) {
      pipe.InitBuffer(idxInt64Buf_, rowAlloc_ * indexDim_ * sizeof(int64_t));
    }
    pipe.InitBuffer(idxOriginBuf_, rowAlloc_ * indexDim_ * sizeof(int));
    pipe.InitBuffer(linearBuf_, rowAlloc_ * sizeof(int));
    pipe.InitBuffer(rangeBuf_, rowAlloc_ * sizeof(int));
    pipe.InitBuffer(addTmpBuf_, rowAlloc_ * sizeof(int));
    pipe.InitBuffer(updBuf_, rowAlloc_ * scatterAlignLength_ * sizeof(T));
    idxOriginLocal_ = idxOriginBuf_.Get<int>();
    linearLocal_ = linearBuf_.Get<int>();
    rangeLocal_ = rangeBuf_.Get<int>();
    addTmpLocal_ = addTmpBuf_.Get<int>();
    updLocal_ = updBuf_.Get<T>();
    if constexpr (IS_INT64_INDICES) {
      idxInt64Local_ = idxInt64Buf_.Get<int64_t>();
    }
  }

  __aicore__ inline void SetGmAddr(GM_ADDR indices,
                                   GM_ADDR updates,
                                   GM_ADDR output) {
    if constexpr (IS_INT64_INDICES) {
      indicesGmInt64_.SetGlobalBuffer((__gm__ int64_t*)indices);
    } else {
      indicesGm_.SetGlobalBuffer((__gm__ int*)indices);
    }
    updatesGm_.SetGlobalBuffer((__gm__ T*)updates);
    outputGm_.SetGlobalBuffer((__gm__ T*)output);
  }

  __aicore__ inline void Process() {
    CopyIndicesIn();
    if constexpr (IS_INT64_INDICES) {
      CastIndicesToInt32();
    }
    ComputeLinearIndex();
    PipeVToS();
    CopyUpdatesIn();
    CopyOutRows();
  }

 private:
  __aicore__ inline void CopyIndicesIn() {
    if constexpr (IS_INT64_INDICES) {
      DataCopyExtParams copyParams{
          1,
          static_cast<uint32_t>(indexRow_ * indexDim_ * sizeof(int64_t)),
          0,
          0,
          0};
      DataCopyPadExtParams<int64_t> padParams{true, 0, 0, 0};
      DataCopyPad(idxInt64Local_, indicesGmInt64_, copyParams, padParams);
    } else {
      DataCopyExtParams copyParams{
          1,
          static_cast<uint32_t>(indexRow_ * indexDim_ * sizeof(int)),
          0,
          0,
          0};
      DataCopyPadExtParams<int> padParams{true, 0, 0, 0};
      DataCopyPad(idxOriginLocal_, indicesGm_, copyParams, padParams);
    }
    PipeMte2ToS();
  }

  __aicore__ inline void CastIndicesToInt32() {
    Cast(idxOriginLocal_,
         idxInt64Local_,
         RoundMode::CAST_NONE,
         indexRow_ * indexDim_);
    PipeBarrier<PIPE_V>();
  }

  // Same vectorized semantics as LinearIndexKernel::Compute4LinearIndex:
  // linearOffset(r) = sum_d indices[r][d] * indicesMask[d], via per-dim Gather.
  // Vector ops run on rowAlloc_ lanes (8-aligned); lanes >= indexRow_ are pad
  // zeros and never used downstream.
  __aicore__ inline void ComputeLinearIndex() {
    int32_t malValue = indexDim_ * sizeof(int);
    Duplicate<int>(linearLocal_, 0, rowAlloc_);
    CreateVecIndex(rangeLocal_, (int)0, rowAlloc_);
    PipeBarrier<PIPE_V>();
    Muls(rangeLocal_, rangeLocal_, malValue, rowAlloc_);
    PipeBarrier<PIPE_V>();
    for (int i = 0; i < indexDim_; ++i) {
      if (i != 0) {
        Adds(rangeLocal_, rangeLocal_, (int)(sizeof(int)), rowAlloc_);
        PipeBarrier<PIPE_V>();
      }
      LocalTensor<uint32_t> rangeCasted =
          rangeLocal_.ReinterpretCast<uint32_t>();
      Gather(addTmpLocal_,
             idxOriginLocal_,
             rangeCasted,
             (uint32_t)0,
             (uint32_t)rowAlloc_);
      PipeBarrier<PIPE_V>();
      Muls(addTmpLocal_, addTmpLocal_, (int)indicesMask_[i], rowAlloc_);
      PipeBarrier<PIPE_V>();
      Add(linearLocal_, linearLocal_, addTmpLocal_, rowAlloc_);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void CopyUpdatesIn() {
    // Per-row pad copy so row r lands at UB offset r*scatterAlignLength_
    // (rows may carry an alignment gap when scatterLength_ is not 32B-aligned).
    // All mte2 issues are independent; a single wait covers the whole batch.
    DataCopyExtParams copyParams{
        1, static_cast<uint32_t>(scatterLength_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    for (uint64_t r = 0; r < indexRow_; ++r) {
      DataCopyPad(updLocal_[r * scatterAlignLength_],
                  updatesGm_[r * scatterLength_],
                  copyParams,
                  padParams);
    }
    PipeMte2ToS();
  }

  // Strictly sequential ascending-row writes: a duplicated index written by an
  // earlier row is overwritten by its later row => last-write semantics.
  __aicore__ inline void CopyOutRows() {
    DataCopyExtParams outParams{
        1, static_cast<uint32_t>(scatterLength_ * sizeof(T)), 0, 0, 0};
    for (uint64_t r = 0; r < indexRow_; ++r) {
      // Match the sorted paths: negative offsets represent padding rows.
      const int64_t linear_offset = linearLocal_.GetValue(r);
      if (linear_offset < 0) {
        continue;
      }
      const uint64_t out_offset = static_cast<uint64_t>(linear_offset);
      DataCopyPad(
          outputGm_[out_offset], updLocal_[r * scatterAlignLength_], outParams);
    }
    PipeMte3ToS();
  }

  // V pipe results consumed by scalar: wait for V completion.
  __aicore__ inline void PipeVToS() {
    event_t eventID =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventID);
    WaitFlag<HardEvent::V_S>(eventID);
  }

 private:
  GlobalTensor<int> indicesGm_;
  GlobalTensor<int64_t> indicesGmInt64_;
  GlobalTensor<T> updatesGm_;
  GlobalTensor<T> outputGm_;

  TBuf<TPosition::VECCALC> idxInt64Buf_;
  TBuf<TPosition::VECCALC> idxOriginBuf_;
  TBuf<TPosition::VECCALC> linearBuf_;
  TBuf<TPosition::VECCALC> rangeBuf_;
  TBuf<TPosition::VECCALC> addTmpBuf_;
  TBuf<TPosition::VECCALC> updBuf_;

  LocalTensor<int64_t> idxInt64Local_;
  LocalTensor<int> idxOriginLocal_;
  LocalTensor<int> linearLocal_;
  LocalTensor<int> rangeLocal_;
  LocalTensor<int> addTmpLocal_;
  LocalTensor<T> updLocal_;

  static constexpr uint64_t ROW_ALLOC_MAX = 32;
  uint64_t rowAlloc_ = ROW_ALLOC_MAX;
  uint64_t indexRow_ = ROW_ALLOC_MAX;
  uint64_t indexDim_ = 1;
  uint64_t scatterLength_ = 1;
  uint64_t scatterAlignLength_ = 1;
  const uint64_t* indicesMask_ = nullptr;
};

}  // namespace XllmKernelScatterNdUpdateV2

#endif  // SCATTER_ND_UPDATE_SMALL_FAST_PATH_H