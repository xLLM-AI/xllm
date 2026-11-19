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
 * \file moe_v3_gather_out_droppad.h
 * \brief
 */
#ifndef MOE_V3_GATHER_OUT_DROPPAD_H
#define MOE_V3_GATHER_OUT_DROPPAD_H

#include "kernel_operator.h"
#include "moe_v3_common.h"

namespace XllmKernelMoeInitRoutingV3 {
using namespace AscendC;

constexpr int64_t GATHER_OUT_DROPPAD_BUFFER_NUM = 2;

template <typename T>
class MoeGatherOutDroppad {
 public:
  __aicore__ inline MoeGatherOutDroppad(){};
  __aicore__ inline void Init(GM_ADDR inputX,
                              GM_ADDR scale,
                              GM_ADDR expandedRowIdx,
                              GM_ADDR expandedX,
                              GM_ADDR expandedScale,
                              GM_ADDR workspace,
                              const MoeInitRoutingV3TilingData* tilingData,
                              TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyInIndices(int64_t progress);
  __aicore__ inline void CopyOut(int64_t progress);

 private:
  TPipe* pipe_;
  TQueBind<QuePosition::VECIN,
           QuePosition::VECOUT,
           GATHER_OUT_DROPPAD_BUFFER_NUM>
      xCopyInQueue_;
  TQue<QuePosition::VECIN, GATHER_OUT_DROPPAD_BUFFER_NUM>
      expandedRowIdxCopyInQueue_;

  GlobalTensor<T> inputXGm_;
  GlobalTensor<float> xGscaleGm_;
  GlobalTensor<T> expandedXGm_;
  GlobalTensor<int32_t> expandedRowIdxGm_;
  GlobalTensor<float> expandedScaleGm_;

  const MoeV3GatherOutComputeTilingData* gatherOutTilingData_;

  int64_t needCoreNum_;
  int64_t blockIdx_;
  int64_t cols_;
  int64_t n_;
  int64_t k_;
  int64_t currentLoopRows_;
  int64_t coreRows_;
  int64_t perLoopRows_;
  int64_t lastLoopRows_;
  int64_t rowLoops_;
  int64_t colsTileLength_;
  int64_t perLoopCols_;
  int64_t lastLoopCols_;
  int64_t colLoops_;

  int64_t indicesOffset_;
  int64_t inputOffset_;
  int64_t outOffset_;
};

template <typename T>
__aicore__ inline void MoeGatherOutDroppad<T>::CopyInIndices(int64_t progress) {
  indicesOffset_ = progress * perLoopRows_;
  LocalTensor<int32_t> indicesLocal =
      expandedRowIdxCopyInQueue_.AllocTensor<int32_t>();
  DataCopyExtParams dataCopyParams{
      1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
  DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
  DataCopyPad(indicesLocal,
              expandedRowIdxGm_[indicesOffset_],
              dataCopyParams,
              dataCopyPadParams);
  expandedRowIdxCopyInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutDroppad<T>::CopyOut(int64_t progress) {
  LocalTensor<int32_t> indicesLocal =
      expandedRowIdxCopyInQueue_.DeQue<int32_t>();
  SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
  colsTileLength_ = perLoopCols_;
  for (int64_t colsLoop = 0; colsLoop < colLoops_; colsLoop++) {
    int64_t initialRow =
        gatherOutTilingData_->perCoreIndicesElements * blockIdx_ +
        perLoopRows_ * progress;
    int64_t curLoopRow = 0;
    if (colsLoop == colLoops_ - 1) {
      colsTileLength_ = lastLoopCols_;
    }
    int64_t currentLoopStartRow = initialRow / k_;
    int64_t currentLoopLastRow = (initialRow + currentLoopRows_ - 1) / k_;
    for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
      inputOffset_ = row * cols_ + colsLoop * perLoopCols_;
      // input row position
      LocalTensor<T> inLocal = xCopyInQueue_.AllocTensor<T>();
      DataCopyExtParams dataCopyParams{
          1, static_cast<uint32_t>(colsTileLength_ * sizeof(T)), 0, 0, 0};
      DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
      DataCopyPad(
          inLocal, inputXGm_[inputOffset_], dataCopyParams, dataCopyPadParams);
      SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
      DataCopyExtParams intriParams{
          1, static_cast<uint32_t>(colsTileLength_ * sizeof(T)), 0, 0, 0};
      while (curLoopRow < currentLoopRows_ && initialRow / k_ == row) {
        int32_t outIndex = indicesLocal.GetValue(curLoopRow);
        curLoopRow++;
        initialRow++;
        if (outIndex == -1) {
          continue;
        }
        outOffset_ = outIndex * cols_ + colsLoop * perLoopCols_;
        DataCopyPad(expandedXGm_[outOffset_], inLocal, intriParams);
      }
      xCopyInQueue_.FreeTensor(inLocal);
    }
  }
  expandedRowIdxCopyInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutDroppad<T>::Init(
    GM_ADDR inputX,
    GM_ADDR scale,
    GM_ADDR expandedRowIdx,
    GM_ADDR expandedX,
    GM_ADDR expandedScale,
    GM_ADDR workspace,
    const MoeInitRoutingV3TilingData* tilingData,
    TPipe* tPipe) {
  pipe_ = tPipe;
  blockIdx_ = GetBlockIdx();
  gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);

  needCoreNum_ = gatherOutTilingData_->needCoreNum;
  cols_ = tilingData->cols;
  n_ = tilingData->n;
  k_ = tilingData->k;

  if (blockIdx_ == needCoreNum_ - 1) {
    coreRows_ = gatherOutTilingData_->lastCoreIndicesElements;
    perLoopRows_ = gatherOutTilingData_->lastCorePerLoopIndicesElements;
    lastLoopRows_ = gatherOutTilingData_->lastCoreLastLoopIndicesElements;
    rowLoops_ = gatherOutTilingData_->lastCoreIndicesLoops;
  } else {
    coreRows_ = gatherOutTilingData_->perCoreIndicesElements;
    perLoopRows_ = gatherOutTilingData_->perCorePerLoopIndicesElements;
    lastLoopRows_ = gatherOutTilingData_->perCoreLastLoopIndicesElements;
    rowLoops_ = gatherOutTilingData_->perCoreIndicesLoops;
  }
  perLoopCols_ = gatherOutTilingData_->perLoopCols;
  lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
  colLoops_ = gatherOutTilingData_->colsLoops;

  inputXGm_.SetGlobalBuffer((__gm__ T*)inputX, coreRows_ * cols_);
  expandedXGm_.SetGlobalBuffer((__gm__ T*)expandedX, n_ * k_ * cols_);
  expandedRowIdxGm_.SetGlobalBuffer(
      (__gm__ int32_t*)expandedRowIdx +
          blockIdx_ * gatherOutTilingData_->perCoreIndicesElements,
      Align(coreRows_, sizeof(int32_t)));

  pipe_->InitBuffer(xCopyInQueue_,
                    GATHER_OUT_DROPPAD_BUFFER_NUM,
                    AlignBytes(perLoopCols_, sizeof(T)));
  pipe_->InitBuffer(expandedRowIdxCopyInQueue_,
                    GATHER_OUT_DROPPAD_BUFFER_NUM,
                    AlignBytes(perLoopRows_, sizeof(int32_t)));
}

template <typename T>
__aicore__ inline void MoeGatherOutDroppad<T>::Process() {
  if (blockIdx_ < needCoreNum_) {
    currentLoopRows_ = perLoopRows_;
    for (int64_t loop = 0; loop < rowLoops_; loop++) {
      if (loop == rowLoops_ - 1) {
        currentLoopRows_ = lastLoopRows_;
      }
      CopyInIndices(loop);
      CopyOut(loop);
    }
  }
}
}  // namespace XllmKernelMoeInitRoutingV3
#endif  // MOE_V3_GATHER_OUT_DROPPAD_H
