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
 * \file grouped_matmul_swiglu_quant.h
 * \brief
 */
#ifndef MOE_GROUPED_MATMUL_SWIGLU_QUANT_H
#define MOE_GROUPED_MATMUL_SWIGLU_QUANT_H

#include "moe_grouped_matmul_swiglu_quant_utils.h"
namespace MOE_GROUPED_MATMUL_SWIGLU_QUANT {
/** @brief internal computation class
 */
template <class mmType, bool sync = false, typename CHANNELDTYPE = float>
class MoeGMMSwigluCompute {
 public:
  using AT = typename mmType::AT::T;
  using BT = typename mmType::BT::T;
  using B = typename mmType::BT;
  using CT = typename mmType::CT::T;
  using BiasT = typename mmType::BiasT::T;
  using WT = int8_t;
  constexpr static bool transposeX = mmType::AT::isTrans;
  constexpr static bool transposeW = mmType::BT::isTrans;

  /** @brief constructor */
  __aicore__ inline MoeGMMSwigluCompute(typename mmType::MT& mm_) : mm(mm_) {}

  __aicore__ inline void Init(
      GM_ADDR x,
      GM_ADDR weight,
      GM_ADDR perChannelScale,
      GM_ADDR perTokenScale,
      GM_ADDR groupList,
      GM_ADDR quantOutput,
      GM_ADDR quantScaleOutput,
      GM_ADDR workspace,
      const MoeGMMSwigluBaseParams* __restrict gmmBaseParamsIN,
      const TCubeTiling* __restrict mmTilingDataIN,
      const MoeGMMSwiglu* __restrict gmmSwigluIN,
      TPipe* tPipeIN);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void MMCompute(uint32_t groupIdx,
                                   MNConfig& mnConfig,
                                   uint32_t coreIdx);

  __aicore__ inline void UpdateMnConfig(MNConfig& mnConfig, uint32_t groupIdx);

  __aicore__ inline void SetMNConfig(const int32_t splitValue,
                                     const uint32_t groupIdx,
                                     MNConfig& mnConfig);

  __aicore__ inline void SetMKN(const int32_t splitValue,
                                const uint32_t groupIdx,
                                MNConfig& mnConfig);

  __aicore__ inline uint64_t GetWOffset(uint32_t tailN, uint32_t k);

  __aicore__ inline void CubeProcess(MNConfig& mnConfig);
  __aicore__ inline void VecProcess(VecConfig& vecConfig);
  __aicore__ inline void MNBlockIdxCompute(MNConfig& mnConfig,
                                           const uint32_t curBlock,
                                           const uint32_t count,
                                           const uint32_t thresholdM_dimN);
  template <typename DTYPE_CS>
  __aicore__ inline void UpdateChannelScale(uint32_t loopidx,
                                            VecConfig& vecConfig);
  __aicore__ inline void VectorCompute(uint32_t loopidx, VecConfig& vecConfig);
  template <typename DTYPE_CS>
  __aicore__ inline void PreLoadTokenAndChannel(
      LocalTensor<float>& channelScaleLocal,
      VecConfig& vecConfig);
  __aicore__ inline void UpdateVecConfig(uint32_t blockIdx,
                                         VecConfig& vecConfig);
  __aicore__ inline void customDataCopyIn(uint32_t outLoopIdx,
                                          VecConfig& vecConfig);
  __aicore__ inline void customDataCopyOut(VecConfig& vecConfig);
  __aicore__ inline void Dequant(uint32_t loopidx, VecConfig& vecConfig);
  __aicore__ inline void Quant(uint32_t loopidx);
  __aicore__ inline void Swiglu(uint32_t loopidx);

 private:
  typename mmType::MT& mm;
  const MoeGMMSwigluBaseParams* __restrict gmmBaseParams;
  const MoeGMMSwiglu* __restrict gmmSwiglu;
  const TCubeTiling* __restrict mmTilingData;
  TPipe* pipe;
  GlobalTensor<int8_t> xGM;
  GlobalTensor<int8_t> weightGM;
  GlobalTensor<CHANNELDTYPE> perChannelScaleGM;
  GlobalTensor<float> perTokenScaleGM;
  GlobalTensor<int64_t> groupListGM;
  GlobalTensor<int8_t> quantOutputGM;
  GlobalTensor<float> quantScaleOutputGM;
  GlobalTensor<int32_t> mmOutGM;
  // define the que
  TQue<QuePosition::VECIN, 1> mmOutQueue;
  TQue<QuePosition::VECIN, 1> perChannelScaleInQueue;
  TQue<QuePosition::VECOUT, 1> quantOutQueue;
  TQue<QuePosition::VECOUT, 1> quantScaleOutQueue;
  TBuf<TPosition::VECCALC> reduceWorkspace;
  uint32_t blockIdx = 0;
  int32_t preOffset = 0;
  int64_t aicCoreNum = 0;
  int64_t aivCoreNum = 0;
  GM_ADDR xTensorPtr;
  GM_ADDR weightTensorPtr;
};

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::Init(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR perChannelScale,
    GM_ADDR perTokenScale,
    GM_ADDR groupList,
    GM_ADDR quantOutput,
    GM_ADDR quantScaleOutput,
    GM_ADDR workspace,
    const MoeGMMSwigluBaseParams* __restrict gmmSwigluBaseParamsIn,
    const TCubeTiling* __restrict mmTilingDataIN,
    const MoeGMMSwiglu* __restrict gmmSwigluIN,
    TPipe* tPipeIN) {
  aicCoreNum = GetBlockNum();
  aivCoreNum = aicCoreNum * 2;
  blockIdx = GetBlockIdx();
  mmTilingData = mmTilingDataIN;
  gmmBaseParams = gmmSwigluBaseParamsIn;
  gmmSwiglu = gmmSwigluIN;
  pipe = tPipeIN;
  xTensorPtr = x;
  weightTensorPtr = weight;
  groupListGM.SetGlobalBuffer((__gm__ int64_t*)groupList,
                              gmmSwiglu->groupListLen);
  mmOutGM.SetGlobalBuffer((__gm__ int32_t*)workspace,
                          gmmBaseParams->M * gmmSwiglu->tokenLen);
  if ASCEND_IS_AIV {
    perChannelScaleGM.SetGlobalBuffer(
        (__gm__ CHANNELDTYPE*)perChannelScale,
        gmmSwiglu->groupListLen * gmmSwiglu->tokenLen);
    perTokenScaleGM.SetGlobalBuffer((__gm__ float*)perTokenScale,
                                    gmmBaseParams->M);
    quantOutputGM.SetGlobalBuffer(
        (__gm__ int8_t*)quantOutput,
        gmmBaseParams->M * gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR);
    quantScaleOutputGM.SetGlobalBuffer((__gm__ float*)quantScaleOutput,
                                       gmmBaseParams->M);
  }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void
MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::Process() {
  MNConfig mnConfig;
  VecConfig vecConfig;
  CubeProcess(mnConfig);
  VecProcess(vecConfig);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
template <typename DTYPE_CS>
__aicore__ inline void
MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::PreLoadTokenAndChannel(
    LocalTensor<float>& channelScaleLocal,
    VecConfig& vecConfig) {
  DataCopyExtParams copyChannelParams{
      1,
      static_cast<uint32_t>(gmmSwiglu->tokenLen * sizeof(DTYPE_CS)),
      0,
      0,
      0};
  DataCopyPadExtParams<DTYPE_CS> padParams{false, 0, 0, 0};
  if constexpr (!IsSameType<DTYPE_CS, float>::value) {
    LocalTensor<DTYPE_CS> dstLocalT =
        channelScaleLocal.template ReinterpretCast<DTYPE_CS>();
    DataCopyPad(dstLocalT[gmmSwiglu->tokenLen],
                perChannelScaleGM[vecConfig.curGroupIdx * gmmSwiglu->tokenLen],
                copyChannelParams,
                padParams);
    PipeBarrier<PIPE_ALL>();
    Cast(channelScaleLocal,
         dstLocalT[gmmSwiglu->tokenLen],
         RoundMode::CAST_NONE,
         gmmSwiglu->tokenLen);
  } else {
    DataCopyPad(channelScaleLocal,
                perChannelScaleGM[vecConfig.curGroupIdx * gmmSwiglu->tokenLen],
                copyChannelParams,
                padParams);
  }
  perChannelScaleInQueue.EnQue(channelScaleLocal);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void
MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::MMCompute(uint32_t groupIdx,
                                                           MNConfig& mnConfig,
                                                           uint32_t coreIdx) {
  uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
  uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1
                            ? mnConfig.singleN
                            : mnConfig.n - tailN;
  uint32_t curSingleM = mnConfig.mIdx < mnConfig.blockDimM - 1
                            ? mnConfig.singleM
                            : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
  uint64_t xOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.k;
  if constexpr (transposeX) {
    xOffset = mnConfig.mIdx * mnConfig.singleM;
  }
  uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
  xGM.SetGlobalBuffer((__gm__ int8_t*)xTensorPtr + mnConfig.xBaseOffset);
  weightGM.SetGlobalBuffer((__gm__ int8_t*)weightTensorPtr +
                           mnConfig.wBaseOffset +
                           GetWOffset(tailN, mnConfig.k));
  if (mnConfig.blockDimM == 1) {
    weightGM.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  }
  mnConfig.workSpaceOffset = outOffset + mnConfig.yBaseOffset;
  mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
  mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
  mm.SetTensorA(xGM[xOffset], transposeX);
  mm.SetTensorB(weightGM, transposeW);
  mm.template IterateAll<sync>(mmOutGM[mnConfig.workSpaceOffset], 0);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void
MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::UpdateMnConfig(
    MNConfig& mnConfig,
    uint32_t groupIdx) {
  mnConfig.mAxisBaseOffset += mnConfig.m;
  mnConfig.xBaseOffset += mnConfig.m * mnConfig.k;
  mnConfig.yBaseOffset += mnConfig.m * mnConfig.n;
  mnConfig.nAxisBaseOffset = groupIdx * mnConfig.n;
  mnConfig.wBaseOffset =
      AlignUp<16>(mnConfig.k) * AlignUp<16>(mnConfig.nAxisBaseOffset);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void
MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::SetMNConfig(
    const int32_t splitValue,
    const uint32_t groupIdx,
    MNConfig& mnConfig) {
  SetMKN(splitValue, groupIdx, mnConfig);
  mnConfig.baseM = BASIC_M;
  mnConfig.baseN = BASIC_N;
  mnConfig.singleM = SINGLE_CORE_M;
  mnConfig.singleN = SINGLE_CORE_N;
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::SetMKN(
    const int32_t splitValue,
    const uint32_t groupIdx,
    MNConfig& mnConfig) {
  mnConfig.m = static_cast<uint32_t>(splitValue);
  mnConfig.k = gmmBaseParams->K;  // tilingData
  mnConfig.n = gmmBaseParams->N;  // tilingData
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline uint64_t
MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::GetWOffset(uint32_t tailN,
                                                            uint32_t k) {
  uint64_t wOffset = 0;
  if constexpr (mmType::BT::format == CubeFormat::NZ) {
    wOffset = tailN * AlignUp<16>(k);  // 16: nz format last two dim size
  } else {
    wOffset = tailN;
  }
  return wOffset;
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void
MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::CubeProcess(
    MNConfig& mnConfig) {
  if ASCEND_IS_AIC {
    preOffset = 0;
    int32_t prevSplitValue = 0;
    for (uint32_t loop = 0, count = 0;
         loop < gmmSwiglu->groupListLen * GROUP_LIST_INNER_SHAPE;
         loop += GROUP_LIST_INNER_SHAPE) {
      // int64_t groupM = groupListGM.GetValue(groupIdx);
      // int32_t currSplitValue = static_cast<int32_t>(groupM);
      // int32_t splitValue = currSplitValue - prevSplitValue;
      // prevSplitValue = currSplitValue;
      int32_t splitValue = static_cast<int32_t>(groupListGM.GetValue(loop + 1));
      if (splitValue <= 0) {
        continue;
      }
      uint32_t groupIdx = static_cast<int32_t>(groupListGM.GetValue(loop));
      mnConfig.mAxisBaseOffset += mnConfig.m;
      mnConfig.xBaseOffset += mnConfig.m * mnConfig.k;
      mnConfig.yBaseOffset += mnConfig.m * mnConfig.n;
      SetMNConfig(splitValue, groupIdx, mnConfig);
      if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0) {
        continue;
      }
      mnConfig.nAxisBaseOffset = groupIdx * mnConfig.n;
      mnConfig.wBaseOffset =
          AlignUp<16>(mnConfig.k) * AlignUp<16>(mnConfig.nAxisBaseOffset);
      mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
      mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.singleN);

      uint32_t curCount = count + mnConfig.blockDimM * mnConfig.blockDimN;
      uint32_t curBlock =
          blockIdx >= count ? blockIdx : blockIdx + gmmBaseParams->coreNum;
      uint32_t thresholdM_dimN = THRESHOLD_BLOCK_NUM * mnConfig.blockDimN;

      while (curBlock < curCount) {
        MNBlockIdxCompute(mnConfig, curBlock, count, thresholdM_dimN);
        // printf("blockIdx %d groupIdx %d splitValue %d m %d k %d n %d
        // blockDimM %d blockDimN %d mIdx %d nIdx %d\n", blockIdx, groupIdx,
        // splitValue, mnConfig.m, mnConfig.k, mnConfig.n, mnConfig.blockDimM,
        // mnConfig.blockDimN, mnConfig.mIdx, mnConfig.nIdx);
        MMCompute(groupIdx, mnConfig, blockIdx);
        curBlock += aicCoreNum;
      }
      count = curCount % gmmBaseParams->coreNum;
    }
    SyncAll<false>();
  }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void
MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::VecProcess(
    VecConfig& vecConfig) {
  if ASCEND_IS_AIV {
    UpdateVecConfig(blockIdx, vecConfig);
    if (blockIdx < vecConfig.usedCoreNum) {
      LocalTensor<float> channelScaleLocal =
          perChannelScaleInQueue.AllocTensor<float>();
      LocalTensor<int32_t> mmLocal = mmOutQueue.AllocTensor<int32_t>();
      LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();
      LocalTensor<float> quantScaleLocal =
          quantScaleOutQueue.AllocTensor<float>();
      mmOutQueue.EnQue(mmLocal);
      quantScaleOutQueue.EnQue(quantScaleLocal);
      quantOutQueue.EnQue(quantLocal);
      PreLoadTokenAndChannel<CHANNELDTYPE>(channelScaleLocal, vecConfig);
    }
    SyncAll<false>();
    if (blockIdx < vecConfig.usedCoreNum) {
      for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum;
           outLoopIdx++) {
        vecConfig.innerLoopNum = outLoopIdx == (vecConfig.outLoopNum - 1)
                                     ? vecConfig.tailLoopNum
                                     : gmmSwiglu->maxProcessRowNum;
        customDataCopyIn(outLoopIdx, vecConfig);
        for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum;
             innerLoopIdx++) {
          // printf("blockIdx %d innerLoopNum %d innerLoopIdx %d
          // nextupdateInterval %d\n", blockIdx, vecConfig.innerLoopNum,
          // innerLoopIdx, vecConfig.nextUpadteInterVal);
          UpdateChannelScale<CHANNELDTYPE>(innerLoopIdx, vecConfig);
          VectorCompute(innerLoopIdx, vecConfig);
        }
        customDataCopyOut(vecConfig);
      }

      LocalTensor<float> channelScaleLocal =
          perChannelScaleInQueue.DeQue<float>();
      LocalTensor<int32_t> mmLocal = mmOutQueue.DeQue<int32_t>();
      LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
      LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
      perChannelScaleInQueue.FreeTensor(channelScaleLocal);
      mmOutQueue.FreeTensor(mmLocal);
      quantScaleOutQueue.FreeTensor(quantScaleLocal);
      quantOutQueue.FreeTensor(quantLocal);
    } else {
      return;
    }
  }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void
MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::MNBlockIdxCompute(
    MNConfig& mnConfig,
    const uint32_t curBlock,
    const uint32_t count,
    const uint32_t thresholdM_dimN) {
  mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;
  mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void
MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::UpdateVecConfig(
    uint32_t blockIdx,
    VecConfig& vecConfig) {
  // 第一步 读取grouplist reduceSum 计算总数据个数
  for (uint32_t loop = 0;
       loop < gmmSwiglu->groupListLen * GROUP_LIST_INNER_SHAPE;
       loop += GROUP_LIST_INNER_SHAPE) {
    int64_t currM = groupListGM.GetValue(loop + 1);
    if (currM <= 0) {
      continue;
    }
    vecConfig.M += currM;
  }

  // 第二步 计算分核
  uint32_t eachCoreTaskNum = (vecConfig.M + aivCoreNum - 1) / aivCoreNum;
  vecConfig.usedCoreNum = vecConfig.M >= aivCoreNum ? aivCoreNum : vecConfig.M;
  uint32_t tailCoreIdx =
      vecConfig.M - (eachCoreTaskNum - 1) * vecConfig.usedCoreNum;
  vecConfig.taskNum =
      blockIdx < tailCoreIdx ? eachCoreTaskNum : eachCoreTaskNum - 1;
  vecConfig.startIdx = blockIdx < tailCoreIdx
                           ? eachCoreTaskNum * blockIdx
                           : ((eachCoreTaskNum - 1) * blockIdx + tailCoreIdx);
  vecConfig.curIdx = vecConfig.startIdx;
  vecConfig.startOffset = vecConfig.startIdx * gmmSwiglu->tokenLen;
  vecConfig.curOffset = vecConfig.startOffset;
  int64_t curStartIdx = vecConfig.startIdx;

  // printf("blockIdx %d curStartIdx %d\n", blockIdx, curStartIdx);
  for (uint32_t loop = 0;
       loop < gmmSwiglu->groupListLen * GROUP_LIST_INNER_SHAPE;
       loop += GROUP_LIST_INNER_SHAPE) {
    int64_t currM = groupListGM.GetValue(loop + 1);
    int64_t groupIdx = groupListGM.GetValue(loop);
    if (currM <= 0) {
      continue;
    }
    if (curStartIdx >= 0 && curStartIdx - currM < 0) {
      vecConfig.curGroupIdx = groupIdx;
      vecConfig.nextUpadteInterVal = currM - curStartIdx;
      vecConfig.groupRowIdx = loop / GROUP_LIST_INNER_SHAPE;

      // printf("blockIdx %d curGroupIdx %d curStartIdx %d nextUpadteInterVal %d
      // groupRowIdx %d\n", blockIdx, vecConfig.curGroupIdx, curStartIdx,
      // vecConfig.nextUpadteInterVal, vecConfig.groupRowIdx);
    }
    curStartIdx -= currM;
  }
  // 第三步 计算总数据量
  vecConfig.outLoopNum = (vecConfig.taskNum + gmmSwiglu->maxProcessRowNum - 1) /
                         gmmSwiglu->maxProcessRowNum;
  vecConfig.tailLoopNum = vecConfig.taskNum % gmmSwiglu->maxProcessRowNum
                              ? vecConfig.taskNum % gmmSwiglu->maxProcessRowNum
                              : gmmSwiglu->maxProcessRowNum;
  pipe->Reset();
  // 第四步 申请空间
  pipe->InitBuffer(
      mmOutQueue,
      DOUBLE_BUFFER,
      gmmSwiglu->maxProcessRowNum * gmmSwiglu->tokenLen * sizeof(int32_t));
  pipe->InitBuffer(perChannelScaleInQueue,
                   DOUBLE_BUFFER,
                   gmmSwiglu->tokenLen * sizeof(float));
  pipe->InitBuffer(quantOutQueue,
                   DOUBLE_BUFFER,
                   gmmSwiglu->maxProcessRowNum * gmmSwiglu->tokenLen /
                       SWIGLU_REDUCE_FACTOR * sizeof(int8_t));
  pipe->InitBuffer(quantScaleOutQueue,
                   DOUBLE_BUFFER,
                   AlignUp<int32_t>(gmmSwiglu->maxProcessRowNum, ALIGN_8_ELE) *
                       sizeof(float));
  // two 32 byte buffer for reduceMax calculation in Quant.
  pipe->InitBuffer(reduceWorkspace,
                   gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(float) +
                       UB_BLOCK_UNIT_SIZE + UB_BLOCK_UNIT_SIZE);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void
MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::customDataCopyIn(
    uint32_t outLoopIdx,
    VecConfig& vecConfig) {
  LocalTensor<int32_t> _inMMLocal_0 = mmOutQueue.DeQue<int32_t>();
  DataCopyExtParams copyParams_0{
      1,
      static_cast<uint32_t>(vecConfig.innerLoopNum * gmmSwiglu->tokenLen *
                            sizeof(int32_t)),
      0,
      0,
      0};
  DataCopyPadExtParams<int32_t> padParams_0{false, 0, 0, 0};
  DataCopyPad(
      _inMMLocal_0, mmOutGM[vecConfig.curOffset], copyParams_0, padParams_0);

  mmOutQueue.EnQue(_inMMLocal_0);

  LocalTensor<int32_t> _inMMLocal_1 = mmOutQueue.DeQue<int32_t>();

  // if (vecConfig.startIdx == 34) {
  //     printf("blockIdx %d groupIdx %d innerLoopNum %d\n", blockIdx,
  //     vecConfig.curGroupIdx, vecConfig.innerLoopNum);
  //     DumpTensor(_inMMLocal_1, 1, gmmSwiglu->tokenLen);
  // }

  Cast(_inMMLocal_1.ReinterpretCast<float>(),
       _inMMLocal_1,
       RoundMode::CAST_NONE,
       vecConfig.innerLoopNum * gmmSwiglu->tokenLen);

  mmOutQueue.EnQue(_inMMLocal_1);
  LocalTensor<float> _inMMLocal_2 = mmOutQueue.DeQue<float>();

  if (gmmBaseParams->use_split_arithmetic) {
    vecConfig.curIdx += vecConfig.innerLoopNum;
  } else {
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    for (uint32_t i = 0; i < vecConfig.innerLoopNum; i++) {
      WaitFlag<HardEvent::S_V>(EVENT_ID0);
      float scale = perTokenScaleGM.GetValue(vecConfig.curIdx);
      SetFlag<HardEvent::S_V>(EVENT_ID0);
      WaitFlag<HardEvent::S_V>(EVENT_ID0);
      Muls(_inMMLocal_2[i * gmmSwiglu->tokenLen],
           _inMMLocal_2[i * gmmSwiglu->tokenLen],
           scale,
           gmmSwiglu->tokenLen);
      SetFlag<HardEvent::S_V>(EVENT_ID0);
      vecConfig.curIdx++;
    }
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
  }
  vecConfig.curOffset = vecConfig.curIdx * gmmSwiglu->tokenLen;
  mmOutQueue.EnQue(_inMMLocal_2);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
template <typename DTYPE_CS>
__aicore__ inline void
MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::UpdateChannelScale(
    uint32_t loopIdx,
    VecConfig& vecConfig) {
  // 更新perChannel
  if (unlikely(vecConfig.nextUpadteInterVal == 0)) {
    int64_t nextLoopIdx = vecConfig.groupRowIdx + 1;
    for (int64_t loop = nextLoopIdx; loop < gmmSwiglu->groupListLen; loop++) {
      int64_t splitValue =
          groupListGM.GetValue(loop * GROUP_LIST_INNER_SHAPE + 1);
      int64_t groupIdx = groupListGM.GetValue(loop * GROUP_LIST_INNER_SHAPE);
      if (splitValue > 0) {
        // printf("blockIdx %d groupIdx %d splitValue %d loop %d\n", blockIdx,
        // groupIdx, splitValue, loop);
        vecConfig.nextUpadteInterVal = splitValue;
        vecConfig.curGroupIdx = groupIdx;
        vecConfig.groupRowIdx = loop;
        break;
      }
      if (splitValue <= 0) {
        continue;
      }
    }
    LocalTensor<float> _inChannel = perChannelScaleInQueue.DeQue<float>();
    DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(gmmSwiglu->tokenLen * sizeof(DTYPE_CS)),
        0,
        0,
        0};
    DataCopyPadExtParams<DTYPE_CS> padParams{false, 0, 0, 0};
    if constexpr (!IsSameType<DTYPE_CS, float>::value) {
      LocalTensor<DTYPE_CS> dstLocalT =
          _inChannel.template ReinterpretCast<DTYPE_CS>();
      DataCopyPad(
          dstLocalT[gmmSwiglu->tokenLen],
          perChannelScaleGM[vecConfig.curGroupIdx * gmmSwiglu->tokenLen],
          copyParams,
          padParams);
      PipeBarrier<PIPE_ALL>();
      Cast(_inChannel,
           dstLocalT[gmmSwiglu->tokenLen],
           RoundMode::CAST_NONE,
           gmmSwiglu->tokenLen);
    } else {
      DataCopyPad(
          _inChannel,
          perChannelScaleGM[vecConfig.curGroupIdx * gmmSwiglu->tokenLen],
          copyParams,
          padParams);
    }
    PipeBarrier<PIPE_ALL>();
    perChannelScaleInQueue.EnQue(_inChannel);
  }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void
MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::VectorCompute(
    uint32_t loopIdx,
    VecConfig& vecConfig) {
  Dequant(loopIdx, vecConfig);
  Swiglu(loopIdx);
  Quant(loopIdx);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::Dequant(
    uint32_t loopIdx,
    VecConfig& vecConfig) {
  // perChanelScale * perTokenScale
  LocalTensor<float> mmLocal = mmOutQueue.DeQue<float>();
  LocalTensor<float> perChannelLocal = perChannelScaleInQueue.DeQue<float>();
  Mul(mmLocal[loopIdx * gmmSwiglu->tokenLen],
      mmLocal[loopIdx * gmmSwiglu->tokenLen],
      perChannelLocal,
      gmmSwiglu->tokenLen);
  if (gmmBaseParams->use_split_arithmetic) {
    PipeBarrier<PIPE_V>();
    const float scale = perTokenScaleGM.GetValue(
        vecConfig.curIdx - vecConfig.innerLoopNum + loopIdx);
    const int32_t scale_event =
        static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(scale_event);
    WaitFlag<HardEvent::S_V>(scale_event);
    Muls(mmLocal[loopIdx * gmmSwiglu->tokenLen],
         mmLocal[loopIdx * gmmSwiglu->tokenLen],
         scale,
         gmmSwiglu->tokenLen);
  }
  vecConfig.nextUpadteInterVal--;
  mmOutQueue.EnQue(mmLocal);
  perChannelScaleInQueue.EnQue(perChannelLocal);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::Swiglu(
    uint32_t loopIdx) {
  // 高阶API swiglu
  LocalTensor<float> _inMMLocal = mmOutQueue.DeQue<float>();
  float beta = 1.0f;
  LocalTensor<float> workspaceLocal = reduceWorkspace.Get<float>();
  LocalTensor<float> src0Local =
      _inMMLocal[loopIdx * gmmSwiglu->tokenLen +
                 gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR];
  LocalTensor<float> src1Local = _inMMLocal[loopIdx * gmmSwiglu->tokenLen];
  // Clamp the linear branch symmetrically and the gate above its limit.
  const float clamp_limit = gmmBaseParams->clamp_limit;
  if (clamp_limit > 0.0F) {
    const uint32_t elements = gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR;
    Mins(src0Local, src0Local, clamp_limit, elements);
    PipeBarrier<PIPE_V>();
    Maxs(src0Local, src0Local, -clamp_limit, elements);
    PipeBarrier<PIPE_V>();
    Mins(src1Local, src1Local, clamp_limit, elements);
    PipeBarrier<PIPE_V>();
  }
  SwiGLU<float, false>(workspaceLocal,
                       src0Local,
                       src1Local,
                       beta,
                       gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR);
  PipeBarrier<PIPE_ALL>();
  DataCopyParams repeatParams{
      1,
      static_cast<uint16_t>((gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR) /
                            ALIGN_8_ELE),
      0,
      0};
  DataCopy(
      _inMMLocal[loopIdx * gmmSwiglu->tokenLen], workspaceLocal, repeatParams);
  mmOutQueue.EnQue(_inMMLocal);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::Quant(
    uint32_t loopIdx) {
  LocalTensor<float> _inMMLocal = mmOutQueue.DeQue<float>();
  uint64_t preOffset = loopIdx * gmmSwiglu->tokenLen;
  uint64_t halfTokenLen = gmmSwiglu->tokenLen / BISECT;
  Abs(_inMMLocal[preOffset + gmmSwiglu->tokenLen / BISECT],
      _inMMLocal[preOffset],
      halfTokenLen);
  PipeBarrier<PIPE_V>();
  // reduceMax
  LocalTensor<float> workLocal = reduceWorkspace.Get<float>(halfTokenLen);
  LocalTensor<float> reduceResLocal = reduceWorkspace.GetWithOffset<float>(
      FLOAT_UB_BLOCK_UNIT_SIZE, halfTokenLen * sizeof(float));
  LocalTensor<float> reduceTmpLocal = reduceWorkspace.GetWithOffset<float>(
      FLOAT_UB_BLOCK_UNIT_SIZE,
      halfTokenLen * sizeof(float) + UB_BLOCK_UNIT_SIZE);
  ReduceMaxTemplate(reduceResLocal,
                    workLocal,
                    _inMMLocal[preOffset + gmmSwiglu->tokenLen / BISECT],
                    reduceTmpLocal,
                    static_cast<uint32_t>(halfTokenLen));

  int32_t eventIdVToS =
      static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
  SetFlag<HardEvent::V_S>(eventIdVToS);
  WaitFlag<HardEvent::V_S>(eventIdVToS);
  float quantScale =
      gmmBaseParams->use_split_arithmetic
          ? reduceResLocal.GetValue(0) * (1.0F / QUANT_SCALE_INT8)
          : reduceResLocal.GetValue(0) / QUANT_SCALE_INT8;
  LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
  quantScaleLocal.SetValue(loopIdx, quantScale);
  int32_t eventIdSToV =
      static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
  SetFlag<HardEvent::S_V>(eventIdSToV);
  WaitFlag<HardEvent::S_V>(eventIdSToV);
  if (gmmBaseParams->use_split_arithmetic) {
    Duplicate(workLocal, quantScale, halfTokenLen);
    PipeBarrier<PIPE_V>();
    Div(_inMMLocal[preOffset], _inMMLocal[preOffset], workLocal, halfTokenLen);
  } else {
    quantScale = 1 / quantScale;
    const int32_t reciprocal_event =
        static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(reciprocal_event);
    WaitFlag<HardEvent::S_V>(reciprocal_event);
    Muls(
        _inMMLocal[preOffset], _inMMLocal[preOffset], quantScale, halfTokenLen);
  }
  PipeBarrier<PIPE_V>();
  LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
  int32_t dstTempOffset = static_cast<int32_t>(preOffset / BISECT);
  int32_t srcTempOffset = static_cast<int32_t>(preOffset);
  int32_t tempCount = static_cast<int32_t>(halfTokenLen);
  LocalTensor<int8_t> castSpace =
      reduceWorkspace.Get<int8_t>(UB_BLOCK_UNIT_SIZE);
  CastFp32ToInt8Template(quantLocal,
                         _inMMLocal,
                         castSpace,
                         dstTempOffset,
                         srcTempOffset,
                         tempCount,
                         gmmBaseParams->use_split_arithmetic);
  mmOutQueue.EnQue(_inMMLocal);
  quantOutQueue.EnQue(quantLocal);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void
MoeGMMSwigluCompute<mmType, sync, CHANNELDTYPE>::customDataCopyOut(
    VecConfig& vecConfig) {
  // perChanelScale * perTokenScale
  LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
  DataCopyParams copyParams_0{
      1, (uint16_t)(vecConfig.innerLoopNum * sizeof(float)), 0, 0};
  PipeBarrier<PIPE_ALL>();
  DataCopyPad(
      quantScaleOutputGM[vecConfig.startIdx], quantScaleLocal, copyParams_0);
  LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
  DataCopyParams copyParams_1{
      1,
      (uint16_t)(vecConfig.innerLoopNum * gmmSwiglu->tokenLen /
                 SWIGLU_REDUCE_FACTOR * sizeof(int8_t)),
      0,
      0};
  PipeBarrier<PIPE_ALL>();

  DataCopyPad(quantOutputGM[vecConfig.startIdx * gmmSwiglu->tokenLen /
                            SWIGLU_REDUCE_FACTOR],
              quantLocal,
              copyParams_1);
  PipeBarrier<PIPE_ALL>();
  vecConfig.startIdx += vecConfig.innerLoopNum;
  vecConfig.startOffset = vecConfig.startIdx * gmmSwiglu->tokenLen;
  quantOutQueue.EnQue(quantLocal);
  quantScaleOutQueue.EnQue(quantScaleLocal);
}

}  // namespace MOE_GROUPED_MATMUL_SWIGLU_QUANT
#endif
