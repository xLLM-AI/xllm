/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "kernel_operator.h"
using namespace AscendC;
class ReplaceToken {
 public:
  __aicore__ inline ReplaceToken() {}
  __aicore__ inline void Init(GM_ADDR a,
                              GM_ADDR b,
                              GM_ADDR Out,
                              int32_t sequenceLength,
                              int32_t blength,
                              int32_t maxTokens) {
    aGM.SetGlobalBuffer((__gm__ int32_t*)a);
    bGM.SetGlobalBuffer((__gm__ int64_t*)b);
    outGM.SetGlobalBuffer((__gm__ int32_t*)Out);
    this->sequenceLength = sequenceLength;
    this->blength = blength;
    this->maxTokens = maxTokens;
    this->tailA = sequenceLength % maxTokens;
    this->roundA = this->tailA == 0 ? sequenceLength / maxTokens
                                    : sequenceLength / maxTokens + 1;
    pipe.InitBuffer(vecOutQue, 1, maxTokens * sizeof(int32_t));
    pipe.InitBuffer(vecLastIn, 1, blength * sizeof(int64_t));
    pipe.InitBuffer(vecInQue, 1, maxTokens * sizeof(int32_t));
    pipe.InitBuffer(tmpBuf, maxTokens * sizeof(int32_t));
  }
  __aicore__ inline void Process() {
    for (int32_t i = 0; i < roundA; i++) {
      compute(i);
    }
  }

 private:
  __aicore__ inline void compute(int32_t singleAId) {
    LocalTensor<int32_t> forkedIn = vecInQue.AllocTensor<int32_t>();
    LocalTensor<int64_t> lastIn = vecLastIn.AllocTensor<int64_t>();
    int32_t currentALength = maxTokens;
    if (singleAId == roundA - 1 && tailA != 0) {
      currentALength = tailA;
    }
    DataCopyPadExtParams<int32_t> forkedInpadParams;
    forkedInpadParams.isPad = true;
    forkedInpadParams.paddingValue = 0;
    DataCopyExtParams forkedInParams;
    forkedInParams.blockLen = currentALength * sizeof(int32_t);
    forkedInParams.blockCount = 1;
    forkedInParams.srcStride = 0;
    forkedInParams.dstStride = 0;
    AscendC::DataCopyPad(forkedIn,
                         aGM[singleAId * maxTokens],
                         forkedInParams,
                         forkedInpadParams);
    DataCopyExtParams lastInParams;
    lastInParams.blockLen = blength * sizeof(int64_t);
    lastInParams.blockCount = 1;
    lastInParams.srcStride = 0;
    lastInParams.dstStride = 0;
    DataCopyPadExtParams<int64_t> lastInpadParams;
    lastInpadParams.isPad = true;
    lastInpadParams.paddingValue = 0;
    AscendC::DataCopyPad(lastIn, bGM, lastInParams, lastInpadParams);
    vecInQue.EnQue(forkedIn);
    vecLastIn.EnQue(lastIn);
    LocalTensor<int32_t> forkedInTmp = vecInQue.DeQue<int32_t>();
    LocalTensor<int32_t> forkedOut = vecOutQue.AllocTensor<int32_t>();
    LocalTensor<int64_t> lastStepOutPutTokenIdsInUb =
        vecLastIn.DeQue<int64_t>();
    Muls(forkedOut, forkedInTmp, int32_t(1), currentALength);
    for (int32_t i = 0; i < currentALength; i++) {
      if (forkedOut.GetValue(i) < 0) {
        forkedOut.SetValue(
            i,
            static_cast<int32_t>(lastStepOutPutTokenIdsInUb.GetValue(
                (0 - forkedOut.GetValue(i)) - 1)));
      }
    }
    AscendC::DataCopyExtParams copyParams{
        1, static_cast<uint32_t>(currentALength * sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPad(outGM[singleAId * maxTokens], forkedOut, copyParams);
    vecInQue.FreeTensor(forkedInTmp);
    vecOutQue.FreeTensor(forkedOut);
    vecLastIn.FreeTensor(lastStepOutPutTokenIdsInUb);
  }

 private:
  AscendC::GlobalTensor<int32_t> aGM;
  AscendC::GlobalTensor<int64_t> bGM;
  AscendC::GlobalTensor<int32_t> outGM;
  int32_t sequenceLength;
  int32_t blength;
  int32_t maxTokens;
  int32_t tailA;
  int32_t roundA;
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> vecOutQue;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> vecInQue;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> vecLastIn;
  AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
};
