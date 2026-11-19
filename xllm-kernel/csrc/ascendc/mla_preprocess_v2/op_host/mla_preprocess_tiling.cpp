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

#include "mla_preprocess_tiling.h"

#include <cmath>
#include <string>

#include "log/log.h"
#include "mla_preprocess_tilingdata.h"
#include "register/op_impl_registry.h"

constexpr uint64_t AXES_ALIGN_SIZE = 512;
constexpr uint64_t BASE_BLOCK_STEP = 2;
constexpr uint64_t CONST_16 = 16;
constexpr uint64_t CONST_32 = 32;
constexpr uint64_t CONST_128 = 128;
constexpr uint64_t CONST_256 = 256;
constexpr uint64_t CONST_512 = 512;
constexpr uint64_t L1_BUFFER_SIZE = 524288;
constexpr uint64_t L1_PINGPONG_BUFFER_LEN = 262144;
constexpr uint64_t L1_SCALE_SIZE = 4096;
constexpr uint64_t L1_BIAS_SIZE = 2048;
constexpr uint64_t L0C_SIZE = 128 * 1024;
constexpr uint64_t CONCAT_SIZE = 512;
constexpr uint64_t HIDDEN_STRATE_ROPE = 192;
constexpr uint64_t HIDDEN_STRATE_MM = 2112;
constexpr uint64_t HIDDEN_STRATE_RMS = 1536;
constexpr uint64_t UB_SIZE = 196352;
constexpr uint64_t HEADDIM = 64;
constexpr uint64_t FP32_REPEAT_MASK = 64;
constexpr uint64_t FP16_REPEAT_MASK = 128;

const int32_t NUM2 = 2;
const int32_t NUM3 = 3;
const int32_t NUM4 = 4;

constexpr uint64_t INDEX_INPUT = 0;
constexpr uint64_t INDEX_WDQKV = 5;
constexpr uint64_t INDEX_WUQ = 12;
constexpr uint64_t INDEX_WUK = 18;

constexpr uint64_t DIM_0 = 0;
constexpr uint64_t DIM_1 = 1;
constexpr uint64_t DIM_2 = 2;

constexpr uint64_t ATTR_EPSILON_IDX = 3;
constexpr uint64_t ATTR_CACHE_MODE_IDX = 9;
constexpr uint64_t ATTR_QUANT_MODE_IDX = 10;
constexpr uint64_t ATTR_DO_RMS_NORM_IDX = 11;

inline uint64_t CeilDiv(const uint64_t dividend, const uint64_t divisor) {
  if (divisor == 0) {
    return UINT32_MAX;
  }
  return (dividend + divisor - 1) / divisor;
}

inline uint64_t RoundUp(const uint64_t val, const uint64_t align = 16) {
  if (align == 0) {
    return 0;
  }
  return (val + align - 1) / align * align;
}

inline uint64_t RoundDown(const uint64_t val, const uint64_t align = 16) {
  if (align == 0) {
    return 0;
  }
  return val / align * align;
}

template <typename T = uint64_t>
inline T Max(const T a, const T b) {
  return a > b ? a : b;
}

template <typename T = uint64_t>
inline T Min(const T a, const T b) {
  return a < b ? a : b;
}

namespace optiling {

using namespace optiling;
using QuantMode = OpParam::MlaPreprocessParam::QuantMode;

class PpMatmulTilingApi {
 public:
  PpMatmulTilingApi(uint64_t numBatch,
                    uint64_t m,
                    uint64_t k,
                    uint64_t n,
                    bool transA,
                    bool transB,
                    bool enDequant,
                    bool deqOnTheFly,
                    uint64_t aicNumPlatForm,
                    uint64_t l0SizePlatForm,
                    uint64_t l2SizePlatForm)
      : numBatch_(numBatch),
        m_(m),
        k_(k),
        n_(n),
        transA_(transA),
        transB_(transB),
        enDequant_(enDequant),
        deqOnTheFly_(deqOnTheFly),
        aicNumPlatForm_(aicNumPlatForm),
        l0SizePlatForm_(l0SizePlatForm),
        l2SizePlatForm_(l2SizePlatForm) {
    inDataSize_ = enDequant ? sizeof(uint8_t) : sizeof(uint16_t);
  }
  void GetTilingData(optiling::MlaPpMatmulTilingData* tiling);

 private:
  void GetTileSize();
  float GetCost(const uint64_t m0, const uint64_t n0);
  void UpdateTileSize(const uint64_t m0, const uint64_t n0);
  void Swizzle();
  uint64_t ComputeL1AbSize();
  uint64_t ComputeK0ForABpingpong(uint64_t l1AbSize);
  bool IsLoadAllAmat(uint64_t l1AbSize);
  uint64_t ComputeK0ForOnlyBpingpong(uint64_t l1AbSize);

 private:
  uint64_t numBatch_{0};
  uint64_t m_{0};
  uint64_t k_{0};
  uint64_t n_{0};
  bool transA_{false};
  bool transB_{false};
  bool enDequant_{false};
  bool deqOnTheFly_{false};
  uint64_t aicNumPlatForm_{0};
  uint64_t l0SizePlatForm_{0};
  uint64_t l2SizePlatForm_{0};
  uint64_t m0_{0};
  uint64_t k0_{0};
  uint64_t n0_{0};
  uint64_t mLoop_{0};
  uint64_t kLoop_{0};
  uint64_t nLoop_{0};
  uint64_t coreLoop_{0};
  uint64_t swizzleCount_{0};
  uint64_t blockDim_{0};
  uint64_t swizzleDirect_{0};
  uint64_t inDataSize_{0};
  uint64_t b0matPingPongBufferLen_{L1_PINGPONG_BUFFER_LEN};
  bool enShuffleK_{false};
  bool enLoadAllAmat_{false};
};

void PpMatmulTilingApi::GetTilingData(optiling::MlaPpMatmulTilingData* tiling) {
  GetTileSize();  // 计算，然后下面赋值
  tiling->set_numBatch(numBatch_);
  tiling->set_m(m_);
  tiling->set_k(k_);
  tiling->set_n(n_);
  tiling->set_m0(m0_);
  tiling->set_k0(k0_);
  tiling->set_n0(n0_);
  tiling->set_mLoop(mLoop_);
  tiling->set_kLoop(kLoop_);
  tiling->set_nLoop(nLoop_);
  tiling->set_coreLoop(coreLoop_);
  tiling->set_swizzleCount(swizzleCount_);
  tiling->set_swizzleDirect(swizzleDirect_);
  tiling->set_enShuffleK(static_cast<uint64_t>(enShuffleK_));
  tiling->set_blockDim(blockDim_);
  tiling->set_enLoadAllAmat(static_cast<uint64_t>(enLoadAllAmat_));
  tiling->set_b0matPingPongBufferLen(b0matPingPongBufferLen_);
}

void PpMatmulTilingApi::GetTileSize() {
  bool priFlag = !(m_ < n_);
  uint64_t roundBase = static_cast<uint64_t>(
      pow(2, ceil(log(CeilDiv(priFlag ? n_ : m_, CONST_16)))) * CONST_16);
  uint64_t priAxes = RoundUp(priFlag ? m_ : n_, CONST_16);
  uint64_t subAxes = RoundUp(priFlag ? n_ : m_, roundBase);
  float minCost = __FLT_MAX__;
  uint64_t maxAxes0 = AXES_ALIGN_SIZE;
  uint64_t maxPriAxes0 = Min(maxAxes0, priAxes);
  uint64_t maxSubAxes0 = Min(maxAxes0, subAxes);
  for (uint64_t priAxes0 = CONST_16; priAxes0 <= maxPriAxes0;
       priAxes0 *= BASE_BLOCK_STEP) {
    for (uint64_t subAxes0 = CONST_16; subAxes0 <= maxSubAxes0;
         subAxes0 *= BASE_BLOCK_STEP) {
      if (priAxes0 * subAxes0 * sizeof(float) > l0SizePlatForm_) {
        continue;
      }
      uint64_t newM0 = priFlag ? priAxes0 : subAxes0;
      uint64_t newN0 = priFlag ? subAxes0 : priAxes0;
      if (newN0 > CONST_256 && enDequant_) {
        continue;
      }
      float cost = GetCost(newM0, newN0);
      if (cost < minCost) {
        minCost = cost;
        UpdateTileSize(newM0, newN0);
      }
    }
  }

  Swizzle();

  uint64_t l1AbSize = ComputeL1AbSize();
  k0_ = ComputeK0ForABpingpong(l1AbSize);
  kLoop_ = CeilDiv(k_, k0_);
  // 对于MM1和MM2, 如果一个核一轮跑不完, 选择全载A, 并更新k0
  if (0) {  // IsLoadAllAmat(l1AbSize)
    k0_ = ComputeK0ForOnlyBpingpong(l1AbSize);
    kLoop_ = CeilDiv(k_, k0_);
  }
}

uint64_t PpMatmulTilingApi::ComputeK0ForOnlyBpingpong(uint64_t l1AbSize) {
  enLoadAllAmat_ = true;
  b0matPingPongBufferLen_ = static_cast<uint64_t>(static_cast<float>(
      (l1AbSize - RoundUp(m_, CONST_16) * RoundUp(k_, CONST_32) * inDataSize_) /
      DIM_2));
  uint64_t k0MaxB0 = static_cast<uint64_t>(static_cast<float>(
      b0matPingPongBufferLen_ / (RoundUp(n0_, CONST_16) * inDataSize_)));
  uint64_t k0B0 = k0MaxB0 < CONST_512 ? RoundDown(k0MaxB0, CONST_32)
                                      : RoundDown(k0MaxB0, CONST_512);
  return k0B0 > CONST_512 ? RoundDown(k0B0, CONST_512) : k0B0;
}

bool PpMatmulTilingApi::IsLoadAllAmat(uint64_t l1AbSize) {
  return (coreLoop_ > blockDim_) && enDequant_ && (kLoop_ > 1) &&
         (l1AbSize >
          RoundUp(m_, CONST_16) * RoundUp(k_, CONST_32) * inDataSize_) &&
         (mLoop_ == 1);
}

uint64_t PpMatmulTilingApi::ComputeK0ForABpingpong(uint64_t l1AbSize) {
  uint64_t k0Max = static_cast<uint64_t>(static_cast<float>(l1AbSize / DIM_2) /
                                         ((m0_ + n0_) * inDataSize_));
  uint64_t tmpK0;
  if (enDequant_) {
    tmpK0 = k0Max < CONST_512 ? RoundDown(k0Max, CONST_32)
                              : RoundDown(k0Max, CONST_512);
  } else {
    tmpK0 = k0Max < CONST_256 ? RoundDown(k0Max, CONST_16)
                              : RoundDown(k0Max, CONST_256);
  }
  if (tmpK0 > CONST_512) {
    tmpK0 = RoundDown(tmpK0, CONST_512);
  }
  return tmpK0;
}

uint64_t PpMatmulTilingApi::ComputeL1AbSize() {
  if (enDequant_ && deqOnTheFly_) {
    return L1_BUFFER_SIZE;
  }
  return enDequant_ ? (L1_BUFFER_SIZE - L1_BIAS_SIZE - L1_SCALE_SIZE)
                    : L1_BUFFER_SIZE;
}

float PpMatmulTilingApi::GetCost(const uint64_t m0, const uint64_t n0) {
  float aCoef = 1.0;
  float bCoef = 1.0;
  float bwCoef = 5.0;
  uint64_t mLoop = CeilDiv(m_, m0);
  uint64_t nLoop = CeilDiv(n_, n0);
  if (mLoop == 0 || nLoop == 0) {
    return __FLT_MAX__;
  }
  uint64_t rqdNumCore = numBatch_ * mLoop * nLoop;
  uint64_t blockDim = Min(rqdNumCore, aicNumPlatForm_);
  uint64_t mOnce = blockDim < nLoop ? m0 : blockDim / nLoop * m0;
  uint64_t nOnce = blockDim < nLoop ? aicNumPlatForm_ * n0 : n_;
  if (mOnce * k_ * sizeof(uint16_t) > l2SizePlatForm_) {
    aCoef = bwCoef;
  }
  if (nOnce * k_ * sizeof(uint16_t) > l2SizePlatForm_) {
    bCoef = bwCoef;
  }
  if (transA_ && m0 % CONST_256 == 0) {
    aCoef *= NUM2;
  }
  if (!transB_ && n0 % CONST_256 == 0) {
    bCoef *= NUM2;
  }
  return 1 / (aCoef * static_cast<float>(n0)) +
         1 / (bCoef * static_cast<float>(m0));
}

void PpMatmulTilingApi::UpdateTileSize(const uint64_t m0, const uint64_t n0) {
  m0_ = m0;
  n0_ = n0;
  mLoop_ = CeilDiv(m_, m0_);
  nLoop_ = CeilDiv(n_, n0_);
  coreLoop_ = numBatch_ * mLoop_ * nLoop_;
  const uint64_t maxNumCubeCore = aicNumPlatForm_;
  if (mLoop_ == 1 && transB_ &&
      coreLoop_ % maxNumCubeCore < maxNumCubeCore / NUM4 * NUM3) {
    uint64_t tmpM0 = RoundUp(m_, CONST_16);
    uint64_t maxN0 = L0C_SIZE / (tmpM0 * sizeof(float));
    if (enDequant_) {
      maxN0 = maxN0 < CONST_256 ? maxN0 : CONST_256;
    }
    uint64_t x = CeilDiv(n_, maxNumCubeCore);
    uint64_t y = CeilDiv(x, maxN0);
    uint64_t tmpN0 = RoundUp(CeilDiv(x, y), CONST_16);
    uint64_t rqdL0cSize = tmpM0 * tmpN0 * sizeof(float);
    if (rqdL0cSize < L0C_SIZE &&
        (tmpM0 + tmpN0) * CONST_256 * inDataSize_ < L1_BUFFER_SIZE) {
      m0_ = tmpM0;
      n0_ = tmpN0;
      nLoop_ = CeilDiv(n_, n0_);
      coreLoop_ = numBatch_ * nLoop_;
    }
  }
  blockDim_ = Min(coreLoop_, maxNumCubeCore);
}

void PpMatmulTilingApi::Swizzle() {
  float minCost = m_ * k_ + k_ * n_;
  for (uint64_t i = 1; i <= blockDim_; ++i) {
    int c = static_cast<int32_t>((blockDim_ + i - 1) / i);
    float cost;
    // B0 + A < A0 + B
    if (i * n0_ + m_ < m0_ * c + n_) {
      swizzleDirect_ = 1;  // Nz
      cost = n0_ * i + m0_ * c;
      if (cost <= minCost) {
        minCost = cost;
        swizzleCount_ = i;
      }
    } else {
      swizzleDirect_ = 0;  // Zn
      cost = m0_ * i + n0_ * c;
      if (cost < minCost) {
        minCost = cost;
        swizzleCount_ = i;
      }
    }
  }
}

void MlaPreprocessTiling::RmsNormQuantTiling(const uint64_t numTokens,
                                             const uint64_t numVectorCore,
                                             const uint64_t hiddtenState) {
  mlaTilingData.set_rmsNumCore1(numVectorCore);
  mlaTilingData.set_rmsNumCol1(hiddtenState);
  mlaTilingData.set_rmsNumRow1(numTokens);
  mlaTilingData.set_rmsQuantMin1(-CONST_128);
  mlaTilingData.set_rmsNumCore2(numVectorCore);
  mlaTilingData.set_rmsNumCol2(HIDDEN_STRATE_MM);
  mlaTilingData.set_rmsNumRow2(numTokens);
  mlaTilingData.set_rmsQuantMin2(-CONST_128);
}

void MlaPreprocessTiling::RopeConcatTiling(
    const OpParam::MlaPreprocessParam& param,
    const uint64_t& aicNum) {
  uint64_t ntokens = param.N;
  uint64_t hiddenSizeQ = HEADDIM * param.headNum;
  uint64_t headDim = HEADDIM;
  uint64_t headNumQ = hiddenSizeQ / headDim;
  uint64_t concatSize = CONCAT_SIZE;
  uint64_t maxCore = aicNum * 2;
  uint64_t maxUbSize = UB_SIZE;
  uint64_t allHeadNum = ntokens * headNumQ;

  uint64_t tempCore = (allHeadNum + maxCore - 1) / maxCore;
  uint64_t realCore = (allHeadNum + tempCore - 1) / tempCore;  // 实际运算核数
  uint64_t nlCoreRun =
      (allHeadNum + realCore - 1) / realCore;  // 前核运算head数
  uint64_t lCoreRun =
      allHeadNum - (realCore - 1) * nlCoreRun;  // 尾核运算head数

  uint64_t dataTypeSize = 2;

  // 计算一次能搬几行 q 4+2、reverseq 4、neg 4、sin 4+2、cos 4+2  + concat 2
  uint64_t allSize = headDim * (3 * (4 + dataTypeSize) + 2 * 4) +
                     concatSize * dataTypeSize;  // rope内部升精度计算
  uint64_t maxNPerLoopForUb =
      maxUbSize / allSize;  // ub每次能载入最大行数（包括所有计算数据）
  uint64_t preCoreLoopTime =
      (nlCoreRun + maxNPerLoopForUb - 1) / maxNPerLoopForUb;  // 前核循环次数
  uint64_t preCoreLoopNLast =
      nlCoreRun -
      (preCoreLoopTime - 1) * maxNPerLoopForUb;  // 前核最后一批处理数据行数
  uint64_t lastCoreLoopTime =
      (lCoreRun + maxNPerLoopForUb - 1) / maxNPerLoopForUb;  // 尾核循环次数
  uint64_t lastCoreLoopNLast =
      lCoreRun -
      (lastCoreLoopTime - 1) * maxNPerLoopForUb;  // 尾核最后一批处理数据行数

  mlaTilingData.set_hiddenSizeQ(hiddenSizeQ);
  mlaTilingData.set_headNumQ(headNumQ);
  mlaTilingData.set_headDim(headDim);
  mlaTilingData.set_concatSize(concatSize);
  mlaTilingData.set_rotaryCoeff(NUM2);
  mlaTilingData.set_ntokens(ntokens);
  mlaTilingData.set_realCore(realCore);
  mlaTilingData.set_nlCoreRun(nlCoreRun);
  mlaTilingData.set_lCoreRun(nlCoreRun);
  mlaTilingData.set_maxNPerLoopForUb(maxNPerLoopForUb);
  mlaTilingData.set_preCoreLoopTime(preCoreLoopTime);
  mlaTilingData.set_preCoreLoopNLast(preCoreLoopNLast);
  mlaTilingData.set_lastCoreLoopTime(lastCoreLoopTime);
  mlaTilingData.set_lastCoreLoopNLast(lastCoreLoopNLast);
}

void MlaPreprocessTiling::EinSumQuantTiling(
    const OpParam::MlaPreprocessParam& param,
    const uint64_t& aicNum,
    const ge::DataType inDtype,
    const bool doRmsQuant) {
  uint64_t aivCore = aicNum * 2;
  uint64_t ubSize = UB_SIZE - 1024;
  // input shape
  // tokenNum  // headNum
  uint64_t esqBatch = param.N;
  uint64_t esqHeadNum = param.headNum;
  uint64_t esqColNum = AXES_ALIGN_SIZE;  // 512

  // split core
  uint64_t esqFrontCore = esqBatch % aivCore;
  uint64_t esqTailCore = aivCore - esqFrontCore;
  uint64_t esqFrontCoreBatch = CeilDiv(esqBatch, aivCore);
  uint64_t esqTailCoreBatch = esqBatch / aivCore;

  // split ub --> calc H' <-- 一次ub循环中搬运处理的行数  //
  // ub每次计算的head行数
  uint64_t splitFactor = 0;
  uint64_t esqHeadPerLoop = 0;
  uint64_t repeatMask = 0;

  if (inDtype == ge::DT_BF16 ||
      param.quantMode == QuantMode::PER_TOKEN_SYMM_QUANT) {
    if (doRmsQuant) {
      // 将scale一次性搬入、广播、缓存 H * 32bytes
      uint64_t scaleUb = RoundUp(esqHeadNum) * CONST_32;
      // bf16 input [H', colNum](f16 + fp32 + int8), ub reuse
      splitFactor =
          esqColNum * (sizeof(uint16_t) + sizeof(float) + sizeof(uint8_t));
      splitFactor *= NUM2;
      esqHeadPerLoop = (ubSize - scaleUb) / splitFactor;  // 26
      repeatMask = FP32_REPEAT_MASK;
    } else {
      splitFactor = esqColNum * (NUM2 * sizeof(uint16_t) + sizeof(uint8_t)) +
                    sizeof(uint16_t) + (CONST_16 * sizeof(uint16_t));
      esqHeadPerLoop = ubSize / splitFactor;
      repeatMask = FP16_REPEAT_MASK;
      esqHeadPerLoop = RoundDown(esqHeadPerLoop);  // 向下16对齐
    }
  } else {
    // fp16 input [H', cloNum](fp16*2 + int8) + [H', 1](fp16) + [H', 16](fp16)
    splitFactor = esqColNum * (NUM2 * sizeof(uint16_t) + sizeof(uint8_t)) +
                  sizeof(uint16_t) + (CONST_16 * sizeof(uint16_t));
    esqHeadPerLoop = ubSize / splitFactor;
    repeatMask = FP16_REPEAT_MASK;
    esqHeadPerLoop = RoundDown(esqHeadPerLoop);  // 向下16对齐
  }
  uint64_t esqUbHeadLoop = esqHeadNum;  // ub完整循环次数
  uint64_t esqHeadTail = esqHeadNum;    // ub最后一次处理head的行数
  uint64_t esqColLoop = esqHeadNum;     // ub完整循环次数
  uint64_t esqColTail = esqHeadNum;     // ub最后一次处理head的行数
  if (esqHeadPerLoop != 0) {
    esqUbHeadLoop = esqHeadNum / esqHeadPerLoop;  // ub完整循环次数
    esqHeadTail = esqHeadNum % esqHeadPerLoop;    // ub最后一次处理head的行数
    esqColLoop = esqColNum / repeatMask;  // 每行按列计算要循环处理的次数
    esqColTail =
        esqColNum % repeatMask;  // colNum非64/128对齐时，最后一次计算列数
  }

  mlaTilingData.set_esqFrontCore(esqFrontCore);
  mlaTilingData.set_esqTailCore(esqTailCore);
  mlaTilingData.set_esqFrontCoreBatch(esqFrontCoreBatch);
  mlaTilingData.set_esqTailCoreBatch(esqTailCoreBatch);
  mlaTilingData.set_esqHeadNum(esqHeadNum);
  mlaTilingData.set_esqColNum(esqColNum);
  mlaTilingData.set_esqUbHeadLoop(esqUbHeadLoop);
  mlaTilingData.set_esqHeadPerLoop(esqHeadPerLoop);
  mlaTilingData.set_esqHeadTail(esqHeadTail);
  mlaTilingData.set_esqColLoop(esqColLoop);
  mlaTilingData.set_esqColTail(esqColTail);
}

void MlaPreprocessTiling::SetTilingKey(const ge::DataType inDtype,
                                       const OpParam::MlaPreprocessParam& param,
                                       const bool doRmsQuant,
                                       gert::TilingContext* context) {
  auto formatWeight1 = static_cast<ge::Format>(ge::GetPrimaryFormat(
      context->GetInputDesc(INDEX_WDQKV)->GetStorageFormat()));
  auto formatWeight2 = static_cast<ge::Format>(ge::GetPrimaryFormat(
      context->GetInputDesc(INDEX_WUQ)->GetStorageFormat()));
  auto formatWeight3 = static_cast<ge::Format>(ge::GetPrimaryFormat(
      context->GetInputDesc(INDEX_WUK)->GetStorageFormat()));

  uint64_t tilingKey = static_cast<uint64_t>(inDtype == ge::DT_BF16);
  tilingKey = (tilingKey << 2) +
              static_cast<uint64_t>(param.cacheMode);  // 2bit for cacheMode.
  tilingKey = (tilingKey << 1) +
              static_cast<uint64_t>(formatWeight1 == ge::FORMAT_FRACTAL_NZ);
  tilingKey = (tilingKey << 1) +
              static_cast<uint64_t>(formatWeight2 == ge::FORMAT_FRACTAL_NZ);
  tilingKey = (tilingKey << 1) +
              static_cast<uint64_t>(formatWeight3 == ge::FORMAT_FRACTAL_NZ);
  tilingKey = (tilingKey << 2) +
              static_cast<uint64_t>(param.quantMode);  // 2bit for quantMode.

  if (!doRmsQuant) {
    tilingKey += 1000;  // 1000 : 代表匹配量化输入分支
  }

  context->SetTilingKey(tilingKey);
}

void MlaPreprocessTiling::SetMlapoWorkSpace(
    const ge::DataType inDtype,
    const OpParam::MlaPreprocessParam& param,
    uint32_t sysWorkSpaceSize,
    gert::TilingContext* context) {
  uint64_t hiddtenState =
      static_cast<uint64_t>(mlaTilingData.get_hiddtenState());
  uint64_t s1wsFactor = static_cast<uint64_t>(
      param.cacheMode == 2
          ? std::max(hiddtenState * sizeof(int8_t),
                     param.headNum * AXES_ALIGN_SIZE * sizeof(uint16_t))
          : hiddtenState * sizeof(int8_t));
  uint64_t workSizeS1 =
      static_cast<uint64_t>(mlaTilingData.get_n()) * s1wsFactor;
  uint64_t workSizeS2 = static_cast<uint64_t>(mlaTilingData.get_n()) *
                        param.headNum * HIDDEN_STRATE_ROPE * sizeof(uint16_t);
  uint64_t workSizeS3 = static_cast<uint64_t>(mlaTilingData.get_n()) *
                        HIDDEN_STRATE_MM * sizeof(uint16_t);
  uint64_t workSizeS4 =
      static_cast<uint64_t>(mlaTilingData.get_n()) *
      std::max(param.headNum * HIDDEN_STRATE_ROPE, HIDDEN_STRATE_MM) *
      sizeof(uint64_t);
  uint64_t pertokenWorkspace =
      static_cast<uint64_t>(mlaTilingData.get_n()) * sizeof(float) * 2;

  uint64_t maxWorkspaceSize = 0;
  maxWorkspaceSize = std::max(maxWorkspaceSize, workSizeS1);
  maxWorkspaceSize = std::max(maxWorkspaceSize, workSizeS2);
  maxWorkspaceSize = std::max(maxWorkspaceSize, workSizeS3);
  maxWorkspaceSize = std::max(maxWorkspaceSize, workSizeS4);

  size_t* currentWorkspace = context->GetWorkspaceSizes(1);

  const int BF16_WORK_NUM = 4;
  if (inDtype == ge::DT_BF16 ||
      param.quantMode == QuantMode::PER_TOKEN_SYMM_QUANT) {
    currentWorkspace[0] =
        maxWorkspaceSize * BF16_WORK_NUM + pertokenWorkspace + sysWorkSpaceSize;
  } else {
    currentWorkspace[0] = maxWorkspaceSize * 3 +
                          sysWorkSpaceSize;  // 3: 使用3份maxWorkspaceSize空间
  }
  mlaTilingData.set_maxWorkspaceSize(maxWorkspaceSize);
}

void MlaPreprocessTiling::PrintTilingData(gert::TilingContext* context) {
  MlaPreprocessTiling::PrintFirstTilingData(context);
  MlaPreprocessTiling::PrintLastTilingData(context);
}

void MlaPreprocessTiling::PrintFirstTilingData(gert::TilingContext* context) {
  /*
  OP_LOGD(context->GetNodeName(), ">>>>>>>>>> Start to print MlaPreprocess
  tiling data <<<<<<<<<<"); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: numCore is %ld.", mlaTilingData.get_numCore());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: n is %ld.",
  mlaTilingData.get_n()); OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling:
  perTaskNum is %ld.", mlaTilingData.get_perTaskNum());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: resTaskNum is %ld.",
  mlaTilingData.get_resTaskNum()); OP_LOGD( context->GetNodeName(),
      "MlaPreprocess_tiling: mm1: bSize is %ld, mSize is %ld, kSize is %ld,
  nSize is %ld,  m0 is %ld, k0 is %ld, " "n0 is %ld, mLoop is %ld, kLoop is %ld,
  nLoop is %ld, coreLoop is %ld, SwizzleCount is %ld, " "SwizzleDirect is %ld,
  blockDim is %ld", mlaTilingData.mm1.get_numBatch(), mlaTilingData.mm1.get_m(),
  mlaTilingData.mm1.get_k(), mlaTilingData.mm1.get_n(),
  mlaTilingData.mm1.get_m0(), mlaTilingData.mm1.get_k0(),
  mlaTilingData.mm1.get_n0(), mlaTilingData.mm1.get_mLoop(),
  mlaTilingData.mm1.get_kLoop(), mlaTilingData.mm1.get_nLoop(),
      mlaTilingData.mm1.get_coreLoop(), mlaTilingData.mm1.get_swizzleCount(),
  mlaTilingData.mm1.get_swizzleDirect(), mlaTilingData.mm1.get_blockDim());
  OP_LOGD(
      context->GetNodeName(),
      "MlaPreprocess_tiling: mm2: bSize is %ld, mSize is %ld, kSize is %ld,
  nSize is %ld,  m0 is %ld, k0 is %ld, " "n0 is %ld, mLoop is %ld, kLoop is %ld,
  nLoop is %ld, coreLoop is %ld, SwizzleCount is %ld, " "SwizzleDirect is %ld,
  blockDim is %ld", mlaTilingData.mm2.get_numBatch(), mlaTilingData.mm2.get_m(),
  mlaTilingData.mm2.get_k(), mlaTilingData.mm2.get_n(),
  mlaTilingData.mm2.get_m0(), mlaTilingData.mm2.get_k0(),
  mlaTilingData.mm2.get_n0(), mlaTilingData.mm2.get_mLoop(),
  mlaTilingData.mm2.get_kLoop(), mlaTilingData.mm2.get_nLoop(),
      mlaTilingData.mm2.get_coreLoop(), mlaTilingData.mm2.get_swizzleCount(),
  mlaTilingData.mm2.get_swizzleDirect(), mlaTilingData.mm2.get_blockDim());
  OP_LOGD(
      context->GetNodeName(),
      "MlaPreprocess_tiling: mm3: bSize is %ld, mSize is %ld, kSize is %ld,
  nSize is %ld,  m0 is %ld, k0 is %ld, " "n0 is %ld, mLoop is %ld, kLoop is %ld,
  nLoop is %ld, coreLoop is %ld, SwizzleCount is %ld, " "SwizzleDirect is %ld,
  blockDim is %ld", mlaTilingData.mm3.get_numBatch(), mlaTilingData.mm3.get_m(),
  mlaTilingData.mm3.get_k(), mlaTilingData.mm3.get_n(),
  mlaTilingData.mm3.get_m0(), mlaTilingData.mm3.get_k0(),
  mlaTilingData.mm3.get_n0(), mlaTilingData.mm3.get_mLoop(),
  mlaTilingData.mm3.get_kLoop(), mlaTilingData.mm3.get_nLoop(),
      mlaTilingData.mm3.get_coreLoop(), mlaTilingData.mm3.get_swizzleCount(),
  mlaTilingData.mm3.get_swizzleDirect(), mlaTilingData.mm3.get_blockDim());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: rmsNumCore1 is %ld.",
  mlaTilingData.get_rmsNumCore1()); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: rmsNumCol1 is %ld.", mlaTilingData.get_rmsNumCol1());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: rmsNumRow1 is %ld.",
  mlaTilingData.get_rmsNumRow1()); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: rmsQuantMin1 is %ld.",
  mlaTilingData.get_rmsQuantMin1());
  */
}

void MlaPreprocessTiling::PrintLastTilingData(gert::TilingContext* context) {
  /*
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: hiddtenState is %ld.",
  mlaTilingData.get_hiddtenState()); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: rmsNumCore2 is %ld.", mlaTilingData.get_rmsNumCore2());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: rmsNumCol2 is %ld.",
  mlaTilingData.get_rmsNumCol2()); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: rmsNumRow2 is %ld.", mlaTilingData.get_rmsNumRow2());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: rmsQuantMin2 is %ld.",
  mlaTilingData.get_rmsQuantMin2()); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: hiddenSizeQ is %ld.", mlaTilingData.get_hiddenSizeQ());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: headNumQ is %ld.",
  mlaTilingData.get_headNumQ()); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: headDim is %ld.", mlaTilingData.get_headDim());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: concatSize is %ld.",
  mlaTilingData.get_concatSize()); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: rotaryCoeff is %ld.", mlaTilingData.get_rotaryCoeff());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: ntokens is %ld.",
  mlaTilingData.get_ntokens()); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: realCore is %ld.", mlaTilingData.get_realCore());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: nlCoreRun is %ld.",
  mlaTilingData.get_nlCoreRun()); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: lCoreRun is %ld.", mlaTilingData.get_lCoreRun());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: maxNPerLoopForUb  is
  %ld.", mlaTilingData.get_maxNPerLoopForUb()); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: preCoreLoopTime   is %ld.",
            mlaTilingData.get_preCoreLoopTime());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: preCoreLoopNLast  is
  %ld.", mlaTilingData.get_preCoreLoopNLast()); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: lastCoreLoopTime  is %ld.",
            mlaTilingData.get_lastCoreLoopTime());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: lastCoreLoopNLast is
  %ld.", mlaTilingData.get_lastCoreLoopNLast()); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: esqFrontCore      is %ld.",
            mlaTilingData.get_esqFrontCore());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: esqTailCore       is
  %ld.", mlaTilingData.get_esqTailCore()); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: esqFrontCoreBatch is %ld.",
            mlaTilingData.get_esqFrontCoreBatch());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: esqTailCoreBatch  is
  %ld.", mlaTilingData.get_esqTailCoreBatch()); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: esqHeadNum        is %ld.",
            mlaTilingData.get_esqHeadNum());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: esqColNum         is
  %ld.", mlaTilingData.get_esqColNum()); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: esqUbHeadLoop     is %ld.",
            mlaTilingData.get_esqUbHeadLoop());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: esqHeadPerLoop    is
  %ld.", mlaTilingData.get_esqHeadPerLoop()); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: esqHeadTail       is %ld.",
            mlaTilingData.get_esqHeadTail());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: esqColLoop        is
  %ld.", mlaTilingData.get_esqColLoop()); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: esqColTail        is %ld.",
            mlaTilingData.get_esqColTail());
  OP_LOGD(context->GetNodeName(), "MlaPreprocess_tiling: epsilon           is
  %f.", mlaTilingData.get_epsilon()); OP_LOGD(context->GetNodeName(),
  "MlaPreprocess_tiling: maxWorkspaceSize  is %ld.",
            mlaTilingData.get_maxWorkspaceSize());
  */
}

ge::graphStatus MlaPreprocessTiling::Init(gert::TilingContext* context) {
  OpParam::MlaPreprocessParam param = MlaPreprocessTiling::GetParam(context);

  bool doRmsNorm =
      *(context->GetAttrs()->GetAttrPointer<bool>(ATTR_DO_RMS_NORM_IDX));
  mlaTilingData.set_doRmsNorm(doRmsNorm);
  mlaTilingData.set_qDownOutFlag(false);
  uint64_t hiddtenState =
      static_cast<uint64_t>(context->GetInputShape(INDEX_INPUT)
                                ->GetStorageShape()
                                .GetDim(DIM_1));  // hiddtenState
  mlaTilingData.set_hiddtenState(hiddtenState);

  bool doRmsNormQuant = true;
  if (context->GetInputDesc(INDEX_WDQKV)->GetDataType() == ge::DT_BF16 &&
      context->GetInputDesc(INDEX_WUQ)->GetDataType() == ge::DT_BF16) {
    doRmsNormQuant = false;
  }

  auto epsilon = context->GetAttrs()->GetAttrPointer<float>(ATTR_EPSILON_IDX);
  mlaTilingData.set_epsilon(*epsilon);

  auto inDtype = context->GetInputDesc(0)->GetDataType();

  auto platformInfo = context->GetPlatformInfo();
  // OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context,"platformInfo is
  // null"), return ge::GRAPH_FAILED);

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  const uint64_t& aicNum = ascendcPlatform.GetCoreNumAic();
  const uint64_t& aivNum = ascendcPlatform.GetCoreNumAiv();
  uint32_t sysWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

  uint64_t l0CSizePlatForm = 0;
  uint64_t l2SizePlatForm = 0;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,
                                 l0CSizePlatForm);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2,
                                 l2SizePlatForm);

  mlaTilingData.set_n(param.N);
  mlaTilingData.set_numCore(aicNum);
  bool deqOnTheFly = false;
  if (doRmsNormQuant && (inDtype == ge::DT_BF16 ||
                         param.quantMode == QuantMode::PER_TOKEN_SYMM_QUANT)) {
    deqOnTheFly = true;
  }

  RmsNormQuantTiling(param.N, aivNum, hiddtenState);
  RopeConcatTiling(param, aicNum);
  EinSumQuantTiling(param, aicNum, inDtype, doRmsNormQuant);
  auto tilingParamMm1 = &mlaTilingData.mm1;
  auto tilingParamMm2 = &mlaTilingData.mm2;
  auto tilingParamMm3 = &mlaTilingData.mm3;

  bool enDequant = doRmsNormQuant;
  // numBatch,m,k,n,transA,transB,enDequant
  PpMatmulTilingApi mm1TilingApi(1,
                                 param.N,
                                 hiddtenState,
                                 HIDDEN_STRATE_MM,
                                 false,
                                 true,
                                 enDequant,
                                 deqOnTheFly,
                                 aicNum,
                                 l0CSizePlatForm,
                                 l2SizePlatForm);
  mm1TilingApi.GetTilingData(tilingParamMm1);
  PpMatmulTilingApi mm2TilingApi(1,
                                 param.N,
                                 HIDDEN_STRATE_RMS,
                                 param.headNum * HIDDEN_STRATE_ROPE,
                                 false,
                                 true,
                                 enDequant,
                                 deqOnTheFly,
                                 aicNum,
                                 l0CSizePlatForm,
                                 l2SizePlatForm);
  mm2TilingApi.GetTilingData(tilingParamMm2);
  PpMatmulTilingApi mm3TilingApi(param.headNum,
                                 param.N,
                                 CONST_128,
                                 CONCAT_SIZE,
                                 false,
                                 false,
                                 false,
                                 deqOnTheFly,
                                 aicNum,
                                 l0CSizePlatForm,
                                 l2SizePlatForm);
  mm3TilingApi.GetTilingData(tilingParamMm3);

  SetMlapoWorkSpace(inDtype, param, sysWorkSpaceSize, context);
  context->SetBlockDim(aicNum);
  SetTilingKey(inDtype, param, doRmsNormQuant, context);
  PrintTilingData(context);

  return ge::GRAPH_SUCCESS;
}

OpParam::MlaPreprocessParam MlaPreprocessTiling::GetParam(
    gert::TilingContext* context) {
  OpParam::MlaPreprocessParam param;

  param.N = static_cast<uint64_t>(context->GetInputShape(INDEX_INPUT)
                                      ->GetStorageShape()
                                      .GetDim(DIM_0));  // token_num
  param.headNum = static_cast<uint64_t>(
      context->GetInputShape(INDEX_WUK)->GetStorageShape().GetDim(
          DIM_0));  // head_num

  auto attrPtr = context->GetAttrs();
  auto cacheModePtr = attrPtr->GetAttrPointer<uint64_t>(ATTR_CACHE_MODE_IDX);
  param.cacheMode = *cacheModePtr;

  auto quantModePtr = attrPtr->GetAttrPointer<QuantMode>(ATTR_QUANT_MODE_IDX);
  param.quantMode = *quantModePtr;
  return param;
}

}  // namespace optiling
