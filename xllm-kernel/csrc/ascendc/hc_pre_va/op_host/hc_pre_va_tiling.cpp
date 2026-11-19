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
 * \file hc_pre_va_tiling.cpp
 * \brief
 */

#include "hc_pre_va_tiling.h"

#include <cmath>
#include <sstream>

using namespace ge;
namespace optiling {
namespace {
constexpr uint64_t WORKSPACE_SIZE = 32;
int64_t CeilDiv(int64_t x, int64_t y) {
  if (y != 0) {
    return (x + y - 1) / y;
  }
  return x;
}
int64_t DownAlign(int64_t x, int64_t y) {
  if (y == 0) {
    return x;
  }
  return (x / y) * y;
}
int64_t RoundUp(int64_t x, int64_t y) { return CeilDiv(x, y) * y; }

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t UB_RESEVED_SIZE = 8192;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t MAX_VECTOR_REPEAT = 255;
constexpr uint64_t M_L1_MAX_SIZE = 256;
constexpr uint64_t K_MULIT_CORE_SPLIT_BASE_SIZE = 256;
constexpr uint64_t A_L1_SIZE = 128 * 256;
constexpr uint64_t K_L1_MAX_SIZE = 1024;
constexpr int64_t K_L1_ALIGN_SIZE = 128;
constexpr int64_t HC_MULT_ATTR_IDX = 0;
constexpr int64_t ITER_TIMES_ATTR_IDX = 1;
constexpr int64_t HC_EPS_ATTR_IDX = 2;
constexpr int64_t NORM_EPS_ATTR_IDX = 3;
constexpr int64_t DEFAULT_ITER_TIMES = 20;
}  // namespace

ge::graphStatus HcPreVaTiling::GetPlatformInfo() {
  auto platformInfo = context_->GetPlatformInfo();
  if (platformInfo == nullptr) {
    auto compileInfoPtr = context_->GetCompileInfo<HcPreVaCompileInfo>();
    OPS_ERR_IF(compileInfoPtr == nullptr,
               OPS_LOG_E(context_, "compile info is null"),
               return ge::GRAPH_FAILED);
    aivCoreNum_ = compileInfoPtr->coreNum;
    aicCoreNum_ = compileInfoPtr->aicCoreNum;
    ubSize_ = compileInfoPtr->ubSize;
  } else {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    aicCoreNum_ = ascendcPlatform.GetCoreNumAic();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,
                                   ubSizePlatForm);
    ubSize_ = ubSizePlatForm;
  }
  OPS_ERR_IF(
      aivCoreNum_ == 0 || aicCoreNum_ == 0 || ubSize_ == 0,
      OPS_LOG_E(context_->GetNodeName(),
                "AIV core count, AIC core count and UB size must be positive"),
      return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreVaTiling::GetAttr() {
  auto* attrs = context_->GetAttrs();
  OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);

  auto hcMultAttr = attrs->GetAttrPointer<int64_t>(HC_MULT_ATTR_IDX);
  const int64_t hc_mult_attribute = hcMultAttr == nullptr ? 4 : *hcMultAttr;
  OPS_ERR_IF(hcMult_ != 4 || hc_mult_attribute != hcMult_,
             OPS_LOG_E(context_->GetNodeName(),
                       "x streams and hc_mult must both equal 4"),
             return ge::GRAPH_FAILED);

  auto iterTimesAttr = attrs->GetAttrPointer<int64_t>(ITER_TIMES_ATTR_IDX);
  iterTimes_ = iterTimesAttr == nullptr ? DEFAULT_ITER_TIMES : *iterTimesAttr;
  OPS_ERR_IF(
      iterTimes_ <= 0,
      OPS_LOG_E(context_->GetNodeName(), "hc_sinkhorn_iters must be positive"),
      return ge::GRAPH_FAILED);

  auto hcEpsAttr = attrs->GetAttrPointer<float>(HC_EPS_ATTR_IDX);
  hcEps_ = hcEpsAttr == nullptr ? 1e-6 : *hcEpsAttr;

  auto normEpsAttr = attrs->GetAttrPointer<float>(NORM_EPS_ATTR_IDX);
  normEps_ = normEpsAttr == nullptr ? 1e-6 : *normEpsAttr;
  OPS_ERR_IF(!std::isfinite(hcEps_) || hcEps_ <= 0 ||
                 !std::isfinite(normEps_) || normEps_ <= 0,
             OPS_LOG_E(context_->GetNodeName(),
                       "hc_eps and norm_eps must be finite and positive"),
             return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreVaTiling::GetShapeAttrsInfoInner() {
  // (b, s, hc_mult, d) or (bs, hc_mult, d)
  auto xShape = context_->GetInputShape(0);
  OPS_LOG_E_IF_NULL(context_, xShape, return ge::GRAPH_FAILED);
  size_t xDimNum = xShape->GetStorageShape().GetDimNum();
  OPS_ERR_IF(xDimNum != 3 && xDimNum != 4,
             OPS_LOG_E(context_->GetNodeName(), "x rank must be 3 or 4"),
             return ge::GRAPH_FAILED);
  for (size_t dimension = 0; dimension < xDimNum; ++dimension) {
    OPS_ERR_IF(
        xShape->GetStorageShape().GetDim(dimension) <= 0,
        OPS_LOG_E(context_->GetNodeName(), "x dimensions must be positive"),
        return ge::GRAPH_FAILED);
  }
  if (xDimNum == 3) {
    bs_ = xShape->GetStorageShape().GetDim(0);
    hcMult_ = xShape->GetStorageShape().GetDim(1);
    d_ = xShape->GetStorageShape().GetDim(2);
  } else if (xDimNum == 4) {
    int64_t b = xShape->GetStorageShape().GetDim(0);
    int64_t s = xShape->GetStorageShape().GetDim(1);
    bs_ = b * s;
    hcMult_ = xShape->GetStorageShape().GetDim(2);
    d_ = xShape->GetStorageShape().GetDim(3);
  }

  auto shapeHcFn = context_->GetInputShape(1);
  OPS_LOG_E_IF_NULL(context_, shapeHcFn, return ge::GRAPH_FAILED);
  OPS_ERR_IF(shapeHcFn->GetStorageShape().GetDimNum() != 2,
             OPS_LOG_E(context_->GetNodeName(), "hc_fn must have rank 2"),
             return ge::GRAPH_FAILED);
  hcMix_ = shapeHcFn->GetStorageShape().GetDim(0);
  OPS_ERR_IF(hcMix_ != 24,
             OPS_LOG_E(context_->GetNodeName(), "hc_fn must have 24 rows"),
             return ge::GRAPH_FAILED);
  OPS_ERR_IF(
      shapeHcFn->GetStorageShape().GetDim(1) != d_ * hcMult_,
      OPS_LOG_E(context_->GetNodeName(),
                "HcFn dim 1 should be equal with d_ * hcMult_  %ld, but is %ld",
                d_ * hcMult_,
                shapeHcFn->GetStorageShape().GetDim(1)),
      return ge::GRAPH_FAILED);

  auto shapeHcScale = context_->GetInputShape(2);
  OPS_LOG_E_IF_NULL(context_, shapeHcScale, return ge::GRAPH_FAILED);
  OPS_ERR_IF(shapeHcScale->GetStorageShape().GetDimNum() != 1,
             OPS_LOG_E(context_->GetNodeName(), "hc_scale must have rank 1"),
             return ge::GRAPH_FAILED);
  int64_t scaleFirstDim = shapeHcScale->GetStorageShape().GetDim(0);
  OPS_ERR_IF(scaleFirstDim != 3,
             OPS_LOG_E(context_->GetNodeName(),
                       "hc_scale size should be equal with 3, but is %ld",
                       scaleFirstDim),
             return ge::GRAPH_FAILED);

  auto shapeHcBase = context_->GetInputShape(3);
  OPS_LOG_E_IF_NULL(context_, shapeHcBase, return ge::GRAPH_FAILED);
  OPS_ERR_IF(shapeHcBase->GetStorageShape().GetDimNum() != 1,
             OPS_LOG_E(context_->GetNodeName(), "hc_base must have rank 1"),
             return ge::GRAPH_FAILED);
  int64_t baseFirstDim = shapeHcBase->GetStorageShape().GetDim(0);
  OPS_ERR_IF(baseFirstDim != hcMix_,
             OPS_LOG_E(context_->GetNodeName(),
                       "hc_base size should be equal with mixhc, but is %ld",
                       baseFirstDim),
             return ge::GRAPH_FAILED);

  // 可选输入 pre_mix: [bs, hc_mult] fp32
  tilingData_.set_hasPreMix(context_->GetInputShape(4) != nullptr ? 1 : 0);
  auto pre_mix_shape = context_->GetInputShape(4);
  if (pre_mix_shape != nullptr) {
    OPS_ERR_IF(pre_mix_shape->GetStorageShape().GetDimNum() != xDimNum - 1,
               OPS_LOG_E(context_->GetNodeName(),
                         "pre_mix rank must equal x rank minus one"),
               return ge::GRAPH_FAILED);
    for (size_t dimension = 0; dimension + 1 < xDimNum; ++dimension) {
      OPS_ERR_IF(pre_mix_shape->GetStorageShape().GetDim(dimension) !=
                     xShape->GetStorageShape().GetDim(dimension),
                 OPS_LOG_E(context_->GetNodeName(),
                           "pre_mix shape must equal x shape without D"),
                 return ge::GRAPH_FAILED);
    }
  }
  // 可选输出 pre: 内部计算的 pre 结果拷出
  tilingData_.set_hasPreOut(context_->GetOutputShape(3) != nullptr ? 1 : 0);

  OPS_ERR_IF(GetAttr() != ge::GRAPH_SUCCESS,
             OPS_LOG_E(context_->GetNodeName(), "get attr failed."),
             return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreVaTiling::CalcMKSplitCoreMembasePart2Tiling() {
  rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(aivCoreNum_));
  usedAivCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_),
                              static_cast<int64_t>(aivCoreNum_));
  rowOfTailBlock_ = bs_ - (usedAivCoreNums_ - 1) * rowOfFormerBlock_;

  int64_t minRowPerCore = 1;
  int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);
  int64_t kBlockNum = tilingData_.get_cubeBlockDimK();
  const int64_t available_ub_size =
      static_cast<int64_t>(ubSize_) - UB_RESEVED_SIZE;

  hcMultAlign_ = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
  int64_t mix0OriginSize =
      kBlockNum * rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
  int64_t mix1OriginSize =
      kBlockNum * rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
  int64_t mix2OriginSize = kBlockNum * rowOnceLoop * hcMult_ * hcMultAlign_ *
                           sizeof(float) * DOUBLE_BUFFER;
  int64_t mix0Size = rowOnceLoop * hcMultAlign_ * sizeof(float);
  int64_t mix1Size = rowOnceLoop * hcMultAlign_ * sizeof(float);
  int64_t mix2Size = rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float);
  int64_t squareSumSize =
      kBlockNum * RoundUp(rowOnceLoop * 16, 16) * sizeof(float) * DOUBLE_BUFFER;
  int64_t rsqrtSize = RoundUp(rowOnceLoop, BLOCK_SIZE / sizeof(float)) *
                      sizeof(float) * DOUBLE_BUFFER;
  int64_t xSize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 *
                  DOUBLE_BUFFER;  // x是bfloat16_t 类型
  int64_t ySize = rowOnceLoop * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
  int64_t postSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
  int64_t combFragSize =
      rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
  int64_t base0Size = hcMultAlign_ * sizeof(float);
  int64_t base1Size = hcMultAlign_ * sizeof(float);
  int64_t base2Size = hcMult_ * hcMultAlign_ * sizeof(float);
  int64_t xCastSize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * sizeof(float);
  int64_t yCastSize = rowOnceLoop * RoundUp(d_, 16) * sizeof(float);
  int64_t rowBrcb0Size = RoundUp(rowOnceLoop, 8) * BLOCK_SIZE;
  int64_t hcBrcb1Size = RoundUp(rowOnceLoop * hcMultAlign_, 8) * BLOCK_SIZE;
  int64_t reduceBufSize = rowOnceLoop * hcMultAlign_ * sizeof(float);
  int64_t maskPatternSize = BLOCK_SIZE * (rowOnceLoop + 15);
  int64_t squareReduceSize = rowOnceLoop * 16 * sizeof(float);
  int64_t preMixSize = rowOnceLoop * hcMultAlign_ * sizeof(float);

  int64_t totalSize = mix0OriginSize + mix1OriginSize + mix2OriginSize +
                      mix0Size + mix1Size + mix2Size + squareSumSize +
                      rsqrtSize + xSize + ySize + postSize + combFragSize +
                      base0Size + base1Size + base2Size + xCastSize +
                      yCastSize + rowBrcb0Size + hcBrcb1Size + reduceBufSize +
                      maskPatternSize + squareReduceSize + preMixSize;
  rowFactor_ = rowOnceLoop;
  if (totalSize <= available_ub_size) {
    // row和d均可以在ub内全载
    dLoop_ = 1;
    dFactor_ = d_;
    tailDFactor_ = dFactor_;
  } else {
    int64_t usedUbSize =
        mix0OriginSize + mix1OriginSize + mix2OriginSize + mix0Size + mix1Size +
        mix2Size + squareSumSize + rsqrtSize + postSize + combFragSize +
        base0Size + base1Size + base2Size + rowBrcb0Size + hcBrcb1Size +
        reduceBufSize + maskPatternSize + squareReduceSize + preMixSize;
    int64_t ubRemain = available_ub_size - usedUbSize;
    OPS_ERR_IF(ubRemain <= 0,
               OPS_LOG_E(context_->GetNodeName(),
                         "stage 2 buffers exceed available UB"),
               return ge::GRAPH_FAILED);
    dFactor_ = d_;
    int64_t base = 2;
    while (1) {
      dFactor_ = CeilDiv(d_, base);
      xSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 16) * 2 *
              DOUBLE_BUFFER;  // x是bfloat16_t 类型
      ySize = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
      xCastSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 16) * sizeof(float);
      yCastSize = rowOnceLoop * RoundUp(dFactor_, 16) * sizeof(float);
      int64_t targetSize = xSize + ySize + xCastSize + yCastSize;
      if (targetSize <= ubRemain) {
        break;
      }
      OPS_ERR_IF(dFactor_ == 1,
                 OPS_LOG_E(context_->GetNodeName(),
                           "a single hidden column does not fit in UB"),
                 return ge::GRAPH_FAILED);
      base++;
    }
    if (dFactor_ > 32) {
      dFactor_ = DownAlign(dFactor_, 32);
    }
    dLoop_ = CeilDiv(d_, dFactor_);
    tailDFactor_ = d_ % dFactor_ == 0 ? dFactor_ : d_ % dFactor_;
  }

  // Softmax reduces hcMult rows per token; vector repeats are limited to 255.
  const int64_t maxRowFactor =
      std::min(rowOfFormerBlock_, MAX_VECTOR_REPEAT / hcMult_);
  // d全载,尝试搬入更多的bs
  if (dFactor_ == d_) {
    while (rowFactor_ <= maxRowFactor) {
      mix0OriginSize =
          kBlockNum * rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
      mix1OriginSize =
          kBlockNum * rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
      mix2OriginSize = kBlockNum * rowFactor_ * hcMult_ * hcMultAlign_ *
                       sizeof(float) * DOUBLE_BUFFER;
      mix0Size = rowFactor_ * hcMultAlign_ * sizeof(float);
      mix1Size = rowFactor_ * hcMultAlign_ * sizeof(float);
      mix2Size = rowFactor_ * hcMult_ * hcMultAlign_ * sizeof(float);
      squareSumSize = kBlockNum * RoundUp(rowFactor_ * 16, 16) * sizeof(float) *
                      DOUBLE_BUFFER;
      rsqrtSize = RoundUp(rowFactor_, BLOCK_SIZE / sizeof(float)) *
                  sizeof(float) * DOUBLE_BUFFER;
      xSize = rowFactor_ * hcMult_ * RoundUp(d_, 16) * 2 *
              DOUBLE_BUFFER;  // x是bfloat16_t 类型
      ySize = rowFactor_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
      postSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
      combFragSize =
          rowFactor_ * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
      xCastSize = rowFactor_ * hcMult_ * RoundUp(d_, 16) * sizeof(float);
      yCastSize = rowFactor_ * RoundUp(d_, 16) * sizeof(float);
      rowBrcb0Size = RoundUp(rowFactor_, 8) * BLOCK_SIZE;
      hcBrcb1Size = RoundUp(rowFactor_ * hcMultAlign_, 8) * BLOCK_SIZE;
      reduceBufSize = rowFactor_ * hcMultAlign_ * sizeof(float);
      maskPatternSize = BLOCK_SIZE * (rowFactor_ + 15);
      squareReduceSize = rowFactor_ * 16 * sizeof(float);
      preMixSize = rowFactor_ * hcMultAlign_ * sizeof(float);
      totalSize = mix0OriginSize + mix1OriginSize + mix2OriginSize + mix0Size +
                  mix1Size + mix2Size + squareSumSize + rsqrtSize + xSize +
                  ySize + postSize + combFragSize + base0Size + base1Size +
                  base2Size + xCastSize + yCastSize + rowBrcb0Size +
                  hcBrcb1Size + reduceBufSize + maskPatternSize +
                  squareReduceSize + preMixSize;
      if (totalSize > available_ub_size) {
        rowFactor_ = rowFactor_ - 1;
        break;
      }
      rowFactor_ = rowFactor_ + 1;
    }
    rowFactor_ = rowFactor_ > maxRowFactor ? rowFactor_ - 1 : rowFactor_;
  }
  rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
  rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
  tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0
                                    ? rowFactor_
                                    : rowOfFormerBlock_ % rowFactor_;
  tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0
                                  ? rowFactor_
                                  : rowOfTailBlock_ % rowFactor_;

  tilingData_.set_bs(bs_);
  tilingData_.set_hcMix(hcMix_);
  tilingData_.set_hcMult(hcMult_);
  tilingData_.set_d(d_);
  tilingData_.set_hcMultAlign(hcMultAlign_);
  tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
  tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
  tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
  tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
  tilingData_.set_stage2RowFactor(rowFactor_);
  tilingData_.set_secondUsedCoreNum(usedAivCoreNums_);
  tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
  tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
  tilingData_.set_dLoop(dLoop_);
  tilingData_.set_dFactor(dFactor_);
  tilingData_.set_tailDFactor(tailDFactor_);
  tilingData_.set_iterTimes(iterTimes_);
  tilingData_.set_hcEps(hcEps_);
  tilingData_.set_normEps(normEps_);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreVaTiling::CalcOpTiling() {
  uint64_t kSize = hcMult_ * d_;
  tilingData_.set_k(kSize);
  // 计算bs_轴切核
  uint64_t mDimNum =
      std::min(aicCoreNum_, static_cast<uint64_t>(CeilDiv(bs_, M_L1_MAX_SIZE)));
  uint64_t singleCoreM = RoundUp(CeilDiv(bs_, mDimNum), AscendC::BLOCK_CUBE);
  uint64_t kDimNum = aicCoreNum_ / mDimNum;
  uint64_t splitKSize =
      RoundUp(CeilDiv(kSize, kDimNum), K_MULIT_CORE_SPLIT_BASE_SIZE);

  tilingData_.set_cubeBlockDimM(mDimNum);
  tilingData_.set_cubeBlockDimK(CeilDiv(kSize, splitKSize));
  tilingData_.set_multCoreSplitMSize(singleCoreM);
  tilingData_.set_mL1Size(std::min(M_L1_MAX_SIZE, singleCoreM));
  tilingData_.set_multCoreSplitKSize(splitKSize);
  tilingData_.set_kL1Size(std::min(A_L1_SIZE / tilingData_.get_mL1Size(),
                                   static_cast<uint64_t>(K_L1_MAX_SIZE)) /
                          K_L1_ALIGN_SIZE * K_L1_ALIGN_SIZE);

  tilingData_.set_cvLoopKSize(1024);

  // vector stage1 tiling
  tilingData_.set_cubeCoreNum(static_cast<int64_t>(aicCoreNum_));
  // x type bfloat16, y type float32 and double
  // exit node 1 b16 input Queue and 1 b32 output Queue
  int64_t lineByteSize = (sizeof(int16_t) + sizeof(int32_t)) * DOUBLE_BUFFER *
                         tilingData_.get_cvLoopKSize();
  int64_t stage1MFactorValue = ubSize_ / lineByteSize;
  tilingData_.set_stage1MFactor(stage1MFactorValue);
  return CalcMKSplitCoreMembasePart2Tiling();
}

ge::graphStatus HcPreVaTiling::DoOpTiling() {
  if (GetPlatformInfo() == ge::GRAPH_FAILED) {
    return ge::GRAPH_FAILED;
  }

  if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
    return ge::GRAPH_FAILED;
  }

  if (CalcOpTiling() == ge::GRAPH_FAILED) {
    return ge::GRAPH_FAILED;
  }

  if (GetWorkspaceSize() == ge::GRAPH_FAILED) {
    return ge::GRAPH_FAILED;
  }

  if (PostTiling() == ge::GRAPH_FAILED) {
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreVaTiling::GetWorkspaceSize() {
  uint64_t xCastFp32BufSize =
      tilingData_.get_mL1Size() * RoundUp(tilingData_.get_cvLoopKSize(), 128);
  uint64_t workspaceSize1 = tilingData_.get_cubeCoreNum() * DOUBLE_BUFFER *
                            xCastFp32BufSize * sizeof(float);

  uint64_t mmLastAxisSize = RoundUp(tilingData_.get_hcMix(), 128);
  uint64_t workspaceSize2 =
      RoundUp(tilingData_.get_cubeBlockDimK() * tilingData_.get_bs() *
                  mmLastAxisSize * sizeof(float),
              512);

  uint64_t squareSumSize =
      RoundUp(tilingData_.get_cubeBlockDimK() *
                  RoundUp(tilingData_.get_bs(), 16) * 16 * sizeof(float),
              512);

  uint64_t requiredSize = workspaceSize1 + workspaceSize2 + squareSumSize +
                          16 * 1024 * 1024;  // 16MB 预留缓冲

  uint64_t defaultSize = 16 * 1024 * 1024 + 192 * 1024 * 1024;  // 208MB
  workspaceSize_ = requiredSize > defaultSize ? requiredSize : defaultSize;

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreVaTiling::PostTiling() {
  context_->SetTilingKey(0);
  context_->SetBlockDim(aicCoreNum_);
  size_t* workspaces = context_->GetWorkspaceSizes(1);
  workspaces[0] = workspaceSize_;
  tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                           context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForHcPreVa(gert::TilingParseContext* context) {
  OPS_ERR_IF(context == nullptr,
             OPS_REPORT_VECTOR_INNER_ERR("XllmKernelHcPreVa",
                                         "Tiling parse context is null"),
             return ge::GRAPH_FAILED);
  auto compileInfo = context->GetCompiledInfo<HcPreVaCompileInfo>();
  OPS_LOG_E_IF_NULL(context, compileInfo, return ge::GRAPH_FAILED);
  auto platformInfo = context->GetPlatformInfo();
  OPS_LOG_E_IF_NULL(context, platformInfo, return ge::GRAPH_FAILED);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
  compileInfo->aicCoreNum = ascendcPlatform.GetCoreNumAic();
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,
                                 compileInfo->ubSize);
  OPS_ERR_IF(
      compileInfo->coreNum == 0 || compileInfo->aicCoreNum == 0 ||
          compileInfo->ubSize == 0,
      OPS_LOG_E(context->GetNodeName(),
                "AIV core count, AIC core count and UB size must be positive"),
      return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForHcPreVa(gert::TilingContext* context) {
  OPS_ERR_IF(context == nullptr,
             OPS_REPORT_VECTOR_INNER_ERR("XllmKernelHcPreVa",
                                         "Tiling context is null"),
             return ge::GRAPH_FAILED);

  HcPreVaTiling hcPreTiling(context);
  return hcPreTiling.DoOpTiling();
}

IMPL_OP_OPTILING(XllmKernelHcPreVa)
    .Tiling(TilingForHcPreVa)
    .TilingParse<HcPreVaCompileInfo>(TilingPrepareForHcPreVa);

}  // namespace optiling
