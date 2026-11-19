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
 * \file gamma_add_rms_norm_tiling.cpp
 * \brief
 */

#include "gamma_add_rms_norm_tiling.h"

#include "op_common/op_host/util/math_util.h"
#include "tiling_util.h"

namespace optiling {
constexpr uint32_t RMS_NORM_KEY = 0;
constexpr uint32_t PRE_RMS_NORM = 100;
constexpr uint32_t POST_RMS_NORM = 1000;
constexpr uint32_t DTYPE_KEY_FP16 = 1;
constexpr uint32_t DTYPE_KEY_FP32 = 2;
constexpr uint32_t DTYPE_KEY_BF16 = 3;
constexpr uint32_t UB_USED = 1024;
constexpr uint32_t UB_FACTOR_B16 = 12288;
constexpr uint32_t UB_FACTOR_B32 = 10240;
constexpr uint32_t UB_FACTOR_B16_CUTD = 12096;
constexpr uint32_t UB_FACTOR_B32_CUTD = 9696;
constexpr uint32_t BLOCK_ALIGN_NUM = 16;
constexpr uint32_t FLOAT_BLOCK_ALIGN_NUM = 8;
constexpr uint32_t SMALL_REDUCE_NUM = 2000;
constexpr uint32_t MODE_NORMAL = 0;
constexpr uint32_t MODE_SPLIT_D = 1;
constexpr uint32_t MODE_MERGE_N = 2;
constexpr uint32_t MODE_SINGLE_N = 3;
constexpr uint32_t MODE_MULTI_N = 4;
constexpr int32_t RMS_INPUT_X1_INDEX = 0;
constexpr int32_t RMS_INPUT_X2_INDEX = 1;
constexpr int32_t RMS_INPUT_GAMMA_INDEX = 2;
constexpr int32_t RMS_OUTPUT_Y_INDEX = 0;
constexpr int32_t RMS_OUTPUT_RSTD_INDEX = 1;
constexpr int32_t RMS_OUTPUT_X_INDEX = 2;
constexpr size_t MAX_DIM_NUM = 8;
constexpr size_t MIN_DIM_X = 1;
constexpr size_t MIN_DIM_GAMMA = 1;
constexpr size_t FP32_WEIGHT = 24;
constexpr size_t OTHER_WEIGHT = 18;
constexpr size_t DIV_FACTOR = 260;
constexpr size_t FLOAT_PER_REPEAT = 64;
constexpr size_t USE_SIZE = 256;
constexpr size_t NUM = 2;
constexpr int32_t TEN = 10;

constexpr int32_t PERFORMANC_DIM_ZERO = 0;
constexpr int32_t PERFORMANC_DIM_ONE = 1;
constexpr int32_t PERFORMANC_DIM_TWO = 2;
constexpr int32_t PERFORMANC_DIM_THREE = 3;
constexpr int32_t PERFORMANC_DIM_ONE_MAX = 512;
constexpr int32_t PERFORMANC_DIM_TWO_MAX = 8;
constexpr int32_t PERFORMANC_DIM_THREE_MAX = 5120;

static uint8_t getPerformanceFlag(uint32_t num_col,
                                  const gert::Shape& x_shape,
                                  const gert::Shape& gamma_shape,
                                  uint32_t xDtypeKey,
                                  platform_ascendc::SocVersion socVersion) {
  uint8_t isPerformance = 0;
  if (socVersion != platform_ascendc::SocVersion::ASCEND910B) {
    return isPerformance;
  }
  size_t xDimNum = x_shape.GetDimNum();
  size_t gammaDimNum = gamma_shape.GetDimNum();
  bool dimOK =
      ((xDimNum == PERFORMANC_DIM_TWO || xDimNum == PERFORMANC_DIM_THREE) &&
       gammaDimNum == PERFORMANC_DIM_ONE);
  bool sizeOk =
      num_col <= PERFORMANC_DIM_THREE_MAX &&
      ((xDimNum == PERFORMANC_DIM_TWO &&
        x_shape.GetDim(PERFORMANC_DIM_ZERO) <= PERFORMANC_DIM_ONE_MAX) ||
       (xDimNum == PERFORMANC_DIM_THREE &&
        x_shape.GetDim(PERFORMANC_DIM_ZERO) <= PERFORMANC_DIM_ONE_MAX &&
        x_shape.GetDim(PERFORMANC_DIM_ONE) <= PERFORMANC_DIM_TWO_MAX));
  bool dtypeOk = (xDtypeKey == DTYPE_KEY_FP16 || xDtypeKey == DTYPE_KEY_BF16);
  if (dimOK && sizeOk && dtypeOk) {
    isPerformance = 1;
  }
  return isPerformance;
}

static void SetByDtype(ge::DataType dataType,
                       uint32_t& dtypeKey,
                       uint32_t& dataPerBlock) {
  switch (dataType) {
    case ge::DT_FLOAT16:
      dtypeKey = DTYPE_KEY_FP16;
      dataPerBlock = BLOCK_ALIGN_NUM;
      break;
    case ge::DT_BF16:
      dtypeKey = DTYPE_KEY_BF16;
      dataPerBlock = BLOCK_ALIGN_NUM;
      break;
    default:
      dtypeKey = DTYPE_KEY_FP32;
      dataPerBlock = FLOAT_BLOCK_ALIGN_NUM;
      break;
  }
}
static bool CheckNullptr(const gert::TilingContext* context,
                         uint32_t& normKey) {
  const gert::StorageShape* x1_shape =
      context->GetInputShape(RMS_INPUT_X1_INDEX);
  const gert::StorageShape* x2_shape =
      context->GetInputShape(RMS_INPUT_X2_INDEX);
  const gert::StorageShape* gamma_shape =
      context->GetInputShape(RMS_INPUT_GAMMA_INDEX);
  const gert::StorageShape* y_shape =
      context->GetOutputShape(RMS_OUTPUT_Y_INDEX);
  const gert::StorageShape* rstd_shape =
      context->GetOutputShape(RMS_OUTPUT_RSTD_INDEX);
  const gert::StorageShape* x_shape =
      context->GetOutputShape(RMS_OUTPUT_X_INDEX);

  OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, x2_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, rstd_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

  normKey = RMS_NORM_KEY;
  if (rstd_shape->GetOriginShape().GetShapeSize() <= 0 &&
      x_shape->GetOriginShape().GetShapeSize() <= 0) {
    normKey = POST_RMS_NORM;
  } else if (rstd_shape->GetOriginShape().GetShapeSize() <= 0) {
    normKey = PRE_RMS_NORM;
  }
  return true;
}
static bool CheckInputOutputDim(const gert::TilingContext* context,
                                uint32_t normKey) {
  const gert::StorageShape* x1_shape =
      context->GetInputShape(RMS_INPUT_X1_INDEX);
  const gert::StorageShape* x2_shape =
      context->GetInputShape(RMS_INPUT_X2_INDEX);
  const gert::StorageShape* gamma_shape =
      context->GetInputShape(RMS_INPUT_GAMMA_INDEX);
  const gert::StorageShape* y_shape =
      context->GetOutputShape(RMS_OUTPUT_Y_INDEX);
  const gert::StorageShape* rstd_shape =
      context->GetOutputShape(RMS_OUTPUT_RSTD_INDEX);
  const gert::StorageShape* x_shape =
      context->GetOutputShape(RMS_OUTPUT_X_INDEX);

  size_t x1DimNum = x1_shape->GetStorageShape().GetDimNum();
  size_t x2DimNum = x2_shape->GetStorageShape().GetDimNum();
  size_t gammaDimNum = gamma_shape->GetStorageShape().GetDimNum();
  size_t yDimNum = y_shape->GetStorageShape().GetDimNum();
  size_t rstdDimNum = rstd_shape->GetStorageShape().GetDimNum();
  size_t xDimNum = x_shape->GetStorageShape().GetDimNum();

  OP_CHECK_IF(x1DimNum > MAX_DIM_NUM || x1DimNum < MIN_DIM_X,
              OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(),
                                           "x1",
                                           std::to_string(x1DimNum).c_str(),
                                           "within the range [1, 8]"),
              return false);
  if (normKey == RMS_NORM_KEY) {
    OP_CHECK_IF(
        gammaDimNum > MAX_DIM_NUM || gammaDimNum < MIN_DIM_GAMMA,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(),
                                     "gamma",
                                     std::to_string(gammaDimNum).c_str(),
                                     "within the range [1, 8]"),
        return false);
    OP_CHECK_IF(
        x1DimNum < gammaDimNum,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(),
            "x1 and gamma",
            (std::to_string(x1DimNum) + " and " + std::to_string(gammaDimNum))
                .c_str(),
            "The shape dim of x1 should be greater than or equal to the shape "
            "dim of gamma"),
        return false);
  } else if (normKey == PRE_RMS_NORM || normKey == POST_RMS_NORM) {
    OP_CHECK_IF(
        gammaDimNum != 2,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(),
                                     "gamma",
                                     std::to_string(gammaDimNum).c_str(),
                                     "2"),
        return false);
  }
  OP_CHECK_IF(x1DimNum != yDimNum,
              OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                  context->GetNodeName(),
                  "x1 and y",
                  (std::to_string(x1DimNum) + " and " + std::to_string(yDimNum))
                      .c_str(),
                  "The shape dims of x1 and y should be the same"),
              return false);

  OP_CHECK_IF(
      x1DimNum != x2DimNum,
      OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
          context->GetNodeName(),
          "x1 and x2",
          (std::to_string(x1DimNum) + " and " + std::to_string(x2DimNum))
              .c_str(),
          "The shape dims of x1 and x2 should be the same"),
      return false);

  if (normKey == RMS_NORM_KEY) {
    OP_CHECK_IF(
        (yDimNum != xDimNum) || (xDimNum != x1DimNum) ||
            (rstdDimNum != x1DimNum),
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(),
            "y, x, rstd and x1",
            (std::to_string(yDimNum) + ", " + std::to_string(xDimNum) + ", " +
             std::to_string(rstdDimNum) + " and " + std::to_string(x1DimNum))
                .c_str(),
            "The shape dims of y, x, rstd and x1 should be the same"),
        return false);
  } else if (normKey == PRE_RMS_NORM) {
    OP_CHECK_IF((yDimNum != xDimNum) || (xDimNum != x1DimNum),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(),
                    "y, x and x1",
                    (std::to_string(yDimNum) + ", " + std::to_string(xDimNum) +
                     " and " + std::to_string(x1DimNum))
                        .c_str(),
                    "The shape dims of y, x and x1 should be the same"),
                return false);
  }
  return true;
}

static bool CheckInputOutputShape(const gert::TilingContext* context,
                                  uint32_t normKey) {
  OP_CHECK_IF(!CheckInputOutputDim(context, normKey),
              OP_LOGE(context, "Input Dim invalid."),
              return false);
  const gert::StorageShape* x1_shape =
      context->GetInputShape(RMS_INPUT_X1_INDEX);
  const gert::StorageShape* x2_shape =
      context->GetInputShape(RMS_INPUT_X2_INDEX);
  const gert::StorageShape* gamma_shape =
      context->GetInputShape(RMS_INPUT_GAMMA_INDEX);
  const gert::StorageShape* y_shape =
      context->GetOutputShape(RMS_OUTPUT_Y_INDEX);
  const gert::StorageShape* rstd_shape =
      context->GetOutputShape(RMS_OUTPUT_RSTD_INDEX);
  const gert::StorageShape* x_shape =
      context->GetOutputShape(RMS_OUTPUT_X_INDEX);

  size_t x1DimNum = x1_shape->GetStorageShape().GetDimNum();
  size_t gammaDimNum = gamma_shape->GetStorageShape().GetDimNum();

  for (uint32_t i = 0; i < x1DimNum; i++) {
    OP_CHECK_IF(x1_shape->GetStorageShape().GetDim(i) == 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(),
                    "x1",
                    Ops::Base::ToString(x1_shape->GetStorageShape()).c_str(),
                    "x1 cannot be an empty tensor"),
                return false);
    OP_CHECK_IF(x2_shape->GetStorageShape().GetDim(i) !=
                    x1_shape->GetStorageShape().GetDim(i),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(),
                    "x2 and x1",
                    (Ops::Base::ToString(x2_shape->GetStorageShape()) +
                     " and " + Ops::Base::ToString(x1_shape->GetStorageShape()))
                        .c_str(),
                    "The shapes of x2 and x1 should be the same"),
                return false);
    OP_CHECK_IF((y_shape->GetStorageShape().GetDim(i) !=
                 x1_shape->GetStorageShape().GetDim(i)),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(),
                    "y and x1",
                    (Ops::Base::ToString(y_shape->GetStorageShape()) + " and " +
                     Ops::Base::ToString(x1_shape->GetStorageShape()))
                        .c_str(),
                    "The shapes of y and x1 should be the same"),
                return false);
    // x out shape check by mode
    if (normKey == RMS_NORM_KEY || normKey == PRE_RMS_NORM) {
      OP_CHECK_IF(
          (x_shape->GetStorageShape().GetDim(i) !=
           x1_shape->GetStorageShape().GetDim(i)),
          OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
              context->GetNodeName(),
              "x and x1",
              (Ops::Base::ToString(x_shape->GetStorageShape()) + " and " +
               Ops::Base::ToString(x1_shape->GetStorageShape()))
                  .c_str(),
              "The shapes of x and x1 should be the same"),
          return false);
    }
  }
  // rstd out shape check by mode
  if (normKey == RMS_NORM_KEY) {
    for (uint32_t i = 0; i < x1DimNum - gammaDimNum; i++) {
      OP_CHECK_IF(
          rstd_shape->GetStorageShape().GetDim(i) !=
              x2_shape->GetStorageShape().GetDim(i),
          OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
              context->GetNodeName(),
              "rstd and x1",
              (Ops::Base::ToString(rstd_shape->GetStorageShape()) + " and " +
               Ops::Base::ToString(x1_shape->GetStorageShape()))
                  .c_str(),
              ("The shape of rstd should be the same as the first " +
               std::to_string(x1DimNum - gammaDimNum) + " dim of x1")
                  .c_str()),
          return false);
    }
    for (uint32_t i = 0; i < gammaDimNum; i++) {
      OP_CHECK_IF(
          gamma_shape->GetStorageShape().GetDim(i) !=
              x1_shape->GetStorageShape().GetDim(x1DimNum - gammaDimNum + i),
          OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
              context->GetNodeName(),
              "gamma and x1",
              (Ops::Base::ToString(gamma_shape->GetStorageShape()) + " and " +
               Ops::Base::ToString(x1_shape->GetStorageShape()))
                  .c_str(),
              ("The shape of gamma should be equal to the last " +
               std::to_string(gammaDimNum) + " dim of x1")
                  .c_str()),
          return false);
      OP_CHECK_IF(
          rstd_shape->GetStorageShape().GetDim(x1DimNum - 1 - i) != 1,
          OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
              context->GetNodeName(),
              "rstd",
              Ops::Base::ToString(rstd_shape->GetStorageShape()).c_str(),
              ("The " + std::to_string(x1DimNum - 1 - i) +
               "th dimension of rstd must be 1")
                  .c_str()),
          return false);
    }
  } else if (normKey == PRE_RMS_NORM || normKey == POST_RMS_NORM) {
    OP_CHECK_IF((gamma_shape->GetStorageShape().GetDim(0) != 1 ||
                 gamma_shape->GetStorageShape().GetDim(gammaDimNum - 1) !=
                     x1_shape->GetStorageShape().GetDim(x1DimNum - 1)),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(),
                    "gamma and x1",
                    (Ops::Base::ToString(gamma_shape->GetStorageShape()) +
                     " and " + Ops::Base::ToString(x1_shape->GetStorageShape()))
                        .c_str(),
                    "The first dim of gamma should be 1 and the last dim of "
                    "gamma and x1 must be the same"),
                return false);
  }
  return true;
}

static void GetCompileParameters(gert::TilingContext* context,
                                 uint32_t& numCore,
                                 uint64_t& ubSize,
                                 platform_ascendc::SocVersion& socVersion) {
  auto ptrCompileInfo = reinterpret_cast<const GammaAddRmsNormCompileInfo*>(
      context->GetCompileInfo());
  if (ptrCompileInfo == nullptr) {
    auto ascendc_platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    socVersion = ascendc_platform.GetSocVersion();
    numCore = ascendc_platform.GetCoreNumAiv();
    ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  } else {
    numCore = ptrCompileInfo->totalCoreNum;
    ubSize = ptrCompileInfo->totalUbSize;
    socVersion = ptrCompileInfo->socVersion;
  }
  ubSize -= UB_USED;
}

static void CalculateRowAndColParameters(gert::TilingContext* context,
                                         uint32_t normKey,
                                         uint32_t& numRow,
                                         uint32_t& numCol) {
  const gert::Shape x1_shape = context->GetInputShape(0)->GetStorageShape();
  const size_t gammaIndex = 2;
  const gert::Shape gamma_shape =
      context->GetInputShape(gammaIndex)->GetStorageShape();
  numCol = gamma_shape.GetShapeSize();

  const size_t x1DimNum = x1_shape.GetDimNum();
  size_t gammaDimNum = gamma_shape.GetDimNum();
  if (normKey == PRE_RMS_NORM || normKey == POST_RMS_NORM) {
    gammaDimNum = gamma_shape.GetDimNum() - 1;
  }
  numRow = 1U;
  for (size_t i = 0; i < x1DimNum - gammaDimNum; ++i) {
    numRow *= x1_shape.GetDim(i);
  }
}

static ge::graphStatus GetEpsilonParameter(gert::TilingContext* context,
                                           float& epsilon) {
  auto attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  epsilon = *attrs->GetFloat(0);
  OP_CHECK_IF(epsilon < 0,
              OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(),
                                        "epsilon",
                                        std::to_string(epsilon).c_str(),
                                        "greater than or equal to zero"),
              return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAddGammaOffsetParameter(gert::TilingContext* context,
                                                  uint32_t& addGammaOffset) {
  auto attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const bool* addGammaOffsetPtr = attrs->GetBool(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, addGammaOffsetPtr);
  addGammaOffset = *addGammaOffsetPtr ? 1U : 0U;
  return ge::GRAPH_SUCCESS;
}

static void CalculateBlockParameters(uint32_t numRow,
                                     uint32_t numCore,
                                     uint32_t& blockFactor,
                                     uint32_t& latsBlockFactor,
                                     uint32_t& useCoreNum) {
  blockFactor = 1U;
  uint32_t tileNum = Ops::Base::CeilDiv(numRow, numCore * blockFactor);
  blockFactor *= tileNum;
  useCoreNum = Ops::Base::CeilDiv(numRow, blockFactor);
  latsBlockFactor = numRow - blockFactor * (useCoreNum - 1);
}

static ge::DataType SetDataTypeParameters(gert::TilingContext* context,
                                          uint32_t& dtype_key,
                                          uint32_t& data_per_block) {
  auto data_type = context->GetInputDesc(0)->GetDataType();
  dtype_key = DTYPE_KEY_FP16;
  SetByDtype(data_type, dtype_key, data_per_block);
  return data_type;
}

static void DetermineModeParameters(GammaAddRMSNormTilingData* tiling,
                                    uint32_t numCol,
                                    uint32_t& ubFactor,
                                    uint32_t& rowFactor,
                                    uint32_t blockFactor,
                                    uint32_t latsBlockFactor,
                                    ge::DataType dataType,
                                    uint32_t dtypKey,
                                    uint64_t ubSize,
                                    uint32_t dataPerBlock,
                                    uint32_t numColAlign,
                                    uint32_t& modeKey,
                                    uint32_t isPerformance) {
  if (numCol > ubFactor) {
    modeKey = MODE_SPLIT_D;
    ubFactor =
        (dataType == ge::DT_FLOAT) ? UB_FACTOR_B32_CUTD : UB_FACTOR_B16_CUTD;
    uint32_t colTileNum = Ops::Base::CeilDiv(numCol, ubFactor);
    ubFactor =
        Ops::Base::CeilDiv(numCol, colTileNum * dataPerBlock) * dataPerBlock;
  } else if (blockFactor == 1) {
    modeKey = MODE_SINGLE_N;
  } else if (numColAlign <= SMALL_REDUCE_NUM) {
    modeKey = MODE_MERGE_N;
    uint64_t numColAlignWeight =
        (dtypKey == DTYPE_KEY_FP32) ? FP32_WEIGHT : OTHER_WEIGHT;
    rowFactor = static_cast<uint32_t>(ubSize) /
                (numColAlign * static_cast<uint32_t>(numColAlignWeight) +
                 static_cast<uint32_t>(DIV_FACTOR));
    ubFactor = rowFactor * numColAlign;

    uint32_t mulLoopFp32 = numColAlign / 64;
    uint32_t mulTailFp32 = numColAlign - mulLoopFp32 * 64;
    uint8_t dstRepStrideFp32 = numColAlign / 8;

    uint32_t mulLoopFp16 = numColAlign / 128;
    uint32_t mulTailFp16 = numColAlign - mulLoopFp16 * 128;
    uint8_t dstRepStrideFp16 = numColAlign / 16;

    tiling->set_is_performance(isPerformance);
    tiling->set_mul_loop_fp32(mulLoopFp32);
    tiling->set_mul_tail_fp32(mulTailFp32);
    tiling->set_dst_rep_stride_fp32(dstRepStrideFp32);
    tiling->set_mul_loop_fp16(mulLoopFp16);
    tiling->set_mul_tail_fp16(mulTailFp16);
    tiling->set_dst_rep_stride_fp16(dstRepStrideFp16);
  } else if ((dataType == ge::DT_FLOAT16) && numCol == numColAlign) {
    modeKey = MODE_MULTI_N;
    rowFactor =
        (static_cast<uint32_t>(ubSize) - static_cast<uint32_t>(USE_SIZE) -
         numColAlign * static_cast<uint32_t>(NUM)) /
        (numColAlign * BLOCK_ALIGN_NUM +
         static_cast<uint32_t>(FLOAT_PER_REPEAT));
    ubFactor = rowFactor * numColAlign;
    if (rowFactor == 0U) {
      modeKey = MODE_NORMAL;
      rowFactor = FLOAT_PER_REPEAT;
      ubFactor = UB_FACTOR_B16;
    }
  }
  uint32_t rowLoop = Ops::Base::CeilDiv(blockFactor, rowFactor);
  uint32_t lastBlockRowLoop = Ops::Base::CeilDiv(latsBlockFactor, rowFactor);
  uint32_t rowTail = blockFactor - (rowLoop - 1) * rowFactor;
  uint32_t lastBlockRowTail =
      latsBlockFactor - (lastBlockRowLoop - 1) * rowFactor;
  tiling->set_row_loop(rowLoop);
  tiling->set_last_block_row_loop(lastBlockRowLoop);
  tiling->set_row_tail(rowTail);
  tiling->set_last_block_row_tail(lastBlockRowTail);
}

static void SetTilingParameters(GammaAddRMSNormTilingData* tiling,
                                uint32_t num_row,
                                uint32_t num_col,
                                uint32_t numColAlign,
                                uint32_t block_factor,
                                uint32_t latsBlockFactor,
                                uint32_t row_factor,
                                uint32_t ub_factor,
                                float epsilon,
                                uint32_t addGammaOffset) {
  const float avg_factor = (num_col == 0) ? 0 : 1.0f / num_col;
  tiling->set_num_row(num_row);
  tiling->set_num_col(num_col);
  tiling->set_num_col_align(numColAlign);
  tiling->set_block_factor(block_factor);
  tiling->set_last_block_factor(latsBlockFactor);
  tiling->set_row_factor(row_factor);
  tiling->set_ub_factor(ub_factor);
  tiling->set_epsilon(epsilon);
  tiling->set_avg_factor(avg_factor);
  tiling->set_add_gamma_offset(addGammaOffset);
}

static void SaveTilingData(gert::TilingContext* context,
                           GammaAddRMSNormTilingData* tiling,
                           uint32_t dtype_key,
                           uint32_t mode_key,
                           uint32_t normKey) {
  const uint32_t tiling_key = (dtype_key * 10 + mode_key) + normKey;
  context->SetTilingKey(tiling_key);
  tiling->SaveToBuffer(context->GetRawTilingData()->GetData(),
                       context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling->GetDataSize());
}

static void SetWorkspaceSize(gert::TilingContext* context) {
  size_t sysWorkspaceSize = 16 * 1024 * 1024;
  constexpr size_t usrSize = 256;
  size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = usrSize + sysWorkspaceSize;
}

static void LogTilingResults(gert::TilingContext* context,
                             GammaAddRMSNormTilingData* tiling,
                             uint32_t mode_key,
                             uint32_t dtype_key,
                             uint32_t use_core_num,
                             float epsilon,
                             uint32_t normKey) {
  OP_LOGI(context, "Tiling Key: %u", (dtype_key * TEN + mode_key) + normKey);
  OP_LOGI(context, "Block Dim: %u", use_core_num);
  OP_LOGI(context, "usr Workspace: 256");
  OP_LOGI(context,
          "num_row: %d, num_col: %d, block_factor: %d, row_factor: %d, "
          "ub_factor: %d, epsilon: %f, avg_factor: %f",
          tiling->get_num_row(),
          tiling->get_num_col(),
          tiling->get_block_factor(),
          tiling->get_row_factor(),
          tiling->get_ub_factor(),
          epsilon,
          tiling->get_avg_factor());
}

static ge::graphStatus Tiling4GammaAddRmsNorm(gert::TilingContext* context) {
  OP_LOGI("Tiling4GammaAddRmsNorm", "Enter Tiling4GammaAddRmsNorm");
  uint32_t normKey = RMS_NORM_KEY;
  OP_CHECK_IF(!CheckNullptr(context, normKey),
              OP_LOGE(context, "Input shape invalid (nullptr)."),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(!CheckInputOutputShape(context, normKey),
              OP_LOGE(context, "Input shape invalid."),
              return ge::GRAPH_FAILED);

  GammaAddRMSNormTilingData tiling;
  uint32_t num_core;
  uint64_t ub_size;
  platform_ascendc::SocVersion socVersion;

  GetCompileParameters(context, num_core, ub_size, socVersion);
  if (Ops::Xllm::OpTiling::IsRegbaseSocVersion(context)) {
    return optiling::gammaAddRmsNormRegbase::TilingGammaAddRmsNormRegbase(
        context);
  }

  uint32_t num_row;
  uint32_t num_col;
  CalculateRowAndColParameters(context, normKey, num_row, num_col);

  float epsilon = 0;
  if (GetEpsilonParameter(context, epsilon) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  uint32_t addGammaOffset = 0;
  if (GetAddGammaOffsetParameter(context, addGammaOffset) !=
      ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  uint32_t block_factor;
  uint32_t latsBlockFactor;
  uint32_t use_core_num;
  CalculateBlockParameters(
      num_row, num_core, block_factor, latsBlockFactor, use_core_num);
  context->SetBlockDim(use_core_num);

  uint32_t dtype_key;
  uint32_t data_per_block;
  ge::DataType data_type =
      SetDataTypeParameters(context, dtype_key, data_per_block);

  uint32_t mode_key = MODE_NORMAL;
  uint32_t row_factor = 64;
  uint32_t ub_factor =
      (dtype_key == DTYPE_KEY_FP32) ? UB_FACTOR_B32 : UB_FACTOR_B16;
  uint32_t numColAlign =
      Ops::Base::CeilDiv(num_col, data_per_block) * data_per_block;
  const gert::Shape x1_shape = context->GetInputShape(0)->GetStorageShape();
  const gert::Shape gamma_shape = context->GetInputShape(2)->GetStorageShape();
  uint8_t isPerformance =
      getPerformanceFlag(num_col, x1_shape, gamma_shape, dtype_key, socVersion);
  DetermineModeParameters(&tiling,
                          num_col,
                          ub_factor,
                          row_factor,
                          block_factor,
                          latsBlockFactor,
                          data_type,
                          dtype_key,
                          ub_size,
                          data_per_block,
                          numColAlign,
                          mode_key,
                          isPerformance);

  SetTilingParameters(&tiling,
                      num_row,
                      num_col,
                      numColAlign,
                      block_factor,
                      latsBlockFactor,
                      row_factor,
                      ub_factor,
                      epsilon,
                      addGammaOffset);
  SaveTilingData(context, &tiling, dtype_key, mode_key, normKey);

  SetWorkspaceSize(context);

  LogTilingResults(
      context, &tiling, mode_key, dtype_key, use_core_num, epsilon, normKey);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GammaAddRmsNorm(
    gert::TilingParseContext* context) {
  OP_LOGI(context, "TilingPrepare4GammaAddRmsNorm running.");
  auto compileInfo = context->GetCompiledInfo<GammaAddRmsNormCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
  auto platformInfo = context->GetPlatformInfo();
  OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

  compileInfo->socVersion = ascendcPlatform.GetSocVersion();
  compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,
                                 compileInfo->totalUbSize);

  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(XllmKernelGammaAddRmsNorm)
    .Tiling(Tiling4GammaAddRmsNorm)
    .TilingParse<GammaAddRmsNormCompileInfo>(TilingPrepare4GammaAddRmsNorm);

}  // namespace optiling
