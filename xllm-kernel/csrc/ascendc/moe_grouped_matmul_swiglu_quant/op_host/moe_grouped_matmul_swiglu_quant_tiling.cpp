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
 * \file grouped_matmul_swiglu_quant_tiling.cpp
 * \brief
 */
#include "moe_grouped_matmul_swiglu_quant_tiling.h"

#include <graph/utils/type_utils.h>

#include <climits>

#include "register/op_impl_registry.h"
using namespace ge;
using namespace AscendC;
using namespace MoeGroupedMatmulSwigluQuantTiling;

#define OP_LOGD(nodeName, fmt, ...) \
  printf(fmt, ##__VA_ARGS__);       \
  printf("\n")
#define OP_LOGE(nodeName, fmt, ...) \
  printf(fmt, ##__VA_ARGS__);       \
  printf("\n")

namespace {
template <typename T>
static inline auto AlignUp(T a, T base) -> T {
  if (base == 0) {
    return 0;
  }
  return (a + base - 1) / base * base;
}
}  // namespace

namespace optiling {

struct MoeGMMSwigluCompileInfo {
  uint64_t ubSize_ = 0;
  uint32_t aicNum_ = 0;
  uint32_t baseM_ = 128;
  uint32_t baseN_ = 256;
};

static int64_t CalMaxRowInUb(const gert::TilingContext* context,
                             const uint64_t ubSize,
                             const uint64_t n) {
  // 512B
  uint64_t tmpBufSize = (n / SWIGLU_REDUCE_FACTOR) * FP32_DTYPE_SIZE;
  // 2K
  uint64_t perchannleBufSize = n * FP32_DTYPE_SIZE * DOUBLE_BUFFER;
  // 32
  uint64_t reduceMaxResBufSize = BLOCK_BYTE;
  // 32
  uint64_t reduceMaxTmpBufSize = BLOCK_BYTE;
  const uint64_t CONSTANT_TERM = 64;
  // 193984
  int64_t remainUbSize = ubSize - tmpBufSize - perchannleBufSize -
                         reduceMaxResBufSize - reduceMaxTmpBufSize;
  int64_t maxRowInUb =
      remainUbSize /
      (n * INT32_DTYPE_SIZE + n / SWIGLU_REDUCE_FACTOR + FP32_DTYPE_SIZE) /
      DOUBLE_BUFFER;
  int64_t curUb =
      DOUBLE_BUFFER *
      (maxRowInUb * (INT32_DTYPE_SIZE * n + n / SWIGLU_REDUCE_FACTOR) +
       AlignUp(maxRowInUb, FP32_BLOCK_SIZE) * FP32_DTYPE_SIZE);
  // printf("ub_size %d tmpBufSize %d perchannleBufSize %d reduceMaxResBufSize
  // %d reduceMaxTmpBufSize %d \n", ubSize, tmpBufSize, perchannleBufSize,
  // reduceMaxResBufSize, reduceMaxTmpBufSize);
  if (curUb > remainUbSize) {
    // 64 : make sure ub does not excceed maxUbSize after align up to 8
    maxRowInUb =
        (remainUbSize - CONSTANT_TERM) /
        (n * INT32_DTYPE_SIZE + n / SWIGLU_REDUCE_FACTOR + FP32_DTYPE_SIZE) /
        DOUBLE_BUFFER;
  }
  if (maxRowInUb < 1) {
    // when n > (ubSize - 72) / 19 = 10330, maxRowInUb < 1
    OP_LOGE(context->GetNodeName(),
            "GMM_SWIGLU_QUANT TILING: n should not be greater than 10240, now "
            "is %lu\n",
            n);
  }
  return maxRowInUb;
}

static void SetTilingKey(gert::TilingContext* context, bool isSplitWorkSpace) {
  if (isSplitWorkSpace) {
    context->SetTilingKey(SPLITWORKSPACE_TILING_KEY_MODE);
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
  } else {
    context->SetTilingKey(COMMON_TILING_KEY_MODE);
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
  }
}

graphStatus TilingMoeGMMSwigluQuant(gert::TilingContext* context) {
  // set info
  auto xDesc = context->GetInputDesc(X_INDEX);
  auto weightDesc = context->GetInputDesc(WEIGHT_INDEX);
  ge::DataType xDType = xDesc->GetDataType();
  ge::DataType weightDType = weightDesc->GetDataType();

  auto compileInfoPtr = context->GetCompileInfo<MoeGMMSwigluCompileInfo>();
  auto xTensor = context->GetInputTensor(X_INDEX);
  const int64_t m = xTensor->GetStorageShape().GetDim(0);
  const int64_t k = xTensor->GetStorageShape().GetDim(1);
  auto wTensor = context->GetInputTensor(WEIGHT_INDEX);
  int64_t n = 0;
  auto dim_num = wTensor->GetStorageShape().GetDimNum();
  if (wTensor->GetStorageShape().GetDimNum() == ND_WEIGHT_DIM_LIMIT) {  // ND
    n = wTensor->GetStorageShape().GetDim(DIM_2);
  } else if (wTensor->GetStorageShape().GetDimNum() ==
             NZ_WEIGHT_DIM_LIMIT) {  // NZ
    n = wTensor->GetStorageShape().GetDim(DIM_1) *
        wTensor->GetStorageShape().GetDim(DIM_4);
  }
  auto wScaleTensor = context->GetInputTensor(WEIGHT_SCALE_INDEX);
  int64_t quantGroupNum = 0;
  if (wScaleTensor->GetStorageShape().GetDimNum() ==
      PERCHANNEL_WSCALE_DIM_LIMIT) {  // perChannel
    quantGroupNum = 1;
  } else if (wScaleTensor->GetStorageShape().GetDimNum() ==
             PERGROUP_WSCALE_DIM_LIMIT) {  // perGroup
    quantGroupNum = wScaleTensor->GetStorageShape().GetDim(1);
  }
  auto groupListTensor = context->GetInputTensor(GROUPLIST_INDEX);
  const int64_t groupNum = groupListTensor->GetStorageShape().GetDim(0);
  MoeGMMSwigluQuantTilingData tilingData;
  const float* clamp_limit = context->GetAttrs()->GetFloat(0);
  tilingData.gmmSwigluBaseParams.set_clamp_limit(
      clamp_limit == nullptr ? 0.0F : *clamp_limit);
  const bool* use_split_arithmetic = context->GetAttrs()->GetBool(1);
  tilingData.gmmSwigluBaseParams.set_use_split_arithmetic(
      use_split_arithmetic != nullptr && *use_split_arithmetic);
  int64_t row = 0;
  row = CalMaxRowInUb(context, compileInfoPtr->ubSize_, n);

  auto core_num = compileInfoPtr->aicNum_;
  auto n_task_num = (n + BASEN - 1) / BASEN;
  auto task_num = m * n_task_num;
  if (task_num < core_num) {
    core_num = task_num;
  }

  tilingData.gmmSwigluBaseParams.set_groupNum(groupNum);
  tilingData.gmmSwigluBaseParams.set_coreNum(core_num);
  tilingData.gmmSwigluBaseParams.set_K(k);
  tilingData.gmmSwigluBaseParams.set_N(n);
  tilingData.gmmSwigluBaseParams.set_M(m);
  tilingData.gmmSwigluBaseParams.set_baseM(BASEM);
  tilingData.gmmSwigluBaseParams.set_baseN(BASEN);
  tilingData.gmmSwiglu.set_maxProcessRowNum(row);
  tilingData.gmmSwiglu.set_groupListLen(groupNum);
  tilingData.gmmSwiglu.set_tokenLen(n);

  tilingData.gmmSwigluBaseParams.set_quantGroupNum(quantGroupNum);

  auto ascendcPlatform =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  using namespace matmul_tiling;

  MatmulApiTiling tiling(ascendcPlatform);
  tiling.SetAType(
      TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT8);
  tiling.SetBType(
      TPosition::GM, CubeFormat::NZ, matmul_tiling::DataType::DT_INT8);
  tiling.SetCType(
      TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
  tiling.SetBias(false);
  tiling.SetShape(BASEM, BASEN, k);
  tiling.SetFixSplit(BASEM, BASEN, BASEK);
  tiling.SetOrgShape(m, n, k);
  tiling.SetBufferSpace(-1, -1, -1);
  if (tiling.GetTiling(tilingData.mmTilingData) == -1) {
    OP_LOGE(context->GetNodeName(),
            "grouped_matmul_swiglu_quant_tiling, get tiling failed");
    return ge::GRAPH_FAILED;
  }
  auto workspaceSizes = context->GetWorkspaceSizes(1);
  int64_t usrWorkspaceLimit = USER_WORKSPACE_LIMIT;
  int64_t mLimit = 0;
  mLimit =
      ((usrWorkspaceLimit / DOUBLE_WORKSPACE_SPLIT) / INT32_DTYPE_SIZE) / n;
  if (mLimit <= 0) {
    OP_LOGE(context->GetNodeName(), "mLimit is %ld must over then 0.", mLimit);
    return ge::GRAPH_FAILED;
  }
  tilingData.gmmSwigluBaseParams.set_mLimit(mLimit);

  int workSpaceMTemp =
      (mLimit * DOUBLE_WORKSPACE_SPLIT > m ? m
                                           : mLimit * DOUBLE_WORKSPACE_SPLIT);
  tilingData.gmmSwigluBaseParams.set_workSpaceOffset1(0);
  tilingData.gmmSwigluBaseParams.set_workSpaceOffset2(0);
  workspaceSizes[0] =
      SYS_WORKSPACE_SIZE + (workSpaceMTemp * n * sizeof(int32_t));

  bool isSplitWorkSpace = m > mLimit * DOUBLE_WORKSPACE_SPLIT;
  // OP_LOGD(context->GetNodeName(), "grouped_matmul_swiglu_quant_tiling.");
  // OP_LOGD(context->GetNodeName(), "gmmSwigluBaseParams.groupNum:      %ld",
  // groupNum); OP_LOGD(context->GetNodeName(), "gmmSwigluBaseParams.coreNum: %u
  // ", core_num); OP_LOGD(context->GetNodeName(), "gmmSwigluBaseParams.M: %ld",
  // m); OP_LOGD(context->GetNodeName(), "gmmSwigluBaseParams.K: %ld", k);
  // OP_LOGD(context->GetNodeName(), "gmmSwigluBaseParams.N:             %ld",
  // n); OP_LOGD(context->GetNodeName(), "gmmSwigluBaseParams.baseM: %ld",
  // BASEM); OP_LOGD(context->GetNodeName(), "gmmSwigluBaseParams.baseN: %ld",
  // BASEN); OP_LOGD(context->GetNodeName(), "gmmSwigluBaseParams.mLimit: %ld",
  // mLimit); OP_LOGD(context->GetNodeName(),
  // "gmmSwigluBaseParams.quantGroupNum: %ld", quantGroupNum);
  // OP_LOGD(context->GetNodeName(), "gmmSwiglu.maxProcessRowNum:        %ld",
  // row); OP_LOGD(context->GetNodeName(), "gmmSwiglu.groupListLen: %ld",
  // groupNum); OP_LOGD(context->GetNodeName(), "gmmSwiglu.tokenLen: %ld", n);
  // OP_LOGD(context->GetNodeName(), "USER_WORKSPACE_LIMIT:              %ld",
  // usrWorkspaceLimit); OP_LOGD(context->GetNodeName(), "workspaceSizes: %lu",
  // workspaceSizes[0]); OP_LOGD(context->GetNodeName(), "isSplitWorkSpace: %s",
  // isSplitWorkSpace ? "true" : "false"); OP_LOGD(context->GetNodeName(),
  // "GMMSWIGLUQUANT_TILING: baseM is %u, baseK is %u, baseN is %u.", BASEM,
  // BASEK, BASEN);
  SetTilingKey(context, isSplitWorkSpace);
  tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                          context->GetRawTilingData()->GetCapacity());
  context->SetBlockDim(core_num);  // block dim is the number of aicube
  context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  // OP_LOGD(context->GetNodeName(), "End Run GMM Swiglu Tiling.");
  return GRAPH_SUCCESS;
}

graphStatus TilingPrepareForMoeGMMSwigluQuant(
    gert::TilingParseContext* context) {
  // get info
  fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
  auto compileInfoPtr = context->GetCompiledInfo<MoeGMMSwigluCompileInfo>();
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  compileInfoPtr->aicNum_ = ascendcPlatform.GetCoreNumAic();
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,
                                 compileInfoPtr->ubSize_);
  // OP_LOGD(context->GetNodeName(), "ubSize is %lu, aicNum is %u.",
  // compileInfoPtr->ubSize_, compileInfoPtr->aicNum_);
  return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(XllmKernelMoeGroupedMatmulSwigluQuant)
    .Tiling(TilingMoeGMMSwigluQuant)
    .TilingParse<MoeGMMSwigluCompileInfo>(TilingPrepareForMoeGMMSwigluQuant);
}  // namespace optiling
