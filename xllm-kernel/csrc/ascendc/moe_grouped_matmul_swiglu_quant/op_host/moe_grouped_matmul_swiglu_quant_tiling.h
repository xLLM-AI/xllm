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

#ifndef OP_IMPL_MOE_GROUPED_MATMUL_SWIGLU_QUANT_H
#define OP_IMPL_MOE_GROUPED_MATMUL_SWIGLU_QUANT_H

#include <set>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
// GMM 基本信息
BEGIN_TILING_DATA_DEF(MoeGMMSwigluBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, K);
TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, M);
TILING_DATA_FIELD_DEF(uint32_t, baseM);
TILING_DATA_FIELD_DEF(uint32_t, baseN);
TILING_DATA_FIELD_DEF(uint32_t, mLimit);
TILING_DATA_FIELD_DEF(uint32_t, workSpaceOffset1);
TILING_DATA_FIELD_DEF(uint32_t, workSpaceOffset2);
TILING_DATA_FIELD_DEF(uint32_t, quantGroupNum);
TILING_DATA_FIELD_DEF(float, clamp_limit);
TILING_DATA_FIELD_DEF(uint32_t, use_split_arithmetic);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeGMMSwigluBaseParamsOp, MoeGMMSwigluBaseParams)

// SwigluQuant部分tiling 基本信息
BEGIN_TILING_DATA_DEF(MoeGMMSwiglu)
TILING_DATA_FIELD_DEF(uint32_t, maxProcessRowNum);
TILING_DATA_FIELD_DEF(uint32_t, groupListLen);
TILING_DATA_FIELD_DEF(uint32_t, tokenLen);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeGMMSwigluOp, MoeGMMSwiglu)

// 结构体集合
BEGIN_TILING_DATA_DEF(MoeGMMSwigluQuantTilingData)
TILING_DATA_FIELD_DEF_STRUCT(MoeGMMSwigluBaseParams, gmmSwigluBaseParams);
TILING_DATA_FIELD_DEF_STRUCT(MoeGMMSwiglu, gmmSwiglu);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(XllmKernelMoeGroupedMatmulSwigluQuant,
                           MoeGMMSwigluQuantTilingData)
}  // namespace optiling

namespace MoeGroupedMatmulSwigluQuantTiling {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t WEIGHT_SCALE_INDEX = 2;
constexpr uint32_t GROUPLIST_INDEX = 4;
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;
constexpr uint32_t DIM_3 = 3;
constexpr uint32_t DIM_4 = 4;
constexpr uint32_t NUM_FOUR = 4;
constexpr uint32_t NUM_EIGHT = 8;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int64_t USER_WORKSPACE_LIMIT = 64 * 1024 * 1024;
constexpr int64_t DOUBLE_WORKSPACE_SPLIT = 2;
constexpr int64_t INT32_DTYPE_SIZE = 4;
constexpr int64_t FP32_DTYPE_SIZE = 4;
constexpr int64_t FP32_BLOCK_SIZE = 8;
constexpr int64_t BLOCK_BYTE = 32;
constexpr int64_t SWIGLU_REDUCE_FACTOR = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t ND_WEIGHT_DIM_LIMIT = 3;
constexpr int64_t NZ_WEIGHT_DIM_LIMIT = 5;
constexpr int64_t DOUBLE_ROW = 2;
constexpr int64_t PERCHANNEL_WSCALE_DIM_LIMIT = 2;
constexpr int64_t PERGROUP_WSCALE_DIM_LIMIT = 3;
constexpr int64_t A8W4_MSD_TILING_KEY_MODE = 2;
constexpr int64_t SPLITWORKSPACE_TILING_KEY_MODE = 1;
constexpr int64_t COMMON_TILING_KEY_MODE = 0;
constexpr int64_t A8W4_TOKEN_THRESHOLD = 32;
constexpr int64_t BASEM = 128;
constexpr int64_t BASEK = 256;
constexpr int64_t BASEN = 256;
}  // namespace MoeGroupedMatmulSwigluQuantTiling

#endif  // OP_IMPL_MOE_GROUPED_MATMUL_SWIGLU_QUANT_H
