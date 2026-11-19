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
 * \file hc_pre_va_tiling.h
 * \brief
 */

#ifndef HC_PRE_VA_TILING_H
#define HC_PRE_VA_TILING_H

#include <iostream>
#include <vector>

#include "common/error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
// ----------公共定义----------
struct TilingRequiredParaInfo {
  const gert::CompileTimeTensorDesc* desc;
  const gert::StorageShape* shape;
};

struct TilingOptionalParaInfo {
  const gert::CompileTimeTensorDesc* desc;
  const gert::Tensor* tensor;
};
// ----------算子TilingData定义----------
BEGIN_TILING_DATA_DEF(HcPreVaTilingData)
TILING_DATA_FIELD_DEF(int64_t, bs);
TILING_DATA_FIELD_DEF(int64_t, hcMix);
TILING_DATA_FIELD_DEF(int64_t, hcMult);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, hcMultAlign);
TILING_DATA_FIELD_DEF(int64_t, rowOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, dLoop);
TILING_DATA_FIELD_DEF(int64_t, dFactor);
TILING_DATA_FIELD_DEF(int64_t, tailDFactor);
TILING_DATA_FIELD_DEF(int64_t, iterTimes);
TILING_DATA_FIELD_DEF(float, hcEps);
TILING_DATA_FIELD_DEF(float, normEps);
TILING_DATA_FIELD_DEF(int64_t, stage2RowFactor);

TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, kL1Size);
TILING_DATA_FIELD_DEF(int64_t, mL1Size);
TILING_DATA_FIELD_DEF(int64_t, cubeBlockDimM);
TILING_DATA_FIELD_DEF(int64_t, cubeBlockDimK);
TILING_DATA_FIELD_DEF(int64_t, cvLoopKSize);         // cv同步使用的k轴长度
TILING_DATA_FIELD_DEF(int64_t, multCoreSplitKSize);  // 多核切K后的单核k处理长度
TILING_DATA_FIELD_DEF(int64_t, multCoreSplitMSize);  // 多核切K后的单核k处理长度

TILING_DATA_FIELD_DEF(int64_t, secondUsedCoreNum);

TILING_DATA_FIELD_DEF(int64_t, cubeCoreNum);
TILING_DATA_FIELD_DEF(int64_t, stage1MFactor);

TILING_DATA_FIELD_DEF(int64_t, hasPreMix);
TILING_DATA_FIELD_DEF(int64_t, hasPreOut);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(XllmKernelHcPreVa, HcPreVaTilingData)

// ----------算子CompileInfo定义----------
struct HcPreVaCompileInfo {
  uint64_t coreNum = 0;
  uint64_t aicCoreNum = 0;
  uint64_t ubSize = 0;
};

// ----------算子Tiling入参信息解析及check类----------
class HcPreVaTiling {
 public:
  explicit HcPreVaTiling(gert::TilingContext* tilingContext)
      : context_(tilingContext) {}
  ~HcPreVaTiling() = default;

  ge::graphStatus GetPlatformInfo();
  ge::graphStatus DoOpTiling();
  ge::graphStatus GetWorkspaceSize();
  ge::graphStatus PostTiling();
  ge::graphStatus GetAttr();
  ge::graphStatus GetShapeAttrsInfoInner();
  ge::graphStatus CalcOpTiling();
  ge::graphStatus CalcMKSplitCoreMembasePart2Tiling();

 private:
  gert::TilingContext* context_ = nullptr;
  uint64_t tilingKey_ = 0;
  HcPreVaTilingData tilingData_;
  uint64_t aivCoreNum_ = 0;
  uint64_t aicCoreNum_ = 0;
  uint64_t workspaceSize_ = 0;
  uint64_t usedAivCoreNums_ = 0;
  uint64_t ubSize_ = 0;
  int64_t bs_ = 0;
  int64_t hcMix_ = 0;
  int64_t hcMult_ = 0;
  int64_t d_ = 0;
  int64_t hcMultAlign_ = 0;
  int64_t rowOfFormerBlock_ = 0;
  int64_t rowOfTailBlock_ = 0;
  int64_t rowLoopOfFormerBlock_ = 0;
  int64_t rowLoopOfTailBlock_ = 0;
  int64_t rowFactor_ = 0;
  int64_t tailRowFactorOfFormerBlock_ = 0;
  int64_t tailRowFactorOfTailBlock_ = 0;
  int64_t dLoop_ = 0;
  int64_t dFactor_ = 0;
  int64_t tailDFactor_ = 0;
  int64_t iterTimes_ = 0;
  double hcEps_ = 0.0;
  double normEps_ = 0.0;
};

}  // namespace optiling
#endif  // HC_PRE_VA_TILING_H
