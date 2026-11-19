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
 * \file mla_preprocess_v2_tiling.cpp
 * \brief
 */

#include "mla_preprocess_tiling.h"
#include "mla_preprocess_tilingdata.h"
#include "register/op_impl_registry.h"
// #include "log/log.h"
#include <cmath>
#include <string>

constexpr uint64_t ATTR_Q_DOWN_OUT_FLAG_IDX = 13;

namespace optiling {

ge::graphStatus TilingMLAPreprocessV2(gert::TilingContext* context) {
  MlaPreprocessTiling mlaTiling;
  mlaTiling.Init(context);
  bool qDownOutFlag =
      *(context->GetAttrs()->GetAttrPointer<bool>(ATTR_Q_DOWN_OUT_FLAG_IDX));
  mlaTiling.mlaTilingData.set_qDownOutFlag(qDownOutFlag);
  mlaTiling.mlaTilingData.SaveToBuffer(
      context->GetRawTilingData()->GetData(),
      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(
      mlaTiling.mlaTilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForMlaPreprocessV2(
    gert::TilingParseContext* context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(XllmKernelMlaPreprocessV2)
    .Tiling(TilingMLAPreprocessV2)
    .TilingParse<MlaPreProcessCompileInfo>(TilingPrepareForMlaPreprocessV2);

}  // namespace optiling
