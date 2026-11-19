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
 * \file moe_init_routing_v3_tiling_base.cpp
 * \brief
 */
#include "common/log/ops_log.h"
#include "moe_init_routing_v3_tiling.h"
#include "register/op_def_registry.h"

using Ops::Transformer::OpTiling::TilingRegistry;

namespace optiling {
static ge::graphStatus TilingForMoeInitRoutingV3(gert::TilingContext* context) {
  return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForMoeInitRountingV3(
    gert::TilingParseContext* context) {
  OPS_LOG_D(context, "TilingPrepareForMoeInitRountingV3 enter.");

  auto compileInfo = context->GetCompiledInfo<MoeInitRoutingV3CompileInfo>();
  OPS_LOG_E_WITHOUT_REPORT(context, compileInfo);
  auto platformInfo = context->GetPlatformInfo();
  OPS_LOG_E_WITHOUT_REPORT(context, platformInfo);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  compileInfo->aivNum = ascendcPlatform.GetCoreNumAiv();
  OPS_CHECK(
      (compileInfo->aivNum <= 0),
      OPS_LOG_E(context,
                "TilingPrepareForMoeInitRountingV3 fail to get core num."),
      return ge::GRAPH_FAILED);

  uint64_t ubSize;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  compileInfo->ubSize = static_cast<int64_t>(ubSize);
  compileInfo->socVersion = ascendcPlatform.GetSocVersion();
  OPS_CHECK((compileInfo->ubSize <= 0),
            OPS_LOG_E(context,
                      "TilingPrepareForMoeInitRountingV3 fail to get ub size."),
            return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(XllmKernelMoeInitRoutingV3)
    .Tiling(TilingForMoeInitRoutingV3)
    .TilingParse<MoeInitRoutingV3CompileInfo>(
        TilingPrepareForMoeInitRountingV3);
}  // namespace optiling
