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

/* !
 * \file gamma_add_rms_norm.cpp
 * \brief
 */

#include "arch35/gamma_add_rms_norm_regbase.h"
#include "arch35/gamma_add_rms_norm_regbase_split_d.h"

using namespace AscendC;
using namespace GammaAddRmsNorm;

extern "C" __global__ __aicore__ void xllm_kernel_gamma_add_rms_norm(
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR gamma,
    GM_ADDR y,
    GM_ADDR rstd,
    GM_ADDR x,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe aptPipe;
  if (TILING_KEY_IS(1000)) {
    GET_TILING_DATA_WITH_STRUCT(
        GammaAddRMSNormRegbaseRFullLoadTilingData, aptTilingDataIn, tiling);
    KernelGammaAddRmsNormRegBase<DTYPE_X1> op(&aptPipe);
    op.Init(x1, x2, gamma, y, rstd, x, &aptTilingDataIn);
    op.Process();
  } else if (TILING_KEY_IS(2000)) {
    GET_TILING_DATA_WITH_STRUCT(
        GammaAddRMSNormRegbaseTilingData, aptTilingDataIn, tiling);
    KernelGammaAddRmsNormRegBaseSplitD<DTYPE_X1> op(&aptPipe);
    op.Init(x1, x2, gamma, y, rstd, x, &aptTilingDataIn);
    op.Process();
  }
}
