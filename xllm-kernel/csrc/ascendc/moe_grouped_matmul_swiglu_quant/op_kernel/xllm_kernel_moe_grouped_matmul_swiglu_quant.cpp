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

#include <typeinfo>

#include "moe_grouped_matmul_swiglu_quant.h"
#include "moe_grouped_matmul_swiglu_quant_split_ws.h"
using namespace AscendC;
using namespace matmul;
using namespace MOE_GROUPED_MATMUL_SWIGLU_QUANT;

#define GMM_CV_SPLIT_IMP(computeClass, dtypeWeightScale, transA, transB, sync) \
  do {                                                                         \
    using xType =                                                              \
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X, false>;    \
    using weightType = MatmulType<AscendC::TPosition::GM,                      \
                                  CubeFormat::NZ,                              \
                                  DTYPE_WEIGHT,                                \
                                  false>;                                      \
    using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>; \
    using matmulType = MMImplTypeStatic<xType, weightType, yType>;             \
    matmulType::MT mm;                                                         \
    GET_TILING_DATA_MEMBER(MoeGMMSwigluQuantTilingData,                        \
                           gmmSwigluBaseParams,                                \
                           gmmSwigluBaseParams_,                               \
                           tiling);                                            \
    GET_TILING_DATA_MEMBER(                                                    \
        MoeGMMSwigluQuantTilingData, mmTilingData, mmTilingData_, tiling);     \
    GET_TILING_DATA_MEMBER(                                                    \
        MoeGMMSwigluQuantTilingData, gmmSwiglu, gmmSwiglu_, tiling);           \
    if ASCEND_IS_AIC {                                                         \
      mm.SetSubBlockIdx(0);                                                    \
      mm.Init(&mmTilingData_, &tPipe);                                         \
    }                                                                          \
    computeClass<matmulType, sync, dtypeWeightScale> computeOp(mm);            \
    computeOp.Init(x,                                                          \
                   weight,                                                     \
                   weightScale,                                                \
                   xScale,                                                     \
                   groupList,                                                  \
                   y,                                                          \
                   yScale,                                                     \
                   user1,                                                      \
                   &gmmSwigluBaseParams_,                                      \
                   &mmTilingData_,                                             \
                   &gmmSwiglu_,                                                \
                   &tPipe);                                                    \
    computeOp.Process();                                                       \
  } while (0)

extern "C" __global__ __aicore__ void
xllm_kernel_moe_grouped_matmul_swiglu_quant(GM_ADDR x,
                                            GM_ADDR weight,
                                            GM_ADDR weightScale,
                                            GM_ADDR xScale,
                                            GM_ADDR groupList,
                                            GM_ADDR y,
                                            GM_ADDR yScale,
                                            GM_ADDR workspace,
                                            GM_ADDR tiling) {
  TPipe tPipe;
  AscendCUtils::SetOverflow(1);
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  GM_ADDR user1 = GetUserWorkspace(workspace);
  if (TILING_KEY_IS(0)) {  // antiquant msd
    KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    GMM_CV_SPLIT_IMP(MoeGMMSwigluCompute,  // computeClass
                     DTYPE_WEIGHT_SCALE,
                     false,  // transA
                     false,  // transB
                     false   // sync
    );
  } else if (TILING_KEY_IS(1)) {
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    GMM_CV_SPLIT_IMP(MoeGMMSwigluSplitWorkSpaceCompute,  // computeClass
                     DTYPE_WEIGHT_SCALE,
                     false,  // transA
                     false,  // transB
                     false   // sync
    );
  }
}
