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
#ifndef OP_API_INC_GROUPED_MATMUL_SWIGLU_QUANT_H
#define OP_API_INC_GROUPED_MATMUL_SWIGLU_QUANT_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus
aclnnXllmKernelMoeGroupedMatmulSwigluQuantGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* weightScale,
    const aclTensor* xScale,
    const aclTensor* groupList,
    aclTensor* output,
    aclTensor* outputScale,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus
aclnnXllmKernelMoeGroupedMatmulSwigluQuant(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

__attribute__((visibility("default"))) aclnnStatus
aclnnXllmKernelMoeGroupedMatmulSwigluQuantV2GetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* weight_scale,
    const aclTensor* x_scale,
    const aclTensor* group_list,
    double clamp_limit,
    aclTensor* output,
    aclTensor* output_scale,
    uint64_t* workspace_size,
    aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus
aclnnXllmKernelMoeGroupedMatmulSwigluQuantV2(void* workspace,
                                             uint64_t workspace_size,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

__attribute__((visibility("default"))) aclnnStatus
aclnnXllmKernelMoeGroupedMatmulSwigluQuantV3GetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* weight_scale,
    const aclTensor* x_scale,
    const aclTensor* group_list,
    double clamp_limit,
    bool use_split_arithmetic,
    aclTensor* output,
    aclTensor* output_scale,
    uint64_t* workspace_size,
    aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus
aclnnXllmKernelMoeGroupedMatmulSwigluQuantV3(void* workspace,
                                             uint64_t workspace_size,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
