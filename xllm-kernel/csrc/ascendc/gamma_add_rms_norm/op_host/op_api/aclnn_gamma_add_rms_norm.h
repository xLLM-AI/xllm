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
#ifndef OP_API_INC_LEVEL2_GAMMA_ADD_RMS_NORM_H_
#define OP_API_INC_LEVEL2_GAMMA_ADD_RMS_NORM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief First-stage aclnnGammaAddRmsNorm API that calculates the required
 * workspace size.
 * @domain aclnn_ops_infer
 *
 * Fuses Add and RMSNorm, returning the sum, normalized result, and reciprocal
 * root mean square. Formula: x = x1 + x2 y = (x/RMS(x))*(gamma +
 * (addGammaOffset ? 1 : 0)) rstd = 1/RMS(x)
 *
 * @param [in] x1:
 * Input `x1` in the formula. Supports BFLOAT16, FLOAT, and FLOAT16 with at most
 * eight dimensions. Non-contiguous tensors and the ND format are supported.
 * @param [in] x2:
 * Input `x2` in the formula. Supports BFLOAT16, FLOAT, and FLOAT16 with at most
 * eight dimensions. Non-contiguous tensors and the ND format are supported.
 * @param [in] gamma:
 * Input `gamma` in the formula. Supports BFLOAT16, FLOAT, and FLOAT16 with at
 * most eight dimensions. Non-contiguous tensors and the ND format are
 * supported.
 * @param [in] epsilon: Double-precision value used to prevent division by zero
 * during normalization.
 * @param [in] addGammaOffset: Whether to apply the Gemma-style `gamma + 1`
 * inside the kernel.
 * @param [in] yOut:
 * Output `y` in the formula. Supports BFLOAT16, FLOAT, and FLOAT16 and must
 * have the same shape as x1. Non-contiguous tensors and the ND format are
 * supported.
 * @param [in] rstdOut:
 * Output `rstd` in the formula. Supports FLOAT with at most eight dimensions.
 * Non-contiguous tensors and the ND format are supported.
 * @param [in] xOut:
 * Output `x` in the formula. Its data type and shape must match x1.
 * Non-contiguous tensors and the ND format are supported.
 * @param [out] workspaceSize: Workspace size that the caller must allocate on
 * the NPU device.
 * @param [out] executor: Operator executor containing the computation flow.
 * @return aclnnStatus: Status code.
 */
ACLNN_API aclnnStatus
aclnnGammaAddRmsNormGetWorkspaceSize(const aclTensor* x1,
                                     const aclTensor* x2,
                                     const aclTensor* gamma,
                                     double epsilon,
                                     bool addGammaOffset,
                                     aclTensor* yOut,
                                     aclTensor* rstdOut,
                                     aclTensor* xOut,
                                     uint64_t* workspaceSize,
                                     aclOpExecutor** executor);

/**
 * @brief Second-stage aclnnGammaAddRmsNorm API that executes the computation.
 *
 * @param [in] workspace: Start address of the workspace allocated on the NPU
 * device.
 * @param [in] workspaceSize: Workspace size returned by
 * aclnnGammaAddRmsNormGetWorkspaceSize.
 * @param [in] executor: Operator executor containing the computation flow.
 * @param [in] stream: ACL stream used to execute the operator.
 * @return aclnnStatus: Status code.
 */
ACLNN_API aclnnStatus aclnnGammaAddRmsNorm(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_GAMMA_ADD_RMS_NORM_H_
