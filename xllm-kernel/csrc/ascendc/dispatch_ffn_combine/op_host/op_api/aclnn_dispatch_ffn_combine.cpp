/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include "aclnn_xllm_kernel_dispatch_ffn_combine.h"
#include "aclnn_xllm_kernel_dispatch_ffn_combine_op.h"

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
  NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
  NNOPBASE_HCCL_SERVER_TYPE_MTE,
  NNOPBASE_HCCL_SERVER_TYPE_END
};

void __attribute__((weak)) NnopbaseSetHcclServerType(
    void* executor,
    NnopbaseHcclServerType sType);

aclnnStatus aclnnXllmKernelDispatchFFNCombineGetWorkspaceSize(
    const aclTensor* x,
    const aclTensorList* weight1,
    const aclTensorList* weight2,
    const aclTensor* expertId,
    const aclTensorList* scale1,
    const aclTensorList* scale2,
    const aclTensor* probs,
    const aclTensor* xActiveMask,
    const char* group,
    int64_t maxOutputSize,
    double swigluLimit,
    const aclTensor* out,
    const aclTensor* expertTokenNums,
    uint64_t* workspaceSize,
    aclOpExecutor** executor) {
  return aclnnXllmKernelDispatchFFNCombineOpGetWorkspaceSize(
      x,
      weight1,
      weight2,
      expertId,
      scale1,
      scale2,
      probs,
      xActiveMask,
      const_cast<char*>(group),
      maxOutputSize,
      /*transB=*/false,
      /*weightNz=*/true,
      swigluLimit,
      out,
      expertTokenNums,
      workspaceSize,
      executor);
}

aclnnStatus aclnnXllmKernelDispatchFFNCombine(void* workspace,
                                              uint64_t workspaceSize,
                                              aclOpExecutor* executor,
                                              aclrtStream stream) {
  if (NnopbaseSetHcclServerType) {
    NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
  }
  return aclnnXllmKernelDispatchFFNCombineOp(
      workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
