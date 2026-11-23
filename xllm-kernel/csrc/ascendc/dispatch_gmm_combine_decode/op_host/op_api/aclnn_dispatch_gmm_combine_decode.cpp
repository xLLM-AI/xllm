/*
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
#include <string.h>

#include "aclnn/opdev/platform.h"
#include "aclnn_xllm_kernel_dispatch_gmm_combine_decode.h"
#include "aclnn_xllm_kernel_dispatch_gmm_combine_decode_op.h"
#include "graph/types.h"

enum NnopbaseHcclServerType {
  NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
  NNOPBASE_HCCL_SERVER_TYPE_MTE,
  NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(
    void* executor,
    NnopbaseHcclServerType sType);

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnXllmKernelDispatchGmmCombineDecodeGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* expertIds,
    const aclTensorList* gmm1PermutedWeight,
    const aclTensorList* gmm1PermutedWeightScale,
    const aclTensorList* gmm2Weight,
    const aclTensorList* gmm2WeightScale,
    const aclTensor* expertScales,
    const aclTensor* expertSmoothScalesOptional,
    const aclTensor* xActiveMaskOptional,
    char* groupEp,
    int64_t epRankSize,
    int64_t epRankId,
    int64_t moeExpertNum,
    int64_t shareExpertNum,
    int64_t shareExpertRankNum,
    int64_t quantMode,
    int64_t globalBs,
    const aclTensor* output,
    const aclTensor* expertTokenNums,
    uint64_t* workspaceSize,
    aclOpExecutor** executor) {
  return aclnnXllmKernelDispatchGmmCombineDecodeOpGetWorkspaceSize(
      x,
      expertIds,
      gmm1PermutedWeight,
      gmm1PermutedWeightScale,
      gmm2Weight,
      gmm2WeightScale,
      expertScales,
      expertSmoothScalesOptional,
      xActiveMaskOptional,
      groupEp,
      epRankSize,
      epRankId,
      moeExpertNum,
      shareExpertNum,
      shareExpertRankNum,
      quantMode,
      globalBs,
      output,
      expertTokenNums,
      workspaceSize,
      executor);
}

aclnnStatus aclnnXllmKernelDispatchGmmCombineDecode(void* workspace,
                                                    uint64_t workspaceSize,
                                                    aclOpExecutor* executor,
                                                    aclrtStream stream) {
  if (NnopbaseSetHcclServerType) {
    if (op::GetCurrentPlatformInfo().GetSocVersion() ==
        op::SocVersion::ASCEND910B) {
      NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
    } else {
      NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
  }
  return aclnnXllmKernelDispatchGmmCombineDecodeOp(
      workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
