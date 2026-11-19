/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus
aclnnXllmKernelMegaChunkGdnGetWorkspaceSize(const aclTensor* q,
                                            const aclTensor* k,
                                            const aclTensor* v,
                                            const aclTensor* g,
                                            const aclTensor* beta,
                                            const aclTensor* maskLower,
                                            const aclTensor* maskFull,
                                            const aclTensor* minusIdentity,
                                            const aclTensor* cuSeqlens,
                                            const aclTensor* initialState,
                                            int64_t numMatrices,
                                            bool hasInitialState,
                                            aclTensor* out,
                                            aclTensor* gSum,
                                            aclTensor* gT,
                                            aclTensor* betaT,
                                            aclTensor* a,
                                            aclTensor* aInvF32,
                                            aclTensor* aInv,
                                            aclTensor* w,
                                            aclTensor* u,
                                            aclTensor* h,
                                            aclTensor* vNew,
                                            aclTensor* finalState,
                                            uint64_t* workspaceSize,
                                            aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus
aclnnXllmKernelMegaChunkGdn(void* workspace,
                            uint64_t workspaceSize,
                            aclOpExecutor* executor,
                            aclrtStream stream);

#ifdef __cplusplus
}
#endif
