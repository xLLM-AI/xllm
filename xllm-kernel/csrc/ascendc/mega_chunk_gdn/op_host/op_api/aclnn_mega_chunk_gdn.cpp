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

#include "aclnn_mega_chunk_gdn.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "mega_chunk_gdn_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnXllmKernelMegaChunkGdnGetWorkspaceSize(
    const aclTensor* q,
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
    aclOpExecutor** executor) {
  L2_DFX_PHASE_1(
      aclnnXllmKernelMegaChunkGdn,
      DFX_IN(q,
             k,
             v,
             g,
             beta,
             maskLower,
             maskFull,
             minusIdentity,
             cuSeqlens,
             initialState,
             numMatrices,
             hasInitialState),
      DFX_OUT(
          out, gSum, gT, betaT, a, aInvF32, aInv, w, u, h, vNew, finalState));

  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  auto qContiguous = l0op::Contiguous(q, uniqueExecutor.get());
  auto kContiguous = l0op::Contiguous(k, uniqueExecutor.get());
  auto vContiguous = l0op::Contiguous(v, uniqueExecutor.get());
  auto gContiguous = l0op::Contiguous(g, uniqueExecutor.get());
  auto betaContiguous = l0op::Contiguous(beta, uniqueExecutor.get());
  auto cuSeqlensContiguous = l0op::Contiguous(cuSeqlens, uniqueExecutor.get());

  auto outputs = l0op::XllmKernelMegaChunkGdn(qContiguous,
                                              kContiguous,
                                              vContiguous,
                                              gContiguous,
                                              betaContiguous,
                                              maskLower,
                                              maskFull,
                                              minusIdentity,
                                              cuSeqlensContiguous,
                                              initialState,
                                              numMatrices,
                                              hasInitialState,
                                              out,
                                              gSum,
                                              gT,
                                              betaT,
                                              a,
                                              aInvF32,
                                              aInv,
                                              w,
                                              u,
                                              h,
                                              vNew,
                                              finalState,
                                              uniqueExecutor.get());
  if (std::get<0>(outputs) == nullptr) {
    return ACLNN_ERR_INNER_NULLPTR;
  }

  const aclTensor* retTensors[] = {std::get<0>(outputs),
                                   std::get<1>(outputs),
                                   std::get<2>(outputs),
                                   std::get<3>(outputs),
                                   std::get<4>(outputs),
                                   std::get<5>(outputs),
                                   std::get<6>(outputs),
                                   std::get<7>(outputs),
                                   std::get<8>(outputs),
                                   std::get<9>(outputs),
                                   std::get<10>(outputs),
                                   std::get<11>(outputs)};
  aclTensor* outTensors[] = {
      out, gSum, gT, betaT, a, aInvF32, aInv, w, u, h, vNew, finalState};
  for (size_t i = 0; i < 12; ++i) {
    auto viewCopy =
        l0op::ViewCopy(retTensors[i], outTensors[i], uniqueExecutor.get());
    if (viewCopy == nullptr) {
      return ACLNN_ERR_INNER_NULLPTR;
    }
  }

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnXllmKernelMegaChunkGdn(void* workspace,
                                        uint64_t workspaceSize,
                                        aclOpExecutor* executor,
                                        aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnXllmKernelMegaChunkGdn);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
