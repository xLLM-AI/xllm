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

#include "mega_chunk_gdn_l0.h"

#include <cstdint>

#include "acl/acl_rt.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(XllmKernelMegaChunkGdn);

using MegaChunkGdnOutputs = std::tuple<const aclTensor*,
                                       const aclTensor*,
                                       const aclTensor*,
                                       const aclTensor*,
                                       const aclTensor*,
                                       const aclTensor*,
                                       const aclTensor*,
                                       const aclTensor*,
                                       const aclTensor*,
                                       const aclTensor*,
                                       const aclTensor*,
                                       const aclTensor*>;

MegaChunkGdnOutputs MakeNullOutputs() {
  return {nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr};
}

MegaChunkGdnOutputs XllmKernelMegaChunkGdn(const aclTensor* q,
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
                                           const aclTensor* out,
                                           const aclTensor* gSum,
                                           const aclTensor* gT,
                                           const aclTensor* betaT,
                                           const aclTensor* a,
                                           const aclTensor* aInvF32,
                                           const aclTensor* aInv,
                                           const aclTensor* w,
                                           const aclTensor* u,
                                           const aclTensor* h,
                                           const aclTensor* vNew,
                                           const aclTensor* finalState,
                                           aclOpExecutor* executor) {
  L0_DFX(XllmKernelMegaChunkGdn,
         q,
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
         hasInitialState);

  auto outRet = executor->AllocTensor(
      out->GetViewShape(), out->GetDataType(), Format::FORMAT_ND);
  auto gSumRet = executor->AllocTensor(
      gSum->GetViewShape(), gSum->GetDataType(), Format::FORMAT_ND);
  auto gTRet = executor->AllocTensor(
      gT->GetViewShape(), gT->GetDataType(), Format::FORMAT_ND);
  auto betaTRet = executor->AllocTensor(
      betaT->GetViewShape(), betaT->GetDataType(), Format::FORMAT_ND);
  auto aRet = executor->AllocTensor(
      a->GetViewShape(), a->GetDataType(), Format::FORMAT_ND);
  auto aInvF32Ret = executor->AllocTensor(
      aInvF32->GetViewShape(), aInvF32->GetDataType(), Format::FORMAT_ND);
  auto aInvRet = executor->AllocTensor(
      aInv->GetViewShape(), aInv->GetDataType(), Format::FORMAT_ND);
  auto wRet = executor->AllocTensor(
      w->GetViewShape(), w->GetDataType(), Format::FORMAT_ND);
  auto uRet = executor->AllocTensor(
      u->GetViewShape(), u->GetDataType(), Format::FORMAT_ND);
  auto hRet = executor->AllocTensor(
      h->GetViewShape(), h->GetDataType(), Format::FORMAT_ND);
  auto vNewRet = executor->AllocTensor(
      vNew->GetViewShape(), vNew->GetDataType(), Format::FORMAT_ND);
  auto finalStateRet = executor->AllocTensor(
      finalState->GetViewShape(), finalState->GetDataType(), Format::FORMAT_ND);

  OP_CHECK(outRet != nullptr && gSumRet != nullptr && gTRet != nullptr &&
               betaTRet != nullptr && aRet != nullptr &&
               aInvF32Ret != nullptr && aInvRet != nullptr && wRet != nullptr &&
               uRet != nullptr && hRet != nullptr && vNewRet != nullptr &&
               finalStateRet != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                   "XllmKernelMegaChunkGdn AllocTensor failed."),
           return MakeNullOutputs());

  void* hardwareSyncAddr = nullptr;
  const aclError aclRet = aclrtGetHardwareSyncAddr(&hardwareSyncAddr);
  const bool hardwareSyncAddrNotRequired =
      aclRet == ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
  OP_CHECK(hardwareSyncAddrNotRequired ||
               (aclRet == ACL_SUCCESS && hardwareSyncAddr != nullptr),
           OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                   "aclrtGetHardwareSyncAddr failed, ret=%d.",
                   aclRet),
           return MakeNullOutputs());
  const uint64_t fftsAddr =
      hardwareSyncAddrNotRequired
          ? 0
          : static_cast<uint64_t>(
                reinterpret_cast<uintptr_t>(hardwareSyncAddr));

  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
      XllmKernelMegaChunkGdn,
      OP_INPUT(q,
               k,
               v,
               g,
               beta,
               maskLower,
               maskFull,
               minusIdentity,
               cuSeqlens,
               initialState),
      OP_OUTPUT(outRet,
                gSumRet,
                gTRet,
                betaTRet,
                aRet,
                aInvF32Ret,
                aInvRet,
                wRet,
                uRet,
                hRet,
                vNewRet,
                finalStateRet),
      OP_ATTR(numMatrices, hasInitialState, static_cast<int64_t>(fftsAddr)));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
      ret != ACLNN_SUCCESS,
      return MakeNullOutputs(),
      "XllmKernelMegaChunkGdn ADD_TO_LAUNCHER_LIST_AICORE failed.");

  return {outRet,
          gSumRet,
          gTRet,
          betaTRet,
          aRet,
          aInvF32Ret,
          aInvRet,
          wRet,
          uRet,
          hRet,
          vNewRet,
          finalStateRet};
}
}  // namespace l0op
