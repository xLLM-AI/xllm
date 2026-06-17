/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <c10/core/Device.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <nlohmann/json.hpp>
#include <unordered_map>
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include "acl/acl.h"
#include "aclnnop/aclnn_apply_top_k_top_p.h"
#include "core/common/macros.h"
#include "core/kernels/npu/utils.h"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

namespace {

class AclWorkspaceCache final {
 public:
  ~AclWorkspaceCache() {
    for (auto& workspace : workspaces_) {
      if (workspace.second.addr_ != nullptr) {
        aclrtSynchronizeStream(workspace.first);
        aclrtFree(workspace.second.addr_);
      }
    }
  }

  void* get(aclrtStream stream, uint64_t workspace_size) {
    if (workspace_size == 0) {
      return nullptr;
    }

    AclWorkspace& workspace = workspaces_[stream];
    if (workspace.addr_ != nullptr && workspace.size_ >= workspace_size) {
      return workspace.addr_;
    }

    if (workspace.addr_ != nullptr) {
      CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                        "top_k_top_p: failed to synchronize stream before "
                        "workspace resize");
      CHECK_ACL_SUCCESS(aclrtFree(workspace.addr_),
                        "top_k_top_p: failed to free resized workspace");
    }

    CHECK_ACL_SUCCESS(
        aclrtMalloc(
            &workspace.addr_, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
        "top_k_top_p: failed to allocate workspace");
    workspace.size_ = workspace_size;
    return workspace.addr_;
  }

 private:
  class AclWorkspace final {
   public:
    void* addr_ = nullptr;
    uint64_t size_ = 0;
  };

  std::unordered_map<aclrtStream, AclWorkspace> workspaces_;
};

AclWorkspaceCache& workspace_cache() {
  thread_local AclWorkspaceCache cache;
  return cache;
}

}  // namespace

// Used by the sampling logits preprocessing path on NPU.
// This wrapper applies top-k and top-p filtering before token sampling so the
// downstream sampler only sees the kept candidates.
// Inputs:
//   topK: top-k threshold tensor for this sampling step.
//   topP: top-p threshold tensor for this sampling step.
// Outputs:
//   logits: logits tensor filtered in place and consumed by the sampler.
void top_k_top_p(torch::Tensor& logits,
                 const torch::Tensor& topK,
                 const torch::Tensor& topP) {
  check_tensor(logits, "logits", "top_k_top_p");
  check_tensor(topK, "topK", "top_k_top_p");
  check_tensor(topP, "topP", "top_k_top_p");
  aclTensor* logits_ids = nullptr;
  aclTensor* topK_ids = nullptr;
  aclTensor* topP_ids = nullptr;
  int32_t device_id = logits.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  create_acltensor(&logits_ids, logits);
  create_acltensor(&topK_ids, topK);
  create_acltensor(&topP_ids, topP);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACL_SUCCESS(aclnnApplyTopKTopPGetWorkspaceSize(logits_ids,
                                                       topP_ids,
                                                       topK_ids,
                                                       logits_ids,
                                                       &workspace_size,
                                                       &executor),
                    "top_k_top_p: failed to get workspace size");
  void* workspace_addr = workspace_cache().get(stream, workspace_size);
  CHECK_ACL_SUCCESS(
      aclnnApplyTopKTopP(workspace_addr, workspace_size, executor, stream),
      "top_k_top_p: failed to apply top k top p");
  aclDestroyTensor(logits_ids);
  aclDestroyTensor(topK_ids);
  aclDestroyTensor(topP_ids);
}
}  // namespace xllm::kernel::npu
