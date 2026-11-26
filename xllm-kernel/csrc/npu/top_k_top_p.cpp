/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <acl/acl.h>
#include <aclnnop/aclnn_apply_top_k_top_p.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <array>
#include <unordered_map>

#include "csrc/npu/sampling.h"
#include "op_plugin/utils/op_api_common_base.h"

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
      CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS)
          << "top_k_top_p: failed to synchronize stream before "
             "workspace resize";
      CHECK_EQ(aclrtFree(workspace.addr_), ACL_SUCCESS)
          << "top_k_top_p: failed to free resized workspace";
    }

    CHECK_EQ(aclrtMalloc(
                 &workspace.addr_, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
             ACL_SUCCESS)
        << "top_k_top_p: failed to allocate workspace";
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
//   top_k: top-k threshold tensor for this sampling step.
//   top_p_threshold: top-p threshold tensor for this sampling step.
// Outputs:
//   logits: logits tensor filtered in place and consumed by the sampler.
void top_k_top_p(torch::Tensor& logits,
                 const torch::Tensor& top_k,
                 const torch::Tensor& top_p_threshold) {
  for (const torch::Tensor* input :
       std::array<const torch::Tensor*, 3>{&logits, &top_k, &top_p_threshold}) {
    CHECK(input->defined());
    CHECK_GT(input->numel(), 0);
    CHECK(input->device().is_privateuseone());
    CHECK_EQ(input->device(), logits.device());
  }
  const torch::DeviceGuard guard(logits.device());
  int32_t device_id = logits.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  // aclnn compares in the logits dtype (unlike the CUDA/CPU fp32 cutoff);
  // named so the converted storage outlives dispatch — ACL tensor conversion
  // keeps only the raw pointer.
  const torch::Tensor top_p = top_p_threshold.to(logits.scalar_type());
  aclTensor* logits_ids = ConvertType(logits);
  aclTensor* top_k_ids = ConvertType(top_k);
  aclTensor* top_p_threshold_ids = ConvertType(top_p);
  CHECK(logits_ids && top_k_ids && top_p_threshold_ids);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_EQ(aclnnApplyTopKTopPGetWorkspaceSize(logits_ids,
                                              top_p_threshold_ids,
                                              top_k_ids,
                                              logits_ids,
                                              &workspace_size,
                                              &executor),
           ACL_SUCCESS)
      << "top_k_top_p: failed to get workspace size";
  void* workspace_addr = workspace_cache().get(stream, workspace_size);
  CHECK_EQ(aclnnApplyTopKTopP(workspace_addr, workspace_size, executor, stream),
           ACL_SUCCESS)
      << "top_k_top_p: failed to apply top k top p";
  aclDestroyTensor(logits_ids);
  aclDestroyTensor(top_k_ids);
  aclDestroyTensor(top_p_threshold_ids);
}
}  // namespace xllm::kernel::npu
