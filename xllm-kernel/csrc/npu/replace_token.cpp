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

#include <aclnn/acl_meta.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "core/common/macros.h"
#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/utils.h"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {

// Used by schedule overlap on NPU.
// This wrapper prepares the next decode-step input under the ACLNN update
// rule.
// Inputs:
//   src: sampled tokens from the previous step.
// Outputs:
//   dst: current-step input token tensor, updated in place after replacement.
void replace_token(torch::Tensor& dst,
                   torch::Tensor& src,
                   bool synchronize_stream) {
  check_tensor(dst, "dst", "replace_token");
  check_tensor(src, "src", "replace_token");
  using GetWorkspaceSize = aclnnStatus (*)(const aclTensor*,
                                           const aclTensor*,
                                           const aclTensor*,
                                           uint64_t*,
                                           aclOpExecutor**);
  using Launch = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
  static const auto get_workspace_size =
      reinterpret_cast<GetWorkspaceSize>(aclnn::detail::get_op_api_func_addr(
          "aclnnXllmKernelReplaceTokenGetWorkspaceSize"));
  static const auto launch = reinterpret_cast<Launch>(
      aclnn::detail::get_op_api_func_addr("aclnnXllmKernelReplaceToken"));
  CHECK(get_workspace_size && launch)
      << "ReplaceToken is missing from the xllm-kernel OPP";

  aclTensor* dst_ids = nullptr;
  aclTensor* src_ids = nullptr;
  int32_t device_id = dst.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  create_acltensor(&dst_ids, dst);
  create_acltensor(&src_ids, src);
  uint64_t workspace_size = 0;
  aclOpExecutor* executor;
  CHECK_ACL_SUCCESS(
      get_workspace_size(dst_ids, src_ids, dst_ids, &workspace_size, &executor),
      "replace_token: failed to get workspace size");
  void* workspace_addr = nullptr;
  CHECK_ACL_SUCCESS(launch(workspace_addr, workspace_size, executor, stream),
                    "replace_token: failed to replace token");
  if (synchronize_stream) {
    CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                      "replace_token: failed to synchronize stream");
  }
  aclDestroyTensor(dst_ids);
  aclDestroyTensor(src_ids);
}
}  // namespace xllm::kernel::npu
