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

#include <c10/core/Device.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <nlohmann/json.hpp>
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include "acl/acl.h"
#include "aclnn_beam_search.h"
#include "core/common/macros.h"
#include "core/kernels/npu/utils.h"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

// Used by the standard BeamSearcher sampling path on NPU.
// This wrapper runs one regular beam-search update from the current beam
// scores and candidate token scores.
// Inputs:
//   logprobs: accumulated scores for the current live beams.
//   top_tokens: candidate token ids for this step.
//   top_logprobs: candidate token scores for this step.
// Outputs:
//   src_seq_idxes: source beam selected for each new beam.
//   out_logprobs: updated accumulated beam scores.
//   out_tokens: chosen next token ids for the next sampling step.
void beam_search(const torch::Tensor& logprobs,
                 const torch::Tensor& top_tokens,
                 const torch::Tensor& top_logprobs,
                 torch::Tensor& src_seq_idxes,
                 torch::Tensor& out_logprobs,
                 torch::Tensor& out_tokens) {
  check_tensor(logprobs, "logprobs", "beam_search");
  check_tensor(top_tokens, "top_tokens", "beam_search");
  check_tensor(top_logprobs, "top_logprobs", "beam_search");
  aclTensor* logprobs_ids = nullptr;
  aclTensor* top_tokens_ids = nullptr;
  aclTensor* top_logprobs_ids = nullptr;
  aclTensor* src_seq_idxes_ids = nullptr;
  aclTensor* out_logprobs_ids = nullptr;
  aclTensor* out_tokens_ids = nullptr;
  int32_t device_id = logprobs.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  create_acltensor(&logprobs_ids, logprobs);
  create_acltensor(&top_tokens_ids, top_tokens);
  create_acltensor(&top_logprobs_ids, top_logprobs);
  create_acltensor(&src_seq_idxes_ids, src_seq_idxes);
  create_acltensor(&out_logprobs_ids, out_logprobs);
  create_acltensor(&out_tokens_ids, out_tokens);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  // LOG(INFO) << "beam_search_ops logprobs: " << logprobs;
  // LOG(INFO) << "beam_search_ops top_tokens: " << top_tokens;
  // LOG(INFO) << "beam_search_ops top_logprobs: " << top_logprobs;

  CHECK_ACL_SUCCESS(aclnnBeamSearchGetWorkspaceSize(logprobs_ids,
                                                    top_tokens_ids,
                                                    top_logprobs_ids,
                                                    out_tokens_ids,
                                                    src_seq_idxes_ids,
                                                    out_logprobs_ids,
                                                    &workspace_size,
                                                    &executor),
                    "beam_search: failed to get workspace size");
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(
        aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
        "beam_search: failed to allocate workspace");
  }
  CHECK_ACL_SUCCESS(
      aclnnBeamSearch(workspace_addr, workspace_size, executor, stream),
      "beam_search: failed to perform beam search");
  CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                    "beam_search: failed to synchronize stream");
  aclDestroyTensor(logprobs_ids);
  aclDestroyTensor(top_tokens_ids);
  aclDestroyTensor(top_logprobs_ids);
  aclDestroyTensor(src_seq_idxes_ids);
  aclDestroyTensor(out_logprobs_ids);
  aclDestroyTensor(out_tokens_ids);
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(aclrtFree(workspace_addr),
                      "beam_search: failed to free workspace");
  }
  // LOG(INFO) << "beam_search_ops src_seq_idxes: " << src_seq_idxes;
  // LOG(INFO) << "beam_search_ops out_logprobs: " << out_logprobs;
  // LOG(INFO) << "beam_search_ops out_tokens: " << out_tokens;
}
}  // namespace xllm::kernel::npu
