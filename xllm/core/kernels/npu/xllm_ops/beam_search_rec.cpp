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
#include "aclnn_beam_search_group.h"
#include "core/common/macros.h"
#include "core/kernels/npu/utils.h"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

namespace {

using BeamSearchGroupGetWorkspaceSizeOldFn = aclnnStatus (*)(const aclTensor*,
                                                             const aclTensor*,
                                                             const aclTensor*,
                                                             const aclTensor*,
                                                             int64_t,
                                                             const aclTensor*,
                                                             const aclTensor*,
                                                             const aclTensor*,
                                                             const aclTensor*,
                                                             const aclTensor*,
                                                             uint64_t*,
                                                             aclOpExecutor**);

using BeamSearchGroupGetWorkspaceSizeNewFn = aclnnStatus (*)(const aclTensor*,
                                                             const aclTensor*,
                                                             const aclTensor*,
                                                             const aclTensor*,
                                                             int64_t,
                                                             int64_t,
                                                             const aclTensor*,
                                                             const aclTensor*,
                                                             const aclTensor*,
                                                             const aclTensor*,
                                                             const aclTensor*,
                                                             uint64_t*,
                                                             aclOpExecutor**);

aclnnStatus beam_search_group_get_workspace_size(
    BeamSearchGroupGetWorkspaceSizeOldFn fn,
    const aclTensor* logprobs_ids,
    const aclTensor* top_tokens_ids,
    const aclTensor* top_logprobs_ids,
    const aclTensor* sequence_group_ids,
    int64_t current_step,
    int64_t /*top_k*/,
    const aclTensor* out_token_ids_ids,
    const aclTensor* out_token_index_ids,
    const aclTensor* out_log_probs_ids,
    const aclTensor* out_beam_count_prefix_sums_ids,
    const aclTensor* out_sequence_ids,
    uint64_t* workspace_size,
    aclOpExecutor** executor) {
  return fn(logprobs_ids,
            top_tokens_ids,
            top_logprobs_ids,
            sequence_group_ids,
            current_step,
            out_token_ids_ids,
            out_token_index_ids,
            out_log_probs_ids,
            out_beam_count_prefix_sums_ids,
            out_sequence_ids,
            workspace_size,
            executor);
}

aclnnStatus beam_search_group_get_workspace_size(
    BeamSearchGroupGetWorkspaceSizeNewFn fn,
    const aclTensor* logprobs_ids,
    const aclTensor* top_tokens_ids,
    const aclTensor* top_logprobs_ids,
    const aclTensor* sequence_group_ids,
    int64_t current_step,
    int64_t top_k,
    const aclTensor* out_token_ids_ids,
    const aclTensor* out_token_index_ids,
    const aclTensor* out_log_probs_ids,
    const aclTensor* out_beam_count_prefix_sums_ids,
    const aclTensor* out_sequence_ids,
    uint64_t* workspace_size,
    aclOpExecutor** executor) {
  return fn(logprobs_ids,
            top_tokens_ids,
            top_logprobs_ids,
            sequence_group_ids,
            current_step,
            top_k,
            out_token_ids_ids,
            out_token_index_ids,
            out_log_probs_ids,
            out_beam_count_prefix_sums_ids,
            out_sequence_ids,
            workspace_size,
            executor);
}

}  // namespace

void run_beam_search_rec_final(const torch::Tensor& logprobs,
                               const torch::Tensor& top_tokens,
                               const torch::Tensor& top_logprobs,
                               torch::Tensor& sequence_group,
                               int64_t current_step,
                               int64_t result_width,
                               torch::Tensor& out_token_ids,
                               torch::Tensor& out_token_index,
                               torch::Tensor& out_log_probs,
                               torch::Tensor& out_beam_count_prefix_sums,
                               torch::Tensor& out_sequence);

// Run one beam-search update in REC multi-round decoding on NPU.
// Inputs:
//   logprobs: accumulated beam scores from the previous round.
//   top_tokens/top_logprobs: current candidate tokens and their scores.
//   sequence_group: beam-to-sequence layout before this round.
//   current_step: decode round index passed to the NPU kernel.
// Outputs:
//   out_token_ids/out_token_index/out_log_probs: selected next tokens, source
//   beam indices, and updated accumulated scores.
//   out_beam_count_prefix_sums/out_sequence: per-request beam offsets and the
//   next sequence layout after this step.
void beam_search_rec(const torch::Tensor& logprobs,
                     const torch::Tensor& top_tokens,
                     const torch::Tensor& top_logprobs,
                     torch::Tensor& sequence_group,
                     int64_t current_step,
                     torch::Tensor& out_token_ids,
                     torch::Tensor& out_token_index,
                     torch::Tensor& out_log_probs,
                     torch::Tensor& out_beam_count_prefix_sums,
                     torch::Tensor& out_sequence) {
  check_tensor(logprobs, "logprobs", "beam_search_rec");
  check_tensor(top_tokens, "top_tokens", "beam_search_rec");
  check_tensor(top_logprobs, "top_logprobs", "beam_search_rec");
  check_tensor(sequence_group, "sequence_group", "beam_search_rec");
  if (top_tokens.dim() < 2) {
    LOG(FATAL) << "beam_search_rec: top_tokens must be at least 2D, got "
               << top_tokens.sizes();
  }
  const int64_t top_k = top_tokens.size(1);

  aclTensor* logprobs_ids = nullptr;
  aclTensor* top_tokens_ids = nullptr;
  aclTensor* top_logprobs_ids = nullptr;
  aclTensor* sequence_group_ids = nullptr;
  aclTensor* out_token_ids_ids = nullptr;
  aclTensor* out_token_index_ids = nullptr;
  aclTensor* out_log_probs_ids = nullptr;
  aclTensor* out_beam_count_prefix_sums_ids = nullptr;
  aclTensor* out_sequence_ids = nullptr;
  int32_t device_id = logprobs.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();

  create_acltensor(&logprobs_ids, logprobs);
  create_acltensor(&top_tokens_ids, top_tokens);
  create_acltensor(&top_logprobs_ids, top_logprobs);
  create_acltensor(&sequence_group_ids, sequence_group);
  create_acltensor(&out_token_ids_ids, out_token_ids);
  create_acltensor(&out_token_index_ids, out_token_index);
  create_acltensor(&out_log_probs_ids, out_log_probs);
  create_acltensor(&out_beam_count_prefix_sums_ids, out_beam_count_prefix_sums);
  create_acltensor(&out_sequence_ids, out_sequence);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACL_SUCCESS(
      beam_search_group_get_workspace_size(aclnnBeamSearchGroupGetWorkspaceSize,
                                           logprobs_ids,
                                           top_tokens_ids,
                                           top_logprobs_ids,
                                           sequence_group_ids,
                                           current_step,
                                           top_k,
                                           out_token_ids_ids,
                                           out_token_index_ids,
                                           out_log_probs_ids,
                                           out_beam_count_prefix_sums_ids,
                                           out_sequence_ids,
                                           &workspace_size,
                                           &executor),
      "beam_search_rec: failed to get workspace size for REC beam search");
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(
        aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
        "beam_search_rec: failed to allocate workspace for REC beam search");
  }
  CHECK_ACL_SUCCESS(
      aclnnBeamSearchGroup(workspace_addr, workspace_size, executor, stream),
      "beam_search_rec: failed to execute REC beam search");
  CHECK_ACL_SUCCESS(
      aclrtSynchronizeStream(stream),
      "beam_search_rec: failed to synchronize stream for REC beam search");
  aclDestroyTensor(logprobs_ids);
  aclDestroyTensor(top_tokens_ids);
  aclDestroyTensor(top_logprobs_ids);
  aclDestroyTensor(sequence_group_ids);
  aclDestroyTensor(out_token_ids_ids);
  aclDestroyTensor(out_token_index_ids);
  aclDestroyTensor(out_log_probs_ids);
  aclDestroyTensor(out_beam_count_prefix_sums_ids);
  aclDestroyTensor(out_sequence_ids);
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(
        aclrtFree(workspace_addr),
        "beam_search_rec: failed to free workspace for REC beam search");
  }
}

void beam_search_rec(const torch::Tensor& logprobs,
                     const torch::Tensor& top_tokens,
                     const torch::Tensor& top_logprobs,
                     torch::Tensor& sequence_group,
                     int64_t current_step,
                     int64_t result_width,
                     torch::Tensor& out_token_ids,
                     torch::Tensor& out_token_index,
                     torch::Tensor& out_log_probs,
                     torch::Tensor& out_beam_count_prefix_sums,
                     torch::Tensor& out_sequence) {
  if (result_width <= 0) {
    beam_search_rec(logprobs,
                    top_tokens,
                    top_logprobs,
                    sequence_group,
                    current_step,
                    out_token_ids,
                    out_token_index,
                    out_log_probs,
                    out_beam_count_prefix_sums,
                    out_sequence);
    return;
  }

  run_beam_search_rec_final(logprobs,
                            top_tokens,
                            top_logprobs,
                            sequence_group,
                            current_step,
                            result_width,
                            out_token_ids,
                            out_token_index,
                            out_log_probs,
                            out_beam_count_prefix_sums,
                            out_sequence);
}
}  // namespace xllm::kernel::npu
