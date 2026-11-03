/* Copyright 2026 The xLLM Authors.

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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "core/runtime/forward_params.h"
#include "core/runtime/mtp_py_executor_pair.h"
#if defined(USE_NPU)
#include "core/kernels/npu/tilelang/mtp_prepare_next_draft.h"
#endif

namespace xllm {

class Stream;
class PyExecutorImpl;

struct UnifiedMtpContinuation {
  torch::Tensor accepted_tokens;
  torch::Tensor accepted_embeddings;
  torch::Tensor base_positions;
  torch::Tensor base_kv_seq_lens;
  std::vector<int32_t> embedding_ids;
  std::vector<std::string> request_ids;
};

struct UnifiedMtpExecutionResult {
  detail::MtpPyGraphOutput output;
  torch::Tensor base_positions;
  torch::Tensor base_kv_seq_lens;
};

// Owns Unified graph variants, continuation, and fused prepare workspaces.
// The worker's compute stream and model executors outlive this object. There
// is no legacy K-step scheduling or Host result publication in this owner.
class __attribute__((visibility("hidden"))) UnifiedMtpExecutor final {
 public:
  explicit UnifiedMtpExecutor(Stream& compute_stream);
  ~UnifiedMtpExecutor();

  template <typename TargetInput>
  const UnifiedMtpContinuation* continuation_for(
      const TargetInput& input) const;
  const UnifiedMtpContinuation& remember(UnifiedMtpContinuation continuation);
  void clear_continuation();

  template <typename TargetInput>
  LlmForwardInput prepare_next(const TargetInput& input,
                               const torch::Tensor& accepted_tokens,
                               const torch::Tensor& accepted_embeddings,
                               const torch::Tensor& base_positions,
                               const torch::Tensor& base_kv_seq_lens,
                               const torch::Tensor& embedding_placeholder,
                               int32_t block_size);

  template <typename TargetInput>
  UnifiedMtpExecutionResult execute(PyExecutorImpl& target_executor,
                                    PyExecutorImpl& draft_executor,
                                    const TargetInput& input,
                                    const LlmForwardInput& current_draft_input,
                                    int32_t speculative_tokens,
                                    int64_t vocab_size,
                                    int32_t block_size,
                                    bool target_step_major_layout,
                                    const torch::Tensor& accepted_prefix = {});

#if defined(USE_NPU)
  kernel::npu::MtpPrepareNextDraftWorkspace& acquire_draft_workspace(
      int64_t batch_size,
      const torch::Tensor& embeddings);
#endif

 private:
  Stream& compute_stream_;
  UnifiedMtpContinuation continuation_;
  std::unique_ptr<detail::MtpPyGraphVariantRegistry> registry_;
#if defined(USE_NPU)
  struct DraftWorkspaceEntry {
    torch::Device device;
    int64_t batch_size;
    int64_t hidden_size;
    torch::ScalarType dtype;
    kernel::npu::MtpPrepareNextDraftWorkspace workspace;
  };
  std::deque<DraftWorkspaceEntry> draft_workspaces_;
#endif
};

}  // namespace xllm
