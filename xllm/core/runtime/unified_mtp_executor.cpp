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

#include "core/runtime/unified_mtp_executor.h"

#include <glog/logging.h>

#include <algorithm>
#include <utility>

#include "core/common/metrics.h"
#include "core/framework/speculative/mtp_async_input_builder.h"
#include "core/platform/stream.h"
#include "core/runtime/py_executor_impl.h"

namespace xllm {
namespace {
constexpr int32_t kMaxUnifiedMtpVariants = 8;
}  // namespace

UnifiedMtpExecutor::UnifiedMtpExecutor(Stream& compute_stream)
    : compute_stream_(compute_stream) {}
UnifiedMtpExecutor::~UnifiedMtpExecutor() {
#if defined(USE_NPU)
  if (registry_ != nullptr || !draft_workspaces_.empty()) {
    CHECK_EQ(compute_stream_.synchronize(), 0)
        << "failed to retire Unified MTP inputs before destruction";
  }
#endif
}

const UnifiedMtpContinuation* UnifiedMtpExecutor::continuation_for(
    const LlmForwardInput& input) const {
  const auto& identity = input.input_params.embedding;
  if (continuation_.embedding_ids != identity.embedding_ids ||
      continuation_.request_ids != identity.request_ids ||
      !continuation_.accepted_tokens.defined() ||
      !continuation_.accepted_embeddings.defined() ||
      !continuation_.base_positions.defined() ||
      !continuation_.base_kv_seq_lens.defined()) {
    return nullptr;
  }
  return &continuation_;
}

const UnifiedMtpContinuation& UnifiedMtpExecutor::remember(
    UnifiedMtpContinuation continuation) {
  continuation_ = std::move(continuation);
  return continuation_;
}

void UnifiedMtpExecutor::clear_continuation() {
  continuation_ = UnifiedMtpContinuation();
}

LlmForwardInput UnifiedMtpExecutor::prepare_next(
    const LlmForwardInput& input,
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& accepted_embeddings,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    const torch::Tensor& embedding_placeholder,
    int32_t block_size) {
  c10::StreamGuard stream_guard = compute_stream_.set_stream_guard();
  CHECK(compute_stream_.wait_event(input.runtime.metadata_ready_event))
      << "failed to wait Unified MTP continuation input";
  LlmForwardInput prepared;
  // The Unified recipe owns its immutable greedy plans. A draft sampling
  // dictionary/parameter copy has no consumer in this fused prepare path.
  mtp_async::prepare_next_draft_from_accepted_state(
      prepared,
      input,
      accepted_tokens,
      accepted_embeddings,
      embedding_placeholder,
      base_positions,
      base_kv_seq_lens,
      /*use_chunked_prefill=*/false,
      /*rebuild_expanded_decode_metadata=*/false,
      block_size,
      /*require_fused_npu_kernel=*/true,
#if defined(USE_NPU)
      &acquire_draft_workspace(input.input_params.meta.num_sequences,
                               accepted_embeddings));
#else
      nullptr);
#endif
  return prepared;
}

UnifiedMtpExecutionResult UnifiedMtpExecutor::execute(
    PyExecutorImpl& target_executor,
    PyExecutorImpl& draft_executor,
    const LlmForwardInput& input,
    const LlmForwardInput& current_draft_input,
    int32_t speculative_tokens,
    int64_t vocab_size,
    int32_t block_size,
    bool target_step_major_layout) {
  const int32_t batch_size = input.input_params.meta.num_sequences;
  CHECK_GT(batch_size, 0);
  CHECK_EQ(current_draft_input.positions.numel(), batch_size * 2)
      << "unified Python MTP graph requires [repair,current] draft rows";
  CHECK_EQ(current_draft_input.token_ids.numel(), batch_size * 2)
      << "unified Python MTP graph requires [repair,current] draft tokens";
  CHECK(current_draft_input.input_params.embedding.input_embedding.defined())
      << "unified Python MTP graph requires draft embeddings";

  c10::StreamGuard stream_guard = compute_stream_.set_stream_guard();
  CHECK(compute_stream_.wait_event(
      current_draft_input.runtime.metadata_ready_event))
      << "failed to wait Unified MTP input producer";

  torch::Tensor draft_rows =
      current_draft_input.token_ids.view({batch_size, 2});
  torch::Tensor draft_positions =
      current_draft_input.positions.view({batch_size, 2});
  torch::Tensor draft_kv_rows =
      current_draft_input.input_params.attention.device.kv_seq_lens.view(
          {batch_size, 2});
  // The Python graph owns int64 token destinations. Let its final copy cast
  // these strided int32 views, without two intermediate device allocations.
  torch::Tensor seed_token_ids = draft_rows.select(/*dim=*/1, /*index=*/1);
  torch::Tensor repair_token_ids = draft_rows.select(/*dim=*/1, /*index=*/0);
  torch::Tensor base_positions = draft_positions.select(/*dim=*/1, /*index=*/1)
                                     .clone(torch::MemoryFormat::Contiguous);
  torch::Tensor kv_seq_lens = draft_kv_rows.select(/*dim=*/1, /*index=*/1)
                                  .to(torch::kInt)
                                  .clone(torch::MemoryFormat::Contiguous);
  torch::Tensor draft_embedding =
      current_draft_input.input_params.embedding.input_embedding;

  if (!registry_) {
    LOG(INFO) << "MTP unified Python graph registry create begin";
    registry_ = detail::MtpPyGraphVariantRegistry::create(
        target_executor, draft_executor, kMaxUnifiedMtpVariants);
    LOG(INFO) << "MTP unified Python graph registry create done";
  }
  detail::MtpPyGraphOutput graph_output = registry_->execute_sparse(
      input.input_params.attention.device.block_tables,
      current_draft_input.input_params.attention.device.kv_seq_lens,
      current_draft_input.input_params.attention.device.new_cache_slots,
      repair_token_ids,
      seed_token_ids,
      base_positions,
      kv_seq_lens,
      draft_embedding,
      batch_size,
      speculative_tokens,
      vocab_size,
      block_size,
      target_step_major_layout,
      input.sampling_params.logprobs,
      input.sampling_params.max_top_logprobs);

#if defined(USE_NPU)
  const torch::Tensor& destination = graph_output.draft_embedding_destination;
  CHECK(destination.defined() && destination.is_contiguous());
  CHECK(destination.sizes() == draft_embedding.sizes());
  CHECK_EQ(destination.scalar_type(), draft_embedding.scalar_type());
  CHECK_EQ(destination.device(), draft_embedding.device());
  // Static input storage is owned by the Python variant. Holding this Tensor
  // retains the allocation across FIFO/capacity transitions; the next fused
  // prepare follows previous graph reads on compute_stream_. No TensorImpl
  // owned by the scheduler is rebound, and old output snapshots stay separate.
  acquire_draft_workspace(batch_size, destination).output.embeddings =
      destination;
#endif

  // Count one completed graph execution, independent of batch size or K.
  COUNTER_INC(speculative_unified_graph_executions_total);

  continuation_.accepted_tokens = graph_output.committed_tokens;
  continuation_.accepted_embeddings = graph_output.target_embeddings;
  continuation_.base_positions = base_positions;
  continuation_.base_kv_seq_lens = kv_seq_lens;
  // Reuse the identity vectors' capacity on steady replay. Constructing a
  // temporary continuation and moving it in would allocate both every time.
  continuation_.embedding_ids = input.input_params.embedding.embedding_ids;
  continuation_.request_ids = input.input_params.embedding.request_ids;
  return {std::move(graph_output),
          std::move(base_positions),
          std::move(kv_seq_lens)};
}

#if defined(USE_NPU)
kernel::npu::MtpPrepareNextDraftWorkspace&
UnifiedMtpExecutor::acquire_draft_workspace(int64_t batch_size,
                                            const torch::Tensor& embeddings) {
  const torch::Device device = embeddings.device();
  const int64_t hidden_size = embeddings.size(-1);
  const torch::ScalarType dtype = embeddings.scalar_type();
  auto existing = std::find_if(
      draft_workspaces_.begin(),
      draft_workspaces_.end(),
      [device, batch_size, hidden_size, dtype](const auto& entry) {
        return entry.device == device && entry.batch_size == batch_size &&
               entry.hidden_size == hidden_size && entry.dtype == dtype;
      });
  if (existing != draft_workspaces_.end()) {
    return existing->workspace;
  }
  if (draft_workspaces_.size() >= static_cast<size_t>(kMaxUnifiedMtpVariants)) {
    // Cold FIFO eviction only: all consumers of the retired workspace finish
    // on this stream. Never discard live input storage to avoid this wait.
    CHECK_EQ(compute_stream_.synchronize(), 0);
    draft_workspaces_.pop_front();
  }
  draft_workspaces_.emplace_back(
      DraftWorkspaceEntry{device, batch_size, hidden_size, dtype, {}});
  return draft_workspaces_.back().workspace;
}
#endif

}  // namespace xllm
