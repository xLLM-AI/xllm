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

#include <array>
#include <unordered_map>
#include <unordered_set>

#include "core/framework/speculative/adaptive/spec_cost_profile_store.h"
#include "core/framework/speculative/adaptive/spec_startup_calibration.h"
#include "core/framework/speculative/adaptive/verify_budget_planner.h"
#include "core/framework/speculative/rejection_sampler.h"
#include "core/runtime/task_execution_pipeline.h"

namespace xllm {

// Admission occurs before any graph execution mutates the task state.
bool supports_unified_block_graph(const SpeculativeTaskCapacity& capacity,
                                  uint32_t rows,
                                  const LlmForwardInput& input);

struct MtpContextTensors {
  torch::Tensor previous_tokens;
  torch::Tensor hidden;
  torch::Tensor positions;
  torch::Tensor kv_seq_lens;
  // Previous round's accepted tokens including its bonus, in [1, K+1].
  torch::Tensor accepted_lengths;
  torch::Tensor repair_required;
};

// Tensor rows use the existing EMBEDDING block ids. The scheduler owns their
// allocation and reuse; this storage has no allocator or retirement protocol.
// Publications run in FIFO order on the task stream. A Prepare snapshot waits
// for the producing generation before reading into its own Slot storage.
struct MtpContextStorage {
  torch::Tensor tokens_;
  MtpContextTensors mtp_state_;
  // Same request-id guard as EmbeddingCache; metadata promises an earlier
  // accepted Task will publish before a following decode reads the row.
  std::vector<std::string> request_ids_;
  std::vector<uint8_t> published_;
  // A nonzero generation promises a prior admitted Task's FIFO publication.
  // Prepare owns this metadata; Launch never reads or mutates it.
  std::vector<uint64_t> draft_prefix_generations_;
  torch::Tensor draft_prefix_hidden_;
};

// One Slot's fixed snapshot; embedding rows belong to the scheduler.
struct MtpContextView {
  MtpContextStorage* context_ = nullptr;
  uint32_t capacity_;
  bool prepared_ = false;
  bool read_published_state_ = false;
  bool reuse_draft_prefix_ = false;
  std::vector<int32_t> sorted_ids_;
  std::vector<uint32_t> identity_rows_;
  std::vector<int32_t> model_to_embedding_;
  std::vector<uint8_t> publication_mask_;
  std::vector<uint8_t> bootstrap_mask_;
  torch::Tensor host_rows_;
  torch::Tensor device_rows_;
  torch::Tensor rows_;
  torch::Tensor token_storage_;
  MtpContextTensors storage_;
  torch::Tensor tokens_;
  MtpContextTensors state_;
  torch::Tensor index_storage_;
  torch::Tensor mask_storage_;
  torch::Tensor last_indices_;
  torch::Tensor previous_indices_;
  torch::Tensor last_hidden_indices_;
  torch::Tensor previous_hidden_indices_;
  torch::Tensor current_token_output_;
  torch::Tensor previous_token_output_;
  torch::Tensor current_hidden_output_;
  torch::Tensor previous_hidden_output_;
  torch::Tensor no_previous_;
  torch::Tensor no_previous_mask_;
  torch::Tensor draft_prefix_hidden_storage_;
  torch::Tensor draft_prefix_hidden_;
};

// An event is never re-recorded while a producer/consumer retains this lease.
// Host submission acknowledgement prevents waiting on an unrecorded event;
// it does not wait for the device publication to complete.
struct MtpContextPublication {
  uint64_t task_id = 0;
  StreamEventPtr ready;
  std::mutex mutex;
  std::condition_variable submitted_cv;
  bool submitted = false;
};

// Fixed storage for the forwards in one speculative round. No scheduler,
// streams or independently retiring execution object lives here.
struct TaskExecutionPipeline::SpeculativeSlot {
  struct PackedVerify {
    // Adaptive verification alone owns packed inputs, confidence and feedback.
    // Fixed blocks use regular request-major Slot inputs instead.
    VerifyBudgetPlan plan;
    uint64_t generation = 0;
    uint32_t max_kv_length = 0;
    uint32_t selected_draft_tokens = 0;
    std::vector<int32_t> embedding_ids;
    std::vector<std::string> request_ids;
    std::vector<double> lagged_confidence;
    std::vector<int32_t> draft_counts;
    // Complete raw-B/G bindings are captured at drained startup for this Slot.
    // Admission checks this set before mutating persistent request context.
    std::unordered_set<uint64_t> unified_graph_bindings;
    // Proposal IDs, confidence and widths share an owned arena. Proposal +
    // confidence consensus is one broadcast; confidence + width feedback is
    // one tiny D2H copy covered by the ordinary result-ready fence.
    torch::Tensor arena_storage;
    // Control aliases otherwise-unused adaptive rejection-mask storage. The
    // pinned source borrows retired feedback until the next result D2H starts.
    torch::Tensor control;
    torch::Tensor host_control;
    torch::Tensor consensus;
    torch::Tensor confidence;
    torch::Tensor widths;
    torch::Tensor feedback;
    torch::Tensor host_feedback_storage;
    torch::Tensor host_feedback;
    torch::Tensor host_confidence;
    torch::Tensor host_widths;
    torch::Tensor previous_storage;
    torch::Tensor previous_tokens;
    torch::Tensor offsets_storage;
    torch::Tensor offsets;
    torch::Tensor owners_storage;
    torch::Tensor owners;
    torch::Tensor columns_storage;
    torch::Tensor columns;
    // Dense refresh/state recovery reuses MTP's existing hidden-state buffer.
    torch::Tensor hidden_columns;
  };
  struct Draft {
    std::vector<int64_t> graph_batch_sizes;
    std::vector<int64_t> captured_graph_batch_sizes;
    // First warmed compact FIA table bound for each physical draft bucket.
    std::unordered_map<uint32_t, uint64_t> captured_graph_kv_capacities;
    uint32_t padded_batch_size = 0;
    std::unique_ptr<SlotBuffer> input;
    std::optional<Sampling> sampling;
    torch::Tensor token_storage;
    torch::Tensor hidden_storage;
    torch::Tensor topk_storage;
    torch::Tensor next_hidden;
    torch::Tensor next_topk;
    ModelOutput output;
    torch::Tensor logits;
    SampleOutput sampled;
  };

  struct Recurrent {
    torch::Tensor host_state_indices;
    torch::Tensor state_indices;
    torch::Tensor accepted_lengths;
  };

  struct Bootstrap {
    torch::Tensor host_rows;
    torch::Tensor row_storage;
    torch::Tensor rows;
    torch::Tensor host_tokens;
    torch::Tensor token_storage;
    torch::Tensor tokens;
    MtpContextTensors host_state;
    MtpContextTensors storage;
    MtpContextTensors state;
  };

  bool decode = false;
  // Immutable until this Slot retires; capacity remains the maximum depth.
  uint32_t draft_steps = 0;
  std::unique_ptr<PackedVerify> verify;
  std::unique_ptr<Recurrent> recurrent;
  bool run_models = false;
  torch::Tensor dummy_hidden;
  bool is_warmup = false;
  bool unified_graph = false;
  bool context_prepared = false;
  bool context_published = false;
  std::shared_ptr<MtpContextPublication> publication;
  std::shared_ptr<MtpContextPublication> predecessor_publication;
  uint32_t rows = 0;
  uint32_t samples = 0;
  uint32_t model_tokens = 0;
  uint32_t target_padded_batch_size = 0;
  std::unique_ptr<SlotBuffer> target_prefill;
  std::unique_ptr<SlotBuffer> draft_prefill;
  std::unique_ptr<SlotBuffer> validate_input;
  std::unique_ptr<MtpContextView> state;
  std::optional<Sampling> target_sampling;
  std::vector<std::unique_ptr<Draft>> drafts;
  std::unique_ptr<Draft> prefix_refresh;
  std::vector<int32_t> upper_positions;
  std::vector<int32_t> upper_kv_lengths;
  SamplingParameters host_validate_sampling;
  Bootstrap bootstrap;
  torch::Tensor draft_first_indices;
  torch::Tensor draft_prefix_indices;
  torch::Tensor draft_block_indices;
  torch::Tensor target_token_storage;
  torch::Tensor target_tokens;
  torch::Tensor proposal_storage;
  torch::Tensor accepted_mask_storage;
  torch::Tensor host_hidden_storage;
  torch::Tensor proposal;
  torch::Tensor accepted_mask;
  torch::Tensor context_slots_storage;
  torch::Tensor context_slots;
  torch::Tensor host_draft_block_map;
  torch::Tensor draft_block_map;
  torch::Tensor block_gumbel;
  torch::Tensor host_hidden;
  ModelOutput target_output;
  torch::Tensor target_logits;
};

struct TaskExecutionPipeline::AdaptiveVerifyState {
  struct Feedback {
    uint64_t generation = 0;
    uint64_t rounds_since_full_width = 0;
    // Host-only history uses the largest compiled width. Only the configured
    // prefix is copied/consumed; Device and pinned buffers use the exact width.
    std::array<double, 7> confidence{};
  };
  struct History {
    std::string request_id;
    uint64_t request_generation = 0;
    std::array<Feedback, 2> versions;
  };
  struct RetiredPlan {
    uint64_t generation = 0;
    std::vector<int32_t> embedding_ids;
    std::vector<std::string> request_ids;
    VerifyBudgetPlan plan;
  };
  uint64_t retired_generation = 0;
  std::vector<History> histories;
  std::array<RetiredPlan, 2> retired_plans;
  std::shared_ptr<SpecCostProfileStore> owned_costs;
  std::optional<SpecCalibrationPoint> calibration_point;
  std::unique_ptr<SpecCalibrationSession> calibration_session;
  bool startup_calibration_complete = false;
};

}  // namespace xllm
