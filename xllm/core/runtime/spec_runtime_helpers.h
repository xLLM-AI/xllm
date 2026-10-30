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

#include <glog/logging.h>
#include <torch/types.h>

#include "core/framework/config/speculative_config.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/platform/stream.h"
#include "core/runtime/forward_runtime_state.h"
#include "runtime/options.h"

namespace xllm::spec_detail {

// The group that speculative sampling state replicates over: the TP group
// when one exists, otherwise the whole process group.
inline ProcessGroup* spec_sampling_group(const ParallelArgs& parallel_args) {
  return parallel_args.tp_group_ != nullptr ? parallel_args.tp_group_
                                            : parallel_args.process_group_;
}

bool uses_orthogonal_cp_consensus(const ParallelArgs& parallel_args);

// Task-pipeline variant over explicit sampling groups: a greedy sample is
// rank-local only when no orthogonal CP group exists to disagree and no
// prior in-group broadcast has produced a divergent copy.
inline bool pipeline_tokens_rank_local(ProcessGroup* cp_sampling_group,
                                       SamplingMode sampling_mode,
                                       bool already_broadcast_in_group) {
  return sampling_mode == SamplingMode::GREEDY &&
         cp_sampling_group == nullptr && !already_broadcast_in_group;
}

// Gates on orthogonal CP consensus (always) or the sampling flag (non-greedy).
inline bool consensus_required(const ParallelArgs& parallel_args,
                               bool enable_spec_token_broadcast,
                               SamplingMode sampling_mode) {
  return uses_orthogonal_cp_consensus(parallel_args) ||
         (enable_spec_token_broadcast && sampling_mode != SamplingMode::GREEDY);
}

// Broadcasts the consensus tokens over the TP/CP groups; the single caller
// (keep_tokens_consistent) has already gated on consensus_required.
void broadcast_consensus_tokens(torch::Tensor& tokens,
                                const ParallelArgs& parallel_args);

template <typename Input>
void record_metadata_ready_event(Stream& stream, Input& input) {
  input.runtime.metadata_ready_event = stream.record_event_or_sync();
}

template <typename Input>
void wait_metadata_ready_event(const Input& input, Stream& stream) {
  CHECK(stream.wait_event(input.runtime.metadata_ready_event))
      << "failed to wait speculative metadata ready event";
}

// Target-side worker options inside a speculative pair: never overlap
// scheduling, always capture aux hidden states for the verifier.
inline runtime::Options spec_target_options(const runtime::Options& options) {
  runtime::Options opts = options;
  opts.enable_schedule_overlap(false).enable_graph_aux_hidden_states(true);
  return opts;
}

// Speculative decoding (including the draft-free Suffix algorithm) sizes
// workers for num_speculative_tokens + 1 decode tokens per step. Shared by
// the LLM and VLM engines.
inline runtime::Options make_speculative_worker_options(
    const runtime::Options& options) {
  auto worker_options = options;
  if (SpeculativeConfig::uses_speculative_pipeline(options)) {
    worker_options.num_decoding_tokens(options.num_speculative_tokens() + 1)
        .enable_speculative_decode(/*value=*/true);
  }
  return worker_options;
}

// Applies the universal draft tweaks to a caller-owned copy so families can
// keep chaining setters on the named lvalue.
inline void mark_spec_draft_options(runtime::Options& opts) {
  opts.enable_schedule_overlap(false).is_draft_model(true);
}

// Waits on the input's current metadata event, then records a fresh one.
template <typename Input>
void record_current_metadata_ready_event(Input& input, Stream& stream) {
  wait_metadata_ready_event(input, stream);
  record_metadata_ready_event(stream, input);
}

}  // namespace xllm::spec_detail
