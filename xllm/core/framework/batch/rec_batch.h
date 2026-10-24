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

#include <limits>
#include <utility>

#include "core/framework/batch/rec_batch_state.h"

namespace xllm {

// Rec engine-facing batch, backed by RecBatchState.
class RecBatch final {
 public:
  explicit RecBatch(BatchInputType input_type);
  BatchInputType input_type() const { return state_.input_type(); }
  RecBatchState& state() { return state_; }
  const RecBatchState& state() const { return state_; }
  void reserve(size_t sequence_count, size_t group_count) {
    state_.reserve(sequence_count, group_count);
  }
  void add(Sequence* sequence,
           uint32_t token_budget = std::numeric_limits<uint32_t>::max()) {
    state_.add(sequence, token_budget);
  }
  void add(SequencesGroup* group) { state_.add(group); }
  void set_batch_id() { state_.set_batch_id(); }
  uint64_t batch_id() const { return state_.batch_id(); }
  bool empty() const { return state_.empty(); }
  size_t num_scheduled_sequences() const {
    return state_.num_scheduled_sequences();
  }
  size_t num_groups() const { return state_.num_groups(); }
  void set_swap_block_transfer_infos(std::vector<BlockTransferInfo> infos) {
    state_.set_swap_block_transfer_infos(std::move(infos));
  }
  const std::vector<SequencesGroup*>& sequence_groups() const {
    return state_.sequence_groups();
  }
  const BatchSequencePlan& sequence_plan() const {
    return state_.sequence_plan();
  }
  const std::vector<uint32_t>& get_allowed_max_tokens() const {
    return state_.get_allowed_max_tokens();
  }
  void refresh_forward_type() { state_.refresh_forward_type(); }
  Sequence* operator[](size_t index) const { return sequence(index); }
  size_t size() const;
  Sequence* sequence(size_t index) const;
  std::vector<Sequence*> get_sequences() const;
  void refresh_sequences_from_groups();
  bool uses_group_input() const;
  RecForwardInput prepare_forward_input(uint32_t num_decoding_tokens,
                                        uint32_t min_decoding_batch_size,
                                        const ModelArgs& args,
                                        int32_t cp_size = 1);
  RecForwardInput prepare_forward_input(const ModelArgs& args,
                                        ThreadPool* thread_pool,
                                        int32_t cp_size = 1);
  RecForwardInput prepare_rec_forward_input(uint32_t num_decoding_tokens,
                                            uint32_t min_decoding_batch_size,
                                            const ModelArgs& args,
                                            MPMCThreadPool* thread_pool);
  void process_sample_output(const RawForwardOutput& output,
                             bool replace_fake_token);
  void process_sample_output(const SampleOutput& output,
                             bool replace_fake_token,
                             bool force_requested_beam_result_size);
  void process_beam_search_output(const RawForwardOutput& output,
                                  bool replace_fake_token);
  void process_beam_sequence_group(const ForwardOutput& output);
  void finish();

 private:
  RecBatchState state_;
};

}  // namespace xllm
