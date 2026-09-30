/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <torch/types.h>

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "core/framework/batch/batch_state.h"
#include "core/framework/request/request.h"

namespace xllm {

struct ModelArgs;

// LLM/VLM engine-facing batch. Rec uses RecBatch directly.
class Batch final {
 public:
  Batch() = default;

  BatchInputType input_type() const;
  void reserve(size_t sequence_count, size_t group_count);

  explicit Batch(Sequence* sequence);
  explicit Batch(const std::vector<Sequence*>& sequences);

  void add(Sequence* sequence,
           uint32_t allowed_max_token = std::numeric_limits<uint32_t>::max());

  void add(const std::vector<Sequence*>& sequences);

  void add(SequencesGroup* sequence_group);

  const std::vector<SequencesGroup*>& sequence_groups() const {
    return state_.sequence_groups();
  }

  void update_forward_type(Sequence* sequence);

  void refresh_forward_type();

  void set_swap_block_transfer_infos(
      std::vector<BlockTransferInfo> swap_block_transfer_infos) {
    state_.set_swap_block_transfer_infos(std::move(swap_block_transfer_infos));
  }

  void set_batch_id();

  uint64_t batch_id() const { return state_.batch_id(); }

  // Logical execution rows.
  size_t size() const;
  bool empty() const { return state_.empty(); }
  size_t num_scheduled_sequences() const {
    return state_.sequence_plan().size();
  }
  size_t num_groups() const { return state_.sequence_groups().size(); }
  const BatchSequencePlan& sequence_plan() const {
    return state_.sequence_plan();
  }

  Sequence* operator[](size_t index) const;

  // prepare forward inputs
  ForwardInput prepare_forward_input(uint32_t num_decoding_tokens,
                                     uint32_t min_decoding_bach_size,
                                     const ModelArgs& args,
                                     int32_t cp_size = 1);

  // Prepare ForwardInput for distributed transport.
  ForwardInput prepare_forward_input(const ModelArgs& args,
                                     ThreadPool* thread_pool,
                                     int32_t cp_size = 1);

  // process output
  //
  // replace_fake_token:
  // In the scenario where enable_schedule_overlap is true,
  // the forward is divided into two stages.
  // The first stage populates the sequence with a fake token,
  // and the second stage replaces the previous fake token with a real token.
  // The boolean parameter `replace_fake_token` indicates
  // whether the current stage is the second stage.
  void process_sample_output(const SampleOutput& sample_output,
                             bool replace_fake_token,
                             bool force_requested_beam_result_size = false);

  void process_sample_output(const RawForwardOutput& raw_output,
                             bool replace_fake_token);

  // process output for beam search kernel
  void process_beam_search_output(const RawForwardOutput& raw_output,
                                  bool replace_fake_token);

  // Start a new sequence view after beam expansion. New beam rows have no
  // scheduler token limit.
  void refresh_sequences_from_groups();

  const std::vector<uint32_t>& get_allowed_max_tokens() const {
    return state_.sequence_plan().budgets();
  }

  std::unordered_map<uint32_t, uint32_t> cal_seq_exchange_index_test(
      std::vector<uint32_t>& kv_cache_tokens_num) {
    return BatchState::cal_seq_exchange_index(kv_cache_tokens_num);
  }

  // Return the current sequence view.
  std::vector<Sequence*> get_sequences();
  std::vector<Sequence*> get_sequences() const;

 private:
  BatchState state_;
};

}  // namespace xllm
