/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#pragma once

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <torch/torch.h>

#include <limits>
#include <vector>

#include "core/framework/multimodal/mm_data.h"
#include "framework/batch/batch_forward_type.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "framework/request/sequences_group.h"
#include "runtime/forward_params.h"
#include "util/threadpool.h"

namespace xllm {

struct ModelArgs;

static uint64_t batch_counter_ = 1;
constexpr uint64_t UNINITIALIZED_BATCH_ID = 0x0;

class Batch {
 public:
  Batch() = default;

  Batch(Sequence* sequence);
  Batch(const std::vector<Sequence*>& sequences);

  void add(Sequence* sequence,
           uint32_t allowed_max_token = std::numeric_limits<uint32_t>::max());

  void add(const std::vector<Sequence*>& sequences);

  void add(SequencesGroup* sequence_group) {
    sequence_groups_.push_back(sequence_group);
  }

  void update_forward_type(Sequence* sequence);

  void refresh_forward_type();

  void set_swap_block_transfer_infos(
      std::vector<BlockTransferInfo> swap_block_transfer_infos) {
    swap_block_transfer_infos_ = std::move(swap_block_transfer_infos);
  }

  void set_batch_id() {
    if (batch_id_ == UNINITIALIZED_BATCH_ID) {
      batch_id_ = batch_counter_;
      batch_counter_++;
      if (batch_counter_ == UINT64_MAX) {
        batch_counter_ = 1;
      }
    }
  }

  uint64_t batch_id() const { return batch_id_; }

  // get the number of sequences in the batch
  size_t size() const { return sequences_.size(); }
  bool empty() const { return sequences_.empty() && sequence_groups_.empty(); }

  Sequence* operator[](size_t i) { return sequences_[i]; }

  // prepare forward inputs
  ForwardInput prepare_forward_input(uint32_t num_decoding_tokens,
                                     uint32_t min_decoding_bach_size,
                                     const ModelArgs& args,
                                     int32_t cp_size = 1);

  ForwardInput prepare_rec_forward_input(uint32_t num_decoding_tokens,
                                         uint32_t min_decoding_batch_size,
                                         const ModelArgs& args,
                                         MPMCThreadPool* thread_pool = nullptr);

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

  void process_beam_sequence_group(const RawForwardOutput& raw_output);
  void process_beam_sequence_group(const ForwardOutput& output);
  // mark all sequences as finished (used by rec model multi-round decoding)
  void finish();

  // Refresh sequences_ from sequence_groups_ after beam search processing.
  // This is needed for RecEngine because SequencesGroup::process_beam_search()
  // replaces its internal sequences_, invalidating pointers in
  // Batch::sequences_.
  void refresh_sequences_from_groups();

  const std::vector<uint32_t>& get_allowed_max_tokens() const {
    return allowed_max_tokens_;
  }

  std::unordered_map<uint32_t, uint32_t> cal_seq_exchange_index_test(
      std::vector<uint32_t>& kv_cache_tokens_num) {
    return cal_seq_exchange_index(kv_cache_tokens_num);
  }

  // Get all sequences from either sequences_ or sequence_groups_
  // Used by RecEngine to access sequences for stopping checker evaluation
  std::vector<Sequence*> get_sequences();

 private:
  struct OutputTarget {
    Sequence* sequence = nullptr;
    size_t sample_id = 0;
    bool from_sample_slot = false;
  };

  void refresh_output_targets();
  void refresh_onerec_prefill_output_targets();
  bool update_sequence_state(Sequence* seq, bool replace_fake_token);

  void append_token_for_sequence(Sequence* seq,
                                 const Token& token,
                                 int token_idx,
                                 bool replace_fake_token);

  void process_beam_search(bool force_requested_result_size = false);

  // Returns true only when the batch is a clean, uniform beam-search batch that
  // the device beam-search kernel can safely consume: every sequence group is a
  // beam group, all groups share one beam_width, each group is fully expanded
  // to exactly beam_width active (unfinished, already-decoding) sequences, and
  // no non-beam sequence is mixed in. When this is false, the kernel path would
  // mis-index across requests, so callers fall back to the host beam path.
  bool beam_kernel_batch_eligible() const;

  std::unordered_map<uint32_t, uint32_t> cal_seq_exchange_index(
      std::vector<uint32_t>& kv_cache_tokens_num);

  void dp_balance_shuffle_seqs();

  std::vector<Sequence*> sequences_;
  std::vector<SequencesGroup*> sequence_groups_;
  std::vector<BlockTransferInfo> swap_block_transfer_infos_;

  // max number of tokens to process for each sequence
  // default to max value
  std::vector<uint32_t> allowed_max_tokens_;

  std::vector<torch::Tensor> input_embeddings_vec_;

  // mm_data in the batch
  std::vector<MMData> mm_data_vec_;

  // Runtime sampling targets that were actually injected for the current
  // forward pass. Sample requests may contribute multiple targets per
  // sequence.
  std::vector<OutputTarget> output_targets_;

  BatchForwardType batch_forward_type_;

  uint64_t batch_id_ = UNINITIALIZED_BATCH_ID;
};

}  // namespace xllm
