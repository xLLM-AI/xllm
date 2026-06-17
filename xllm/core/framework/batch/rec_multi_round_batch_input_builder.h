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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

#include "batch_input_builder.h"
#include "core/framework/multimodal/mm_data.h"
#include "framework/request/sequence.h"
#include "framework/request/sequences_group.h"
#include "rec_batch_input_builder.h"
#include "runtime/forward_params.h"
#include "util/threadpool.h"

namespace xllm {

struct ModelArgs;

class RecMultiRoundBatchInputBuilder : public RecBatchInputBuilder {
 public:
  explicit RecMultiRoundBatchInputBuilder(
      const std::vector<SequencesGroup*>& sequence_groups,
      const std::vector<uint32_t>& allowed_max_tokens,
      const std::vector<torch::Tensor>& input_embeddings_vec,
      const std::vector<MMData>& mm_data_vec,
      // for beam-search
      std::vector<BlockTransferInfo>* swap_block_transfer_infos,
      const uint64_t batch_id,
      const ModelArgs* args,
      BatchForwardType batch_forward_type,
      MPMCThreadPool* thread_pool = nullptr);

  ~RecMultiRoundBatchInputBuilder() override = default;

  // Override base class method
  ForwardInput build_rec_forward_input(
      uint32_t num_decoding_tokens,
      uint32_t min_decoding_batch_size) override;

 private:
  // Build Rec multi-round forward input for the whole batch (internal
  // implementation).
  ForwardInput build_forward_input();

  // Local builder state (copy of the legacy BatchInputBuilder::BuilderState)
  struct BuilderState {
    // Token and position data
    std::vector<int32_t> flatten_tokens_vec;
    std::vector<int32_t> flatten_positions_vec;
    std::vector<torch::Tensor> mrope_positions_vec;

    // Sampling data
    std::vector<const RequestSamplingParam*> sampling_params;
    std::vector<int32_t> selected_token_idxes;
    std::vector<int32_t> sample_idxes;

    // Unique token tracking
    std::vector<std::vector<int64_t>> unique_token_ids_vec;
    std::vector<std::vector<int32_t>> unique_token_counts_vec;
    std::vector<int32_t> unique_token_lens_vec;

    // Sequence metadata
    BatchForwardType batch_forward_type;
    uint32_t max_seq_len = 0;
    uint32_t q_max_seq_len = 0;
#if defined(USE_NPU)
    std::vector<int32_t> seq_lens;
    std::vector<int32_t> q_seq_lens;
#elif defined(USE_MLU) || defined(USE_CUDA) || defined(USE_ILU) || \
    defined(USE_MUSA) || defined(USE_DCU)
    std::vector<int32_t> seq_lens = {0};    // cu_seq_lens
    std::vector<int32_t> q_seq_lens = {0};  // q_cu_seq_len
#endif

    // Cache and block data
    std::vector<int32_t> new_token_slot_ids;
    std::vector<std::vector<int32_t>> block_tables_vec;

    // beam search kernel input
    std::vector<float> acc_logprob_vec;

    // Additional data
    std::vector<int32_t> embedding_ids;
    std::vector<int32_t> linear_state_ids;
    std::vector<int32_t> extra_token_ids;
    uint32_t prefill_seq_len = 0;
    std::vector<TransferKVInfo> transfer_kv_infos;

    // for continuous kvcache
    std::vector<int64_t> new_cache_slot_offsets;  //[n_tokens]
    std::vector<int64_t> kv_cache_start_offsets;  //[n_seq]

    // for flashinfer
    std::vector<int32_t> paged_kv_indptr = {0};
    std::vector<int32_t> paged_kv_indices;
    std::vector<int32_t> paged_kv_last_page_len;
  };

 protected:
  // Core building methods - provide Rec multi-round specific logic.
  void process_single_sequence(
      int32_t seq_index,
      BuilderState* state_ptr = nullptr,
      std::unordered_set<int32_t>* write_block_ids_ptr = nullptr);

 private:
  // State management for RecMultiRound
  struct RecMultiRoundBuilderState {
    // Base state compatible with single-round builder
    BuilderState base_state;

    // Rec multi-round step tracking data
    std::vector<int32_t> step_tokens_vec;
    std::vector<int32_t> step_positions_vec;
    std::vector<torch::Tensor> step_mrope_positions_vec;

    // Rec multi-round decode state buffers
    std::vector<int32_t> decode_selected_token_idxes;
    std::vector<const RequestSamplingParam*> decode_sampling_params;
    std::vector<std::vector<int64_t>> decode_unique_token_ids_vec;
    std::vector<std::vector<int32_t>> decode_unique_token_counts_vec;
    std::vector<int32_t> decode_unique_token_lens_vec;
    std::vector<int32_t> decode_sample_idxes;
    std::vector<int32_t> decode_positions_vec;

    // Rec multi-round specific metadata
    uint32_t total_steps = 0;
  };

  // Enhanced state
  RecMultiRoundBuilderState rec_multi_round_state_;

 private:
  // Override extract_tokens_and_positions to handle Rec multi-round decode
  // logic.
  void extract_tokens_and_positions(Sequence* sequence,
                                    uint32_t n_kv_cache_tokens,
                                    uint32_t seq_len,
                                    RecMultiRoundBuilderState* state_ptr);

  // Rec multi-round specific forward input conversion functions
  ForwardInput state_to_forward_input();

  void setup_kv_cache_info(Sequence* sequence,
                           uint32_t n_kv_cache_tokens,
                           uint32_t seq_len,
                           uint32_t q_seq_len,
                           BuilderState* state_ptr,
                           std::unordered_set<int32_t>* write_block_ids_ptr);

  // Input data (same semantics as in BatchInputBuilder)
  // Extracted from sequence_groups_ in constructor
  std::vector<Sequence*> sequences_;
  const std::vector<uint32_t>& allowed_max_tokens_;
  const std::vector<torch::Tensor>& input_embeddings_vec_;
  const std::vector<MMData>& mm_data_vec_;
  const ModelArgs* args_;
  BatchForwardType batch_forward_type_;

  // Configuration/state
  int32_t num_sequences_ = 0;
  bool use_mrope_ = false;

  // swap blocks between device/host/global memory (optional)
  std::unordered_set<int32_t> write_block_ids_;
  std::vector<BlockTransferInfo>* swap_block_transfer_infos_ = nullptr;

  // thread pool for potential future multithreaded processing, not owned
  MPMCThreadPool* thread_pool_ = nullptr;
  uint64_t batch_id_ = 0;

  // whether prepare draft input for MTP(EAGLE) at Decode phase.
  bool is_mtp_decode_ = false;
};

}  // namespace xllm
