/* Copyright 2025-2026 The xLLM Authors.

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

// batch_input_builder.h
#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/framework/multimodal/mm_data.h"
#include "framework/request/sequence.h"
#include "runtime/forward_params.h"
#include "util/threadpool.h"

namespace xllm {

class BatchInputBuilderTestPeer;
struct ModelArgs;

class BatchInputBuilder {
 public:
  explicit BatchInputBuilder(
      const std::vector<Sequence*>& sequences,
      const std::vector<uint32_t>& allowed_max_tokens,
      const std::vector<torch::Tensor>& input_embeddings_vec,
      const std::vector<MMData>& mm_data_vec,
      // for beam-search
      std::vector<BlockTransferInfo>* swap_block_transfer_infos,
      const uint64_t batch_id,
      const ModelArgs* args,
      BatchForwardType batch_forward_type,
      int32_t cp_size = 1,
      ThreadPool* thread_pool = nullptr);

  ForwardInput build_forward_input(uint32_t num_decoding_tokens,
                                   uint32_t min_decoding_batch_size);

 private:
  friend class BatchInputBuilderTestPeer;

  // Core building methods
  void process_sequences();
  void process_sequences_multithreaded();
  void process_multi_modal_inputs(Sequence* sequence,
                                  uint32_t n_kv_cache_tokens,
                                  uint32_t q_seq_len,
                                  int32_t seq_index);
  ForwardInput state_to_forward_input();
  void padding_decode_batch_size(uint32_t num_decoding_tokens,
                                 uint32_t min_decoding_batch_size);

  static TransferKVInfo build_step_transfer_info(
      const TransferKVInfo& full_info,
      const std::vector<uint64_t>& local_block_ids,
      size_t next_transfer_block_idx,
      uint32_t seq_len,
      uint32_t block_size,
      size_t* advanced_transfer_block_idx);

  static KVBlockTransferGroup build_group_step_transfer(
      const KVBlockTransferGroup& full_group,
      const std::vector<int32_t>& local_block_ids,
      size_t next_transfer_block_idx,
      uint32_t seq_len,
      uint32_t block_size,
      size_t* advanced_transfer_block_idx);

  void process_swap_block_infos(ForwardInput& forward_input);

  // State management
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
    // Tracking token counts in KV cache，only used for deepseek chunked prefill
    // ops on npu device
    std::vector<int32_t> kv_cache_tokens_nums;
#if defined(USE_NPU) || defined(USE_MUSA)
    std::vector<int32_t> seq_lens;
    std::vector<int32_t> q_seq_lens;
#elif defined(USE_MLU) || defined(USE_CUDA) || defined(USE_ILU) || \
    defined(USE_DCU)
    std::vector<int32_t> seq_lens = {0};    // cu_seq_lens
    std::vector<int32_t> q_seq_lens = {0};  // q_cu_seq_len
#endif

    // Cache and block data
    std::vector<int32_t> new_token_slot_ids;
    std::vector<std::vector<int32_t>> block_tables_vec;
    // Grouped cache block tables.
    // [manager_num][batch_size][block_ids]
    std::vector<std::vector<std::vector<int32_t>>> multi_block_tables;

    // beam search kernel input
    std::vector<float> acc_logprob_vec;

    // Additional data
    std::vector<int32_t> embedding_ids;
    std::vector<int32_t> linear_state_ids;
    std::vector<LinearStateCacheOp> linear_state_cache_ops;
    std::vector<std::string> request_ids;
    std::vector<int32_t> extra_token_ids;
    std::vector<int32_t> mtp_shifted_token_ids;
    std::vector<int32_t> mtp_bootstrap_row_idxes;
    std::vector<torch::Tensor> mtp_bootstrap_embeddings;
    std::vector<TransferKVInfo> transfer_kv_infos;

    // for continuous kvcache
    std::vector<int64_t> new_cache_slot_offsets;  //[n_tokens]
    std::vector<int64_t> kv_cache_start_offsets;  //[n_seq]

    // for flashinfer
    std::vector<int32_t> paged_kv_indptr = {0};
    std::vector<int32_t> paged_kv_indices;
    std::vector<int32_t> paged_kv_last_page_len;
  };

  // Helper methods for sequence processing
  void process_single_sequence(
      int32_t seq_index,
      BuilderState* state_ptr = nullptr,
      std::unordered_set<int32_t>* write_block_ids_ptr = nullptr);
  void extract_tokens_and_positions(Sequence* sequence,
                                    uint32_t n_kv_cache_tokens,
                                    uint32_t seq_len,
                                    uint32_t padded_seq_len,
                                    BuilderState* state_ptr = nullptr);
  // Append this batch row's linear-state transport fields: the live slot id
  // (always, so rows stay aligned) plus a LinearStateCacheOp carrying the
  // resolved restore/save plan for linear-attention models.
  void append_linear_state_row(Sequence* sequence,
                               uint32_t n_kv_cache_tokens,
                               uint32_t seq_len,
                               BuilderState& state);
  torch::Tensor get_mrope_positions(Sequence* sequence,
                                    uint32_t start,
                                    uint32_t end);

  void handle_sampling_parameters(Sequence* sequence,
                                  BuilderState* state_ptr = nullptr);
  void setup_kv_cache_info(
      Sequence* sequence,
      uint32_t n_kv_cache_tokens,
      uint32_t seq_len,
      uint32_t q_seq_len,
      BuilderState* state_ptr = nullptr,
      std::unordered_set<int32_t>* write_block_ids_ptr = nullptr);

  // Input data
  const std::vector<Sequence*>& sequences_;
  const std::vector<uint32_t>& allowed_max_tokens_;
  const std::vector<torch::Tensor>& input_embeddings_vec_;
  std::vector<MMData> mm_data_vec_;
  const ModelArgs* args_;

  // Builder state
  BuilderState state_;

  // Configuration
  bool use_mrope_ = false;
  uint32_t num_sequences_ = 0;
  bool need_unique_tokens_ = true;
  int32_t cp_size_ = 1;

  // copy in and out cache contents
  std::unordered_set<int32_t> write_block_ids_;
  std::vector<BlockTransferInfo>* swap_block_transfer_infos_ = nullptr;

  // thread pool for multithreaded processing, not owned
  ThreadPool* thread_pool_ = nullptr;
  uint64_t batch_id_;
};

}  // namespace xllm
