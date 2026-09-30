/* Copyright 2026 The xLLM Authors.
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

#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/batch/batch_input_data.h"
#include "core/framework/batch/batch_output_handler.h"
#include "core/framework/batch/batch_sequence_plan.h"
#include "core/runtime/forward_params.h"
#include "core/util/threadpool.h"

namespace xllm {

struct ModelArgs;
constexpr uint64_t UNINITIALIZED_BATCH_ID = 0x0;

// Shared sequence execution state, composed by Batch and RecBatchState. It
// maintains scheduling invariants and forward resources across model domains.
class BatchState final {
 public:
  void reserve(size_t sequence_count, size_t group_count);
  void add(Sequence* sequence, uint32_t token_budget);
  void add(SequencesGroup* group);
  void set_batch_id();
  uint64_t batch_id() const { return batch_id_; }
  bool empty() const {
    return sequence_plan_.empty() && sequence_groups_.empty();
  }
  BatchSequencePlan& sequence_plan() { return sequence_plan_; }
  const BatchSequencePlan& sequence_plan() const { return sequence_plan_; }
  const std::vector<SequencesGroup*>& sequence_groups() const {
    return sequence_groups_;
  }
  BatchOutputHandler& output_handler() { return output_handler_; }
  void set_swap_block_transfer_infos(std::vector<BlockTransferInfo> infos) {
    swap_block_transfer_infos_ = std::move(infos);
  }
  void update_forward_type(Sequence* sequence);
  void refresh_forward_type(const std::vector<Sequence*>& sequences);
  void refresh_sequences_from_groups();
  std::vector<Sequence*> group_sequences() const;
  size_t num_group_sequences() const;
  Sequence* group_sequence(size_t index) const;
  BatchInputData input_data(const BatchSequencePlan& plan);
  ForwardInput prepare_sequence_input(uint32_t num_decoding_tokens,
                                      uint32_t min_decoding_batch_size,
                                      const ModelArgs& args,
                                      int32_t cp_size);
  ForwardInput prepare_distributed_input(const ModelArgs& args,
                                         ThreadPool* thread_pool,
                                         int32_t cp_size);
  static std::unordered_map<uint32_t, uint32_t> cal_seq_exchange_index(
      std::vector<uint32_t>& kv_cache_tokens_num);

 private:
  void dp_balance_shuffle_seqs();
  bool has_partial_finished_beam_group() const;
  BatchSequencePlan sequence_plan_;
  std::vector<SequencesGroup*> sequence_groups_;
  std::vector<BlockTransferInfo> swap_block_transfer_infos_;
  std::vector<torch::Tensor> input_embeddings_vec_;
  std::vector<MMData> mm_data_vec_;
  // Keep serialized restore sources alive through worker-result processing.
  std::vector<Block> linear_restore_src_blocks_;
  BatchOutputHandler output_handler_;
  BatchForwardType batch_forward_type_;
  uint64_t batch_id_ = UNINITIALIZED_BATCH_ID;
};

}  // namespace xllm
