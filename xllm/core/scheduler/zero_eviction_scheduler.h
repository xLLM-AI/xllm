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

#pragma once

#include "scheduler/continuous_scheduler.h"

namespace xllm {

struct SequenceStatus {
  SequenceStatus(uint32_t need_to_use, uint32_t release_block)
      : num_block_need_to_use_(need_to_use),
        num_release_block_(release_block) {}

  void print() const {
    LOG(INFO) << "SequenceStatus { "
              << "need_to_use: " << num_block_need_to_use_ << ", "
              << "release: " << num_release_block_ << " }";
  }

  uint32_t num_block_need_to_use_;
  uint32_t num_release_block_;
};

template <typename Func>
class ResourceGuard {
 public:
  explicit ResourceGuard(Func&& release_func)
      : release_func_(std::forward<Func>(release_func)) {}

  ~ResourceGuard() {
    if (is_necessory_release_) {
      release_func_();
    }
  }

  void success() { is_necessory_release_ = false; }

 private:
  bool is_necessory_release_ = true;
  Func release_func_;
};

class BlockCapacityGuard {
 public:
  BlockCapacityGuard(KVCacheManager* kv_cache_manager);

  bool if_accept_candidate_sequences(
      const std::vector<Sequence*>& candidate_sequences,
      const std::unique_ptr<RequestPriorityQueue>& running_queue,
      const std::vector<Sequence*>& running_sequences);

 private:
  int32_t block_size() const { return kv_cache_manager_->block_size(); }

  uint32_t num_block() const { return kv_cache_manager_->num_blocks(); }

  uint32_t num_blocks_in_useless() const {
    // TODO for mutil dp
    return num_block() - kv_cache_manager_->num_used_blocks()[0];
  }

  bool simulate_is_satisfied_for_candidate_sequences();

  void compute_reserved_block_num();

  void prefix_cache_for_candidate_sequences();

  uint32_t get_needed_block_num_for_prefill();

  std::vector<SequenceStatus> get_all_sequence_status();

  uint32_t num_block_need_to_use_for(const Sequence* sequence);

  uint32_t num_release_block_for(Sequence* sequence);

  std::vector<SequenceStatus> get_running_sequence_status();

 private:
  KVCacheManager* kv_cache_manager_;

  std::vector<Sequence*> candidate_sequences_;
  std::vector<Sequence*> decode_queue_;
  std::vector<Sequence*> running_sequences_;

  uint32_t num_reserved_block_for_prefill_;
};

class ZeroEvictionScheduler final : public ContinuousScheduler {
 public:
  ZeroEvictionScheduler(Engine* engine, const Options& options);
  virtual ~ZeroEvictionScheduler();

 private:
  void handle_prefill_requests(
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      std::vector<std::shared_ptr<Request>>& finished_requests);

  bool try_allocate_block_for(std::shared_ptr<Request> request,
                              std::vector<Sequence*>* prefill_sequences,
                              std::vector<size_t>* prefill_sequences_budget,
                              size_t* allocated_tokens,
                              size_t* allocated_seqs,
                              size_t remaining_token_budget,
                              size_t remaining_seq_budget);

  std::unique_ptr<BlockCapacityGuard> block_capacity_guard_;
};

}  // namespace xllm
