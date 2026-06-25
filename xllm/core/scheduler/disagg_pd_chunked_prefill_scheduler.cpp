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

#include "scheduler/disagg_pd_chunked_prefill_scheduler.h"

#include <algorithm>

#include "common/metrics.h"
#include "core/framework/config/scheduler_config.h"
#include "framework/batch/batch_factory.h"
#include "util/utils.h"

namespace xllm {

namespace {

bool exceeds_block_capacity(Sequence* sequence, KVCacheManager* manager) {
  const size_t block_size = static_cast<size_t>(manager->block_size());
  if (block_size == 0) {
    return true;
  }
  const size_t needed_blocks = util::ceil_div(
      static_cast<size_t>(sequence->num_prompt_tokens()), block_size);
  const std::vector<size_t> free_blocks = manager->num_free_blocks();
  const std::vector<size_t> used_blocks = manager->num_used_blocks();
  size_t max_blocks = 0;
  const size_t num_ranks = std::min(free_blocks.size(), used_blocks.size());
  for (size_t i = 0; i < num_ranks; ++i) {
    max_blocks = std::max(max_blocks, free_blocks[i] + used_blocks[i]);
  }
  return needed_blocks > max_blocks;
}

void update_block_metrics(KVCacheManager* manager) {
  CHECK(manager != nullptr);
  GAUGE_SET(kv_cache_utilization_perc, manager->kv_cache_utilization());
  GAUGE_SET(num_blocks_in_prefix_cache,
            util::min(manager->num_blocks_in_prefix_cache()));
  GAUGE_SET(num_free_blocks, util::max(manager->num_free_blocks()));
  GAUGE_SET(num_used_blocks, util::min(manager->num_used_blocks()));
}

}  // namespace

PDChunkBudget pick_pd_chunk_budget(size_t kv_tokens,
                                   size_t num_tokens,
                                   size_t max_chunk,
                                   size_t remaining_budget) {
  PDChunkBudget budget;
  budget.max_tokens = kv_tokens;
  if (kv_tokens >= num_tokens || remaining_budget == 0 || max_chunk == 0) {
    return budget;
  }
  const size_t remain = num_tokens - kv_tokens;
  budget.next_tokens = std::min({remain, max_chunk, remaining_budget});
  budget.max_tokens = kv_tokens + budget.next_tokens;
  return budget;
}

DisaggPDChunkedPrefillScheduler::DisaggPDChunkedPrefillScheduler(
    Engine* engine,
    const Options& options)
    : DisaggPDScheduler(engine, options) {}

void DisaggPDChunkedPrefillScheduler::match_prefix_blocks(Sequence* sequence) {
  CHECK(sequence != nullptr);
  if (!enable_prefix_cache_) {
    return;
  }

  if (sequence->kv_state().num_blocks(BlockType::KV) == 0) {
    kv_cache_manager_->allocate_shared(sequence);
    return;
  }
  if (!sequence->is_chunked_prefill_stage()) {
    return;
  }

  const size_t max_tokens_per_chunk = static_cast<size_t>(
      std::max(options_.max_tokens_per_chunk_for_prefill(), 64));
  const size_t total_chunked_size =
      util::ceil_div(sequence->num_tokens(), max_tokens_per_chunk);
  const int32_t match_frequency =
      ::xllm::SchedulerConfig::get_instance().chunked_match_frequency();
  CHECK_GT(match_frequency, 0);
  if (total_chunked_size < static_cast<size_t>(match_frequency)) {
    kv_cache_manager_->allocate_shared(sequence);
    return;
  }

  const size_t prefix_cache_interval =
      util::ceil_div(total_chunked_size, static_cast<size_t>(match_frequency));
  const size_t cur_chunked_index =
      sequence->kv_state().kv_cache_tokens_num() / max_tokens_per_chunk;
  if (cur_chunked_index % prefix_cache_interval == 0) {
    kv_cache_manager_->allocate_shared(sequence);
  }
}

bool DisaggPDChunkedPrefillScheduler::alloc_chunk(Sequence* sequence,
                                                  size_t token_budget,
                                                  size_t* actual_tokens) {
  CHECK(sequence != nullptr);
  CHECK(actual_tokens != nullptr);

  match_prefix_blocks(sequence);

  const size_t kv_tokens = sequence->kv_cache_tokens_num();
  const PDChunkBudget budget = pick_pd_chunk_budget(
      kv_tokens,
      sequence->num_tokens(),
      static_cast<size_t>(options_.max_tokens_per_chunk_for_prefill()),
      token_budget);
  *actual_tokens = budget.next_tokens;
  if (budget.next_tokens == 0) {
    return false;
  }
  return kv_cache_manager_->allocate(sequence, budget.max_tokens);
}

void DisaggPDChunkedPrefillScheduler::schedule_waiting_prefill(
    RequestPriorityQueue& queue,
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    std::vector<std::shared_ptr<Request>>& done) {
  while (!queue.empty() && remaining_token_budget > 0 &&
         remaining_seq_budget > 0) {
    std::shared_ptr<Request> request(queue.top());
    if (request->finished() || request->cancelled()) {
      kv_cache_manager_->deallocate(request.get());
      done.emplace_back(request);
      queue.pop_top();
      continue;
    }

    CHECK(!request->sequences().empty());
    if (!kv_cache_manager_->update_prefetch_result(
            request, options_.prefetch_timeout())) {
      queue.pop_top();
      queue.push(request);
      break;
    }

    Sequence* sequence = request->sequences()[0].get();
    size_t actual_tokens = 0;
    if (!alloc_chunk(sequence, remaining_token_budget, &actual_tokens)) {
      if (running_sequences_.empty() &&
          exceeds_block_capacity(sequence, kv_cache_manager_)) {
        queue.pop_top();
        kv_cache_manager_->deallocate(request.get());
        LOG(ERROR) << "Request prompt is too long, no enough resource to "
                      "schedule a single pd chunked prefill sequence.";
        response_processor_->process_failed_request(
            request,
            {StatusCode::RESOURCE_EXHAUSTED,
             "No enough resource to schedule a single pd chunked prefill "
             "sequence"});
      } else {
        queue.pop_top();
        queue.push(request);
      }
      break;
    }

    queue.pop_top();
    running_requests_.emplace_back(request);
    running_sequences_.emplace_back(sequence);
    running_sequences_budgets_.emplace_back(actual_tokens);
    remaining_token_budget -= actual_tokens;
    --remaining_seq_budget;

    const size_t kv_tokens = sequence->kv_cache_tokens_num();
    if (kv_tokens + actual_tokens >= sequence->num_prompt_tokens()) {
      last_step_prefill_ = true;
    }
  }
}

std::vector<Batch> DisaggPDChunkedPrefillScheduler::prepare_batch() {
  if (options_.instance_role() == InstanceRole::DECODE) {
    return ContinuousScheduler::prepare_batch();
  }

  std::shared_ptr<Request> request;
  while (request_queue_.read(request)) {
    CHECK(request);
    // PREFILL/MIX path in disagg PD only handles the first sequence.
    // For best_of_n, expansion to best_of sequences is deferred to the
    // DECODE instance (where prefix cache lets seq[1..best_of-1] reuse
    // seq[0]'s prompt KV). Expanding here would waste N x prefill compute.
    if (request->offline()) {
      waiting_priority_queue_offline_->push(request);
    } else {
      waiting_priority_queue_->push(request);
    }
  }

  std::vector<std::shared_ptr<Request>> done;
  for (auto it = running_requests_.rbegin(); it != running_requests_.rend();
       ++it) {
    if (*it == nullptr) {
      continue;
    }

    std::shared_ptr<Request> running = *it;
    running->update_connection_status();
    if (running->finished() || running->cancelled()) {
      kv_cache_manager_->deallocate(running.get());
      done.emplace_back(running);
      *it = nullptr;
      continue;
    }

    if (running->is_chunked_prefill_stage()) {
      if (running->offline()) {
        waiting_priority_queue_offline_->push(running);
      } else {
        waiting_priority_queue_->push(running);
      }
      *it = nullptr;
    }
  }

  last_step_prefill_ = false;
  running_requests_.clear();
  running_sequences_.clear();
  running_sequences_budgets_.clear();

  size_t remaining_token_budget =
      static_cast<size_t>(options_.max_tokens_per_batch());
  const size_t max_seq_budget =
      static_cast<size_t>(std::max(options_.max_seqs_per_batch(), 1));
  size_t remaining_seq_budget = max_seq_budget;
  running_requests_.reserve(max_seq_budget);
  running_sequences_.reserve(max_seq_budget);
  running_sequences_budgets_.reserve(max_seq_budget);
  schedule_waiting_prefill(*waiting_priority_queue_,
                           remaining_token_budget,
                           remaining_seq_budget,
                           done);
  schedule_waiting_prefill(*waiting_priority_queue_offline_,
                           remaining_token_budget,
                           remaining_seq_budget,
                           done);

  if (!done.empty()) {
    response_processor_->process_completed_requests(done);
  }

  if (running_sequences_.empty()) {
    update_block_metrics(kv_cache_manager_);
    return {};
  }

  std::vector<Batch> batches = BatchFactory::get_instance(options_.dp_size())
                                   ->create_batches(running_requests_,
                                                    running_sequences_,
                                                    running_sequences_budgets_);
  update_block_metrics(kv_cache_manager_);
  return batches;
}

}  // namespace xllm
