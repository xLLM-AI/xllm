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

#include "scheduler/scheduler_policy.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>

#include "async_response_processor.h"
#include "common/metrics.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/kv_cache_store_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "framework/batch/batch_factory.h"
#include "framework/request/priority_comparator.h"
#include "platform/platform.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {

// LCM helper for size_t values. Returns 0 when either argument is 0.
inline size_t lcm_size_t(size_t a, size_t b) {
  if (a == 0 || b == 0) {
    return 0;
  }
  return (a / std::gcd(a, b)) * b;
}

// Round a chunked-prefill chunk size DOWN to a multiple of
// lcm(2 * cp_size, kv_split_size * block_size).
inline size_t maybe_align_cp_chunk_tokens(size_t num_tokens,
                                          int32_t cp_size,
                                          int32_t kv_split_size,
                                          int32_t block_size,
                                          size_t remaining_in_seq) {
  if (cp_size <= 1 && kv_split_size <= 1) {
    return num_tokens;
  }
  if (block_size <= 0 || num_tokens == 0) {
    return num_tokens;
  }
  if (num_tokens >= remaining_in_seq) {
    return num_tokens;
  }
  const size_t cp_term =
      cp_size > 1 ? static_cast<size_t>(2) * static_cast<size_t>(cp_size) : 1;
  const size_t kv_term =
      kv_split_size > 1
          ? static_cast<size_t>(kv_split_size) * static_cast<size_t>(block_size)
          : 1;
  const size_t alignment = lcm_size_t(cp_term, kv_term);
  if (alignment == 0 || num_tokens < alignment) {
    return num_tokens;
  }
  return (num_tokens / alignment) * alignment;
}

// Estimate extra blocks needed for a decode step with speculative tokens.
size_t estimate_decode_extra_blocks(Sequence* sequence,
                                    size_t updated_num_tokens,
                                    size_t block_size) {
  const size_t num_blocks = sequence->kv_state().num_blocks(BlockType::KV);
  const size_t num_blocks_needed =
      (updated_num_tokens + block_size - 1) / block_size;
  if (num_blocks_needed > num_blocks) {
    return num_blocks_needed - num_blocks;
  }
  if (sequence->check_beam_search() &&
      !sequence->kv_state().src_blocks().empty() &&
      sequence->kv_state().need_swap()) {
    return 1;
  }
  return 0;
}

size_t get_sequence_free_blocks_for_rank(KVCacheManager* kv_cache_manager,
                                         int32_t dp_rank) {
  const auto free_blocks = kv_cache_manager->num_free_blocks();
  if (free_blocks.empty()) {
    return 0;
  }
  if (dp_rank >= 0 && static_cast<size_t>(dp_rank) < free_blocks.size()) {
    return free_blocks[dp_rank];
  }
  return util::max(free_blocks);
}

}  // namespace

// =============================================================================
// Construction
// =============================================================================

SchedulerPolicy::SchedulerPolicy(const BatchMode& mode,
                                 const ContinuousScheduler::Options& options)
    : batch_mode_(mode), options_(options) {}

void SchedulerPolicy::adjust_latency_budget_and_reorder(
    RequestPriorityQueue* /*first_queue*/,
    RequestPriorityQueue* /*second_queue*/,
    double& /*latency_budget*/,
    bool /*for_prefill*/,
    const SchedulerState& /*state*/) {}

// =============================================================================
// Common phases
// =============================================================================

void SchedulerPolicy::drain_request_queue(
    SchedulerState& state,
    folly::MPMCQueue<std::shared_ptr<Request>>& request_queue) {
  std::shared_ptr<Request> request;
  while (request_queue.read(request)) {
    CHECK(request);

    if (!state.enable_prefix_cache) {
      request->expand_sequences(/*force=*/false);
    }

    if (request->sequences()[0]->kv_state().kv_cache_tokens_num() == 0) {
      // New request goes to waiting queue (back = FIFO from MPMC).
      state.prefill_queue.push(request, /*if_back=*/true);
    } else {
      // Request from prefill instance in disagg PD mode -- already has KV.
      state.running_requests.emplace_back(request);
    }
  }
}

std::vector<std::shared_ptr<Request>> SchedulerPolicy::collect_finished(
    SchedulerState& state) {
  std::vector<std::shared_ptr<Request>> finished_requests;
  for (auto it = state.running_requests.rbegin();
       it != state.running_requests.rend();
       ++it) {
    if (*it == nullptr) {
      continue;
    }
    std::shared_ptr<Request> request = *it;
    request->update_connection_status();
    if (request->finished() || request->cancelled()) {
      clear_mtp_bootstrap(request.get(), state);
      state.kv_cache_manager->deallocate(request.get());
      finished_requests.emplace_back(request);
      *it = nullptr;
    }
  }
  return finished_requests;
}

void SchedulerPolicy::reset_batch_state(SchedulerState& state) {
  state.last_step_prefill = false;
  state.running_requests.clear();
  state.running_sequences.clear();
  state.running_sequences_budgets.clear();
}

// =============================================================================
// Requeue helper
// =============================================================================
// Prefill scheduling
// =============================================================================

void SchedulerPolicy::schedule_prefill_from_queue(
    RequestPriorityQueue* queue,
    SchedulerState& state,
    ScheduleBudget& budget,
    std::vector<std::shared_ptr<Request>>& finished) {
  if (queue == nullptr || queue->empty()) {
    return;
  }

  bool budget_exhausted = false;
  bool blocks_exhausted = false;

  while (!queue->empty() && budget.remaining_seq_budget > 0 &&
         budget.remaining_token_budget > 0 &&
         budget.latency_budget > budget.estimate_latency) {
    // Memory-utilization short-circuit.
    if (!options_.enable_disagg_pd() &&
        state.kv_cache_manager->kv_cache_utilization() >=
            SchedulerConfig::get_instance()
                .prefill_scheduling_memory_usage_threshold()) {
      blocks_exhausted = true;
      break;
    }

    std::shared_ptr<Request> request(queue->top());
    if (request->finished() || request->cancelled()) {
      clear_mtp_bootstrap(request.get(), state);
      state.kv_cache_manager->deallocate(request.get());
      finished.emplace_back(request);
      queue->pop_top();
      continue;
    }

    const size_t num_sequences = request->sequences().size();
    if (!request->preempted()) {
      CHECK(num_sequences == 1 || num_sequences == request->best_of())
          << "Waiting request should have either 1 or best_of("
          << request->best_of() << ") sequences, got " << num_sequences;
    }

    if (!state.kv_cache_manager->update_prefetch_result(
            request, options_.prefetch_timeout())) {
      queue->pop_top();
      queue->push(request);
      continue;
    }

    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    double allocated_estimate_latency = 0;
    bool can_schedule = true;
    std::vector<Sequence*> prefill_sequences;
    std::vector<size_t> prefill_sequences_budget;
    prefill_sequences.reserve(request->sequences().size());
    prefill_sequences_budget.reserve(request->sequences().size());

    for (auto& prefill_sequence : request->sequences()) {
      if (prefill_sequence->finished()) {
        continue;
      }

      size_t num_tokens = compute_prefill_tokens(
          prefill_sequence.get(),
          budget.remaining_token_budget - allocated_tokens,
          state);

      if (budget.remaining_token_budget < allocated_tokens + num_tokens ||
          budget.remaining_seq_budget < allocated_seqs + 1) {
        can_schedule = false;
        budget_exhausted = true;
        break;
      }

      // Latency check.
      double seq_estimate_latency = 0;
      if (options_.enable_latency_aware_schedule()) {
        size_t kv_cache_tokens_num =
            prefill_sequence->kv_state().kv_cache_tokens_num();
        seq_estimate_latency = state.profile_manager->predict_step_time(
            num_tokens + kv_cache_tokens_num, kv_cache_tokens_num, false);
        if (budget.estimate_latency + allocated_estimate_latency +
                seq_estimate_latency >
            budget.latency_budget) {
          can_schedule = false;
          budget_exhausted = true;
          break;
        }
      }

      size_t actual_tokens = 0;
      if (!allocate_for_prefill(
              prefill_sequence.get(), num_tokens, &actual_tokens, state)) {
        can_schedule = false;
        state.kv_cache_manager->deallocate(prefill_sequence.get());
        blocks_exhausted = true;
        break;
      }

      prefill_sequences_budget.emplace_back(actual_tokens);
      prefill_sequences.emplace_back(prefill_sequence.get());
      allocated_tokens += actual_tokens;
      allocated_seqs += 1;
      allocated_estimate_latency += seq_estimate_latency;
    }

    if (!can_schedule) {
      for (auto& seq : prefill_sequences) {
        state.kv_cache_manager->deallocate(seq);
      }
      break;
    }

    budget.remaining_token_budget -= allocated_tokens;
    budget.remaining_seq_budget -= allocated_seqs;
    budget.estimate_latency += allocated_estimate_latency;
    queue->pop_top();
    state.running_requests.emplace_back(request);
    request->record_num_prefix_cache_tokens();
    state.running_sequences.insert(state.running_sequences.end(),
                                   prefill_sequences.begin(),
                                   prefill_sequences.end());
    state.running_sequences_budgets.insert(
        state.running_sequences_budgets.end(),
        prefill_sequences_budget.begin(),
        prefill_sequences_budget.end());
    cache_in_batch_prefix(prefill_sequences, prefill_sequences_budget, state);
  }

  // Handle unschedulable head request.
  handle_unschedulable_head(
      queue, state, finished, budget_exhausted, blocks_exhausted);
}

size_t SchedulerPolicy::compute_prefill_tokens(Sequence* seq,
                                               size_t remaining_budget,
                                               const SchedulerState& state) {
  if (!batch_mode_.enable_chunked_prefill) {
    // Full prefill: compute all remaining tokens.
    size_t num_tokens = seq->num_need_compute_tokens();
    // CP alignment for full prefill (skip when model handles CP partition).
    const int32_t worker_cp_size =
        Platform::uses_model_cp_partition() ? 1 : state.options.cp_size();
    if (worker_cp_size > 1 && seq->is_prefill_stage() && num_tokens > 0) {
      const size_t alignment = static_cast<size_t>(worker_cp_size) * 2;
      num_tokens = ((num_tokens + alignment - 1) / alignment) * alignment;
    }
    return num_tokens;
  }

  // Chunked prefill: compute min(remaining, max_chunk, budget).
  const size_t max_tokens_per_chunk =
      options_.max_tokens_per_chunk_for_prefill();
  size_t num_tokens = seq->num_tokens();
  size_t assume_max = std::min(max_tokens_per_chunk, remaining_budget);
  num_tokens = std::min(assume_max, num_tokens);

  // CP-aware chunk alignment.
  const size_t kv_cache_tokens_num = seq->kv_state().kv_cache_tokens_num();
  const size_t remaining_in_seq = seq->num_tokens() > kv_cache_tokens_num
                                      ? seq->num_tokens() - kv_cache_tokens_num
                                      : 0;
  const int32_t kv_split_for_align =
      ::xllm::ParallelConfig::get_instance().kv_split_size_effective();
  const int32_t worker_cp_size =
      Platform::uses_model_cp_partition() ? 1 : options_.cp_size();
  num_tokens = maybe_align_cp_chunk_tokens(num_tokens,
                                           worker_cp_size,
                                           kv_split_for_align,
                                           state.kv_cache_manager->block_size(),
                                           remaining_in_seq);

  return num_tokens;
}

bool SchedulerPolicy::allocate_for_prefill(Sequence* seq,
                                           size_t token_budget,
                                           size_t* actual_tokens,
                                           SchedulerState& state,
                                           bool skip_shared) {
  if (!batch_mode_.enable_chunked_prefill) {
    // Full prefill: match prefix cache first.
    if (!skip_shared) {
      allocate_shared_blocks_for(seq, state);
    }
    // Full prefill: allocate for full prompt.
    *actual_tokens = seq->num_need_compute_tokens();
    // CP alignment (skip when model handles CP partition internally).
    const int32_t worker_cp_size =
        Platform::uses_model_cp_partition() ? 1 : state.options.cp_size();
    if (worker_cp_size > 1 && seq->is_prefill_stage() && *actual_tokens > 0) {
      const size_t alignment = static_cast<size_t>(worker_cp_size) * 2;
      *actual_tokens =
          ((*actual_tokens + alignment - 1) / alignment) * alignment;
    }
    const size_t target = seq->kv_cache_tokens_num() + *actual_tokens;
    return state.kv_cache_manager->allocate(seq, target);
  }

  // Chunked prefill: allocate shared blocks first (skip during redistribution
  // to avoid matching a sequence's own in-batch-published blocks).
  if (!skip_shared) {
    allocate_shared_blocks_for(seq, state);
  }

  const size_t kv_cache_tokens_num = seq->kv_cache_tokens_num();
  size_t max_handle_num_tokens =
      std::min(kv_cache_tokens_num + token_budget, seq->num_tokens());

  // Linear-state block alignment: for models with linear attention layers +
  // prefix cache, chunk boundaries must align to block_size so linear-state
  // checkpoints land at recoverable positions.
  if (state.has_linear_attention_layers && state.enable_prefix_cache &&
      seq->is_prefill_stage()) {
    const size_t block_size =
        static_cast<size_t>(state.kv_cache_manager->block_size());
    const size_t aligned = (max_handle_num_tokens / block_size) * block_size;
    if (aligned <= kv_cache_tokens_num) {
      if (max_handle_num_tokens == seq->num_tokens()) {
        // Final chunk: allow unaligned to complete the sequence.
      } else {
        *actual_tokens = 0;
        return false;
      }
    } else {
      max_handle_num_tokens = aligned;
    }
  }

  CHECK_GT(max_handle_num_tokens, kv_cache_tokens_num);
  *actual_tokens = max_handle_num_tokens - kv_cache_tokens_num;
  return state.kv_cache_manager->allocate(seq, max_handle_num_tokens);
}

void SchedulerPolicy::allocate_shared_blocks_for(Sequence* seq,
                                                 SchedulerState& state) {
  if (seq->kv_state().num_blocks(BlockType::KV) == 0) {
    state.kv_cache_manager->allocate_shared(seq);
    return;
  }
  if (seq->is_chunked_prefill_stage()) {
    if (state.has_linear_attention_layers && state.enable_prefix_cache) {
      // Linear-state prefix cache can only resume at saved state checkpoints.
      // Re-match at every chunk boundary.
      state.kv_cache_manager->allocate_shared(seq);
      return;
    }
    const size_t max_tokens_per_chunk =
        std::max(options_.max_tokens_per_chunk_for_prefill(), 64);
    size_t total_chunked_size =
        (seq->num_tokens() + max_tokens_per_chunk - 1) / max_tokens_per_chunk;
    if (total_chunked_size <
        ::xllm::SchedulerConfig::get_instance().chunked_match_frequency()) {
      state.kv_cache_manager->allocate_shared(seq);
      return;
    }
    size_t prefix_cache_interval =
        (total_chunked_size +
         ::xllm::SchedulerConfig::get_instance().chunked_match_frequency() -
         1) /
        ::xllm::SchedulerConfig::get_instance().chunked_match_frequency();
    size_t cur_chunked_index =
        seq->kv_state().kv_cache_tokens_num() / max_tokens_per_chunk;
    if (cur_chunked_index % prefix_cache_interval == 0) {
      state.kv_cache_manager->allocate_shared(seq);
    }
  }
}

// =============================================================================
// Decode scheduling
// =============================================================================

void SchedulerPolicy::schedule_decode_from_queue(RequestPriorityQueue* queue,
                                                 SchedulerState& state,
                                                 ScheduleBudget& budget) {
  if (queue == nullptr || queue->empty()) {
    return;
  }

  while (!queue->empty() &&
         budget.remaining_token_budget >
             static_cast<size_t>(state.min_speculative_tokens_required) &&
         budget.latency_budget > budget.estimate_latency &&
         budget.remaining_seq_budget > 0) {
    std::shared_ptr<Request> request = queue->top();

    const size_t num_sequences = request->sequences().size();
    std::vector<Sequence*> candidate_sequences;
    std::vector<size_t> candidate_token_budgets;
    candidate_sequences.reserve(num_sequences);
    candidate_token_budgets.reserve(num_sequences);

    bool has_enough_budget = true;
    bool has_enough_blocks = true;
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    double allocated_estimate_latency = 0;

    if (request->check_beam_search()) {
      // Beam search path.
      std::vector<Sequence*> active_sequences;
      active_sequences.reserve(num_sequences);
      for (auto& seq : request->sequences()) {
        if (!seq->finished()) {
          active_sequences.emplace_back(seq.get());
        }
      }
      if (active_sequences.empty()) {
        queue->pop_top();
        continue;
      }

      const size_t decode_step_tokens =
          state.min_speculative_tokens_required + 1;
      if (decode_step_tokens * active_sequences.size() >
              budget.remaining_token_budget ||
          active_sequences.size() > budget.remaining_seq_budget) {
        has_enough_budget = false;
      }

      if (has_enough_budget && options_.enable_latency_aware_schedule() &&
          !(options_.instance_role().has_value() &&
            options_.instance_role().value() == InstanceRole::PREFILL)) {
        for (auto* sequence : active_sequences) {
          const double seq_latency =
              state.profile_manager->predict_step_time(sequence, false);
          if (budget.estimate_latency + allocated_estimate_latency +
                  seq_latency >
              budget.latency_budget) {
            has_enough_budget = false;
            break;
          }
          allocated_estimate_latency += seq_latency;
        }
      }
      allocated_estimate_latency = 0.0;

      if (has_enough_budget) {
        const size_t block_size = state.kv_cache_manager->block_size();
        size_t needed_blocks = 0;
        for (auto* sequence : active_sequences) {
          const size_t updated_num_tokens =
              sequence->num_tokens() + state.min_speculative_tokens_required;
          needed_blocks += estimate_decode_extra_blocks(
              sequence, updated_num_tokens, block_size);
        }
        const int32_t dp_rank = active_sequences.front()->dp_rank();
        const size_t free_blocks =
            get_sequence_free_blocks_for_rank(state.kv_cache_manager, dp_rank);
        if (needed_blocks > free_blocks) {
          has_enough_blocks = false;
        }
      }

      if (has_enough_budget && has_enough_blocks) {
        bool allocate_failed = false;
        for (auto* sequence : active_sequences) {
          const size_t updated_num_tokens =
              sequence->num_tokens() + state.min_speculative_tokens_required;
          if (!state.kv_cache_manager->allocate(sequence, updated_num_tokens)) {
            allocate_failed = true;
            break;
          }
          if (sequence->if_cache_block_for_prefill()) {
            state.kv_cache_manager->cache(sequence);
          }
          candidate_sequences.emplace_back(sequence);
          candidate_token_budgets.emplace_back(decode_step_tokens);
          allocated_tokens += decode_step_tokens;
          allocated_seqs += 1;
        }

        if (allocate_failed) {
          LOG(ERROR) << "Beam strict scheduling allocation failed. "
                     << "request_id=" << request->request_id()
                     << ", beam=" << request->check_beam_search();
          state.kv_cache_manager->deallocate(request.get());
          queue->pop_top();
          request->set_preempted();
          state.prefill_queue.push(request);
          continue;
        }

        if (options_.enable_latency_aware_schedule() &&
            !(options_.instance_role().has_value() &&
              options_.instance_role().value() == InstanceRole::PREFILL)) {
          for (auto* sequence : candidate_sequences) {
            allocated_estimate_latency +=
                state.profile_manager->predict_step_time(sequence, false);
          }
        }
      }
    } else {
      // Non-beam-search decode path.
      for (auto& sequence : request->sequences()) {
        if (sequence->finished()) {
          continue;
        }
        double seq_estimate_latency = 0;
        if (options_.enable_latency_aware_schedule() &&
            !(options_.instance_role().has_value() &&
              options_.instance_role().value() == InstanceRole::PREFILL)) {
          seq_estimate_latency =
              state.profile_manager->predict_step_time(sequence.get(), false);
          if (budget.estimate_latency + allocated_estimate_latency +
                  seq_estimate_latency >
              budget.latency_budget) {
            has_enough_budget = false;
            break;
          }
        }
        if (allocated_tokens + state.min_speculative_tokens_required >=
                budget.remaining_token_budget ||
            allocated_seqs >= budget.remaining_seq_budget) {
          has_enough_budget = false;
          break;
        }
        size_t updated_num_tokens =
            sequence->num_tokens() + state.min_speculative_tokens_required;
        if (!state.kv_cache_manager->allocate(sequence.get(),
                                              updated_num_tokens)) {
          has_enough_blocks = false;
          break;
        }
        if (sequence->if_cache_block_for_prefill()) {
          state.kv_cache_manager->cache(sequence.get());
        }
        allocated_tokens += state.min_speculative_tokens_required + 1;
        allocated_seqs += 1;
        allocated_estimate_latency += seq_estimate_latency;
        candidate_sequences.emplace_back(sequence.get());
        candidate_token_budgets.emplace_back(
            state.min_speculative_tokens_required + 1);
      }
    }

    if (has_enough_budget && has_enough_blocks) {
      queue->pop_top();
      state.running_requests.emplace_back(request);
      state.running_sequences.insert(state.running_sequences.end(),
                                     candidate_sequences.begin(),
                                     candidate_sequences.end());
      state.running_sequences_budgets.insert(
          state.running_sequences_budgets.end(),
          candidate_token_budgets.begin(),
          candidate_token_budgets.end());
      budget.remaining_token_budget -= allocated_tokens;
      budget.remaining_seq_budget -= allocated_seqs;
      budget.estimate_latency += allocated_estimate_latency;
      continue;
    }

    // Budget exhausted: partial schedule or fail.
    if (!has_enough_budget) {
      handle_abnormal_decode_request(queue,
                                     state,
                                     budget,
                                     request,
                                     candidate_sequences,
                                     candidate_token_budgets,
                                     allocated_tokens,
                                     allocated_seqs,
                                     allocated_estimate_latency,
                                     /*budget_exhausted=*/true);
      break;
    }

    // Blocks exhausted: first try preempting from chunk_queue (has KV blocks
    // to free), then from the decode queue's lowest priority.
    if (!has_enough_blocks && !state.chunk_queue.empty()) {
      std::shared_ptr<Request> request_to_preempt = state.chunk_queue.back();
      ++budget.num_preempted_requests;
      state.kv_cache_manager->deallocate(request_to_preempt.get());
      state.chunk_queue.pop_back();
      request_to_preempt->set_preempted();
      state.prefill_queue.push(request_to_preempt);
      continue;
    }
    if (!has_enough_blocks && queue->size() > 1) {
      std::shared_ptr<Request> request_to_preempt = queue->back();
      if (request_to_preempt.get() != request.get()) {
        ++budget.num_preempted_requests;
        state.kv_cache_manager->deallocate(request_to_preempt.get());
        queue->pop_back();
        request_to_preempt->set_preempted();
        state.prefill_queue.push(request_to_preempt);
        continue;
      }
    }

    // Cannot preempt or single request left.
    handle_abnormal_decode_request(queue,
                                   state,
                                   budget,
                                   request,
                                   candidate_sequences,
                                   candidate_token_budgets,
                                   allocated_tokens,
                                   allocated_seqs,
                                   allocated_estimate_latency,
                                   /*budget_exhausted=*/false);
    break;
  }
}

// =============================================================================
// Preemption and error handling
// =============================================================================

void SchedulerPolicy::handle_abnormal_decode_request(
    RequestPriorityQueue* queue,
    SchedulerState& state,
    ScheduleBudget& budget,
    std::shared_ptr<Request>& request,
    const std::vector<Sequence*>& candidate_sequences,
    const std::vector<size_t>& candidate_token_budgets,
    size_t allocated_tokens,
    size_t allocated_seqs,
    double allocated_estimate_latency,
    bool budget_exhausted) {
  if (!candidate_sequences.empty() && !request->check_beam_search()) {
    // Partial schedule: commit the sequences we already allocated.
    queue->pop_top();
    state.running_requests.emplace_back(request);
    state.running_sequences.insert(state.running_sequences.end(),
                                   candidate_sequences.begin(),
                                   candidate_sequences.end());
    state.running_sequences_budgets.insert(
        state.running_sequences_budgets.end(),
        candidate_token_budgets.begin(),
        candidate_token_budgets.end());
    budget.remaining_token_budget -= allocated_tokens;
    budget.remaining_seq_budget -= allocated_seqs;
    budget.estimate_latency += allocated_estimate_latency;
  } else if (candidate_sequences.empty() && state.running_sequences.empty()) {
    // Nothing scheduled at all -- request is unschedulable.
    queue->pop_top();
    clear_mtp_bootstrap(request.get(), state);
    state.kv_cache_manager->deallocate(request.get());
    if (budget_exhausted) {
      state.response_processor->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough budget to schedule a single sequence."});
    } else {
      state.response_processor->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough memory to schedule a single sequence."});
    }
  }
}

void SchedulerPolicy::handle_unschedulable_head(
    RequestPriorityQueue* queue,
    SchedulerState& state,
    std::vector<std::shared_ptr<Request>>& finished,
    bool budget_exhausted,
    bool blocks_exhausted) {
  if (state.running_sequences.empty() && !queue->empty() &&
      state.decode_queue.empty() && state.chunk_queue.empty()) {
    std::shared_ptr<Request> request(queue->top());
    queue->pop_top();
    clear_mtp_bootstrap(request.get(), state);
    state.kv_cache_manager->deallocate(request.get());
    if (blocks_exhausted) {
      LOG(ERROR) << "Request prompt is too long, no enough memory to schedule "
                    "a single sequence.";
      state.response_processor->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough memory to schedule single sequence"});
    } else if (budget_exhausted) {
      LOG(ERROR) << "Request prompt is too long, no enough budget to schedule "
                    "a single sequence. Please set a larger budget.";
      state.response_processor->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough budget to schedule single sequence."});
    } else {
      LOG(FATAL) << "Unexpected error: blocks and budget are enough but can "
                    "not schedule.";
    }
  }
}

// =============================================================================
// Metrics reporting
// =============================================================================

void SchedulerPolicy::report_metrics(const SchedulerState& state,
                                     double elapsed_seconds,
                                     size_t num_preempted_requests) {
  GAUGE_SET(num_running_requests, state.running_requests.size());
  GAUGE_SET(num_waiting_requests, state.prefill_queue.size());
  GAUGE_SET(num_preempted_requests, num_preempted_requests);
  GAUGE_SET(num_running_sequences, state.running_sequences.size());
  GAUGE_SET(kv_cache_utilization_perc,
            state.kv_cache_manager->kv_cache_utilization());
  GAUGE_SET(num_blocks_in_prefix_cache,
            util::min(state.kv_cache_manager->num_blocks_in_prefix_cache()));
  GAUGE_SET(num_free_blocks,
            util::max(state.kv_cache_manager->num_free_blocks()));
  GAUGE_SET(num_used_blocks,
            util::min(state.kv_cache_manager->num_used_blocks()));
}

// =============================================================================
// Helpers
// =============================================================================

void SchedulerPolicy::handle_running_requests(std::shared_ptr<Request> request,
                                              SchedulerState& state) {
  if (request->finished() || request->cancelled()) {
    LOG(FATAL) << "Unknown error, finished/cancelled request should have been "
                  "handled before. request_id is "
               << request->request_id();
  }

  if (request->expand_sequences()) {
    state.kv_cache_manager->cache(request->sequences()[0].get());
  }

  for (auto& sequence : request->sequences()) {
    if (sequence->finished()) {
      state.kv_cache_manager->deallocate(sequence.get());
    }
  }
}

void SchedulerPolicy::cache_in_batch_prefix(
    const std::vector<Sequence*>& sequences,
    const std::vector<size_t>& token_budgets,
    SchedulerState& state) {
  if (!state.enable_prefix_cache || sequences.empty()) {
    return;
  }
  bool enable_in_batch =
      ::xllm::KVCacheConfig::get_instance().enable_in_batch_prefix_cache();
  if (!enable_in_batch) {
    return;
  }
  CHECK_EQ(sequences.size(), token_budgets.size());
  for (size_t i = 0; i < sequences.size(); ++i) {
    Sequence* sequence = sequences[i];
    if (sequence == nullptr || !sequence->is_prefill_stage()) {
      continue;
    }
    const size_t max_handle_num_tokens =
        sequence->kv_state().kv_cache_tokens_num() + token_budgets[i];
    state.kv_cache_manager->cache(sequence, max_handle_num_tokens);
  }
}

void SchedulerPolicy::clear_mtp_bootstrap(Request* request,
                                          const SchedulerState& state) {
  if (!state.options.enable_disagg_pd() ||
      state.options.num_speculative_tokens() <= 0 || request == nullptr ||
      request->sequences().empty()) {
    return;
  }
  Sequence* sequence = request->sequences()[0].get();
  if (sequence == nullptr) {
    return;
  }
  sequence->clear_mtp_bootstrap_embedding();
}

// =============================================================================
// Factory
// =============================================================================

std::unique_ptr<SchedulerPolicy> create_scheduler_policy(
    const BatchMode& mode,
    const ContinuousScheduler::Options& options) {
  if (mode.enable_mix_batch && mode.priority_strategy == "multi_slo_and_prio") {
    return std::make_unique<UnifiedPolicy>(mode, options);
  } else if (mode.enable_mix_batch) {
    return std::make_unique<DecodeFirstPolicy>(mode, options);
  } else {
    return std::make_unique<PrefillFirstPolicy>(mode, options);
  }
}

}  // namespace xllm
