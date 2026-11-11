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

#include "scheduler/scheduler_metrics.h"

#include <absl/time/clock.h>
#include <glog/logging.h>

#include <cstdint>
#include <string>

#include "common/metrics.h"
#include "framework/block/kv_cache_manager.h"

namespace xllm {

namespace {

int64_t microseconds_to_milliseconds(int64_t microseconds) {
  return (microseconds + 500) / 1000;
}

}  // namespace

int64_t SchedulerMetrics::amortized_token_latency(int64_t latency,
                                                  size_t num_tokens) {
  const int64_t n = static_cast<int64_t>(num_tokens);
  return (latency + n / 2) / n;
}

SchedulerMetrics::SchedulerMetrics(KVCacheManager* kv_cache_manager,
                                   int32_t dp_size,
                                   int32_t num_speculative_tokens,
                                   bool collect_recent_latency)
    : kv_cache_manager_(kv_cache_manager),
      dp_size_(dp_size),
      num_speculative_tokens_(num_speculative_tokens),
      collect_recent_latency_(collect_recent_latency) {
  CHECK(kv_cache_manager_ != nullptr);
  CHECK_GT(dp_size_, 0);
}

void SchedulerMetrics::update(std::vector<Sequence*>& sequences) {
  if (sequences.empty()) {
    return;
  }
  update_token_latency_metrics(sequences);
  update_memory_metrics(sequences);
}

void SchedulerMetrics::update_token_latency_metrics(
    std::vector<Sequence*>& sequences) {
  std::unique_lock<std::mutex> lock(latency_metrics_mutex_, std::defer_lock);
  if (collect_recent_latency_) {
    lock.lock();
  }
  update_token_latency_metrics_impl(sequences);
}

void SchedulerMetrics::get_latency_metrics(std::vector<int64_t>& ttft,
                                           std::vector<int64_t>& tbt) {
  std::lock_guard<std::mutex> lock(latency_metrics_mutex_);
  ttft = std::move(recent_ttft_);
  tbt = std::move(recent_tbt_);
}

void SchedulerMetrics::update_token_latency_metrics_impl(
    const std::vector<Sequence*>& sequences) {
  const auto now = absl::Now();
  const bool speculative_metrics_enabled = num_speculative_tokens_ > 0;
  for (Sequence* sequence : sequences) {
    if (sequence->cancelled() || sequence->is_chunked_prefill_stage() ||
        sequence->last_token_handled()) {
      continue;
    }
    // Read the committed-token count before tbt_microseconds(), which resets
    // it. Overlap can advance KV state to decode before any real token
    // arrives; keep the latency clock until there is a committed token.
    const size_t committed_tokens = sequence->generated_tokens_since_latency();
    if (committed_tokens == 0) {
      continue;
    }
    const int64_t tbt_microseconds = sequence->tbt_microseconds(now);
    const int64_t tbt_milliseconds =
        microseconds_to_milliseconds(tbt_microseconds);
    if (sequence->is_first_token()) {
      HISTOGRAM_OBSERVE(time_to_first_token_latency_milliseconds,
                        tbt_milliseconds);
      sequence->set_time_to_first_token_latency_seconds(
          static_cast<double>(tbt_milliseconds) / 1000);
      if (collect_recent_latency_) {
        recent_ttft_.emplace_back(tbt_milliseconds);
      }
    } else {
      int64_t inter_token_latency_us = tbt_microseconds;
      if (collect_recent_latency_) {
        recent_tbt_.emplace_back(tbt_milliseconds);
      }
      if (speculative_metrics_enabled) {
        inter_token_latency_us = SchedulerMetrics::amortized_token_latency(
            tbt_microseconds, committed_tokens);
      }
      HISTOGRAM_OBSERVE(inter_token_latency_microseconds,
                        inter_token_latency_us);
      HISTOGRAM_OBSERVE(inter_token_latency_milliseconds,
                        microseconds_to_milliseconds(inter_token_latency_us));
    }
  }
}

std::vector<int64_t> SchedulerMetrics::get_num_occupied_slots(
    const std::vector<Sequence*>& sequences) const {
  std::vector<int64_t> num_occupied_slots(static_cast<size_t>(dp_size_));
  std::vector<int64_t> num_unfilled_blocks(static_cast<size_t>(dp_size_));
  const std::vector<size_t> num_used_blocks =
      kv_cache_manager_->num_used_blocks();
  const int32_t block_size = kv_cache_manager_->block_size();
  const size_t full_block_units =
      kv_cache_manager_->block_resource_units(BlockType::KV, 1);
  for (Sequence* sequence : sequences) {
    const int32_t dp_rank = sequence->dp_rank();
    const int32_t last_block_len =
        sequence->kv_state().kv_cache_tokens_num() % block_size;
    num_occupied_slots[static_cast<size_t>(dp_rank)] +=
        last_block_len * full_block_units;
    num_unfilled_blocks[static_cast<size_t>(dp_rank)] +=
        last_block_len > 0 ? full_block_units : 0;
  }
  for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
    num_occupied_slots[static_cast<size_t>(dp_rank)] +=
        (num_used_blocks[static_cast<size_t>(dp_rank)] -
         num_unfilled_blocks[static_cast<size_t>(dp_rank)]) *
        block_size;
  }
  return num_occupied_slots;
}

void SchedulerMetrics::update_memory_metrics(
    const std::vector<Sequence*>& sequences) {
  const std::vector<int64_t> num_occupied_slots =
      get_num_occupied_slots(sequences);
  for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
    const double num_total_slots =
        static_cast<double>(kv_cache_manager_->admission_capacity(dp_rank)) *
        kv_cache_manager_->block_size();
    const double occupied_slots_ratio =
        num_total_slots > 0
            ? static_cast<double>(
                  num_occupied_slots[static_cast<size_t>(dp_rank)]) /
                  num_total_slots
            : 0;
    const double active_kv_cache_size_in_kilobytes =
        occupied_slots_ratio * GAUGE_VALUE(total_kv_cache_size_in_kilobytes);
    MULTI_HISTOGRAM_OBSERVE(
        active_kv_cache_size_in_kilobytes,
        std::to_string(dp_rank),
        static_cast<int64_t>(active_kv_cache_size_in_kilobytes));
  }
}

}  // namespace xllm
