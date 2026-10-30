/* Copyright 2025-2026 The xLLM Authors.

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

#include <absl/time/time.h>
#include <folly/MPMCQueue.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <vector>

#include "core/common/macros.h"
#include "core/common/types.h"
#include "core/distributed_runtime/rec_engine.h"
#include "core/framework/batch/rec_batch_factory.h"
#include "core/framework/batch/rec_batch_group.h"
#include "core/framework/block/kv_cache_manager.h"
#include "core/framework/request/request.h"
#include "core/framework/request/sequence.h"
#include "core/scheduler/dp_rank_selector.h"
#include "core/scheduler/request_output_dispatcher.h"
#include "core/scheduler/request_priority_queue.h"
#include "core/scheduler/scheduler.h"
#include "core/scheduler/scheduler_notifier.h"
#include "core/util/threadpool.h"

namespace xllm {

// Return value structure for schedule_request
struct ScheduleResult {
  RecBatchGroup batches;
  std::vector<std::shared_ptr<Request>> requests;
  std::vector<Sequence*> sequences;
};

class FixedStepsScheduler : public Scheduler {
 public:
  using Options = SchedulerOptions;

  FixedStepsScheduler(RecEngine* engine, const Options& options);

  ~FixedStepsScheduler() override;

  // step the scheduler forward by one step
  // may get blocked if there are no requests to process
  void step(const absl::Duration& timeout) override;
  void generate() override;

  bool add_request(std::shared_ptr<Request>& request) override;

  void incr_pending_requests(size_t count) override {
    pending_requests_.fetch_add(count, std::memory_order_relaxed);
  }

  void decr_pending_requests() override {
    const size_t old_value =
        pending_requests_.fetch_sub(1, std::memory_order_relaxed);
    CHECK_GT(old_value, 0) << "pending requests underflow";
  }

  size_t num_pending_requests() override {
    return pending_requests_.load(std::memory_order_relaxed);
  }

  bool has_pending_prefetch() const override {
    return prefetching_requests_.load(std::memory_order_relaxed) > 0;
  }

  uint32_t get_waiting_requests_num() const override {
    return static_cast<uint32_t>(
        prefill_queue_->size() +
        prefetching_requests_.load(std::memory_order_relaxed));
  }

  void get_latency_metrics(std::vector<int64_t>& /*ttft*/,
                           std::vector<int64_t>& /*tbt*/) override {}

  const InstanceInfo& get_instance_info() override { return instance_info_; }

 protected:
  RecBatchGroup prepare_rec_batch();

  std::vector<std::shared_ptr<Request>> running_requests_;

 private:
  // Scheduler pipeline for different rec types
  class SchedulerPipeline {
   public:
    virtual ~SchedulerPipeline() = default;
    virtual BatchInputType input_type() const = 0;
    virtual bool requires_kv_cache() const = 0;
    // Describe the reservation before the scheduler checks and allocates it.
    virtual std::optional<KVCacheManager::KVCacheReservation>
    kv_cache_reservation(Sequence* sequence) const = 0;
  };

  class LlmRecSchedulerPipeline final : public SchedulerPipeline {
   public:
    BatchInputType input_type() const override {
      return BatchInputType::SEQUENCE;
    }
    bool requires_kv_cache() const override { return true; }
    std::optional<KVCacheManager::KVCacheReservation> kv_cache_reservation(
        Sequence* sequence) const override;
  };

  class OneRecSchedulerPipeline final : public SchedulerPipeline {
   public:
    BatchInputType input_type() const override {
      return BatchInputType::ONEREC;
    }
    bool requires_kv_cache() const override { return false; }
    std::optional<KVCacheManager::KVCacheReservation> kv_cache_reservation(
        Sequence* /*sequence*/) const override {
      return std::nullopt;
    }
  };

  class OneRecXAttentionSchedulerPipeline final : public SchedulerPipeline {
   public:
    BatchInputType input_type() const override {
      return BatchInputType::ONEREC_XATTENTION;
    }
    bool requires_kv_cache() const override { return true; }
    std::optional<KVCacheManager::KVCacheReservation> kv_cache_reservation(
        Sequence* sequence) const override;
  };

  class RecMultiRoundSchedulerPipeline final : public SchedulerPipeline {
   public:
    BatchInputType input_type() const override {
      return BatchInputType::REC_MULTI_ROUND;
    }
    bool requires_kv_cache() const override { return false; }
    std::optional<KVCacheManager::KVCacheReservation> kv_cache_reservation(
        Sequence* /*sequence*/) const override {
      return std::nullopt;
    }
  };

  // Factory method to create scheduler pipeline
  static std::unique_ptr<SchedulerPipeline> create_scheduler_pipeline(
      RecType rec_type,
      bool is_rec_multi_round);

  static std::optional<KVCacheManager::KVCacheReservation>
  make_kv_cache_reservation(Sequence* sequence, size_t max_generated_tokens);

  ScheduleResult schedule_request(const absl::Duration& timeout);

  // Returns true when an oversized request was rejected and more local work
  // remains to be scheduled in this pass.
  bool handle_prefill_requests(
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      std::vector<std::shared_ptr<Request>>& finished_requests);

  void drain_prefetch_admissions();
  void drain_completed_prefetches();
  void drain_prefetch_pipeline();
  void apply_cancel_requests();

  const Options options_;

  // RecMaster owns the engine and destroys the scheduler first.
  RecEngine* engine_;
  KVCacheManager* kv_cache_manager_;
  DpRankSelector dp_rank_selector_;

  folly::MPMCQueue<std::shared_ptr<Request>> request_queue_;
  std::atomic<size_t> prefetching_requests_{0};
  std::mutex prefetch_admission_mutex_;
  std::deque<std::shared_ptr<Request>> prefetch_admissions_;
  std::deque<std::shared_ptr<Request>> completed_prefetches_;

  std::vector<Sequence*> running_sequences_;
  std::vector<size_t> running_sequences_budgets_;

  std::shared_ptr<CancelRequestQueue> cancel_request_queue_;
  std::unique_ptr<RequestOutputDispatcher> output_dispatcher_;

  bool enable_prefix_cache_ = false;
  std::atomic<size_t> pending_requests_{0};
  std::unique_ptr<RequestPriorityQueue> prefill_queue_;
  InstanceInfo instance_info_;

  // Lazy-initialized pipeline
  std::unique_ptr<SchedulerPipeline> scheduler_pipeline_;
  std::unique_ptr<RecBatchFactory> rec_batch_factory_;

  std::shared_ptr<SchedulerNotifier> notifier_;

  // Scheduler thread pool for parallel execution of step()
  std::unique_ptr<ThreadPool> step_threadpool_;

  // Semaphore to control concurrent execution of step()
  std::counting_semaphore<10000> step_semaphore_;
};

}  // namespace xllm
