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

#include <absl/time/time.h>
#include <folly/MPMCQueue.h>
#include <folly/futures/Future.h>

#include <limits>
#include <memory>
#include <queue>
#include <semaphore>

#include "async_response_processor.h"
#include "common/macros.h"
#include "common/types.h"
#include "framework/batch/batch.h"
#include "framework/batch/batch_factory.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "runtime/xservice_client.h"
#include "scheduler.h"
#include "scheduler/continuous_scheduler.h"
#include "util/threadpool.h"

namespace xllm {
class Engine;

// Return value structure for schedule_request
struct ScheduleResult {
  std::vector<Batch> batches;
  std::vector<std::shared_ptr<Request>> requests;
  std::vector<Sequence*> sequences;
};

class FixedStepsScheduler final : public ContinuousScheduler {
 public:
  FixedStepsScheduler(Engine* engine, const Options& options);
  ~FixedStepsScheduler() override = default;

  bool add_request(std::shared_ptr<Request>& request) override;

  // step the scheduler forward by one step
  // may get blocked if there are no requests to process
  void step(const absl::Duration& timeout) override;

 private:
  // Scheduler pipeline for different rec types
  class SchedulerPipeline {
   public:
    virtual ~SchedulerPipeline() = default;
    virtual std::vector<Batch> create_batches(FixedStepsScheduler& scheduler,
                                              BatchFactory* batch_factory) = 0;
    virtual bool requires_kv_cache() const = 0;
    // Allocate KV cache for sequence, implemented by each pipeline
    virtual bool allocate_kv_cache(KVCacheManager* kv_cache_manager,
                                   Sequence* sequence) = 0;
  };

  class LlmRecSchedulerPipeline final : public SchedulerPipeline {
   public:
    std::vector<Batch> create_batches(FixedStepsScheduler& scheduler,
                                      BatchFactory* batch_factory) override;
    bool requires_kv_cache() const override { return true; }
    bool allocate_kv_cache(KVCacheManager* kv_cache_manager,
                           Sequence* sequence) override;
  };

  class OneRecSchedulerPipeline final : public SchedulerPipeline {
   public:
    std::vector<Batch> create_batches(FixedStepsScheduler& scheduler,
                                      BatchFactory* batch_factory) override;
    bool requires_kv_cache() const override { return false; }
    bool allocate_kv_cache(KVCacheManager* /*kv_cache_manager*/,
                           Sequence* /*sequence*/) override {
      return true;
    }
  };

  class OneRecXAttentionSchedulerPipeline final : public SchedulerPipeline {
   public:
    std::vector<Batch> create_batches(FixedStepsScheduler& scheduler,
                                      BatchFactory* batch_factory) override;
    bool requires_kv_cache() const override { return true; }
    bool allocate_kv_cache(KVCacheManager* kv_cache_manager,
                           Sequence* sequence) override;
  };

  class RecMultiRoundSchedulerPipeline final : public SchedulerPipeline {
   public:
    std::vector<Batch> create_batches(FixedStepsScheduler& scheduler,
                                      BatchFactory* batch_factory) override;
    bool requires_kv_cache() const override { return false; }
    bool allocate_kv_cache(KVCacheManager* /*kv_cache_manager*/,
                           Sequence* /*sequence*/) override {
      return true;  // RecMultiRound mode does not need KV cache allocation
    }
  };

  // Factory method to create scheduler pipeline
  static std::unique_ptr<SchedulerPipeline> create_scheduler_pipeline(
      RecType rec_type,
      bool is_rec_multi_round);

  ScheduleResult schedule_request(const absl::Duration& timeout);

  // build a batch of requests from the priority queue
  std::vector<Batch> prepare_batch() override;

  void handle_prefill_requests(
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      std::vector<std::shared_ptr<Request>>& finished_requests);

  // Lazy-initialized pipeline
  std::unique_ptr<SchedulerPipeline> scheduler_pipeline_;

  // Holds a request consumed by the blocking wait in schedule_request() while
  // the queue was empty. prepare_batch() drains it first, through the same path
  // as request_queue_, so the blocking wait does not lose requests.
  std::shared_ptr<Request> prefetched_request_;

  // Scheduler thread pool for parallel execution of step()
  std::unique_ptr<ThreadPool> step_threadpool_;

  // Semaphore to control concurrent execution of step()
  std::counting_semaphore<10000> step_semaphore_;
};

}  // namespace xllm
