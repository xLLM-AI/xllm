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
#include <folly/futures/Future.h>

#include <limits>
#include <memory>
#include <queue>
#include <semaphore>

#include "core/common/macros.h"
#include "core/common/types.h"
#include "core/framework/batch/rec_batch_factory.h"
#include "core/framework/batch/rec_batch_group.h"
#include "core/framework/request/request.h"
#include "core/framework/request/sequence.h"
#include "core/runtime/xservice_client.h"
#include "core/scheduler/async_response_processor.h"
#include "core/scheduler/continuous_scheduler.h"
#include "core/scheduler/scheduler.h"
#include "core/util/threadpool.h"

namespace xllm {
class Engine;

// Return value structure for schedule_request
struct ScheduleResult {
  RecBatchGroup batches;
  std::vector<std::shared_ptr<Request>> requests;
  std::vector<Sequence*> sequences;
};

class FixedStepsScheduler : public ContinuousScheduler {
 public:
  FixedStepsScheduler(Engine* engine, const Options& options);
  ~FixedStepsScheduler() override = default;

  // step the scheduler forward by one step
  // may get blocked if there are no requests to process
  void step(const absl::Duration& timeout) override;

 protected:
  RecBatchGroup prepare_rec_batch();

 private:
  // Scheduler pipeline for different rec types
  class SchedulerPipeline {
   public:
    virtual ~SchedulerPipeline() = default;
    virtual BatchInputType input_type() const = 0;
    virtual bool requires_kv_cache() const = 0;
    // Allocate KV cache for sequence, implemented by each pipeline
    virtual bool allocate_kv_cache(KVCacheManager* kv_cache_manager,
                                   Sequence* sequence) = 0;
  };

  class LlmRecSchedulerPipeline final : public SchedulerPipeline {
   public:
    BatchInputType input_type() const override {
      return BatchInputType::SEQUENCE;
    }
    bool requires_kv_cache() const override { return true; }
    bool allocate_kv_cache(KVCacheManager* kv_cache_manager,
                           Sequence* sequence) override;
  };

  class OneRecSchedulerPipeline final : public SchedulerPipeline {
   public:
    BatchInputType input_type() const override {
      return BatchInputType::ONEREC;
    }
    bool requires_kv_cache() const override { return false; }
    bool allocate_kv_cache(KVCacheManager* /*kv_cache_manager*/,
                           Sequence* /*sequence*/) override {
      return true;
    }
  };

  class OneRecXAttentionSchedulerPipeline final : public SchedulerPipeline {
   public:
    BatchInputType input_type() const override {
      return BatchInputType::ONEREC_XATTENTION;
    }
    bool requires_kv_cache() const override { return true; }
    bool allocate_kv_cache(KVCacheManager* kv_cache_manager,
                           Sequence* sequence) override;
  };

  class RecMultiRoundSchedulerPipeline final : public SchedulerPipeline {
   public:
    BatchInputType input_type() const override {
      return BatchInputType::REC_MULTI_ROUND;
    }
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

  void handle_prefill_requests(
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      std::vector<std::shared_ptr<Request>>& finished_requests);

  // Lazy-initialized pipeline
  std::unique_ptr<SchedulerPipeline> scheduler_pipeline_;
  std::unique_ptr<RecBatchFactory> rec_batch_factory_;

  // Holds a request consumed by the blocking wait in schedule_request() while
  // the queue was empty. prepare_rec_batch() drains it first, through the same
  // path as request_queue_, so the blocking wait does not lose requests.
  std::shared_ptr<Request> prefetched_request_;

  // Scheduler thread pool for parallel execution of step()
  std::unique_ptr<ThreadPool> step_threadpool_;

  // Semaphore to control concurrent execution of step()
  std::counting_semaphore<10000> step_semaphore_;
};

}  // namespace xllm
