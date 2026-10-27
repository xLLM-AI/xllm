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

#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/framework/batch/batch_group.h"
#include "core/framework/block/block_manager_pool.h"
#include "core/framework/model/model_args.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/runtime/forward_params.h"
#include "core/scheduler/continuous_scheduler.h"
#include "core/scheduler/scheduler_callbacks.h"
#include "scheduler/profile/profile_manager.h"
#include "scheduler_notifier_test_peer.h"

namespace xllm {

class FakeSchedulingTokenizer : public Tokenizer {
 public:
  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeSchedulingTokenizer>();
  }
};

class FakeSchedulingEngine {
 public:
  virtual ~FakeSchedulingEngine() = default;

  explicit FakeSchedulingEngine(
      std::unique_ptr<BlockManagerPool> block_manager = nullptr)
      : tokenizer_(std::make_unique<FakeSchedulingTokenizer>()),
        block_manager_(std::move(block_manager)) {}

  virtual ForwardOutput step(BatchGroup& /*batch*/) { return {}; }
  virtual std::vector<RawForwardOutput> collect_step_outputs(
      const BatchGroup& /*batch*/) {
    return {};
  }
  const Tokenizer* tokenizer() const { return tokenizer_.get(); }
  BlockManagerPool* block_manager_pool() const { return block_manager_.get(); }
  const ModelArgs& model_args() const { return model_args_; }

 protected:
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<BlockManagerPool> block_manager_;
  ModelArgs model_args_;
};

class TestableContinuousScheduler : public ContinuousScheduler {
 public:
  TestableContinuousScheduler(FakeSchedulingEngine* engine,
                              const Options& options)
      : ContinuousScheduler(engine->block_manager_pool(),
                            &engine->model_args(),
                            engine->tokenizer(),
                            std::move(make_scheduler_callbacks(engine)),
                            options,
                            /*model_residency_coordinator=*/nullptr,
                            /*distributed_worker_manager=*/nullptr) {}

  void drain_prefetch_for_test() {
    if (kv_cache_prefetcher_) {
      kv_cache_prefetcher_->drain();
    }
  }

  ProfileManager* get_profile_manager() { return profile_manager_.get(); }

  bool in_batch_prefix_cache_enabled() const {
    return enable_in_batch_prefix_cache_;
  }

  uint64_t wakeup_generation() const {
    return notifier_->snapshot().generation;
  }

  bool wait_for_wakeup(uint64_t observed_generation) {
    notifier_->wait_for(observed_generation, std::chrono::seconds(5));
    return notifier_->snapshot().generation != observed_generation;
  }

  bool wait_until_armed() const {
    return SchedulerNotifierTestPeer::wait_until_armed(*notifier_);
  }

  void stop_scheduling() { notifier_->request_stop(); }

  void enqueue_decode_restore(const std::shared_ptr<Request>& request) {
    decode_restore_waiting_.emplace_back(
        DecodeRestoreEntry{request, absl::Now()});
    incr_pending_requests(1);
  }

  void notify_after_next_prepare_batch(std::promise<void>& notification) {
    prepared_notification_ = &notification;
  }

  void process_batch_output_test() { process_batch_output(running_requests_); }

  std::vector<std::shared_ptr<Request>> get_running_requests() {
    return running_requests_;
  }

  std::vector<size_t> get_running_sequences_budgets() {
    return running_sequences_budgets_;
  }

  void reject_stream(const std::shared_ptr<Request>& request) {
    output_dispatcher_->process_stream_request(request);
    output_dispatcher_->wait_completion();
  }

  void reject_streams(const std::vector<std::shared_ptr<Request>>& requests) {
    std::vector<std::shared_ptr<Request>> request_copy = requests;
    output_dispatcher_->batch_process_stream_requests(request_copy);
    output_dispatcher_->wait_completion();
  }

  size_t scheduler_queue_size() { return request_queue_.size(); }

  void wait_for_responses() { output_dispatcher_->wait_completion(); }
  void seed_pending_batch(const std::shared_ptr<Request>& request) {
    pending_step_.emplace();
    pending_step_->requests.emplace_back(request);
    pending_step_->batches = BatchGroup(/*dp_size=*/1);
    pending_step_->batches.front().add(request->sequences().front().get());
  }

  void schedule_batch_for_test(std::vector<std::shared_ptr<Request>> requests) {
    scripted_scheduling_ = true;
    scripted_requests_ = std::move(requests);
  }

  BatchGroup prepare_batch() override {
    if (scripted_scheduling_) {
      running_requests_ = std::move(scripted_requests_);
      refresh_sequences_from_requests(running_requests_, running_sequences_);
      BatchGroup batch(/*dp_size=*/1);
      batch.front().add(running_sequences_);
      return batch;
    }
    BatchGroup batch = ContinuousScheduler::prepare_batch();
    std::promise<void>* notification =
        std::exchange(prepared_notification_, nullptr);
    if (notification != nullptr) {
      notification->set_value();
    }
    return batch;
  }

 private:
  bool scripted_scheduling_ = false;
  std::vector<std::shared_ptr<Request>> scripted_requests_;
  std::promise<void>* prepared_notification_ = nullptr;
};

inline ProfileManager make_profile_manager(FakeSchedulingEngine& engine,
                                           ProfileManager::Options options) {
  return ProfileManager(engine.block_manager_pool(),
                        &engine.model_args(),
                        options,
                        make_scheduler_callbacks(&engine));
}

}  // namespace xllm
