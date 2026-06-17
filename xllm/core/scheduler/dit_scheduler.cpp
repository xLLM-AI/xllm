/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "dit_scheduler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <folly/MPMCQueue.h>
#include <glog/logging.h>

#include <atomic>
#include <cstdint>
#include <memory>

#include "common/metrics.h"
#include "distributed_runtime/dit_engine.h"
#include "framework/request/dit_request.h"
#include "util/utils.h"

namespace xllm {

namespace {
constexpr size_t kRequestQueueSize = 100;
}  // namespace

void DiTAsyncResponseProcessor::process_completed_request(
    std::shared_ptr<DiTRequest> request) {
  response_threadpool_.schedule([request = std::move(request)]() {
    double end_2_end_latency_seconds = request->elapsed_seconds();
    HISTOGRAM_OBSERVE(end_2_end_latency_milliseconds,
                      static_cast<int64_t>(end_2_end_latency_seconds * 1000.0));

    request->log_statistic(end_2_end_latency_seconds);
    request->state().output_func()(request->generate_output());
  });
}

void DiTAsyncResponseProcessor::process_failed_request(
    std::shared_ptr<DiTRequest> request,
    Status status) {}

DiTDynamicBatchScheduler::DiTDynamicBatchScheduler(Engine* engine,
                                                   const Options& options)
    : options_(options),
      engine_(dynamic_cast<DiTEngine*>(engine)),
      request_queue_(kRequestQueueSize) {
  CHECK(engine_ != nullptr);

  response_handler_ = std::make_unique<DiTAsyncResponseProcessor>();
}

DiTDynamicBatchScheduler::~DiTDynamicBatchScheduler() {
  running_requests_.clear();
}

bool DiTDynamicBatchScheduler::add_request(
    std::shared_ptr<DiTRequest>& request) {
  CHECK(request != nullptr);

  if (request_queue_.write(request)) {
    return true;
  }

  LOG(WARNING) << " request queue is full, size is " << request_queue_.size();
  return false;
}

void DiTDynamicBatchScheduler::step(const absl::Duration& timeout) {
  // get a new batch of requests
  std::vector<DiTBatch> batches = schedule_request(timeout);
  bool all_empty =
      std::all_of(batches.begin(), batches.end(), [](const DiTBatch& batch) {
        return batch.empty();
      });

  if (all_empty) {
    return;
  }

  auto output = engine_->step(batches);

  // process request output in batch
  process_batch_output();
}

void DiTDynamicBatchScheduler::generate() {}

std::vector<DiTBatch> DiTDynamicBatchScheduler::prepare_batch() {
  Timer timer;

  int count = 0;
  std::shared_ptr<DiTRequest> request;
  while (request_queue_.read(request)) {
    running_requests_.emplace_back(request);

    if (++count == options_.max_request_per_batch()) break;
  }

  DiTBatch batches;
  for (size_t idx = 0; idx < running_requests_.size(); ++idx) {
    auto request = running_requests_[idx];
    batches.add(request);
  }

  GAUGE_SET(num_pending_requests,
            pending_requests_.load(std::memory_order_relaxed));
  GAUGE_SET(num_running_requests, running_requests_.size());
  GAUGE_SET(num_waiting_requests, request_queue_.size());

  return {batches};
}

std::vector<DiTBatch> DiTDynamicBatchScheduler::schedule_request(
    const absl::Duration& timeout) {
  const auto deadline = absl::Now() + timeout;
  std::vector<DiTBatch> batches;

  while (true) {
    batches = prepare_batch();
    bool all_empty =
        std::all_of(batches.begin(), batches.end(), [](const DiTBatch& batch) {
          return batch.empty();
        });

    if (!all_empty) {
      return batches;
    }

    const auto now = absl::Now();
    if (now > deadline) {
      break;
    }
    // wait for new requests to arrive
    constexpr uint64_t kStepSleepTimeMs = 10;
    const auto time_to_sleep =
        std::min(absl::Milliseconds(kStepSleepTimeMs), deadline - now);
    absl::SleepFor(time_to_sleep);
  }

  // return an empty batch
  return batches;
}

void DiTDynamicBatchScheduler::process_batch_output() {
  for (auto& request : running_requests_) {
    response_handler_->process_completed_request(request);
  }

  running_requests_.clear();
}

}  // namespace xllm
