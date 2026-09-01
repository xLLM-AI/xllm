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

#include "dit_scheduler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <folly/MPMCQueue.h>
#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "common/metrics.h"
#include "distributed_runtime/dit_engine.h"
#include "framework/config/service_config.h"
#include "framework/request/dit_request.h"
#include "util/blocking_counter.h"
#include "util/tensor_helper.h"
#include "util/utils.h"

namespace xllm {

namespace {
constexpr size_t kRequestQueueSize = 100;

bool image_batch_signature_matches(const DiTInputParams& lhs,
                                   const DiTInputParams& rhs) {
  if (!tensor_batch_signature_matches(lhs.image, rhs.image)) {
    return false;
  }
  if (lhs.images.size() != rhs.images.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.images.size(); ++i) {
    if (!tensor_batch_signature_matches(lhs.images[i], rhs.images[i])) {
      return false;
    }
  }
  return true;
}

bool stacked_tensor_inputs_match(const DiTInputParams& lhs,
                                 const DiTInputParams& rhs) {
  return tensor_batch_signature_matches(lhs.prompt_embed, rhs.prompt_embed) &&
         tensor_batch_signature_matches(lhs.pooled_prompt_embed,
                                        rhs.pooled_prompt_embed) &&
         tensor_batch_signature_matches(lhs.negative_prompt_embed,
                                        rhs.negative_prompt_embed) &&
         tensor_batch_signature_matches(lhs.negative_pooled_prompt_embed,
                                        rhs.negative_pooled_prompt_embed) &&
         tensor_batch_signature_matches(lhs.latent, rhs.latent) &&
         tensor_batch_signature_matches(lhs.mask_image, rhs.mask_image) &&
         tensor_batch_signature_matches(lhs.control_image, rhs.control_image) &&
         tensor_batch_signature_matches(lhs.masked_image_latent,
                                        rhs.masked_image_latent) &&
         tensor_batch_signature_matches(lhs.last_image, rhs.last_image);
}

bool prompt_audio_allows_batching(const DiTInputParams& lhs,
                                  const DiTInputParams& rhs) {
  return !lhs.prompt_audio.defined() && !rhs.prompt_audio.defined() &&
         lhs.audio_prompt_text.empty() && rhs.audio_prompt_text.empty();
}

int32_t true_cfg_condition_type(const std::shared_ptr<DiTRequest>& request) {
  const auto& generation_params = request->state().generation_params();
  if (generation_params.true_cfg_scale <= 1.0) {
    return 0;
  }

  const auto& input_params = request->state().input_params();
  if (!input_params.negative_prompt.empty()) {
    return 1;
  }
  if (input_params.negative_prompt_embed.defined()) {
    return 2;
  }
  return 3;
}

bool is_compatible_dit_batch_request(
    const std::shared_ptr<DiTRequest>& batch_request,
    const std::shared_ptr<DiTRequest>& candidate_request) {
  const auto& batch_state = batch_request->state();
  const auto& candidate_state = candidate_request->state();
  if (candidate_state.generation_params() != batch_state.generation_params()) {
    return false;
  }
  if (true_cfg_condition_type(candidate_request) !=
      true_cfg_condition_type(batch_request)) {
    return false;
  }
  const auto& batch_input = batch_state.input_params();
  const auto& candidate_input = candidate_state.input_params();
  return image_batch_signature_matches(batch_input, candidate_input) &&
         stacked_tensor_inputs_match(batch_input, candidate_input) &&
         prompt_audio_allows_batching(batch_input, candidate_input);
}

}  // namespace

DiTAsyncResponseProcessor::DiTAsyncResponseProcessor(bool disable_log_stats)
    : response_threadpool_(
          ServiceConfig::get_instance().num_response_handling_threads(),
          /*cpu_binding=*/false,
          /*pool_name=*/"DiTAsyncResponseProcessor.response"),
      disable_log_stats_(disable_log_stats) {}

DiTAsyncResponseProcessor::~DiTAsyncResponseProcessor() { wait_completion(); }

void DiTAsyncResponseProcessor::wait_completion() {
  size_t thread_num = response_threadpool_.size();
  BlockingCounter counter(static_cast<int32_t>(thread_num));
  for (size_t i = 0; i < thread_num; ++i) {
    response_threadpool_.schedule_with_tid(
        [&counter]() mutable { counter.decrement_count(); }, i);
  }
  counter.wait();
}

void DiTAsyncResponseProcessor::process_completed_request(
    std::shared_ptr<DiTRequest> request) {
  const bool disable_log_stats = disable_log_stats_;
  response_threadpool_.schedule([disable_log_stats,
                                 request = std::move(request)]() {
    // Skip expensive generate_output() if the client has already disconnected.
    request->update_connection_status();
    if (request->cancelled()) {
      return;
    }

    double end_2_end_latency_seconds = request->elapsed_seconds();
    HISTOGRAM_OBSERVE(end_2_end_latency_milliseconds,
                      static_cast<int64_t>(end_2_end_latency_seconds * 1000.0));

    if (!disable_log_stats) {
      request->log_statistic(end_2_end_latency_seconds);
    }

    request->state().output_func()(request->generate_output());
  });
}

void DiTAsyncResponseProcessor::process_failed_request(
    std::shared_ptr<DiTRequest> request,
    Status status) {
  response_threadpool_.schedule(
      [request = std::move(request), status = std::move(status)]() mutable {
        DiTRequestOutput output;
        output.request_id = request->request_id();
        output.service_request_id = request->service_request_id();
        output.status = std::move(status);
        output.finished = request->finished();
        output.cancelled = request->cancelled();
        request->state().output_func()(output);
      });
}

DiTDynamicBatchScheduler::DiTDynamicBatchScheduler(Engine* engine,
                                                   const Options& options)
    : options_(options),
      engine_(dynamic_cast<DiTEngine*>(engine)),
      request_queue_(kRequestQueueSize) {
  CHECK(engine_ != nullptr);

  response_handler_ =
      std::make_unique<DiTAsyncResponseProcessor>(options_.disable_log_stats());
}

DiTDynamicBatchScheduler::~DiTDynamicBatchScheduler() {
  abort_all_requests();
  response_handler_->wait_completion();
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
  auto read_next_request = [this](std::shared_ptr<DiTRequest>& next_request) {
    if (!deferred_requests_.empty()) {
      next_request = std::move(deferred_requests_.front());
      deferred_requests_.pop_front();
      return true;
    }
    return request_queue_.read(next_request);
  };

  auto read_next_active_request =
      [&read_next_request](std::shared_ptr<DiTRequest>& next_request) {
        while (read_next_request(next_request)) {
          next_request->update_connection_status();
          if (!next_request->cancelled()) {
            return true;
          }
          next_request.reset();
        }
        return false;
      };

  if (read_next_active_request(request)) {
    std::shared_ptr<DiTRequest> batch_request = request;
    running_requests_.emplace_back(request);
    count = 1;

    while (count < options_.max_request_per_batch() &&
           read_next_active_request(request)) {
      if (!is_compatible_dit_batch_request(batch_request, request)) {
        deferred_requests_.emplace_back(std::move(request));
        break;
      }

      running_requests_.emplace_back(request);
      ++count;
    }
  }

  DiTBatch batches;
  for (size_t idx = 0; idx < running_requests_.size(); ++idx) {
    auto request = running_requests_[idx];
    batches.add(request);
  }

  GAUGE_SET(num_pending_requests,
            pending_requests_.load(std::memory_order_relaxed));
  GAUGE_SET(num_running_requests, running_requests_.size());
  GAUGE_SET(num_waiting_requests,
            request_queue_.size() + deferred_requests_.size());

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
  // Remove requests whose clients have already disconnected before
  // scheduling them into the (potentially multi-threaded) response pool.
  for (auto& request : running_requests_) {
    request->update_connection_status();
  }
  auto it = std::remove_if(running_requests_.begin(),
                           running_requests_.end(),
                           [](const auto& req) { return req->cancelled(); });
  running_requests_.erase(it, running_requests_.end());

  for (auto& request : running_requests_) {
    response_handler_->process_completed_request(request);
  }

  running_requests_.clear();
}

void DiTDynamicBatchScheduler::abort_all_requests() {
  auto abort_request = [this](std::shared_ptr<DiTRequest> request) {
    if (request == nullptr) {
      return;
    }
    request->set_cancel();
    response_handler_->process_failed_request(
        std::move(request),
        {StatusCode::CANCELLED, "Request aborted during scheduler shutdown"});
  };

  for (auto& request : running_requests_) {
    abort_request(std::move(request));
  }
  running_requests_.clear();

  for (auto& request : deferred_requests_) {
    abort_request(std::move(request));
  }
  deferred_requests_.clear();

  std::shared_ptr<DiTRequest> request;
  while (request_queue_.read(request)) {
    abort_request(std::move(request));
  }
}

}  // namespace xllm
