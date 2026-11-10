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

#include "core/runtime/schedule_overlap_state.h"

#include <glog/logging.h>

#include <algorithm>
#include <iterator>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace xllm {
namespace {

bool has_request_overlap(const std::vector<std::string>& current_request_ids,
                         const std::vector<std::string>& prior_request_ids) {
  if (current_request_ids.empty() || prior_request_ids.empty()) {
    return true;
  }
  const auto first_current = std::find_if(
      current_request_ids.begin(),
      current_request_ids.end(),
      [](const std::string& request_id) { return !request_id.empty(); });
  if (first_current == current_request_ids.end()) {
    return false;
  }
  // Stable decode batches usually match before a lookup table is needed.
  if (std::find(prior_request_ids.begin(),
                prior_request_ids.end(),
                *first_current) != prior_request_ids.end()) {
    return true;
  }

  std::unordered_set<std::string_view> prior_requests;
  prior_requests.reserve(prior_request_ids.size());
  for (const std::string& request_id : prior_request_ids) {
    if (!request_id.empty()) {
      prior_requests.emplace(request_id);
    }
  }
  return std::any_of(std::next(first_current),
                     current_request_ids.end(),
                     [&prior_requests](const std::string& request_id) {
                       return !request_id.empty() &&
                              prior_requests.contains(request_id);
                     });
}

}  // namespace

ScheduleOverlapState::ScheduleOverlapState(bool publish_to_host,
                                           bool retain_tokenless_output,
                                           bool retire_unconsumed_inputs)
    : publish_to_host_(publish_to_host),
      retain_tokenless_output_(retain_tokenless_output),
      retire_unconsumed_inputs_(publish_to_host || retire_unconsumed_inputs) {}

ScheduleOverlapState::~ScheduleOverlapState() {
  if (!retire_unconsumed_inputs_) {
    return;
  }
  retire_deferred_inputs(/*wait=*/true);
  if (prior_output_.ready_event != nullptr &&
      !prior_output_.retained_inputs.empty()) {
    CHECK(prior_output_.ready_event->synchronize())
        << "failed to retire final asynchronous overlap input";
  }
}

bool ScheduleOverlapState::record(
    std::optional<ForwardOutput> output,
    std::vector<std::string> request_ids,
    std::vector<std::string> sample_sequence_ids) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (publish_to_host_) {
    condition_.wait(lock, [this] { return !pending_ || shutting_down_; });
  }
  if (shutting_down_) {
    return false;
  }
  // Wait for the consumer to finish copying before moving cached owners.
  // Tokenless publications may be filtered, leaving no host copy to retain
  // their asynchronous inputs. Keep device waits outside the publication
  // mutex so they cannot block shutdown or the host consumer.
  if (retire_unconsumed_inputs_) {
    lock.unlock();
    retire_prior_inputs();
    lock.lock();
    if (shutting_down_) {
      return false;
    }
  }
  update_prior_output(std::move(output),
                      std::move(request_ids),
                      std::move(sample_sequence_ids));
  if (publish_to_host_) {
    pending_ = true;
    condition_.notify_all();
  }
  return true;
}

ForwardOutput ScheduleOverlapState::consume() {
  CHECK(publish_to_host_) << "overlap output has no host consumer";
  std::unique_lock<std::mutex> lock(mutex_);
  condition_.wait(lock, [this] { return pending_ || shutting_down_; });

  ForwardOutput output =
      pending_ && has_output() ? prior_output_ : ForwardOutput();
  pending_ = false;
  condition_.notify_all();
  return output;
}

void ScheduleOverlapState::shutdown() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    shutting_down_ = true;
  }
  condition_.notify_all();
}

bool ScheduleOverlapState::can_reuse_tokens(
    const std::vector<std::string>& request_ids) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return has_tokens_ && has_request_overlap(request_ids, request_ids_);
}

void ScheduleOverlapState::retire_prior_inputs() {
  retire_deferred_inputs(/*wait=*/false);
  // Cache-only and filtered tokenless results have no host copy to retain
  // their device inputs. Defer owners before replacing or clearing the cache.
  if (prior_output_.retained_inputs.empty() ||
      prior_output_.ready_event == nullptr) {
    return;
  }
  if (prior_output_.ready_event->query()) {
    prior_output_.retained_inputs.clear();
    return;
  }
  // Only one deferred release can be pending; a full one waits for its device
  // work before being replaced.
  if (deferred_release_.has_value()) {
    CHECK(deferred_release_->ready_event->synchronize())
        << "failed to retire deferred overlap input";
  }
  deferred_release_ = DeferredRelease{prior_output_.ready_event,
                                      take_retained_inputs(prior_output_)};
}

void ScheduleOverlapState::retire_deferred_inputs(bool wait) {
  if (!deferred_release_.has_value()) {
    return;
  }
  const bool complete = wait ? deferred_release_->ready_event->synchronize()
                             : deferred_release_->ready_event->query();
  CHECK(complete || !wait) << "failed to retire deferred overlap input";
  if (complete) {
    deferred_release_.reset();
  }
}

void ScheduleOverlapState::update_prior_output(
    std::optional<ForwardOutput> output,
    std::vector<std::string> request_ids,
    std::vector<std::string> sample_sequence_ids) {
  prior_output_ =
      output.has_value() ? std::move(output.value()) : ForwardOutput();
  request_ids_ = std::move(request_ids);
  sample_sequence_ids_ = std::move(sample_sequence_ids);
  has_tokens_ = prior_output_.sample_output.next_tokens.defined();
  if (!output.has_value() || !has_output()) {
    request_ids_.clear();
    sample_sequence_ids_.clear();
  }
}

}  // namespace xllm
