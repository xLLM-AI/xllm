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

#include <utility>

#include "core/runtime/json_object_output_rows.h"

namespace xllm {

ScheduleOverlapState::ScheduleOverlapState(bool publish_to_host,
                                           bool retain_tokenless_output,
                                           bool retire_unconsumed_inputs)
    : publish_to_host_(publish_to_host),
      retain_tokenless_output_(retain_tokenless_output),
      retire_unconsumed_inputs_(!publish_to_host && retire_unconsumed_inputs) {}

bool ScheduleOverlapState::record(
    std::optional<ForwardOutput> output,
    std::vector<std::string> request_ids,
    std::vector<std::string> sample_sequence_ids) {
  // Only the producer replaces prior_output_. Wait outside the publication
  // mutex so device work cannot block shutdown or the host consumer.
  if (retire_unconsumed_inputs_) {
    retire_prior_inputs();
  }
  std::unique_lock<std::mutex> lock(mutex_);
  if (publish_to_host_) {
    condition_.wait(lock, [this] { return !pending_ || shutting_down_; });
  }
  if (shutting_down_) {
    return false;
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

  ForwardOutput output;
  if (pending_ && (has_tokens_ || retain_tokenless_output_ ||
                   !prior_output_.json_object_errors.empty())) {
    output = prior_output_;
  }
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
  return has_tokens_ &&
         detail::has_request_id_overlap(request_ids, request_ids_);
}

void ScheduleOverlapState::retire_prior_inputs() const {
#if defined(USE_NPU)
  // Cache-only DP speculative results have no host consumer to wait for their
  // device work. Retire temporary input owners before replacing or clearing
  // the prior output; a device-stream wait alone cannot protect host owners.
  if (has_tokens_ && !prior_output_.retained_inputs.empty() &&
      prior_output_.ready_event != nullptr) {
    CHECK(prior_output_.ready_event->synchronize())
        << "failed to retire previous DP speculative input";
  }
#endif
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
  if (!output.has_value() || (!has_tokens_ && !retain_tokenless_output_ &&
                              prior_output_.json_object_errors.empty())) {
    request_ids_.clear();
    sample_sequence_ids_.clear();
  }
}

}  // namespace xllm
