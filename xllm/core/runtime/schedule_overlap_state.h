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

#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "core/runtime/forward_params.h"

namespace xllm {

// Owns the one-step device continuation and the one-slot host publication
// protocol used by schedule overlap. The producer thread updates the prior
// output, while the host consumer only clears the publication flag. A
// consumed result therefore remains available for the next device step.
class ScheduleOverlapState final {
 public:
  ScheduleOverlapState(bool publish_to_host,
                       bool retain_tokenless_output,
                       bool retire_unconsumed_inputs = false);
  ~ScheduleOverlapState() = default;

  ScheduleOverlapState(const ScheduleOverlapState&) = delete;
  ScheduleOverlapState& operator=(const ScheduleOverlapState&) = delete;

  // Called by the single producer to replace the device continuation and, for
  // host-visible ranks, publish it with one-slot backpressure. Retires cached
  // asynchronous inputs before replacement when no host consumer can do so.
  // Returns false after shutdown; the caller must drain device work before
  // releasing the rejected output.
  bool record(std::optional<ForwardOutput> output,
              std::vector<std::string> request_ids,
              std::vector<std::string> sample_sequence_ids);

  bool publishes_to_host() const { return publish_to_host_; }

  // Only valid for host-visible ranks. Waits for publication or shutdown and
  // consumes the publication flag. Returns a copy when relevant to the caller.
  // The cached prior output is retained for the next overlapped device step.
  ForwardOutput consume();

  // Wakes blocked producer and consumer calls. Cached device state is kept
  // intact so destruction can drain already submitted work safely.
  void shutdown();

  // These accessors are for the single producer thread that executes the next
  // overlapped step. They borrow state owned by this object.
  const ForwardOutput& prior_output() const { return prior_output_; }
  const std::vector<std::string>& request_ids() const { return request_ids_; }
  const std::vector<std::string>& sample_sequence_ids() const {
    return sample_sequence_ids_;
  }
  bool has_tokens() const { return has_tokens_; }

  // Returns whether the cached output can provide continuation tokens for the
  // current request set. The request matching and token availability are one
  // state decision, so evaluate them under one lock.
  bool can_reuse_tokens(const std::vector<std::string>& request_ids) const;

 private:
  void retire_prior_inputs() const;
  void update_prior_output(std::optional<ForwardOutput> output,
                           std::vector<std::string> request_ids,
                           std::vector<std::string> sample_sequence_ids);

  const bool publish_to_host_;
  const bool retain_tokenless_output_;
  const bool retire_unconsumed_inputs_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  ForwardOutput prior_output_;
  std::vector<std::string> request_ids_;
  std::vector<std::string> sample_sequence_ids_;
  bool has_tokens_ = false;
  bool pending_ = false;
  bool shutting_down_ = false;
};

}  // namespace xllm
