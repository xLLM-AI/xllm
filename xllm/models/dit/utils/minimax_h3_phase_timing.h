/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <glog/logging.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>

namespace xllm {

enum class H3TimingPhase : int8_t { LOAD, PREPARE, DENOISE, DECODE };

struct H3PhaseTimingMetadata {
  std::string profile;
  std::string request_id;
  int64_t global_rank = 0;
  int64_t world_size = 1;
  int64_t tp_size = 1;
  int64_t sp_size = 1;
  int64_t denoise_steps = 0;
  bool output_owner = false;
  bool rolling_load = false;
  bool correctness_diagnostics = false;
};

// Opt-in diagnostic timing. No synchronization, clocks or log output when off.
// The caller injects device synchronization; this class never adds collectives
// or synchronizes individual denoiser steps/blocks. Durations use local clocks;
// aggregate complete per-request rank sets offline instead of averaging ranks.
class H3PhaseTimer final {
 public:
  H3PhaseTimer(bool enabled,
               H3PhaseTimingMetadata metadata,
               std::function<void()> synchronize)
      : enabled_(enabled),
        metadata_(std::move(metadata)),
        synchronize_(std::move(synchronize)) {}

  void start(H3TimingPhase phase) {
    if (!enabled_) {
      return;
    }
    CHECK(!started_);
    CHECK(phase == H3TimingPhase::LOAD || phase == H3TimingPhase::PREPARE);
    CHECK(phase == H3TimingPhase::LOAD || !metadata_.request_id.empty());
    synchronize_();
    started_ = true;
    active_ = phase;
    boundary_ = std::chrono::steady_clock::now();
  }

  void advance(H3TimingPhase phase) {
    if (!enabled_) {
      return;
    }
    CHECK(started_ && !finished_);
    CHECK(
        (active_ == H3TimingPhase::PREPARE &&
         phase == H3TimingPhase::DENOISE) ||
        (active_ == H3TimingPhase::DENOISE && phase == H3TimingPhase::DECODE));
    synchronize_();
    const auto now = std::chrono::steady_clock::now();
    durations_[static_cast<size_t>(active_)] =
        std::chrono::duration<double, std::milli>(now - boundary_).count();
    boundary_ = now;
    active_ = phase;
  }

  nlohmann::ordered_json finish() {
    if (!enabled_) {
      return nullptr;
    }
    CHECK(started_ && !finished_);
    CHECK(active_ == H3TimingPhase::LOAD || active_ == H3TimingPhase::DECODE);
    synchronize_();
    durations_[static_cast<size_t>(active_)] =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - boundary_)
            .count();
    finished_ = true;
    const bool model_load = active_ == H3TimingPhase::LOAD;
    nlohmann::ordered_json result = {
        {"kind", model_load ? "h3_phase_model_load" : "h3_phase_request"},
        {"status", "success"},
        {"measurement_mode", "phase_boundary_device_sync"},
        {"scope", model_load ? "pipeline_load_model" : "pipeline_forward"},
        {"profile", metadata_.profile},
        {"global_rank", metadata_.global_rank},
        {"world_size", metadata_.world_size},
        {"tp_size", metadata_.tp_size},
        {"sp_size", metadata_.sp_size},
        {"output_owner", metadata_.output_owner},
        {"correctness_diagnostics", metadata_.correctness_diagnostics}};
    if (model_load) {
      result["load_ms"] = durations_[0];
    } else {
      result["request_id"] = metadata_.request_id;
      result["rolling_load"] = metadata_.rolling_load;
      result["denoise_steps"] = metadata_.denoise_steps;
      result["prepare_ms"] = durations_[1];
      result["denoise_ms"] = durations_[2];
      result["decode_ms"] = durations_[3];
      result["pipeline_ms"] = durations_[1] + durations_[2] + durations_[3];
    }
    LOG(INFO) << "H3_PHASE_JSON " << result.dump();
    return result;
  }

 private:
  bool enabled_;
  H3PhaseTimingMetadata metadata_;
  std::function<void()> synchronize_;
  bool started_ = false;
  bool finished_ = false;
  H3TimingPhase active_ = H3TimingPhase::LOAD;
  std::chrono::steady_clock::time_point boundary_;
  std::array<double, 4> durations_ = {};
};

}  // namespace xllm
