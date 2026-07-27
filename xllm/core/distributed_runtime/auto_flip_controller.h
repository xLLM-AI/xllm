/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <atomic>
#include <deque>
#include <memory>
#include <thread>
#include <utility>

namespace xllm {

class LLMEngine;
class ContinuousScheduler;
class ModeSwitchService;

// AutoFlipController watches recent request statistics (long-prompt ratio,
// KV-cache pressure) and drives runtime CP<->DP switch autonomously,
// without an external xllm_service. It runs a background thread that
// wakes up every FLAGS_auto_flip_check_interval_s seconds, snapshots the
// signals from the ContinuousScheduler's AutoFlipStats + kv_cache_manager,
// applies the same hysteresis policy as xllm_service's
// InstanceMgr::decide_mode_switch_target, and, when a flip is warranted,
// invokes ModeSwitchService::SwitchMode in-process (no RPC round trip).
//
// Feature-gated by FLAGS_enable_auto_flip. When disabled the controller
// still starts (idempotent, cheap) but its tick loop is a no-op, so
// enabling/disabling at runtime via brpc /flags gives an operator lever
// without restart.
//
// Non-owning pointers: the caller (LLMMaster) owns engine + scheduler +
// mode_switch_service and outlives this controller.
class AutoFlipController {
 public:
  AutoFlipController(LLMEngine* engine,
                     ContinuousScheduler* scheduler,
                     ModeSwitchService* mode_switch);
  ~AutoFlipController();

  // Non-copyable, non-movable.
  AutoFlipController(const AutoFlipController&) = delete;
  AutoFlipController& operator=(const AutoFlipController&) = delete;

  void start();
  void stop();

 private:
  void run_loop();

  // Return the target mode we should be in given current mode + signals,
  // matching xllm_service/scheduler/managers/instance_mgr.cpp's
  // decide_mode_switch_target. Both call sites should stay in sync so
  // colocate + disagg deployments produce the same policy.
  //
  // active_dp_size is the scheduler's live dp_size (post-flip);
  // max_pending_in_window is the peak pending across the recent heal
  // window (see record_pending_and_max). Together they gate the DP-side
  // lopsided-hang mitigation: a DP burst that starves one dp_rank drives
  // step() into the >100ms backdoor and the fake-input path in
  // worker_impl (v14 traced this to an ATB placeholder hang inside the
  // shared decoder layer's DP decode node, not fixable from xllm). To
  // avoid falling into that path we heal DP -> CP as soon as concurrency
  // drops below what fills every dp_rank. Using max-in-window rather
  // than instant pending (v20 -> v21) prevents spurious heal when a
  // burst briefly drains between ticks.
  int8_t decide_target(int8_t cur_mode,
                       double long_ratio,
                       double pool_pressure,
                       uint64_t total_in_window,
                       int32_t active_dp_size,
                       size_t max_pending_in_window) const;

  // Heal path predicate shared with decide_target. Returns true when we're
  // in DP_DECODE with a lopsided dp layout that would drive step() into the
  // fake-input hang. The tick loop also uses this to bypass the dwell-time
  // gate: waiting out the dwell window during a hang would leave the
  // instance dark.
  //
  // max_pending_in_window: peak pending across the last few ticks (see
  // pending_history_). We heal only when the WHOLE window has been thin,
  // not just the instantaneous sample -- otherwise a request burst that
  // completes in <tick_interval leaves pending==0 at sample time and
  // triggers a spurious heal that races with in-flight requests
  // (verify_switch 11/18 PARTIAL, 2026-07-07). If the peak in the window
  // was tall enough to fill every dp_rank, the instance is healthy.
  static bool heal_active(int8_t cur_mode,
                          int32_t active_dp_size,
                          size_t max_pending_in_window);

  // Push a (now, pending) sample into pending_history_ and prune anything
  // older than heal_window_ms. Returns the peak pending in the pruned
  // history. Called once per tick from run_loop.
  size_t record_pending_and_max(size_t current_pending);

  LLMEngine* engine_ = nullptr;
  ContinuousScheduler* scheduler_ = nullptr;
  ModeSwitchService* mode_switch_ = nullptr;

  std::atomic<bool> stop_flag_{false};
  std::thread thread_;

  // Timestamp (absl millis-since-epoch) of the last successful flip.
  // Used to enforce FLAGS_auto_flip_persist_ms minimum dwell time.
  int64_t last_flip_ms_ = 0;

  // Sliding window of (timestamp_ms, pending) samples pushed once per
  // tick. Used to compute max_pending_in_window for the heal path so
  // brief request-burst gaps (pending==0 for one tick between a burst
  // of finished requests and the next arrivals) don't trigger a
  // spurious heal. Bounded by FLAGS_auto_flip_heal_window_ms.
  std::deque<std::pair<int64_t, size_t>> pending_history_;
};

}  // namespace xllm
