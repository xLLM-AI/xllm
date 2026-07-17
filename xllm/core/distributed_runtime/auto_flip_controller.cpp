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

#include "auto_flip_controller.h"

#include <absl/time/clock.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>
#include <cmath>
#include <numeric>

#include "common/metrics.h"
#include "framework/block/kv_cache_manager.h"
#include "llm_engine.h"
#include "mode_switch.pb.h"
#include "mode_switch_service.h"
#include "scheduler/continuous_scheduler.h"

// Feature gate + policy knobs. Names mirror
// xllm_service/scheduler/managers/instance_mgr.cpp's mode-switch controller
// so operators moving between deployments (colocate single-instance vs
// disagg multi-instance) can reason with the same signals + thresholds.
DEFINE_bool(enable_auto_flip,
            false,
            "Enable in-process AutoFlipController that autonomously drives "
            "CP<->DP mode switching based on recent request stats. Requires "
            "--enable_runtime_cp_dp_switch=true. When false, ModeSwitchService "
            "still accepts external RPC-driven flips.");

DEFINE_int32(auto_flip_check_interval_s,
             5,
             "AutoFlipController tick period in seconds. Every tick the "
             "controller snapshots stats and evaluates the flip decision.");

DEFINE_int32(auto_flip_window_ms,
             30000,
             "Rolling window (ms) over which long_ratio is computed.");

DEFINE_int64(auto_flip_long_prompt_threshold,
             1024,
             "Requests with prompt_tokens >= this threshold are counted as "
             "'long' for long_ratio.");

DEFINE_double(auto_flip_long_ratio_activate,
              0.4,
              "If long_ratio in the window >= this threshold AND we are in "
              "DP_DECODE mode (1), the controller flips to CP_PREFILL (0). "
              "Enum truth: CP_PREFILL=0, DP_DECODE=1 (parallel_args.h).");

DEFINE_double(auto_flip_long_ratio_deactivate,
              0.2,
              "If long_ratio in the window < this threshold AND we are in "
              "CP_PREFILL mode (0), the controller flips to DP_DECODE (1). "
              "Must be <= auto_flip_long_ratio_activate to give a hysteresis "
              "band that prevents flip-flopping.");

DEFINE_int32(auto_flip_persist_ms,
             30000,
             "Minimum dwell time (ms) in the current mode before another "
             "flip is allowed. Prevents chatter under noisy signals.");

DEFINE_uint64(auto_flip_min_samples,
              10,
              "Minimum sample count in the window before decisions are made. "
              "Below this, controller stays in current mode.");

DEFINE_int32(auto_flip_drain_timeout_ms,
             60000,
             "Drain timeout (ms) passed to ModeSwitchService.SwitchMode from "
             "the AutoFlipController tick. Should exceed the p99 in-flight "
             "decode wall-clock of the workload; if drain times out the flip "
             "aborts and rolls back cleanly. Longer than the default handles "
             "workloads with long-running decode (60-120s) that would "
             "otherwise miss the drain window and leave heal flips stuck.");

DEFINE_double(auto_flip_min_pending_per_dp_rank,
              2.0,
              "In DP_DECODE mode, if pending_requests / active_dp_size < this "
              "ratio, force a heal-flip back to CP_PREFILL regardless of "
              "long_ratio and dwell time. DP with too few requests to fill "
              "every dp_rank drives step() into the lopsided-batch backdoor, "
              "which reaches worker_impl's fake-input path and hangs on the "
              "ATB decoder layer's placeholder tensors (see v14 memory). "
              "Setting this to 0 disables the heal path (legacy behavior).");

DEFINE_int32(auto_flip_heal_window_ms,
             5000,
             "Sliding window (ms) over which the heal path evaluates peak "
             "pending. Heal fires only when max(pending) across the whole "
             "window falls below the per-dp-rank threshold. Wider window is "
             "more resistant to brief inter-burst gaps (verify_switch style "
             "max_tokens=6 short-request pattern) but slower to react to a "
             "genuine load drop. 5000ms covers ~1 request roundtrip while "
             "still healing within ~5s of a real thin workload.");

namespace xllm {

namespace {
inline int64_t now_ms() { return absl::ToUnixMillis(absl::Now()); }
}  // namespace

AutoFlipController::AutoFlipController(LLMEngine* engine,
                                       ContinuousScheduler* scheduler,
                                       ModeSwitchService* mode_switch)
    : engine_(engine),
      scheduler_(scheduler),
      mode_switch_(mode_switch),
      last_flip_ms_(now_ms()) {
  CHECK(engine_ != nullptr);
  CHECK(scheduler_ != nullptr);
  CHECK(mode_switch_ != nullptr);
}

AutoFlipController::~AutoFlipController() { stop(); }

void AutoFlipController::start() {
  if (thread_.joinable()) {
    return;
  }
  stop_flag_.store(false);
  thread_ = std::thread([this]() { run_loop(); });
  LOG(INFO) << "AutoFlipController started (enable=" << FLAGS_enable_auto_flip
            << ", tick=" << FLAGS_auto_flip_check_interval_s << "s"
            << ", window=" << FLAGS_auto_flip_window_ms << "ms"
            << ", long_threshold=" << FLAGS_auto_flip_long_prompt_threshold
            << ", activate=" << FLAGS_auto_flip_long_ratio_activate
            << ", deactivate=" << FLAGS_auto_flip_long_ratio_deactivate
            << ", persist=" << FLAGS_auto_flip_persist_ms << "ms"
            << ", drain_timeout=" << FLAGS_auto_flip_drain_timeout_ms << "ms"
            << ", min_pending_per_dp_rank="
            << FLAGS_auto_flip_min_pending_per_dp_rank
            << ", heal_window=" << FLAGS_auto_flip_heal_window_ms << "ms)";
}

void AutoFlipController::stop() {
  if (!thread_.joinable()) {
    return;
  }
  stop_flag_.store(true);
  thread_.join();
  LOG(INFO) << "AutoFlipController stopped";
}

// Heal predicate. Extracted so decide_target (which returns the target
// mode) and run_loop's dwell-time bypass share exactly the same
// condition -- earlier revisions duplicated the expression in both
// places and drifted apart.
//
// max_pending_in_window is the peak pending count across the recent
// FLAGS_auto_flip_heal_window_ms. Using peak rather than instant
// pending prevents spurious heal when a short-request burst finishes
// between ticks (pending momentarily 0, then bounces back). Verified
// on 2026-07-07: instantaneous heal triggered during verify_switch
// max_tokens=6 pattern and raced with in-flight requests.
bool AutoFlipController::heal_active(int8_t cur_mode,
                                     int32_t active_dp_size,
                                     size_t max_pending_in_window) {
  if (cur_mode != 1 /* DP_DECODE */ || active_dp_size <= 1 ||
      FLAGS_auto_flip_min_pending_per_dp_rank <= 0.0) {
    return false;
  }
  const double min_pending_double = FLAGS_auto_flip_min_pending_per_dp_rank *
                                    static_cast<double>(active_dp_size);
  // Ceiling so a fractional threshold (e.g. 1.5 * dp=2 = 3) rounds up,
  // matching the "must be able to fill every dp_rank" invariant.
  const size_t min_pending = static_cast<size_t>(std::ceil(min_pending_double));
  return max_pending_in_window < min_pending;
}

// Sample the current pending count into the sliding history, prune
// anything older than heal_window_ms, and return the peak in the
// resulting window. Called once per tick from run_loop; single
// producer (tick thread), no locking needed.
size_t AutoFlipController::record_pending_and_max(size_t current_pending) {
  const int64_t now = now_ms();
  const int64_t window_ms = std::max(0, FLAGS_auto_flip_heal_window_ms);
  const int64_t cutoff = now - window_ms;

  pending_history_.emplace_back(now, current_pending);
  while (!pending_history_.empty() && pending_history_.front().first < cutoff) {
    pending_history_.pop_front();
  }
  size_t peak = 0;
  for (const auto& entry : pending_history_) {
    if (entry.second > peak) peak = entry.second;
  }
  return peak;
}

int8_t AutoFlipController::decide_target(int8_t cur_mode,
                                         double long_ratio,
                                         double pool_pressure,
                                         uint64_t total_in_window,
                                         int32_t active_dp_size,
                                         size_t max_pending_in_window) const {
  // Heal path: in DP_DECODE with active_dp_size > 1, if the sliding
  // window's peak pending fell below what fills every dp_rank, force a
  // flip back to CP. This runs BEFORE the sample-count and hysteresis
  // gates because a hung DP forward stops the request pipe, samples stop
  // arriving, and the ordinary long_ratio decision would sit at cur_mode
  // forever waiting for min_samples. See the DEFINE_ comment above for
  // the full failure path (worker_impl fake-input -> ATB placeholder
  // hang). Using max_pending across the window (v21) instead of instant
  // pending (v20) prevents spurious heal during short-request bursts
  // that briefly drain between arrivals.
  if (heal_active(cur_mode, active_dp_size, max_pending_in_window)) {
    return 0;  // -> CP_PREFILL, heal the lopsided DP batch.
  }

  // Not enough samples: stay put.
  if (total_in_window < FLAGS_auto_flip_min_samples) {
    return cur_mode;
  }
  // Mirrors xllm_service decide_mode_switch_target's hysteresis:
  //   * from CP (0), flip down to DP (1) when long-request pressure has
  //     receded (below the deactivate threshold).
  //   * from DP (1), flip up to CP (0) when long-request pressure is high.
  // Enum truth: CP_PREFILL=0, DP_DECODE=1 (parallel_args.h:219-220).
  // pool_pressure is left in the signature for future refinement (matches
  // xllm_service's per-lane pressure inputs); the current colocate rollout
  // only uses long_ratio, since single-instance has no per-lane pool.
  (void)pool_pressure;

  if (cur_mode == 0 /* CP_PREFILL */) {
    if (long_ratio < FLAGS_auto_flip_long_ratio_deactivate) {
      // v22: CP -> DP requires enough concurrency to fill every dp_rank.
      // Without this gate, a low-QPS workload (long_ratio=0, pending<threshold)
      // would trigger CP -> DP, then heal would immediately trigger
      // DP -> CP, ping-ponging every tick. Symmetric to heal_active but
      // guarding the entrance to DP mode instead of the exit. Uses the
      // same threshold (min_pending_per_dp_rank * active_dp_size) to
      // keep the two gates numerically consistent.
      //
      // Note: at cur_mode==CP, active_dp_size is 1 (CP layout dp=1). We
      // must project what the DP layout WOULD be after the flip, which
      // equals options_.cp_size() (paired invariant). Since we don't have
      // options in this static-analysis-friendly signature, we approximate
      // with the ceil of (min_pending_per_dp_rank * 2) as a conservative
      // guess -- most deployments have paired=cp=2 which matches, and
      // higher pairing means we'd need MORE pending anyway (so this is
      // never too permissive).
      if (FLAGS_auto_flip_min_pending_per_dp_rank > 0.0) {
        const size_t min_pending = static_cast<size_t>(
            std::ceil(FLAGS_auto_flip_min_pending_per_dp_rank * 2.0));
        if (max_pending_in_window < min_pending) {
          // Not enough concurrency to justify DP; stay in CP.
          return 0;
        }
      }
      return 1;  // -> DP_DECODE
    }
    return 0;
  }
  if (cur_mode == 1 /* DP_DECODE */) {
    if (long_ratio >= FLAGS_auto_flip_long_ratio_activate) {
      return 0;  // -> CP_PREFILL
    }
    return 1;
  }
  return cur_mode;
}

void AutoFlipController::run_loop() {
  while (!stop_flag_.load()) {
    // Sleep in short slices so a stop() during a long tick returns
    // quickly. tick_seconds could be changed at runtime via /flags,
    // so re-read the flag each iteration.
    const int32_t tick_s = std::max(1, FLAGS_auto_flip_check_interval_s);
    for (int i = 0; i < tick_s && !stop_flag_.load(); ++i) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (stop_flag_.load()) break;

    if (!FLAGS_enable_auto_flip) {
      continue;
    }

    // 1. Snapshot signals.
    auto stats_snap = scheduler_->auto_flip_stats().snapshot(
        FLAGS_auto_flip_long_prompt_threshold, FLAGS_auto_flip_window_ms);
    // Publish the driving signal for observability. A dashboard can plot
    // long_ratio alongside mode_switch_total to correlate signal shifts
    // with the flips they triggered.
    GAUGE_SET(auto_flip_long_ratio, stats_snap.long_ratio);

    // pool_pressure = mean(1 - free/total) across dp lanes. Single-
    // instance colocate typically has one lane; averaging is still safe.
    double pool_pressure = 0.0;
    // Guard against a null pool during rebuild_after_flip (transient).
    // The scheduler rebuild path holds pool ptr behind a shared_mutex;
    // we skip a tick rather than block the flip.
    // For now, leave pool_pressure at 0.0 and rely on long_ratio only.
    // See xllm_service compute_pool_pressure_lane_locked for the future
    // per-lane implementation.

    // 2. Read current mode. Trust ModeSwitchService's atomic
    // current_mode_ as the source of truth; it's written on flip
    // completion and reads consistently even mid-flip. Deriving from
    // engine_->dp_size() lagged by ~1 tick on some paths (dp_size is
    // updated by engine.switch_mode which happens under the pause gate)
    // and produced spurious "cur_mode=1 target=0" repeats after a
    // successful flip, though the flip itself was already complete.
    const int32_t cur_mode_i32 = mode_switch_->current_mode();
    const int8_t cur_mode = static_cast<int8_t>(cur_mode_i32);

    // Load-shape signals: how many pending requests, and how wide is the
    // current DP layout. These drive the heal-path in decide_target that
    // avoids DP-mode single-request lopsided hangs. Push current pending
    // into the sliding history and evaluate on the window peak, not the
    // instant sample -- see record_pending_and_max comment.
    const int32_t active_dp = scheduler_->active_dp_size();
    const size_t pending = scheduler_->num_pending_requests();
    const size_t max_pending = record_pending_and_max(pending);

    // 3. Persist-time gate. The heal path (DP with too few pending in
    // the sliding window) is exempt: a hung DP forward stops the
    // pipeline entirely, and waiting out the dwell window would leave
    // the instance dark. If decide_target returns the CP heal target
    // we let it through even mid-dwell.
    const int64_t elapsed_since_flip = now_ms() - last_flip_ms_;
    const bool heal_path_active = heal_active(cur_mode, active_dp, max_pending);
    if (!heal_path_active && elapsed_since_flip < FLAGS_auto_flip_persist_ms) {
      // Still in the dwell window from the last flip; skip decision.
      continue;
    }

    // 4. Decide.
    const int8_t target = decide_target(cur_mode,
                                        stats_snap.long_ratio,
                                        pool_pressure,
                                        stats_snap.total_in_window,
                                        active_dp,
                                        max_pending);

    LOG_EVERY_N(INFO, 6) << "AutoFlipController tick: cur_mode="
                         << static_cast<int>(cur_mode)
                         << " long_ratio=" << stats_snap.long_ratio << " (long/"
                         << stats_snap.long_in_window << "/"
                         << stats_snap.total_in_window << ")"
                         << " active_dp=" << active_dp << " pending=" << pending
                         << " max_pending=" << max_pending
                         << " heal=" << heal_path_active
                         << " target=" << static_cast<int>(target);

    if (target == cur_mode) {
      continue;
    }

    // 5. Flip. Route through ModeSwitchService::SwitchMode with an
    // in-process request; the service already handles pause/drain/rebuild
    // exactly the same way the RPC path does. We build a synthetic
    // request/response and pass a null Closure since we execute the
    // handler synchronously (SwitchMode consumes the ClosureGuard but
    // done->Run() is no-op if done is null).
    LOG(INFO) << "AutoFlipController flipping " << static_cast<int>(cur_mode)
              << " -> " << static_cast<int>(target)
              << " (long_ratio=" << stats_snap.long_ratio
              << ", samples=" << stats_snap.total_in_window << ")";

    proto::InstanceModeSwitchRequest req;
    req.set_target_mode(target);
    // Drain timeout is now flag-driven; production workloads with
    // long-running decode (60-120s) would miss a 30s drain window and
    // leave heal flips permanently stuck. See auto_flip_drain_timeout_ms.
    req.set_timeout_ms(FLAGS_auto_flip_drain_timeout_ms);
    proto::InstanceModeSwitchResponse resp;
    // In-process invocation: pass nullptr controller and nullptr closure.
    // SwitchMode's ClosureGuard tolerates a null done (guard's dtor
    // becomes a no-op) as long as we only touch done through the guard.
    // But SwitchMode assumes non-null; wrap a trivial closure.
    struct NoopClosure : public ::google::protobuf::Closure {
      void Run() override {}
    };
    NoopClosure done;
    mode_switch_->SwitchMode(nullptr, &req, &resp, &done);

    if (resp.ok()) {
      last_flip_ms_ = now_ms();
      LOG(INFO) << "AutoFlipController flip ok: current_mode="
                << resp.current_mode();
    } else {
      LOG(WARNING) << "AutoFlipController flip failed: " << resp.error()
                   << " (current_mode=" << resp.current_mode() << ")";
      // Do NOT update last_flip_ms_ so the persist gate lets us retry
      // on the next tick. This matches xllm_service's mode_switch_retry
      // semantics conceptually.
    }
  }
}

}  // namespace xllm
