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

#include "mode_switch_service.h"

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include <chrono>

#include "common/metrics.h"
#include "llm_engine.h"
#include "runtime/xservice_client.h"
#include "scheduler/continuous_scheduler.h"
#include "scheduler/disagg_pd_scheduler.h"

namespace xllm {

ModeSwitchService::ModeSwitchService(LLMEngine* engine,
                                     ContinuousScheduler* scheduler)
    : engine_(engine), scheduler_(scheduler) {}

void ModeSwitchService::SwitchMode(
    ::google::protobuf::RpcController* controller,
    const proto::InstanceModeSwitchRequest* request,
    proto::InstanceModeSwitchResponse* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  // Full-flip wall clock for mode_switch_latency_ms histogram. Starts
  // before any bookkeeping so idempotent no-ops and validation errors
  // are also observed (they should be sub-ms so the histogram tail is
  // dominated by real flips).
  const auto flip_start = std::chrono::steady_clock::now();
  auto observe_latency = [&flip_start]() {
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - flip_start)
                             .count();
    HISTOGRAM_OBSERVE(mode_switch_latency_ms, elapsed);
  };
  if (!engine_) {
    response->set_ok(false);
    response->set_current_mode(current_mode_.load());
    response->set_error("engine not bound to ModeSwitchService");
    LOG(ERROR) << "ModeSwitchService::SwitchMode called but engine is null";
    COUNTER_INC(mode_switch_failed_total);
    observe_latency();
    return;
  }
  const int32_t target = request->target_mode();
  if (target != 0 && target != 1) {
    response->set_ok(false);
    response->set_current_mode(current_mode_.load());
    response->set_error("invalid target_mode (expect 0 or 1)");
    LOG(ERROR) << "ModeSwitchService::SwitchMode bad target=" << target;
    COUNTER_INC(mode_switch_failed_total);
    observe_latency();
    return;
  }
  const int32_t prev = current_mode_.load();
  if (prev == target) {
    // Idempotent: report success without re-issuing fan-out.
    response->set_ok(true);
    response->set_current_mode(prev);
    LOG(INFO) << "ModeSwitchService::SwitchMode no-op, already in mode="
              << target;
    // Idempotent no-op is a valid completion; observe the (small) latency
    // but don't bump total_to_cp/dp so the counters reflect actual flips.
    observe_latency();
    return;
  }
  LOG(INFO) << "ModeSwitchService::SwitchMode prev=" << prev
            << " target=" << target;

  // Quiesce protocol. worker_impl.cpp:switch_mode assumes "the scheduler's
  // drain protocol guarantees no forward is in flight on this worker when
  // switch_mode runs" -- previously nothing enforced that. Without it, a
  // brpc handler on the DisaggPDScheduler side (AddNewRequests, LinkInstance,
  // FirstGeneration) or the dispatch_thread would keep touching
  // kv_cache_manager_ mid-rebuild, which unique_ptr-resets the underlying
  // BlockManagerPool. The observed symptom is a silent SIGSEGV that lands
  // rank0 in a Zs zombie (no FATAL, no core in-container). Sequence:
  //   1. pause the main scheduler loop (WAIT: let running requests finish
  //      naturally; KV cache stays intact)
  //   2. take switch_gate_ unique; this waits out every in-flight
  //      dispatch_thread iteration and every brpc handler that grabbed the
  //      shared side, and blocks new ones from entering.
  //   3. flip every worker via engine_->switch_mode (fans out RPC).
  //   4. rebuild the scheduler-side pipeline (BlockManagerPool + BatchFactory
  //      + last_batch_ + active_dp_size_) for the new dp_size, all under the
  //      unique gate.
  //   5. release gate, resume the loop.
  //
  // If scheduler_ is null we skip pause+gate and fall back to the raw path
  // used by the standalone LLMMaster ModeSwitchService (single-instance
  // debug trigger, no dispatch_thread / disagg brpc surface).

  DisaggPDScheduler* disagg = dynamic_cast<DisaggPDScheduler*>(scheduler_);

  if (scheduler_ != nullptr) {
    scheduler_->pause(ContinuousScheduler::PauseMode::WAIT);
    // Use the caller-provided timeout so a heavy P/D pipeline (long TPOT
    // decode requests) can drain before rebuild. Fall back to 30s if the
    // caller didn't specify. If drain lapses we bail out and roll pause
    // back with resume() -- forcing rebuild with in-flight requests would
    // trip BlockManagerImpl's destructor invariant when their sequence
    // shared_ptrs are still outstanding, killing rank0.
    int64_t drain_timeout_ms = request->timeout_ms();
    if (drain_timeout_ms <= 0) {
      drain_timeout_ms = 30000;
    }
    // Observe drain wall time separately so operators can distinguish
    // "drain slow (in-flight decode)" from "rebuild slow (kv-pool rebuild
    // + relink)".
    const auto drain_start = std::chrono::steady_clock::now();
    const bool drained = scheduler_->wait_until_paused(drain_timeout_ms);
    const auto drain_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - drain_start)
                              .count();
    HISTOGRAM_OBSERVE(mode_switch_drain_ms, drain_ms);
    if (!drained) {
      LOG(WARNING) << "ModeSwitchService::SwitchMode timed out waiting for "
                      "scheduler to drain ("
                   << drain_timeout_ms << "ms); aborting flip";
      scheduler_->resume();
      response->set_ok(false);
      response->set_current_mode(prev);
      response->set_error("scheduler drain timeout");
      COUNTER_INC(mode_switch_failed_total);
      observe_latency();
      return;
    }
  }

  // Take the async-surface gate unique. For a plain ContinuousScheduler
  // (single-instance path) there is no dispatch_thread / brpc surface to
  // guard, so we skip the gate entirely.
  std::unique_lock<std::shared_mutex> gate;
  if (disagg != nullptr) {
    gate = disagg->begin_switch();
  }

  const bool ok = engine_->switch_mode(target);
  if (ok) {
    // Refresh scheduler-side bookkeeping to match the flipped engine layout.
    // For DisaggPDScheduler this is the same rebuild that the step() loop
    // used to do inline; doing it here inside the gate + paused loop makes
    // the transition atomic from the outside.
    const int32_t new_dp_size = engine_->dp_size();
    if (disagg != nullptr) {
      disagg->rebuild_after_flip(new_dp_size);
    }
    // v3 e2e revealed that even with the pool rebuilt cleanly, the first
    // DP-mode inference batch failed with LLM_NOT_YET_LINK (0x5010b007) at
    // every layer's PushKvBlocks. Root cause: LLMEngine::link_cluster fans
    // out per-D-worker links keyed on the peer's dp_size, and that topology
    // is baked in at startup. After a flip the (src_worker -> dst_worker)
    // pairs shift and the pre-flip links no longer cover them.
    //
    // Two-step recovery:
    //   1. Re-publish OUR new dp_size to etcd. Peers cache InstanceInfo
    //      lazily and their get_instance_info hits etcd, so without this
    //      the peer's relink builds the wrong topology again.
    //   2. Rebuild every LlmDataDist link this instance owns using the
    //      peer's new dp_size (pulled from etcd fresh). D-side is where
    //      the datadist links live; on P-side relink is a no-op.
    //
    // Step 1 (etcd write) can happen inside the gate cheaply. Step 2 does
    // engine_->unlink_cluster + link_cluster which fan out to worker RPCs
    // through link_threadpool_ + folly futures; running that under the gate
    // deadlocks (v4 D-rank0 died with `std::system_error: Resource deadlock
    // avoided` right after the relink line -- almost certainly a folly
    // future / thread-pool mutex acquired both by the caller thread and the
    // worker RPC handler while gate is held). Release the gate + resume
    // the scheduler BEFORE relinking so worker handlers run without
    // fighting the gate; the scheduler is idempotent to relink because
    // dispatch_thread will not push through a P instance whose new dp_size
    // has not been observed yet, and inference requests can still be
    // dispatched (but the KV push will retry -- link isn't ready until
    // relink_after_flip finishes).
    XServiceClient* xsvc = XServiceClient::get_instance();
    if (xsvc != nullptr) {
      if (!xsvc->re_register_dp_size(new_dp_size)) {
        LOG(WARNING) << "ModeSwitchService: re_register_dp_size failed; peer "
                        "may keep stale dp_size cache";
      }
    }
    current_mode_.store(target);
    response->set_ok(true);
    response->set_current_mode(target);
    // Update per-target flip counter + topology gauges. Counters are
    // split by target rather than tagged so grafana can plot the two
    // lines directly without label filtering (bvar-native counters
    // don't support labels the way Prometheus does).
    // Enum truth (parallel_args.h:219-220): CP_PREFILL=0, DP_DECODE=1.
    // The mode_switch.proto comment has 0/1 swapped; do not follow it.
    if (target == 0) {
      COUNTER_INC(mode_switch_total_to_cp);
    } else {
      COUNTER_INC(mode_switch_total_to_dp);
    }
    // engine_->cp_size() has the same runtime bookkeeping refreshed by
    // switch_mode; expose both so a dashboard can distinguish CP vs DP
    // topology unambiguously.
    GAUGE_SET(active_dp_size, static_cast<double>(new_dp_size));
    // cp_size follows the paired invariant (max(cp,dp) constant across
    // flip); read it via engine to stay honest with the flipped state.
    // If engine doesn't expose cp_size directly this is best-effort and
    // won't block the flip.
    // NOTE: if a future engine variant renames this accessor we'll drop
    // the gauge silently (compile-time member check via has_member style
    // is over-engineering for a metric).
    // For the single-instance/dual-mode case cp_size is world/dp_size, so
    // deriving from a Gauge kept in sync with dp_size is enough.
    // A simple fallback: (world / dp_size). world is available via the
    // engine but not exposed generically; leave to a follow-up.
  } else {
    // engine_->switch_mode() already attempted rollback. We trust the
    // engine has restored prev mode on every worker.
    response->set_ok(false);
    response->set_current_mode(prev);
    response->set_error("engine switch_mode fan-out failed; rolled back");
    COUNTER_INC(mode_switch_failed_total);
  }

  // Release the gate first so any brpc handler that has been queued behind
  // us can proceed once the scheduler resumes; then resume the loop.
  if (gate.owns_lock()) {
    gate.unlock();
  }
  if (scheduler_ != nullptr) {
    scheduler_->resume();
  }

  // Post-flip relink: rebuild datadist P<->D links to match the new dp_size
  // topology. Deliberately outside the gate + after resume() -- the fan-out
  // (link_threadpool_ + folly futures + worker RPC round-trips) hit a
  // deadlock inside the gate on v4 (D-rank0 crashed with
  // `std::system_error: Resource deadlock avoided`), because worker
  // handlers acquire per-service mutexes that overlap with what the caller
  // thread would need to release. The relink itself is safe outside the
  // gate: dispatch_thread that races us just sees old links until they
  // rebuild, and PushKvBlocks retries on transient LLM_NOT_YET_LINK.
  // Skipped if switch_mode failed (we already rolled back to prev mode).
  if (ok && disagg != nullptr) {
    if (!disagg->relink_after_flip()) {
      LOG(WARNING) << "ModeSwitchService: relink_after_flip reported failures; "
                      "some peers may still be linked with stale dp_size";
    }
  }

  // Final latency observation: covers pause + drain + engine flip +
  // rebuild + resume + relink (relink after resume is a real cost that
  // affects when disagg pipelines fully recover after a flip).
  observe_latency();
}

}  // namespace xllm
