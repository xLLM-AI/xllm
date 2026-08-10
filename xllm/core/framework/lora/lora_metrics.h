/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
==============================================================================*/

// Per-adapter observability metrics for multi-LoRA (P1-D).
//
// Design (2026-07-09 explore report):
// - Latency (TTFT / inter-token / E2E) uses
// bvar::MultiDimension<LatencyRecorder>
//   with a single "lora_name" label. get_stats({name}) lazily creates a
//   sub-recorder on first observation; delete_stats({name}) removes it when
//   the adapter is drained. Mirrors the existing dp_rank labelled histograms
//   in metrics.cpp so /vars scrape format stays consistent.
//
// - Counters (requests_total, tokens_total, errors_total) and gauges
//   (active_state, queue_depth, cache_hit_rate) use heap-allocated
//   bvar::Adder<double> / bvar::Status<double> stored in shared_mutex-guarded
//   std::unordered_map keyed by lora_name. On adapter load a full set of
//   named bvars is created (auto-exposed to brpc built-in /vars endpoint);
//   on final removal from LoRARegistry (drain complete) the map slot is
//   erased and the bvars are destroyed (auto-hidden from /vars).
//
// - Info line "xllm:lora_requests_info{lora_name,state}" mirrors vLLM #45030
//   for llm-d / router integrations. Emitted via a single global
//   MultiDimension<Adder> keyed by (lora_name, state) so scrapers can see
//   both running and waiting counts per adapter without a JSON parse.
//
// All access is thread-safe. Reads take a shared_lock; add_adapter and
// remove_adapter take a unique_lock. Hot-path metric observations use
// the sub-bvars directly under a shared_lock -- <100ns overhead when the
// adapter's entry already exists.

#pragma once

#include <bvar/bvar.h>
#include <bvar/multi_dimension.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {

// Per-adapter mutable counter/gauge bvars. One instance per loaded adapter,
// destroyed on unload_final_removal.
struct LoRAAdapterBvars {
  // Counters -- monotonically increasing.
  std::unique_ptr<bvar::Adder<double>> requests_total;
  std::unique_ptr<bvar::Adder<double>> tokens_prompt_total;
  std::unique_ptr<bvar::Adder<double>> tokens_generated_total;
  std::unique_ptr<bvar::Adder<double>> errors_total;
  // Gauges -- point-in-time values.
  //   active_state: 0 = unloaded, 1 = active (device pool populated), 2 =
  //   draining.
  std::unique_ptr<bvar::Status<double>> active_state;
  //   device_slots: how many per_proj slots the adapter currently owns
  //   (36 layers x 7 projs = 252 for a full Qwen adapter).
  std::unique_ptr<bvar::Status<double>> device_slots;
};

class LoRAMetrics {
 public:
  static LoRAMetrics& instance();

  // ---- Lifecycle: called by LoRARegistry on register / final_removal ----

  // Create the per-adapter bvars and expose them under snake_case names of
  // the form "xllm_lora_<lora_name>_<metric>". Also binds int_id -> name so
  // hot-path emit points can index by adapter_id without another map lookup
  // through LoRARegistry.
  void add_adapter(uint64_t int_id, const std::string& lora_name);

  // Destroy the per-adapter bvars AND the labelled sub-recorders in the
  // shared MultiDimension histograms. Idempotent.
  void remove_adapter(uint64_t int_id);

  // ---- Hot-path emit points (called from scheduler / response processor) ----

  // TTFT: observed once per sequence when its first output token materialises.
  void observe_ttft(uint64_t adapter_id, int64_t ms);
  // Inter-token latency (TPOT): observed for each token after the first.
  void observe_inter_token(uint64_t adapter_id, int64_t ms);
  // End-to-end latency: observed on request completion.
  void observe_e2e(uint64_t adapter_id, int64_t ms);

  // Bump the request/token/error counters. Safe if adapter_id == 0
  // (non-LoRA request) -- becomes a no-op.
  void inc_requests(uint64_t adapter_id);
  void add_tokens_prompt(uint64_t adapter_id, int64_t n);
  void add_tokens_generated(uint64_t adapter_id, int64_t n);
  void inc_errors(uint64_t adapter_id);

  // ---- State transitions (called from LoRARuntime) ----

  // 0 = pending / 1 = active / 2 = draining. Mirrors LoRARegistry entry
  // lifetime; used by ops dashboards.
  void set_state(uint64_t adapter_id, int state);
  void set_device_slots(uint64_t adapter_id, int64_t slots);

  // ---- vLLM #45030 -style info counters ----
  // Emit "xllm:lora_requests_info{lora_name=name,state=running|waiting}".
  // Called by scheduler when a sequence transitions running <-> waiting.
  void inc_requests_info(uint64_t adapter_id, const std::string& state);
  void dec_requests_info(uint64_t adapter_id, const std::string& state);

  // ---- Dump for /v1/lora_stats HTTP endpoint (JSON-friendly snapshot) ----

  struct Snapshot {
    std::string lora_name;
    double requests_total = 0;
    double tokens_prompt_total = 0;
    double tokens_generated_total = 0;
    double errors_total = 0;
    double active_state = 0;
    double device_slots = 0;
    int64_t ttft_p50_ms = 0;
    int64_t ttft_p99_ms = 0;
    int64_t e2e_p50_ms = 0;
    int64_t e2e_p99_ms = 0;
    int64_t qps = 0;
  };
  std::vector<Snapshot> snapshot_all() const;

  // W4-A4: batch-level LoRA metrics (global, not per-adapter). Called from
  // scheduler prepare_batch once per assembled batch.
  //   distinct_count = number of distinct non-zero adapter_ids in the batch
  //   total_seqs     = total sequences in the batch (for hit-rate ratio)
  // fast_path = (distinct_count <= 1), slow_path = (distinct_count >= 2)
  void observe_batch_lora(size_t distinct_count, size_t total_seqs);

  // For test / debug: how many adapters are currently tracked.
  size_t adapter_count() const;

 private:
  LoRAMetrics();
  ~LoRAMetrics();
  LoRAMetrics(const LoRAMetrics&) = delete;
  LoRAMetrics& operator=(const LoRAMetrics&) = delete;

  // Adapter-id -> name lookup so hot-path emit can avoid the LoRARegistry
  // mutex. Populated by add_adapter, cleared by remove_adapter.
  mutable std::shared_mutex mu_;
  std::unordered_map<uint64_t, std::string> id_to_name_;
  std::unordered_map<std::string, LoRAAdapterBvars> per_adapter_;

  // Shared labelled histograms. Sub-recorders lazily materialised by
  // MultiDimension::get_stats({name}) on first observation, and pruned in
  // remove_adapter.
  std::unique_ptr<bvar::MultiDimension<bvar::LatencyRecorder>> ttft_ms_;
  std::unique_ptr<bvar::MultiDimension<bvar::LatencyRecorder>> inter_token_ms_;
  std::unique_ptr<bvar::MultiDimension<bvar::LatencyRecorder>> e2e_ms_;
  // Requests-info gauge with 2 labels (lora_name, state).
  std::unique_ptr<bvar::MultiDimension<bvar::Adder<int64_t>>> requests_info_;

  // Global gauge: how many adapters are currently in `active` state.
  std::unique_ptr<bvar::Status<int64_t>> active_adapters_count_;

  // W4-A4 batch-level counters (global, not per-adapter).
  std::unique_ptr<bvar::Adder<int64_t>> batch_fast_path_total_;
  std::unique_ptr<bvar::Adder<int64_t>> batch_slow_path_total_;
  std::unique_ptr<bvar::LatencyRecorder> batch_distinct_adapters_;
};

}  // namespace xllm
