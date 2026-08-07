/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "lora_metrics.h"

#include <glog/logging.h>

namespace xllm {

namespace {
// Sanitise arbitrary user-provided lora_name into a valid bvar identifier.
// bvar names allow [A-Za-z0-9_], so replace anything else with '_'.
std::string sanitise_bvar_name(const std::string& in) {
  std::string out;
  out.reserve(in.size());
  for (char c : in) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '_') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  return out;
}
}  // namespace

LoRAMetrics& LoRAMetrics::instance() {
  static LoRAMetrics g;
  return g;
}

LoRAMetrics::LoRAMetrics() {
  ttft_ms_ = std::make_unique<bvar::MultiDimension<bvar::LatencyRecorder>>(
      "xllm_lora_ttft_milliseconds", std::list<std::string>{"lora_name"});
  inter_token_ms_ =
      std::make_unique<bvar::MultiDimension<bvar::LatencyRecorder>>(
          "xllm_lora_inter_token_latency_milliseconds",
          std::list<std::string>{"lora_name"});
  e2e_ms_ = std::make_unique<bvar::MultiDimension<bvar::LatencyRecorder>>(
      "xllm_lora_end_2_end_latency_milliseconds",
      std::list<std::string>{"lora_name"});
  // vLLM #45030 -compatible info gauge with 2 labels.
  requests_info_ = std::make_unique<bvar::MultiDimension<bvar::Adder<int64_t>>>(
      "xllm_lora_requests_info", std::list<std::string>{"lora_name", "state"});
  active_adapters_count_ = std::make_unique<bvar::Status<int64_t>>(
      "xllm_lora_active_adapters_count", 0);
  // W4-A4 batch-level metrics (global).
  batch_fast_path_total_ =
      std::make_unique<bvar::Adder<int64_t>>("xllm_lora_batch_fast_path_total");
  batch_slow_path_total_ =
      std::make_unique<bvar::Adder<int64_t>>("xllm_lora_batch_slow_path_total");
  batch_distinct_adapters_ = std::make_unique<bvar::LatencyRecorder>(
      "xllm_lora_batch_distinct_adapters");
  LOG(INFO) << "[LoRAMetrics] initialised";
}

LoRAMetrics::~LoRAMetrics() = default;

void LoRAMetrics::add_adapter(uint64_t int_id, const std::string& lora_name) {
  const std::string safe = sanitise_bvar_name(lora_name);
  std::unique_lock lock(mu_);
  id_to_name_[int_id] = lora_name;
  if (per_adapter_.count(lora_name)) {
    // Idempotent: same-name re-register (e.g. after unload+load) reuses
    // the existing bvars so historical counters/histograms carry over.
    return;
  }
  LoRAAdapterBvars b;
  b.requests_total = std::make_unique<bvar::Adder<double>>("xllm_lora_" + safe +
                                                           "_requests_total");
  b.tokens_prompt_total = std::make_unique<bvar::Adder<double>>(
      "xllm_lora_" + safe + "_tokens_prompt_total");
  b.tokens_generated_total = std::make_unique<bvar::Adder<double>>(
      "xllm_lora_" + safe + "_tokens_generated_total");
  b.errors_total = std::make_unique<bvar::Adder<double>>("xllm_lora_" + safe +
                                                         "_errors_total");
  b.active_state = std::make_unique<bvar::Status<double>>(
      "xllm_lora_" + safe + "_active_state", 0.0);
  b.device_slots = std::make_unique<bvar::Status<double>>(
      "xllm_lora_" + safe + "_device_slots", 0.0);
  per_adapter_.emplace(lora_name, std::move(b));
  LOG(INFO) << "[LoRAMetrics] registered bvars for '" << lora_name
            << "' int_id=" << int_id;
}

void LoRAMetrics::remove_adapter(uint64_t int_id) {
  std::unique_lock lock(mu_);
  auto id_it = id_to_name_.find(int_id);
  if (id_it == id_to_name_.end()) return;
  const std::string lora_name = id_it->second;
  id_to_name_.erase(id_it);
  auto it = per_adapter_.find(lora_name);
  if (it == per_adapter_.end()) return;
  // Prune the labelled sub-recorders in the shared histograms too so
  // /vars stops emitting stale zero-observation rows for this adapter.
  if (ttft_ms_) ttft_ms_->delete_stats({lora_name});
  if (inter_token_ms_) inter_token_ms_->delete_stats({lora_name});
  if (e2e_ms_) e2e_ms_->delete_stats({lora_name});
  if (requests_info_) {
    requests_info_->delete_stats({lora_name, "running"});
    requests_info_->delete_stats({lora_name, "waiting"});
  }
  per_adapter_.erase(it);
  LOG(INFO) << "[LoRAMetrics] removed bvars for '" << lora_name
            << "' int_id=" << int_id;
}

// ---- Hot path helpers -----------------------------------------------------

namespace {
// Look up name for an adapter_id under shared_lock. Returns empty string
// on miss (caller should treat as "not tracked").
inline std::string name_of(
    const std::unordered_map<uint64_t, std::string>& id_to_name,
    uint64_t adapter_id) {
  auto it = id_to_name.find(adapter_id);
  if (it == id_to_name.end()) return std::string();
  return it->second;
}
}  // namespace

void LoRAMetrics::observe_ttft(uint64_t adapter_id, int64_t ms) {
  if (adapter_id == 0) return;
  std::shared_lock lock(mu_);
  std::string name = name_of(id_to_name_, adapter_id);
  if (name.empty() || !ttft_ms_) return;
  auto* rec = ttft_ms_->get_stats({name});
  if (rec) *rec << ms;
}

void LoRAMetrics::observe_inter_token(uint64_t adapter_id, int64_t ms) {
  if (adapter_id == 0) return;
  std::shared_lock lock(mu_);
  std::string name = name_of(id_to_name_, adapter_id);
  if (name.empty() || !inter_token_ms_) return;
  auto* rec = inter_token_ms_->get_stats({name});
  if (rec) *rec << ms;
}

void LoRAMetrics::observe_e2e(uint64_t adapter_id, int64_t ms) {
  if (adapter_id == 0) return;
  std::shared_lock lock(mu_);
  std::string name = name_of(id_to_name_, adapter_id);
  if (name.empty() || !e2e_ms_) return;
  auto* rec = e2e_ms_->get_stats({name});
  if (rec) *rec << ms;
}

void LoRAMetrics::inc_requests(uint64_t adapter_id) {
  if (adapter_id == 0) return;
  std::shared_lock lock(mu_);
  std::string name = name_of(id_to_name_, adapter_id);
  if (name.empty()) return;
  auto it = per_adapter_.find(name);
  if (it == per_adapter_.end() || !it->second.requests_total) return;
  *it->second.requests_total << 1;
}

void LoRAMetrics::add_tokens_prompt(uint64_t adapter_id, int64_t n) {
  if (adapter_id == 0 || n <= 0) return;
  std::shared_lock lock(mu_);
  std::string name = name_of(id_to_name_, adapter_id);
  if (name.empty()) return;
  auto it = per_adapter_.find(name);
  if (it == per_adapter_.end() || !it->second.tokens_prompt_total) return;
  *it->second.tokens_prompt_total << static_cast<double>(n);
}

void LoRAMetrics::add_tokens_generated(uint64_t adapter_id, int64_t n) {
  if (adapter_id == 0 || n <= 0) return;
  std::shared_lock lock(mu_);
  std::string name = name_of(id_to_name_, adapter_id);
  if (name.empty()) return;
  auto it = per_adapter_.find(name);
  if (it == per_adapter_.end() || !it->second.tokens_generated_total) return;
  *it->second.tokens_generated_total << static_cast<double>(n);
}

void LoRAMetrics::inc_errors(uint64_t adapter_id) {
  if (adapter_id == 0) return;
  std::shared_lock lock(mu_);
  std::string name = name_of(id_to_name_, adapter_id);
  if (name.empty()) return;
  auto it = per_adapter_.find(name);
  if (it == per_adapter_.end() || !it->second.errors_total) return;
  *it->second.errors_total << 1;
}

void LoRAMetrics::set_state(uint64_t adapter_id, int state) {
  if (adapter_id == 0) return;
  std::shared_lock lock(mu_);
  std::string name = name_of(id_to_name_, adapter_id);
  if (name.empty()) return;
  auto it = per_adapter_.find(name);
  if (it == per_adapter_.end() || !it->second.active_state) return;
  it->second.active_state->set_value(static_cast<double>(state));
  // Recompute active_adapters_count_ (state == 1) via a full pass. This
  // is O(N adapters) but state changes are rare (load / drain).
  int64_t active = 0;
  for (const auto& [_, bv] : per_adapter_) {
    if (bv.active_state && bv.active_state->get_value() == 1.0) ++active;
  }
  if (active_adapters_count_) active_adapters_count_->set_value(active);
}

void LoRAMetrics::set_device_slots(uint64_t adapter_id, int64_t slots) {
  if (adapter_id == 0) return;
  std::shared_lock lock(mu_);
  std::string name = name_of(id_to_name_, adapter_id);
  if (name.empty()) return;
  auto it = per_adapter_.find(name);
  if (it == per_adapter_.end() || !it->second.device_slots) return;
  it->second.device_slots->set_value(static_cast<double>(slots));
}

void LoRAMetrics::inc_requests_info(uint64_t adapter_id,
                                    const std::string& state) {
  if (adapter_id == 0) return;
  std::shared_lock lock(mu_);
  std::string name = name_of(id_to_name_, adapter_id);
  if (name.empty() || !requests_info_) return;
  auto* adder = requests_info_->get_stats({name, state});
  if (adder) *adder << 1;
}

void LoRAMetrics::dec_requests_info(uint64_t adapter_id,
                                    const std::string& state) {
  if (adapter_id == 0) return;
  std::shared_lock lock(mu_);
  std::string name = name_of(id_to_name_, adapter_id);
  if (name.empty() || !requests_info_) return;
  auto* adder = requests_info_->get_stats({name, state});
  if (adder) *adder << -1;
}

std::vector<LoRAMetrics::Snapshot> LoRAMetrics::snapshot_all() const {
  std::vector<Snapshot> out;
  std::shared_lock lock(mu_);
  out.reserve(per_adapter_.size());
  for (const auto& [name, bv] : per_adapter_) {
    Snapshot s;
    s.lora_name = name;
    if (bv.requests_total) s.requests_total = bv.requests_total->get_value();
    if (bv.tokens_prompt_total)
      s.tokens_prompt_total = bv.tokens_prompt_total->get_value();
    if (bv.tokens_generated_total)
      s.tokens_generated_total = bv.tokens_generated_total->get_value();
    if (bv.errors_total) s.errors_total = bv.errors_total->get_value();
    if (bv.active_state) s.active_state = bv.active_state->get_value();
    if (bv.device_slots) s.device_slots = bv.device_slots->get_value();
    // Latency percentiles: read the MultiDimension sub-recorder if it
    // materialised (i.e. at least one observation happened). Skip on miss.
    if (ttft_ms_) {
      auto* r = ttft_ms_->get_stats({name});
      if (r) {
        s.ttft_p50_ms = r->latency_percentile(0.5);
        s.ttft_p99_ms = r->latency_percentile(0.99);
      }
    }
    if (e2e_ms_) {
      auto* r = e2e_ms_->get_stats({name});
      if (r) {
        s.e2e_p50_ms = r->latency_percentile(0.5);
        s.e2e_p99_ms = r->latency_percentile(0.99);
        s.qps = r->qps();
      }
    }
    out.push_back(std::move(s));
  }
  return out;
}

size_t LoRAMetrics::adapter_count() const {
  std::shared_lock lock(mu_);
  return per_adapter_.size();
}

// W4-A4: observe a scheduler batch's LoRA distribution. Fast path is
// (0 or 1) distinct adapters, slow path is 2+. total_seqs unused today
// but included for future hit-rate weighting.
void LoRAMetrics::observe_batch_lora(size_t distinct_count, size_t total_seqs) {
  (void)total_seqs;
  if (batch_distinct_adapters_) {
    (*batch_distinct_adapters_) << static_cast<int64_t>(distinct_count);
  }
  if (distinct_count <= 1) {
    if (batch_fast_path_total_) (*batch_fast_path_total_) << 1;
  } else {
    if (batch_slow_path_total_) (*batch_slow_path_total_) << 1;
  }
}

}  // namespace xllm
