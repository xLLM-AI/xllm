/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "lora_config.h"

#include <glog/logging.h>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Master switch. When off, no other LoRA flag has any effect and the LoRA
// HTTP endpoints return 400.
DEFINE_bool(enable_lora,
            false,
            "Enable multi-tenant LoRA serving. When false, the engine runs as "
            "a plain single-model server.");

// Number of adapters that can live on GPU simultaneously. Directly caps the
// batch-level active adapter count.
DEFINE_int32(max_loras,
             16,
             "Maximum number of LoRA adapters kept on device at the same "
             "time. Also caps how many distinct adapters can appear in one "
             "batch.");

// Size of the pinned CPU pool. Should be a proper superset of max_loras so
// that reload from CPU is much cheaper than reload from disk.
DEFINE_int32(max_cpu_loras,
             64,
             "Maximum number of LoRA adapters kept in the pinned CPU pool.");

// Ceiling on r. Loads with r > this are refused. Chosen so 8 / 16 / 32 all
// fit; the largest common practical rank in production is 32.
DEFINE_int32(max_lora_rank, 32, "Upper bound on any single adapter's rank r.");

// Whitelist of PEFT target_modules the engine will accept. HuggingFace
// adapters that name modules outside this set are refused so we never
// silently ignore a Q/K/V-only vs full attention mismatch.
DEFINE_string(
    lora_target_modules,
    "qkv_proj,o_proj,gate_up_proj,down_proj,q_proj,k_proj,v_proj,gate_proj,up_"
    "proj",
    "Comma-separated whitelist of PEFT target_modules that adapters may "
    "target. Anything else is refused at load time.");

// name=path,name=path. Static adapters registered before the server begins
// serving so /v1/models is populated on first request.
DEFINE_string(
    lora_modules,
    "",
    "Comma-separated list of static adapters: name1=path1,name2=path2. "
    "Each is loaded before the HTTP server starts.");

// If false, only the static set is available and POST /v1/load_lora_adapter
// returns 403. Useful for tenants that want a frozen deployment.
DEFINE_bool(
    allow_runtime_lora_updating,
    true,
    "Whether HTTP endpoints /v1/load_lora_adapter and /v1/unload_lora_adapter "
    "are honoured at runtime.");

// Row-parallel LoRA (o_proj / down_proj) requires an extra rank-dim
// all-reduce per layer per forward to keep the delta TP-correct. Under NPU
// HCCL this launch cost is non-trivial and can subtract ~15-20% throughput
// even though the bytes moved are tiny. Set to false for pure-tps deployments
// where the adapters are known to target only column-parallel projections
// (q_proj / k_proj / v_proj / gate_proj / up_proj); the wrapper then skips
// the collective and the row-parallel delta silently no-ops (same behaviour
// as pre-fix). Default true because most PEFT adapters do include o_proj /
// down_proj and precision improvement of ~9pp compliance typically outweighs
// the throughput cost.
DEFINE_bool(
    enable_lora_row_parallel_all_reduce,
    true,
    "Whether LoRARowParallelLinear applies its delta under TP>1 via an extra "
    "rank-dim all-reduce. Setting false restores pre-fix behaviour: row-"
    "parallel LoRA deltas (o_proj, down_proj) are silently skipped and only "
    "column-parallel deltas (q/k/v_proj, gate/up_proj) apply. Trades ~+9pp "
    "compliance-rate precision for ~+17% throughput headroom.");

// Fused variant: sum the LoRA delta into the base's partial [T, out] BEFORE
// the base row-parallel's all-reduce, so both base output and delta ride the
// same collective. Cuts per-layer AR count from 2 (rank-dim + base) to 1
// (base only), same as vLLM RowParallelLinearWithShardedLoRA / LoRAX /
// HF-TGI. Requires B to be replicated on every rank and A to be sliced on
// in-dim (matches base's shard layout).
//
// When true, this SUPERSEDES enable_lora_row_parallel_all_reduce — the base
// linear is constructed with enable_result_reduction=false and the wrapper
// owns the collective for the fused output.
//
// Default false while we validate. Enabling it should reclaim most of the
// ~17% throughput cost of enable_lora_row_parallel_all_reduce=true without
// giving up the +9pp precision benefit.
DEFINE_bool(
    enable_lora_row_parallel_fused_ar,
    false,
    "Fuse LoRA row-parallel delta into the base's all-reduce (S-LoRA / vLLM "
    "RowParallelLinearWithShardedLoRA pattern). Cuts per-layer AR count "
    "from 2 to 1. Supersedes enable_lora_row_parallel_all_reduce when true.");

namespace xllm {

namespace {
// Split "a,b,,c" into {"a", "b", "c"}. Empty tokens are dropped so a
// dangling trailing comma is tolerated.
std::vector<std::string> split_comma(const std::string& raw) {
  std::vector<std::string> out;
  std::string tok;
  std::stringstream ss(raw);
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) {
      out.push_back(tok);
    }
  }
  return out;
}
}  // namespace

std::vector<std::pair<std::string, std::string>> LoRAConfig::parse_modules(
    const std::string& raw) {
  std::vector<std::pair<std::string, std::string>> out;
  for (const auto& tok : split_comma(raw)) {
    // Accept both name=path (vLLM style) and name:path (business style).
    // Whichever separator appears FIRST wins so paths containing the other
    // char still parse.
    size_t sep = std::string::npos;
    for (size_t i = 0; i < tok.size(); ++i) {
      if (tok[i] == '=' || tok[i] == ':') {
        sep = i;
        break;
      }
    }
    if (sep == std::string::npos || sep == 0 || sep == tok.size() - 1) {
      LOG(ERROR) << "[LoRAConfig] malformed --lora-modules entry '" << tok
                 << "', expected name=path or name:path";
      continue;
    }
    out.emplace_back(tok.substr(0, sep), tok.substr(sep + 1));
  }
  return out;
}

void LoRAConfig::load_from_flags() {
  enable_lora = FLAGS_enable_lora;
  max_loras = FLAGS_max_loras;
  max_cpu_loras = FLAGS_max_cpu_loras;
  max_lora_rank = FLAGS_max_lora_rank;
  lora_target_modules = split_comma(FLAGS_lora_target_modules);
  lora_modules = parse_modules(FLAGS_lora_modules);
  allow_runtime_lora_updating = FLAGS_allow_runtime_lora_updating;

  if (enable_lora) {
    LOG(INFO) << "[LoRAConfig] enabled: max_loras=" << max_loras
              << " max_cpu_loras=" << max_cpu_loras
              << " max_lora_rank=" << max_lora_rank
              << " target_modules=" << lora_target_modules.size()
              << " static_modules=" << lora_modules.size()
              << " runtime_updates="
              << (allow_runtime_lora_updating ? "on" : "off");
  }
}

}  // namespace xllm
