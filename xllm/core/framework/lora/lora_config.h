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

#pragma once

#include <gflags/gflags.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace xllm {

// ---------------------------------------------------------------------------
// LoRA runtime configuration.
//
// Mirrors the vLLM CLI surface (see reference_multilora_engines memory) so
// the same launch flags work on business gateways that already integrate
// with vLLM.
//
// The struct is populated once at startup from gflags and then held on the
// LLMEngine as an immutable read-only view. Adapter registration / eviction
// happens inside LoRARegistry / LoraCacheManager, not here.
// ---------------------------------------------------------------------------
struct LoRAConfig {
  // Master switch. When false, all LoRA APIs report 400 and the model runs
  // exactly as baseline (no delta path).
  bool enable_lora = false;

  // Maximum number of adapters that may be materialised on a single device
  // simultaneously. Directly caps the GPU-side slot pool. vLLM default 4;
  // we set 16 since business is 5-20 tenants.
  int32_t max_loras = 16;

  // Maximum number of adapters the CPU-pinned pool can hold. Should be a
  // superset of max_loras so cache hits are cheap when a request lands on
  // an adapter that was recently evicted from GPU.
  int32_t max_cpu_loras = 64;

  // Upper bound on r for any adapter. Weights above this are refused at
  // load time. Keeps per-slot memory predictable.
  int32_t max_lora_rank = 32;

  // Comma-separated module names that adapters may target. The loader
  // rejects an adapter whose adapter_config.json target_modules is not a
  // subset of this. Kept as a std::vector for cheap membership checks.
  std::vector<std::string> lora_target_modules = {
      "qkv_proj",
      "o_proj",
      "gate_up_proj",
      "down_proj",
      "q_proj",
      "k_proj",
      "v_proj",
      "gate_proj",
      "up_proj",
  };

  // Static adapters registered at startup, one per name=path pair. Each
  // pair is loaded before the HTTP server starts serving so /v1/models
  // returns them immediately.
  std::vector<std::pair<std::string, std::string>> lora_modules;

  // When true, POST /v1/load_lora_adapter is accepted at runtime. When
  // false, only the static --lora-modules set is served.
  bool allow_runtime_lora_updating = true;

  // Parse the raw --lora-modules CLI value ("name1=path1,name2=path2") into
  // the vector form. Bad tokens are logged and skipped.
  static std::vector<std::pair<std::string, std::string>> parse_modules(
      const std::string& raw);

  // Fill this instance from the current gflags values. Idempotent.
  void load_from_flags();
};

}  // namespace xllm

// gflags declarations. The DEFINE_* live in lora_config.cpp so the ODR
// stays consistent with the rest of xllm.
DECLARE_bool(enable_lora);
DECLARE_int32(max_loras);
DECLARE_int32(max_cpu_loras);
DECLARE_int32(max_lora_rank);
DECLARE_string(lora_target_modules);
DECLARE_string(lora_modules);
DECLARE_bool(allow_runtime_lora_updating);
DECLARE_bool(enable_lora_row_parallel_all_reduce);
DECLARE_bool(enable_lora_row_parallel_fused_ar);
