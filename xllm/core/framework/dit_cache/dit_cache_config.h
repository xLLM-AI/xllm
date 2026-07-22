/* Copyright 2025-2026 The xLLM Authors.

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
#include <cstdint>
#include <string>

namespace xllm {

enum class PolicyType {
  None,
  FBCache,
  TaylorSeer,
  FBCacheTaylorSeer,
  ResidualCache,
  RACFGCache
};

struct DiTBaseCacheOptions {
  // the number of warmup steps.
  int warmup_steps = 0;
};

struct FBCacheOptions : public DiTBaseCacheOptions {
  // the residual difference threshold for cache reuse.
  float residual_diff_threshold = 0.09f;
};

struct TaylorSeerOptions : public DiTBaseCacheOptions {
  // the number of derivatives to use in TaylorSeer.
  int n_derivatives = 3;

  // the interval steps to skip for derivative calculation.
  int skip_interval_steps = 3;
};

struct FBCacheTaylorSeerOptions : public DiTBaseCacheOptions {
  // the residual difference threshold for cache reuse.
  float residual_diff_threshold = 0.09f;

  // the number of derivatives to use in TaylorSeer.
  int n_derivatives = 3;
};

struct ResidualCacheOptions {
  // The number of steps to skip at the start.
  int64_t dit_cache_start_steps = 5;

  // The number of steps to skip at the end.
  int64_t dit_cache_end_steps = 5;

  // The number of blocks to skip at the start.
  int64_t dit_cache_start_blocks = 5;

  // The number of blocks to skip at the end.
  int64_t dit_cache_end_blocks = 5;

  // the interval steps to skip for derivative calculation.
  int64_t skip_interval_steps = 3;
};

struct RACFGCacheOptions : public DiTBaseCacheOptions {
  // Number of blocks to run before making the joint decision.
  int64_t probe_depth = 2;

  // Joint accumulated-risk threshold.
  float tau = 0.0f;

  // True CFG scale used in the guided combination.
  float true_cfg_scale = 3.0f;

  // Whether to use propagation-aware reweighting.
  bool use_prop_weight = true;

  // Propagation-aware fitted parameters.
  float prop_a = 0.4806166f;
  float prop_alpha = 0.4782565f;
  float prop_b = 0.0641170f;

  // Branch-local proxy error choice:
  //   0 -> delta_y
  //   1 -> delta_minus
  int64_t proxy_error_type = 0;

  // Offline rho table path.
  std::string rho_table_path = "";
  std::string model_name = "";
  // Optional matched CFG scale recorded with the rho table.
  float matched_cfg_scale = -1.0f;
};

struct DiTCacheConfig {
  DiTCacheConfig() = default;

  // the selected cache policy.
  PolicyType selected_policy = PolicyType::None;

  // the configuration for FBCache policy.
  FBCacheOptions fbcache;

  // the configuration for TaylorSeer policy.
  TaylorSeerOptions taylorseer;

  // the configuration for combined FBCache with TaylorSeer policy.
  FBCacheTaylorSeerOptions fbcachetaylorseer;

  // the configuration for ResidualCache policy.
  ResidualCacheOptions residual_cache;

  // the configuration for RACFGCache policy.
  RACFGCacheOptions racfgcache;
};

}  // namespace xllm
