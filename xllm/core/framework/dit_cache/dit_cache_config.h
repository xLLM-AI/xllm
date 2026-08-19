/* Copyright 2025-2026 The xLLM Authors.

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

namespace xllm {

enum class PolicyType {
  None,
  FBCache,
  TaylorSeer,
  FBCacheTaylorSeer,
  ResidualCache
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
};

}  // namespace xllm
