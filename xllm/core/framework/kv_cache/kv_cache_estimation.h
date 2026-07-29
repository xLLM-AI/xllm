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

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

#include "framework/kv_cache/kv_cache_capacity.h"

namespace xllm {

class ModelArgs;

struct KVCacheEstimateOptions {
  torch::ScalarType dtype = torch::kBFloat16;
  std::string kv_cache_dtype = "auto";
  std::string indexer_cache_dtype = "auto";
  int64_t cache_size_in_bytes = 0;
  int64_t block_size = 0;
  int64_t world_size = 1;
  int64_t n_local_kv_heads = 0;
  int64_t n_local_linear_k_heads = 0;
  int64_t n_local_linear_v_heads = 0;
  int64_t max_seqs_per_batch = 0;
  int64_t num_speculative_tokens = 0;
  int64_t max_tokens_per_batch = 0;
  int64_t max_linear_state_cache_slots = 0;
  bool is_draft_engine = false;
  bool enable_prefix_cache = false;
  bool enable_rdma_scale_padding = false;
  const ModelArgs* draft_model_args = nullptr;
  const KVCacheEstimateOptions* draft_options = nullptr;
};

struct Dsv4KVCacheEstimateCost {
  int64_t swa_count = 0;
  int64_t n_c4_layers = 0;
  int64_t n_c128_layers = 0;
  int64_t constant_swa_bytes = 0;
  int64_t token_unit_bytes = 0;
  int64_t manager_blocks_per_unit = 1;
};

std::vector<bool> resolve_indexer_cache_enabled_layers(
    const ModelArgs& model_args,
    int64_t num_cache_layers);

KVCacheCapacity estimate_kv_cache_capacity(
    const ModelArgs& model_args,
    const KVCacheEstimateOptions& options);

}  // namespace xllm
