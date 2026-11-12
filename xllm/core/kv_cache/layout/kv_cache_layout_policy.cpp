/* Copyright 2026 The xLLM Authors.

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

#include "core/kv_cache/layout/kv_cache_layout_policy.h"

#include <algorithm>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/platform/platform.h"

namespace xllm {

bool is_linear_attention_layer(int64_t layer_idx,
                               int64_t full_attention_interval,
                               const std::vector<std::string>& layer_types) {
  if (!layer_types.empty()) {
    if (layer_idx < 0 ||
        layer_idx >= static_cast<int64_t>(layer_types.size())) {
      return false;
    }
    const std::string& layer_type = layer_types[static_cast<size_t>(layer_idx)];
    // Explicit full-attention markers (including the glm5_next
    // deepseek_sparse_attention label) use paged KV, not the linear
    // (conv/ssm) cache.
    if (is_paged_kv_cache_layer(layer_type)) {
      return false;
    }
    // Only an explicit linear marker flips the layer to the conv/ssm cache.
    // Unknown tags defer to the interval heuristic below instead of guessing
    // linear — a wrong guess flips the whole KV layout and corrupts serving.
    if (layer_type == "linear_attention") {
      return true;
    }
  }
  if (full_attention_interval <= 1) {
    return false;
  }
  return (layer_idx + 1) % full_attention_interval != 0;
}

bool use_npu_nz_kv_cache_layout(const std::string& model_type) {
  return (model_type == "deepseek_v3" || model_type == "deepseek_v3_mtp") &&
         ::xllm::KVCacheConfig::get_instance().enable_prefix_cache();
}

int64_t num_index_pages_per_block() {
  if (!Platform::supports_dsa_indexer_cache_sharding()) {
    return 1;
  }
  return std::max<int64_t>(
      ParallelConfig::get_instance().kv_split_size_effective(), /*b=*/1);
}

}  // namespace xllm
