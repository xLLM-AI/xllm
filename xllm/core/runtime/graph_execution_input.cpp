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

#include "core/runtime/graph_execution_input.h"

namespace xllm {

std::optional<std::pair<torch::Tensor, torch::Tensor>>
find_first_full_attention_cache(const std::vector<KVCache>& kv_caches) {
  for (const auto& cache : kv_caches) {
    if (cache.empty()) {
      continue;
    }
    auto k_cache = cache.get_k_cache();
    auto v_cache = cache.get_v_cache();
    if (k_cache.defined() && v_cache.defined() && k_cache.numel() > 0 &&
        v_cache.numel() > 0) {
      return std::make_pair(std::move(k_cache), std::move(v_cache));
    }
  }
  return std::nullopt;
}

}  // namespace xllm
