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

#pragma once

#include <torch/types.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/kv_cache/layout/virtual_kv_cache_page_layout.h"

namespace xllm {

class KVCacheCapacity;
class ModelArgs;

// Tensors sharing one scheduler block ID space. A speculative pair combines
// its target and draft tensors before calculating their physical page budget.
class VmmKVCacheLayout final {
 public:
  VmmKVCacheLayout(const KVCacheCapacity& capacity,
                   const ModelArgs& model_args,
                   int64_t world_size,
                   torch::ScalarType dtype,
                   bool is_draft_engine = false);

  void append(VmmKVCacheLayout layout);

  VirtualKVCachePageLayout page_layout(size_t page_size) const;
  size_t physical_pages(size_t page_size) const;
  size_t block_capacity(size_t physical_page_budget, size_t page_size) const;
  void validate_pd_page_size(size_t page_size) const;

 private:
  std::vector<size_t> tensor_block_bytes_;
  std::vector<size_t> tensor_counts_;
};

}  // namespace xllm
