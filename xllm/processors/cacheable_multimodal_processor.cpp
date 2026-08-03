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

#include "processors/cacheable_multimodal_processor.h"

#include <glog/logging.h>

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace xllm {

CacheableMultimodalProcessor::CacheableMultimodalProcessor(
    std::unique_ptr<MultimodalProcessorBase> inner,
    int64_t max_cache_items)
    : MultimodalProcessorBase(/*tokenizer=*/nullptr),
      inner_(std::move(inner)),
      cache_(std::make_unique<ProcessorCache>(max_cache_items)) {
  CHECK(inner_ != nullptr);
}

bool CacheableMultimodalProcessor::process_prompt(
    std::string& prompt,
    MMData& mm_data,
    std::vector<int32_t>& token_ids) {
  return inner_->process_prompt(prompt, mm_data, token_ids);
}

bool CacheableMultimodalProcessor::process_multimodal(const MMInput& inputs,
                                                      MMData& data) const {
  ProcessorCacheLookupVisitor cache_lookup_visitor(*cache_, inputs.size());
  CHECK(inputs.foreach (cache_lookup_visitor));

  MMItemVec miss_items;
  if (!process_misses(cache_lookup_visitor.miss_inputs_, miss_items)) {
    return false;
  }

  assemble(cache_lookup_visitor.cache_hits_, std::move(miss_items), data);
  return true;
}

bool CacheableMultimodalProcessor::process_misses(
    const std::vector<MMInputItem>& miss_inputs,
    MMItemVec& miss_items) const {
  if (miss_inputs.empty()) {
    return true;
  }

  MMInput inputs;
  inputs.insert(miss_inputs);
  MMData miss_data;
  if (!inner_->process_multimodal(inputs, miss_data)) {
    return false;
  }
  CHECK_EQ(miss_data.items<MMItemVec>().size(), miss_inputs.size())
      << "Multimodal processor returned mismatched item count.";
  ProcessorCacheInsertVisitor insert(*cache_);
  CHECK(miss_data.foreach (insert));
  miss_items = std::move(miss_data.items<MMItemVec>());
  return true;
}

void CacheableMultimodalProcessor::assemble(
    std::vector<std::optional<MMDataItem>>& cache_hits,
    MMItemVec miss_items,
    MMData& data) const {
  uint32_t full_type = MMType::NONE;
  MMItemVec full_items;
  full_items.reserve(cache_hits.size());

  size_t miss_index = 0;
  for (std::optional<MMDataItem>& cache_hit : cache_hits) {
    if (cache_hit.has_value()) {
      MMDataItem& item = cache_hit.value();
      full_type |= item.type();
      full_items.emplace_back(std::move(item));
      continue;
    }

    MMDataItem& produced = miss_items[miss_index++];
    full_type |= produced.type();
    full_items.emplace_back(std::move(produced));
  }
  CHECK_EQ(miss_index, miss_items.size());

  data.set(full_type, std::move(full_items));
}

}  // namespace xllm
