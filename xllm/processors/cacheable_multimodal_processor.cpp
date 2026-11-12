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

#include "processors/cacheable_multimodal_processor.h"

#include <glog/logging.h>

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace xllm {
namespace {

class ProcessorCacheLookupVisitor final : public MMInputItem::IVisitor {
 public:
  ProcessorCacheLookupVisitor(ProcessorCache& cache, size_t item_count);
  bool visit(const MMInputItem& input) override;

  std::vector<std::optional<MMDataItem>> cache_hits_;
  std::vector<MMInputItem> miss_inputs_;

 private:
  ProcessorCache& cache_;
};

class ProcessorCacheInsertVisitor final : public MMDataItem::IVisitor {
 public:
  explicit ProcessorCacheInsertVisitor(ProcessorCache& cache);
  bool visit(MMDataItem& item) override;

 private:
  ProcessorCache& cache_;
};

ProcessorCacheLookupVisitor::ProcessorCacheLookupVisitor(ProcessorCache& cache,
                                                         size_t item_count)
    : cache_(cache) {
  cache_hits_.reserve(item_count);
  miss_inputs_.reserve(item_count);
}

bool ProcessorCacheLookupVisitor::visit(const MMInputItem& input) {
  std::optional<MMDataItem> cache_hit;
  if (!input.is_embedding() && input.hash_key.has_value()) {
    cache_hit = cache_.lookup(input.hash_key.value());
  }
  if (!cache_hit.has_value()) {
    miss_inputs_.emplace_back(input);
  }
  cache_hits_.emplace_back(std::move(cache_hit));
  return true;
}

ProcessorCacheInsertVisitor::ProcessorCacheInsertVisitor(ProcessorCache& cache)
    : cache_(cache) {}

bool ProcessorCacheInsertVisitor::visit(MMDataItem& item) {
  if (!item.is_embedded()) {
    cache_.insert(item.state().schedule_data().key, item);
  }
  return true;
}

}  // namespace

CacheableMultimodalProcessor::CacheableMultimodalProcessor(
    std::unique_ptr<MultimodalProcessorBase> inner,
    int64_t max_cache_items)
    : inner_(std::move(inner)),
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
  inputs.foreach (cache_lookup_visitor);

  MMItemVec miss_items;
  MMInput miss_inputs(std::move(cache_lookup_visitor.miss_inputs_));
  if (!miss_inputs.empty()) {
    MMData miss_data;
    if (!inner_->process_multimodal(miss_inputs, miss_data)) {
      return false;
    }
    CHECK_EQ(miss_data.items<MMItemVec>().size(), miss_inputs.size())
        << "Multimodal processor returned mismatched item count.";
    ProcessorCacheInsertVisitor insert(*cache_);
    CHECK(miss_data.foreach (insert));
    miss_items = std::move(miss_data.items<MMItemVec>());
  }
  assemble(cache_lookup_visitor.cache_hits_, std::move(miss_items), data);
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

  data.set(full_type, std::move(full_items));
}

}  // namespace xllm
