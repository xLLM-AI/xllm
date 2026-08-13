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
  return inner_->process_multimodal(inputs, data);
}

bool CacheableMultimodalProcessor::process_mm_input(
    const std::vector<Message>& messages,
    std::string payload,
    MMData& out) {
  MMInput mm_inputs(std::move(payload));
  std::vector<MMSourceRef> refs;
  if (transfer_.collect(messages, mm_inputs, refs) != MMErrCode::SUCCESS) {
    return false;
  }
  if (mm_inputs.empty()) {
    return true;
  }

  UuidPrefilterVisitor prefilter(*cache_, mm_inputs.size());
  CHECK(mm_inputs.foreach (prefilter));

  if (transfer_.materialize(refs, prefilter.miss_indices_, mm_inputs) !=
      MMErrCode::SUCCESS) {
    return false;
  }

  const std::vector<MMInputItem>& items = mm_inputs.items();
  MMInput uuid_misses;
  for (int32_t index : prefilter.miss_indices_) {
    uuid_misses.insert(items[index]);
  }
  ProcessorCacheLookupVisitor raw_hash_lookup(*cache_, uuid_misses.size());
  CHECK(uuid_misses.foreach (raw_hash_lookup));
  CHECK_EQ(raw_hash_lookup.cache_hits_.size(), prefilter.miss_indices_.size());

  MMData produced_data;
  if (!raw_hash_lookup.miss_inputs_.empty()) {
    MMInput preprocess_inputs;
    preprocess_inputs.insert(raw_hash_lookup.miss_inputs_);
    if (!inner_->process_multimodal(preprocess_inputs, produced_data)) {
      return false;
    }
    CHECK_EQ(produced_data.items<MMItemVec>().size(),
             raw_hash_lookup.miss_inputs_.size());
    ProcessorCacheInsertVisitor insert(*cache_);
    CHECK(produced_data.foreach (insert));
  }

  MMItemVec fresh_items;
  if (produced_data.hold<MMItemVec>()) {
    fresh_items = std::move(produced_data.items<MMItemVec>());
  }
  assemble(prefilter, raw_hash_lookup, fresh_items, out);
  return true;
}

void CacheableMultimodalProcessor::assemble(
    UuidPrefilterVisitor& prefilter,
    ProcessorCacheLookupVisitor& raw_hash_lookup,
    MMItemVec& fresh_items,
    MMData& out) const {
  const int32_t total = static_cast<int32_t>(prefilter.hit_indices_.size() +
                                             prefilter.miss_indices_.size());
  MMItemVec slots(total, MMDataItem(MMType::NONE));
  uint32_t full_type = MMType::NONE;

  for (size_t i = 0; i < prefilter.hit_indices_.size(); ++i) {
    MMDataItem& item = prefilter.hit_items_[i];
    full_type |= item.type();
    slots[prefilter.hit_indices_[i]] = std::move(item);
  }

  size_t fresh_index = 0;
  for (size_t j = 0; j < prefilter.miss_indices_.size(); ++j) {
    const int32_t global_index = prefilter.miss_indices_[j];
    std::optional<MMDataItem>& cache_hit = raw_hash_lookup.cache_hits_[j];
    MMDataItem& item =
        cache_hit.has_value() ? cache_hit.value() : fresh_items[fresh_index++];
    full_type |= item.type();
    slots[global_index] = std::move(item);
  }
  CHECK_EQ(fresh_index, fresh_items.size());

  out = MMData(full_type, std::move(slots));
}

}  // namespace xllm
