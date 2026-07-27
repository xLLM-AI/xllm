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

#include "processors/caching_multimodal_processor.h"

#include <glog/logging.h>

#include <cstddef>
#include <optional>
#include <utility>

#include "util/hash_util.h"

namespace xllm {

CachingMultimodalProcessor::CachingMultimodalProcessor(
    std::unique_ptr<MultimodalProcessorBase> inner,
    int64_t max_cache_items)
    : MultimodalProcessorBase(/*tokenizer=*/nullptr),
      inner_(std::move(inner)),
      cache_(max_cache_items) {
  CHECK(inner_ != nullptr);
}

bool CachingMultimodalProcessor::process_prompt(
    std::string& prompt,
    MMData& mm_data,
    std::vector<int32_t>& token_ids) {
  return inner_->process_prompt(prompt, mm_data, token_ids);
}

bool CachingMultimodalProcessor::process_multimodal(const MMInput& inputs,
                                                    MMData& data) const {
  const std::vector<MMInputItem>& input_items = inputs.items();
  const size_t input_size = input_items.size();
  std::vector<bool> cacheable(input_size, false);
  std::vector<XXH3Key> cache_keys(input_size);
  std::vector<std::optional<MMDataItem>> cached_items(input_size);
  std::vector<MMInputItem> miss_input_items;
  miss_input_items.reserve(input_size);

  for (size_t index = 0; index < input_size; ++index) {
    const MMInputItem& input_item = input_items[index];
    cacheable[index] =
        !input_item.raw_data.empty() && !input_item.is_embedding();
    if (cacheable[index]) {
      cache_keys[index] = hash_string(input_item.raw_data);
      std::optional<MMDataItem> cached = cache_.lookup(cache_keys[index]);
      if (cached.has_value()) {
        cached->mutable_state().mutable_schedule_data().key = cache_keys[index];
        cached_items[index] = std::move(cached.value());
        continue;
      }
    }
    miss_input_items.emplace_back(input_item);
  }

  MMItemVec miss_items;
  if (!miss_input_items.empty()) {
    MMInput miss_inputs;
    miss_inputs.insert(miss_input_items);
    MMData miss_data;
    if (!inner_->process_multimodal(miss_inputs, miss_data)) {
      return false;
    }
    CHECK(miss_data.hold<MMItemVec>());
    CHECK_EQ(miss_data.items<MMItemVec>().size(), miss_input_items.size())
        << "Multimodal processor returned mismatched item count.";
    miss_items = std::move(miss_data.items<MMItemVec>());
  }

  uint32_t full_type = MMType::NONE;
  MMItemVec full_items;
  full_items.reserve(input_size);
  size_t miss_index = 0;
  for (size_t index = 0; index < input_size; ++index) {
    const bool cache_hit = cached_items[index].has_value();
    MMDataItem item = cache_hit ? std::move(cached_items[index].value())
                                : std::move(miss_items[miss_index++]);
    if (cacheable[index] && !cache_hit) {
      cache_.insert(cache_keys[index], item);
    }
    full_type |= item.type();
    full_items.emplace_back(std::move(item));
  }
  CHECK_EQ(miss_index, miss_items.size());

  data.set(full_type, std::move(full_items));
  return true;
}

}  // namespace xllm
