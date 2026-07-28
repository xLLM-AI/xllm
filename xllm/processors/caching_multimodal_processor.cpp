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
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace xllm {
namespace {

std::optional<XXH3Key> cache_key_for_content(const MMContent& content) {
  if (!is_url_type(content.type) || content.uuid.empty()) {
    return std::nullopt;
  }
  return hash_string(content.uuid);
}

bool all_contents_uuid_cacheable(const std::vector<Message>& messages) {
  for (const Message& message : messages) {
    const MMContentVec& contents = std::get<MMContentVec>(message.content);
    for (const MMContent& content : contents) {
      if (content.type == "text") {
        continue;
      }
      if (!cache_key_for_content(content).has_value()) {
        return false;
      }
    }
  }
  return true;
}

void assemble_in_order(std::vector<std::optional<MMDataItem>> cached,
                       MMItemVec produced,
                       MMData& data) {
  uint32_t full_type = MMType::NONE;
  MMItemVec full_items;
  full_items.reserve(cached.size());
  size_t produced_index = 0;
  for (std::optional<MMDataItem>& hit : cached) {
    MMDataItem item = hit.has_value() ? std::move(hit.value())
                                      : std::move(produced[produced_index++]);
    full_type |= item.type();
    full_items.emplace_back(std::move(item));
  }
  CHECK_EQ(produced_index, produced.size());
  data.set(full_type, std::move(full_items));
}

}  // namespace

CachingMultimodalProcessor::CachingMultimodalProcessor(
    std::unique_ptr<MultimodalProcessorBase> inner,
    int64_t max_cache_items)
    : MultimodalProcessorBase(/*tokenizer=*/nullptr),
      inner_(std::move(inner)),
      cache_(std::make_unique<ProcessorCache>(max_cache_items)) {
  CHECK(inner_ != nullptr);
}

MMErrCode CachingMultimodalProcessor::process_multimodal_request(
    const std::vector<Message>& messages,
    std::string payload,
    MMData& data) {
  if (all_contents_uuid_cacheable(messages)) {
    return process_by_uuid(messages, std::move(payload), data);
  }
  return process_by_content(messages, std::move(payload), data);
}

MMErrCode CachingMultimodalProcessor::process_by_uuid(
    const std::vector<Message>& messages,
    std::string payload,
    MMData& data) {
  std::vector<std::optional<MMDataItem>> cached_items;
  std::vector<bool> selected_items;
  std::vector<XXH3Key> miss_keys;
  for (const Message& message : messages) {
    const MMContentVec& contents = std::get<MMContentVec>(message.content);
    for (const MMContent& content : contents) {
      if (content.type == "text") {
        continue;
      }

      const XXH3Key key = cache_key_for_content(content).value();
      std::optional<MMDataItem> cached = cache_->lookup(key);
      if (cached.has_value()) {
        cached->mutable_state().mutable_schedule_data().key = key;
      } else {
        miss_keys.emplace_back(key);
      }
      selected_items.emplace_back(!cached.has_value());
      cached_items.emplace_back(std::move(cached));
    }
  }

  MMItemVec processed_items;
  if (!miss_keys.empty()) {
    MMInput inputs(std::move(payload));
    MMErrCode code = input_transfer_.trans(messages, selected_items, inputs);
    if (code != MMErrCode::SUCCESS) {
      return code;
    }
    CHECK_EQ(inputs.size(), miss_keys.size());

    MMData processed_data;
    if (!inner_->process_multimodal(inputs, processed_data)) {
      return MMErrCode::PROCESS_ERR;
    }
    processed_items = std::move(processed_data.items<MMItemVec>());
    CHECK_EQ(processed_items.size(), miss_keys.size());

    for (size_t index = 0; index < processed_items.size(); ++index) {
      cache_->insert(miss_keys[index], processed_items[index]);
    }
  }

  assemble_in_order(std::move(cached_items), std::move(processed_items), data);
  return MMErrCode::SUCCESS;
}

MMErrCode CachingMultimodalProcessor::process_by_content(
    const std::vector<Message>& messages,
    std::string payload,
    MMData& data) {
  MMInput inputs(std::move(payload));
  MMErrCode code = input_transfer_.trans(messages, inputs);
  if (code != MMErrCode::SUCCESS) {
    return code;
  }
  if (!process_multimodal(inputs, data)) {
    return MMErrCode::PROCESS_ERR;
  }
  return MMErrCode::SUCCESS;
}

bool CachingMultimodalProcessor::process_prompt(
    std::string& prompt,
    MMData& mm_data,
    std::vector<int32_t>& token_ids) {
  return inner_->process_prompt(prompt, mm_data, token_ids);
}

bool CachingMultimodalProcessor::process_multimodal(const MMInput& inputs,
                                                    MMData& data) const {
  CacheProbeVisitor probe(*cache_);
  inputs.foreach (probe);

  MMItemVec miss_items;
  if (!probe.miss_input_items_.empty()) {
    MMInput miss_inputs;
    miss_inputs.insert(probe.miss_input_items_);
    MMData miss_data;
    if (!inner_->process_multimodal(miss_inputs, miss_data)) {
      return false;
    }
    CHECK_EQ(miss_data.items<MMItemVec>().size(),
             probe.miss_input_items_.size())
        << "Multimodal processor returned mismatched item count.";
    miss_items = std::move(miss_data.items<MMItemVec>());
  }

  for (size_t index = 0; index < miss_items.size(); ++index) {
    if (probe.miss_keys_[index].has_value()) {
      cache_->insert(probe.miss_keys_[index].value(), miss_items[index]);
    }
  }

  assemble_in_order(
      std::move(probe.cached_items_), std::move(miss_items), data);
  return true;
}

}  // namespace xllm
