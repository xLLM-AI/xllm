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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/common/message.h"
#include "core/framework/multimodal/mm_visitor.h"
#include "core/framework/multimodal/processor_cache.h"
#include "processors/multimodal_processor.h"

namespace xllm {

class CachingMultimodalProcessor final : public MultimodalProcessorBase {
 public:
  CachingMultimodalProcessor(std::unique_ptr<MultimodalProcessorBase> inner,
                             int64_t max_cache_items);
  ~CachingMultimodalProcessor() override = default;

  MMErrCode process_multimodal_request(const std::vector<Message>& messages,
                                       std::string payload,
                                       MMData& data);
  bool process_prompt(std::string& prompt,
                      MMData& mm_data,
                      std::vector<int32_t>& token_ids) override;
  bool process_multimodal(const MMInput& inputs, MMData& data) const override;

 private:
  // Cache hits are keyed by client uuid and resolved before any URL is loaded,
  // so cached items skip downloading and processing entirely.
  MMErrCode process_by_uuid(const std::vector<Message>& messages,
                            std::string payload,
                            MMData& data);
  // Full load followed by content-keyed caching inside process_multimodal.
  MMErrCode process_by_content(const std::vector<Message>& messages,
                               std::string payload,
                               MMData& data);

  std::unique_ptr<MultimodalProcessorBase> inner_;
  std::unique_ptr<ProcessorCache> cache_;
  MMInputTransfer input_transfer_;
};

}  // namespace xllm
