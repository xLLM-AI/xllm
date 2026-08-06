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
#include <optional>
#include <string>
#include <vector>

#include "processors/multimodal_processor.h"
#include "processors/processor_cache.h"

namespace xllm {

class CacheableMultimodalProcessor final : public MultimodalProcessorBase {
 public:
  CacheableMultimodalProcessor(std::unique_ptr<MultimodalProcessorBase> inner,
                               int64_t max_cache_items);
  ~CacheableMultimodalProcessor() override = default;

  bool process_prompt(std::string& prompt,
                      MMData& mm_data,
                      std::vector<int32_t>& token_ids) override;
  bool process_multimodal(const MMInput& inputs, MMData& data) const override;

 private:
  bool process_misses(const std::vector<MMInputItem>& miss_inputs,
                      MMItemVec& miss_items) const;
  void assemble(std::vector<std::optional<MMDataItem>>& cache_hits,
                MMItemVec miss_items,
                MMData& data) const;

  std::unique_ptr<MultimodalProcessorBase> inner_;
  std::unique_ptr<ProcessorCache> cache_;
};

}  // namespace xllm
