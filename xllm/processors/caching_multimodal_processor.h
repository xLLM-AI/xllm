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

#include "core/framework/multimodal/processor_cache.h"
#include "processors/multimodal_processor.h"

namespace xllm {

class CachingMultimodalProcessor final : public MultimodalProcessorBase {
 public:
  CachingMultimodalProcessor(std::unique_ptr<MultimodalProcessorBase> inner,
                             int64_t max_cache_items);
  ~CachingMultimodalProcessor() override = default;

  bool process_prompt(std::string& prompt,
                      MMData& mm_data,
                      std::vector<int32_t>& token_ids) override;
  bool process_multimodal(const MMInput& inputs, MMData& data) const override;

 private:
  std::unique_ptr<MultimodalProcessorBase> inner_;
  mutable ProcessorCache cache_;
};

}  // namespace xllm
