/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_input.h"
#include "core/util/hash_util.h"

namespace xllm {

class PromptProcessor {
 public:
  virtual ~PromptProcessor() = default;

  virtual void process(std::string& prompt, const MMData& mm_data) = 0;
  virtual void find_mm_spans(const std::vector<int32_t>& token_ids,
                             MMData& mm_data) = 0;
};

}  // namespace xllm
