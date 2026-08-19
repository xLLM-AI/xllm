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

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "core/framework/tokenizer/tokenizer.h"

namespace xllm {

struct SampleSlot {
  std::string request_id;
  size_t sample_id = 0;
  size_t token_position = 0;
};

bool build_sample_slots(const std::string& request_id,
                        const std::string& prompt,
                        const std::string& literal,
                        const Tokenizer& tokenizer,
                        std::vector<SampleSlot>* sample_slots);

}  // namespace xllm
