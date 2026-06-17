/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

namespace xllm {

struct Usage {
  // the number of tokens in the prompt.
  int32_t num_prompt_tokens = 0;

  // the number of tokens in the generated completion.
  int32_t num_generated_tokens = 0;

  // the total number of tokens used in the request (prompt + completion).
  int32_t num_total_tokens = 0;

  // the number of prompt tokens served from prefix cache.
  int32_t num_cached_tokens = 0;
};

}  // namespace xllm
