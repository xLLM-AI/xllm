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

#include <string>
#include <utility>

#include "core/common/types.h"

namespace xllm {

// Normalizes OpenAI-style completion JSON before protobuf parsing. An integer
// array prompt is moved into "token_ids" and "prompt" is cleared; a string
// prompt (or a missing/scalar prompt) is passed through unchanged. Batch and
// malformed array forms are rejected with INVALID_ARGUMENT.
std::pair<Status, std::string> preprocess_completion_prompt(
    std::string json_str);

}  // namespace xllm
