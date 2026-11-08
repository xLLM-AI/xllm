/* Copyright 2026 The xLLM Authors.

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

#include <torch/types.h>

#include <string>

#include "core/util/hash_util.h"

namespace xllm {

// Content identity shares the configured seed with prefix-cache identities.
XXH3Key hash_tensor(const torch::Tensor& tensor);
XXH3Key hash_string(const std::string& value);

}  // namespace xllm
