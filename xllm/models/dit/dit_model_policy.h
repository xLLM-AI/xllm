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

#include <cstdint>
#include <optional>
#include <string>

namespace xllm {

class Options;
struct DiTRequestParams;

using DiTRequestValidator = const char* (*)(const DiTRequestParams&);

struct DiTRequestPolicy {
  DiTRequestValidator validator = nullptr;
  std::optional<int32_t> required_batch_size;
};

std::string resolve_dit_model_type(const std::string& checkpoint_model_type,
                                   const std::string& model_id);

DiTRequestPolicy resolve_dit_request_policy(const Options& options,
                                            const std::string& model_impl);

}  // namespace xllm
