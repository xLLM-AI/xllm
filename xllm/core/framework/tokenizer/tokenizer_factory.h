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

#include <memory>
#include <string>

namespace xllm {

class Tokenizer;
struct TokenizerArgs;

class TokenizerFactory {
 public:
  static std::unique_ptr<Tokenizer> create_tokenizer(
      const std::string& model_weights_path,
      TokenizerArgs tokenizer_args,
      bool proxy = true);
};

}  // namespace xllm
