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

#include "sample_slot.h"

#include <string_view>

namespace xllm {

namespace {

bool get_stable_special_token_id(const std::string& literal,
                                 const Tokenizer& tokenizer,
                                 int32_t* literal_token_id) {
  if (literal_token_id == nullptr || literal.empty()) {
    return false;
  }

  const auto token_id = tokenizer.token_to_id(literal);
  if (!token_id.has_value() ||
      tokenizer.id_to_token(token_id.value()) != literal) {
    return false;
  }

  std::vector<int32_t> literal_tokens;
  if (!tokenizer.encode(std::string_view(literal), &literal_tokens, false) ||
      literal_tokens.size() != 1 ||
      literal_tokens.front() != token_id.value()) {
    return false;
  }

  *literal_token_id = token_id.value();
  return true;
}

}  // namespace

bool build_sample_slots(const std::string& request_id,
                        const std::string& prompt,
                        const std::string& literal,
                        const Tokenizer& tokenizer,
                        std::vector<SampleSlot>* sample_slots) {
  if (sample_slots == nullptr) {
    return false;
  }

  sample_slots->clear();
  if (literal.empty()) {
    return true;
  }

  int32_t literal_token_id = 0;
  if (!get_stable_special_token_id(literal, tokenizer, &literal_token_id)) {
    return false;
  }

  std::vector<int32_t> prompt_tokens;
  if (!tokenizer.encode(std::string_view(prompt), &prompt_tokens, false)) {
    return false;
  }

  for (size_t token_idx = 0; token_idx < prompt_tokens.size(); ++token_idx) {
    if (prompt_tokens[token_idx] != literal_token_id) {
      continue;
    }
    SampleSlot slot;
    slot.request_id = request_id;
    slot.sample_id = sample_slots->size();
    slot.token_position = token_idx;
    sample_slots->push_back(std::move(slot));
  }

  return true;
}

}  // namespace xllm
