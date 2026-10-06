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
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "core/common/types.h"
#include "framework/request/request_output.h"
#include "framework/tokenizer/tokenizer.h"

namespace xllm::test {

// Deterministic tokenizer for request factory tests. It encodes each character
// to a token id inside the vocabulary range, and fails encoding of any text
// containing the marker "FAIL" so tokenize and stop-sequence error paths can be
// exercised independently.
class FakeTokenizer final : public Tokenizer {
 public:
  explicit FakeTokenizer(int32_t vocab_size) : vocab_size_(vocab_size) {}

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool /*add_special_tokens*/ = true) const override {
    if (text.find("FAIL") != std::string_view::npos) {
      return false;
    }
    ids->clear();
    for (const char c : text) {
      ids->push_back(static_cast<int32_t>(static_cast<unsigned char>(c)) %
                     vocab_size_);
    }
    if (ids->empty()) {
      ids->push_back(1);
    }
    return true;
  }

  size_t vocab_size() const override {
    return static_cast<size_t>(vocab_size_);
  }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>(*this);
  }

 private:
  int32_t vocab_size_ = 1000;
};

// Records the last error surfaced through the OutputCallback.
struct CallbackCapture {
  bool called = false;
  std::optional<Status> status;
};

inline OutputCallback make_capture_callback(CallbackCapture* capture) {
  return [capture](RequestOutput output) {
    capture->called = true;
    capture->status = output.status;
    return false;
  };
}

}  // namespace xllm::test
