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

#include <cstdint>

namespace xllm::dspark_detail {

enum class VocabularyWeightSource : uint8_t {
  NONE,
  TARGET_FALLBACK,
  DEDICATED,
};

class VocabularyWeightSelector final {
 public:
  bool should_load(bool dedicated) const {
    if (dedicated) {
      return source_ != VocabularyWeightSource::DEDICATED;
    }
    return source_ == VocabularyWeightSource::NONE;
  }

  void mark_loaded(bool dedicated) {
    source_ = dedicated ? VocabularyWeightSource::DEDICATED
                        : VocabularyWeightSource::TARGET_FALLBACK;
  }

  bool loaded() const { return source_ != VocabularyWeightSource::NONE; }

  const char* source_name() const {
    switch (source_) {
      case VocabularyWeightSource::TARGET_FALLBACK:
        return "target vocabulary fallback";
      case VocabularyWeightSource::DEDICATED:
        return "dedicated DSpark vocabulary";
      case VocabularyWeightSource::NONE:
        return "not loaded";
    }
    return "unknown";
  }

 private:
  VocabularyWeightSource source_ = VocabularyWeightSource::NONE;
};

}  // namespace xllm::dspark_detail
