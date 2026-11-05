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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/common/message.h"
#include "core/common/types.h"

namespace xllm {

enum class SpeechToTextTask : int8_t {
  TRANSCRIBE = 0,
  TRANSLATE = 1,
};

struct SpeechToTextParams {
  SpeechToTextTask task = SpeechToTextTask::TRANSCRIBE;
  std::string request_prompt;
  std::string hotwords;
  std::string mime_type;
  size_t audio_bytes = 0;
  double max_duration_s = 0.0;
};

// Model-specific request adaptation; inference stays in the VLM engine.
class SpeechModelInterface {
 public:
  virtual ~SpeechModelInterface() = default;

  virtual bool supports_translation() const = 0;
  virtual bool supports_verbose_json() const = 0;
  virtual bool supports_segment_timestamps() const = 0;
  virtual Status validate_language(const std::string& language) const = 0;
  virtual std::vector<Message> build_messages(
      const SpeechToTextParams& params) const = 0;
};

}  // namespace xllm
