/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <functional>
#include <memory>

#include "common/macros.h"
#include "parser/reasoning_detector.h"

namespace xllm {

using DetectorFactory =
    std::function<std::unique_ptr<ReasoningDetector>(bool, bool)>;
class DetectorRegistry {
 public:
  static DetectorRegistry& get_instance() {
    static DetectorRegistry instance;
    return instance;
  }

  std::unique_ptr<ReasoningDetector> get_detector(const std::string& model_type,
                                                  bool stream_reasoning,
                                                  bool force_reasoning);

  bool has_detector(const std::string& parser_name) const;

  std::string get_supported_parsers() const;

  // Get the reasoning parser name for auto mode based on model_type
  // Returns empty string if not found
  std::string get_parser_name_by_model_type(
      const std::string& model_type) const;

 private:
  DetectorRegistry() = default;
  ~DetectorRegistry() = default;

  DISALLOW_COPY_AND_ASSIGN(DetectorRegistry);
};

}  // namespace xllm
