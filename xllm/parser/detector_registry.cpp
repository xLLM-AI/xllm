/* Copyright 2025-2026 The xLLM Authors.
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

#include "detector_registry.h"

#include <glog/logging.h>

#include <algorithm>

#include "absl/strings/str_join.h"

namespace xllm {

namespace {
#define REGISTER_DETECTOR_DEFAULT_FORCE(name, start, end, default_force) \
  {name, [](bool stream, bool force) {                                   \
     return std::make_unique<ReasoningDetector>(                         \
         start, end, default_force, stream);                             \
   }}

#define REGISTER_DETECTOR(name, start, end)                                 \
  {name, [](bool stream, bool force) {                                      \
     return std::make_unique<ReasoningDetector>(start, end, force, stream); \
   }}

// Maps reasoning_parser name to supported model_types
const std::unordered_map<std::string, std::string> auto_paser_map = {
    // {"deepseek_v3", "deepseek-v3"},
    // {"qwen3", "qwen3"},
    {"glm4_moe", "glm45"},
    {"deepseek_v32", "deepseekv32"},
    {"kimi_k2", "kimi"},
    {"kimi_k25", "kimi_k25"},
    {"step3", "step3"},
};

std::string get_auto_paser_map_supported() {
  std::vector<std::string> keys;
  keys.reserve(auto_paser_map.size());
  for (const auto& pair : auto_paser_map) {
    keys.push_back(pair.first);
  }
  return absl::StrJoin(keys, ", ");
}

const std::unordered_map<std::string, DetectorFactory> paser_factories = {
    REGISTER_DETECTOR_DEFAULT_FORCE("deepseek-r1", "<think>", "</think>", true),
    REGISTER_DETECTOR("deepseek-v3", "<think>", "</think>"),
    REGISTER_DETECTOR("glm45", "<think>", "</think>"),
    REGISTER_DETECTOR("glm47", "<think>", "</think>"),
    REGISTER_DETECTOR("glm5", "<think>", "</think>"),
    REGISTER_DETECTOR("deepseek-v4", "<think>", "</think>"),
    REGISTER_DETECTOR_DEFAULT_FORCE("kimi", "◁think▷", "◁/think▷", false),
    REGISTER_DETECTOR("kimi_k25", "<think>", "</think>"),
    REGISTER_DETECTOR("qwen3", "<think>", "</think>"),
    REGISTER_DETECTOR("qwen35", "<think>", "</think>"),
    REGISTER_DETECTOR_DEFAULT_FORCE("qwen3-thinking",
                                    "<think>",
                                    "</think>",
                                    true),
    REGISTER_DETECTOR_DEFAULT_FORCE("qwen35-thinking",
                                    "<think>",
                                    "</think>",
                                    true),
    REGISTER_DETECTOR("step3", "<think>", "</think>"),
};

}  // namespace

std::unique_ptr<ReasoningDetector> DetectorRegistry::get_detector(
    const std::string& model_type,
    bool stream_reasoning,
    bool force_reasoning) {
  auto it = paser_factories.find(model_type);
  if (it == paser_factories.end()) {
    std::vector<std::string> keys;
    for (const auto& pair : paser_factories) {
      keys.push_back(pair.first);
    }
    LOG(FATAL) << "Unsupported model type for reasoning parser: " << model_type
               << "\nAvailable model types are: " << absl::StrJoin(keys, ", ");
  }

  return it->second(stream_reasoning, force_reasoning);
};

bool DetectorRegistry::has_detector(const std::string& parser_name) const {
  return paser_factories.find(parser_name) != paser_factories.end();
}

std::string DetectorRegistry::get_supported_parsers() const {
  std::vector<std::string> keys;
  keys.reserve(paser_factories.size());
  for (const auto& pair : paser_factories) {
    keys.push_back(pair.first);
  }
  return absl::StrJoin(keys, ", ");
}

std::string DetectorRegistry::get_parser_name_by_model_type(
    const std::string& model_type) const {
  auto it = auto_paser_map.find(model_type);
  if (it != auto_paser_map.end()) {
    return it->second;
  }
  LOG(FATAL) << "Unsupported model type for reasoning parser: " << model_type
             << ". Supported model types are: "
             << get_auto_paser_map_supported();
  return "";
}
}  // namespace xllm
