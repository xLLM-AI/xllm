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
#include <absl/strings/str_split.h>

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace xllm {

// an thin wrapper around nlohmann/json to read json files.
// it supports read keys with dot notation from json.
// for example: value_or("a.b.c", 0) will return 100 for following json:
// {
//   "a": {
//     "b": {
//       "c": 100
//     }
//   }
// }
//
class JsonReader {
 public:
  // parse the json file, return true if success
  bool parse(const std::string& json_file_path);

  // parse json content from a string, return true if success
  bool parse_text(const std::string& json_text);

  // check if the json contains the key, key can be nested with dot notation
  bool contains(const std::string& key) const;

  template <typename T, typename T2>
  T value_or(const std::vector<std::string>& keys, T2 default_value) const {
    for (const auto& key : keys) {
      if (auto data = value<T>(key)) {
        return data.value();
      }
    }
    // may introduce implicit conversion from T2 to T
    return default_value;
  }

  template <typename T, typename T2>
  T value_or(const std::string& key, T2 default_value) const {
    if (auto data = value<T>(key)) {
      return data.value();
    }
    // may introduce implicit conversion from T2 to T
    return default_value;
  }

  template <typename T>
  std::optional<T> value(const std::string& key) const {
    if (auto data = resolve(key)) {
      if (data->is_null() || data->is_object()) {
        // cannot convert null or object data to T
        return std::nullopt;
      }
      return data->get<T>();
    }
    return std::nullopt;
  }

  // Resolve a dot-separated key path against a json object; returns nullptr if
  // any segment is missing.
  static const nlohmann::json* resolve_path(const nlohmann::json& root,
                                            const std::string& key) {
    const std::vector<std::string> keys = absl::StrSplit(key, '.');
    const nlohmann::json* data = &root;
    for (const auto& k : keys) {
      if (data->contains(k)) {
        data = &(*data)[k];
      } else {
        return nullptr;
      }
    }
    return data;
  }

  // Look up a key, falling back to the same key under a "text_config" subtree
  // when the top-level lookup misses. Multimodal configs (e.g. glm5_next
  // full weights) nest the text-model fields under "text_config"; single-model
  // configs keep them flat. This lets model args loaders use the same flat key
  // ("num_hidden_layers") for both layouts.
  const nlohmann::json* resolve(const std::string& key) const {
    if (auto* data = resolve_path(data_, key)) {
      return data;
    }
    if (data_.contains("text_config") && data_["text_config"].is_object()) {
      return resolve_path(data_["text_config"], key);
    }
    return nullptr;
  }

  nlohmann::json data() const { return data_; }

 private:
  nlohmann::json data_;
};

}  // namespace xllm
