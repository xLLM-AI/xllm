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
#include <cstdlib>
#include <optional>
#include <string>

namespace xllm::spawn_worker_protocol {

inline constexpr int32_t kArgumentCount = 41;
inline constexpr int32_t kMinimumArgumentCount = 34;
inline constexpr int32_t kIndexerCacheDtypeArgumentIndex = 34;
inline constexpr int32_t kEnableMtpDraftBodyTp1ArgumentIndex = 35;
inline constexpr int32_t kTextEncoderTpSizeArgumentIndex = 36;
inline constexpr int32_t kEnableProbabilisticDraftArgumentIndex = 37;
inline constexpr int32_t kKvSplitSizeArgumentIndex = 38;
inline constexpr int32_t kModelImplArgumentIndex = 39;
inline constexpr int32_t kExtensionImplArgumentIndex = 40;
inline constexpr char kDefaultIndexerCacheDtype[] = "auto";

inline std::optional<std::string> parse_indexer_cache_dtype(
    int32_t argc,
    char* const argv[]) {
  if (argc < kMinimumArgumentCount || argv == nullptr) {
    return std::nullopt;
  }

  if (argc == kMinimumArgumentCount) {
    return std::string(kDefaultIndexerCacheDtype);
  }

  if (argv[kIndexerCacheDtypeArgumentIndex] == nullptr) {
    return std::nullopt;
  }
  return std::string(argv[kIndexerCacheDtypeArgumentIndex]);
}

inline std::string parse_model_impl(int32_t argc, char* const argv[]) {
  if (argv == nullptr || argc <= kModelImplArgumentIndex ||
      argv[kModelImplArgumentIndex] == nullptr) {
    return {};
  }
  return std::string(argv[kModelImplArgumentIndex]);
}

inline bool parse_bool_flag(int32_t argc, char* const argv[], int32_t index) {
  if (argv == nullptr || argc <= index || argv[index] == nullptr) {
    return false;
  }
  return atoi(argv[index]) > 0;
}

inline std::string parse_extension_impl(int32_t argc, char* const argv[]) {
  if (argv == nullptr || argc <= kExtensionImplArgumentIndex ||
      argv[kExtensionImplArgumentIndex] == nullptr) {
    return {};
  }
  return std::string(argv[kExtensionImplArgumentIndex]);
}

}  // namespace xllm::spawn_worker_protocol
