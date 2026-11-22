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

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

#include "core/framework/eplb/eplb_options.h"

namespace xllm::spawn_worker_protocol {

inline constexpr int32_t kArgumentCount = 46;
inline constexpr int32_t kMinimumArgumentCount = 33;
inline constexpr int32_t kIndexerCacheDtypeArgumentIndex = 33;
inline constexpr int32_t kEnableMtpDraftBodyTp1ArgumentIndex = 34;
inline constexpr int32_t kTextEncoderTpSizeArgumentIndex = 35;
inline constexpr int32_t kEnableProbabilisticDraftArgumentIndex = 36;
inline constexpr int32_t kKvSplitSizeArgumentIndex = 37;
inline constexpr int32_t kModelImplArgumentIndex = 38;
inline constexpr int32_t kExtensionImplArgumentIndex = 39;
inline constexpr int32_t kEnableEplbArgumentIndex = 40;
inline constexpr int32_t kRedundantExpertsNumArgumentIndex = 41;
inline constexpr int32_t kUseDecodeOnlyLoadArgumentIndex = 42;
inline constexpr int32_t kEplbArgumentCount = 43;
inline constexpr int32_t kPrefillChunkArgumentIndex = 43;
inline constexpr int32_t kEnableChunkedPrefillArgumentIndex = 44;
inline constexpr int32_t kEnableDpFairTokenBudgetArgumentIndex = 45;
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

struct PrefillConfig {
  int32_t max_chunk_tokens = -1;
  bool enable_chunked_prefill = true;
  bool enable_dp_fair_token_budget = true;
};

inline std::optional<PrefillConfig> parse_prefill_config(int32_t argc,
                                                         char* const argv[]) {
  if (argv == nullptr || argc < kMinimumArgumentCount) {
    return std::nullopt;
  }
  if (argc <= kPrefillChunkArgumentIndex) {
    return PrefillConfig{};
  }
  if (argc < kArgumentCount || argv[kPrefillChunkArgumentIndex] == nullptr ||
      argv[kEnableChunkedPrefillArgumentIndex] == nullptr ||
      argv[kEnableDpFairTokenBudgetArgumentIndex] == nullptr) {
    return std::nullopt;
  }
  const std::string_view chunk(argv[kPrefillChunkArgumentIndex]);
  const std::string_view chunked(argv[kEnableChunkedPrefillArgumentIndex]);
  const std::string_view fair(argv[kEnableDpFairTokenBudgetArgumentIndex]);
  if ((chunked != "0" && chunked != "1") || (fair != "0" && fair != "1")) {
    return std::nullopt;
  }
  PrefillConfig config;
  const auto [end, error] = std::from_chars(
      chunk.data(), chunk.data() + chunk.size(), config.max_chunk_tokens);
  if (error != std::errc{} || end != chunk.data() + chunk.size() ||
      config.max_chunk_tokens <= 0) {
    return std::nullopt;
  }
  config.enable_chunked_prefill = chunked == "1";
  config.enable_dp_fair_token_budget = fair == "1";
  return config;
}

inline std::optional<EplbWorkerConfig> parse_eplb_worker_config(
    int32_t argc,
    char* const argv[]) {
  if (argv == nullptr || argc < kMinimumArgumentCount) {
    return std::nullopt;
  }
  if (argc <= kEnableEplbArgumentIndex) {
    return EplbWorkerConfig{};
  }
  if (argc < kEplbArgumentCount || argv[kEnableEplbArgumentIndex] == nullptr ||
      argv[kRedundantExpertsNumArgumentIndex] == nullptr ||
      argv[kUseDecodeOnlyLoadArgumentIndex] == nullptr) {
    return std::nullopt;
  }

  const std::string_view enabled(argv[kEnableEplbArgumentIndex]);
  const std::string_view redundant_experts_num(
      argv[kRedundantExpertsNumArgumentIndex]);
  const std::string_view use_decode_only_load(
      argv[kUseDecodeOnlyLoadArgumentIndex]);
  if ((enabled != "0" && enabled != "1") ||
      (use_decode_only_load != "0" && use_decode_only_load != "1")) {
    return std::nullopt;
  }

  EplbWorkerConfig config;
  const auto [end, error] = std::from_chars(
      redundant_experts_num.data(),
      redundant_experts_num.data() + redundant_experts_num.size(),
      config.redundant_experts_num);
  if (error != std::errc{} ||
      end != redundant_experts_num.data() + redundant_experts_num.size() ||
      config.redundant_experts_num < 0) {
    return std::nullopt;
  }
  config.enabled = enabled == "1";
  config.use_decode_only_load = use_decode_only_load == "1";
  return config;
}

}  // namespace xllm::spawn_worker_protocol
