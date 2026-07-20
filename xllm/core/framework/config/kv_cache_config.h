/* Copyright 2025-2026 The xLLM Authors.

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
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

class KVCacheConfig final {
 public:
  KVCacheConfig() = default;
  ~KVCacheConfig() = default;

  static KVCacheConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();
  void validate() const;

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "KV CACHE OPTIONS",
        {"block_size",
         "max_cache_size",
         "max_memory_utilization",
         "kv_cache_dtype",
         "indexer_cache_dtype",
         "enable_prefix_cache",
         "enable_in_batch_prefix_cache",
         "max_linear_state_cache_slots",
         "xxh3_128bits_seed",
         "enable_xtensor",
         "phy_page_granularity_size"}};
    return kOptionCategory;
  }

  PROPERTY(int32_t, block_size) = 128;

  PROPERTY(int64_t, max_cache_size) = 0;

  PROPERTY(double, max_memory_utilization) = 0.8;

  PROPERTY(std::string, kv_cache_dtype) = "auto";

  PROPERTY(std::string, indexer_cache_dtype) = "auto";

  PROPERTY(bool, enable_prefix_cache) = true;

  PROPERTY(bool, enable_in_batch_prefix_cache) = false;

  PROPERTY(int64_t, max_linear_state_cache_slots) = 0;

  PROPERTY(uint32_t, xxh3_128bits_seed) = 1024;

  PROPERTY(bool, enable_xtensor) = false;

  PROPERTY(int64_t, phy_page_granularity_size) = 2 * 1024 * 1024;
};

}  // namespace xllm
