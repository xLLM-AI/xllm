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

#pragma once

#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <utility>

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

  [[nodiscard]] bool max_memory_utilization_configured() const {
    return max_memory_utilization_configured_;
  }

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "KV CACHE OPTIONS",
        {"block_size",
         "max_cache_size",
         "max_memory_utilization",
         "kv_cache_dtype",
         "indexer_cache_dtype",
         "enable_prefix_cache",
         "kv_events_config",
         "enable_in_batch_prefix_cache",
         "max_linear_state_cache_slots",
         "xxh3_128bits_seed",
         "enable_vmm_kvcache",
         "enable_shared_kv_cache",
         "vmm_elastic",
         "vmm_memory_reserve_bytes",
         "vmm_startup_timeout_ms",
         "phy_page_granularity_size"}};
    return kOptionCategory;
  }

  PROPERTY(int32_t, block_size) = 128;

  PROPERTY(int64_t, max_cache_size) = 0;

  [[nodiscard]] double max_memory_utilization() const noexcept {
    return max_memory_utilization_;
  }
  KVCacheConfig& max_memory_utilization(double value) & {
    max_memory_utilization_ = value;
    max_memory_utilization_configured_ = true;
    return *this;
  }
  void max_memory_utilization(double value) && = delete;

  PROPERTY(std::string, kv_cache_dtype) = "auto";

  PROPERTY(std::string, indexer_cache_dtype) = "auto";

  PROPERTY(bool, enable_prefix_cache) = true;
  PROPERTY(std::string, kv_events_config);

  PROPERTY(bool, enable_in_batch_prefix_cache) = false;

  PROPERTY(int64_t, max_linear_state_cache_slots) = 0;

  PROPERTY(uint32_t, xxh3_128bits_seed) = 1024;

  PROPERTY(bool, enable_vmm_kvcache) = false;

  // Fixed arena shared by Full KV blocks and recurrent state extents.
  PROPERTY(bool, enable_shared_kv_cache) = false;

  // Fixed arenas remain explicit for transports requiring dense registration.
  PROPERTY(bool, vmm_elastic) = true;

  PROPERTY(int64_t, vmm_memory_reserve_bytes) = 0;

  PROPERTY(int32_t, vmm_startup_timeout_ms) = 120000;

  PROPERTY(int64_t, phy_page_granularity_size) = 2 * 1024 * 1024;

 private:
  double max_memory_utilization_ = 0.8;
  bool max_memory_utilization_configured_ = false;
};

}  // namespace xllm
