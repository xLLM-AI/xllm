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

#include "core/framework/config/kv_cache_config.h"

#include <glog/logging.h>

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

DEFINE_int32(block_size,
             128,
             "Number of slots per kv cache block. Default is 128.");

DEFINE_int64(max_cache_size,
             0,
             "Max gpu memory size for kv cache. Default is 0, which means "
             "cache size is caculated by available memory.");

DEFINE_double(max_memory_utilization,
              0.8,
              "The fraction of GPU memory to be used for model inference, "
              "including model weights and kv cache.");

DEFINE_string(
    kv_cache_dtype,
    "auto",
    "KV cache data type for quantization. \"auto\" (default): KV "
    "cache dtype aligns with model dtype (no quantization). "
    "\"int8\": Enables INT8 quantization. Only supported on MLU backend.");

DEFINE_string(indexer_cache_dtype,
              "auto",
              "Indexer cache data type for quantization. \"auto\" (default): "
              "Indexer cache dtype aligns with model dtype (no "
              "quantization). \"int8\": Enables INT8 quantization when "
              "supported. Only supported on MLU backend.");

DEFINE_bool(enable_prefix_cache,
            true,
            "Whether to enable the prefix cache for the block manager.");

DEFINE_bool(enable_in_batch_prefix_cache,
            false,
            "Whether to cache admitted prefill full blocks into the prefix "
            "cache so that later requests in the same batch can share them.");

DEFINE_int64(max_linear_state_cache_slots,
             0,
             "Maximum active linear-attention state cache slots. 0 derives an "
             "automatic capacity from the available KV cache budget.");

DEFINE_uint32(xxh3_128bits_seed, 1024, "Default XXH3 128-bits hash seed.");

DEFINE_bool(
    enable_xtensor,
    false,
    "Whether to enable xtensor for model weights with physical page pool.");

DEFINE_int64(
    phy_page_granularity_size,
    2 * 1024 * 1024,
    "Granularity size for one physical page in bytes, default 2MB, when enable "
    "continuous kv cache.");

namespace xllm {

void KVCacheConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(block_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_cache_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_memory_utilization);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(kv_cache_dtype);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(indexer_cache_dtype);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_prefix_cache);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_in_batch_prefix_cache);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_linear_state_cache_slots);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(xxh3_128bits_seed);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_xtensor);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(phy_page_granularity_size);
}

void KVCacheConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(block_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_cache_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_memory_utilization);
  XLLM_CONFIG_ASSIGN_FROM_JSON(kv_cache_dtype);
  XLLM_CONFIG_ASSIGN_FROM_JSON(indexer_cache_dtype);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_prefix_cache);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_in_batch_prefix_cache);
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_linear_state_cache_slots);
  XLLM_CONFIG_ASSIGN_FROM_JSON(xxh3_128bits_seed);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_xtensor);
  XLLM_CONFIG_ASSIGN_FROM_JSON(phy_page_granularity_size);
}

void KVCacheConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const KVCacheConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, block_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_cache_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_memory_utilization);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, kv_cache_dtype);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, indexer_cache_dtype);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_prefix_cache);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_in_batch_prefix_cache);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_linear_state_cache_slots);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, xxh3_128bits_seed);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_xtensor);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, phy_page_granularity_size);
}

KVCacheConfig& KVCacheConfig::get_instance() {
  static KVCacheConfig config;
  return config;
}

void KVCacheConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
  validate();
}

void KVCacheConfig::validate() const {
  if (indexer_cache_dtype_ != "auto" && indexer_cache_dtype_ != "int8") {
    LOG(FATAL) << "Invalid indexer_cache_dtype=\"" << indexer_cache_dtype_
               << "\". Supported values are exactly \"auto\" and \"int8\".";
  }
}

}  // namespace xllm
