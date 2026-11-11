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

#include "core/framework/config/kv_cache_config.h"

#include <glog/logging.h>

#include <cmath>

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

DEFINE_int32(block_size,
             128,
             "Number of slots per kv cache block. Default is 128.");

DEFINE_int64(max_cache_size,
             0,
             "Max gpu memory size for kv cache. Default is 0, which means "
             "cache size is calculated by available memory.");

DEFINE_double(max_memory_utilization,
              0.8,
              "The fraction of GPU memory to be used for model inference, "
              "including model weights and kv cache. Elastic VMM ignores "
              "this ratio and uses live memory and execution peaks.");

DEFINE_bool(vmm_elastic,
            true,
            "Grow and release virtual-memory physical pages on demand. Set "
            "false for transports requiring a fixed, fully resident arena.");

DEFINE_int64(vmm_memory_reserve_bytes,
             0,
             "Additional device memory kept free for execution when elastic "
             "VMM grows. This supplements runtime execution headroom.");

DEFINE_int32(vmm_startup_timeout_ms,
             120000,
             "Timeout in milliseconds for each VMM startup wait: master node "
             "registration and non-master physical pool readiness. Must be "
             "positive; independent of per-request RPC timeouts.");

DEFINE_string(kv_cache_dtype,
              "auto",
              "KV cache data type for quantization. \"auto\" (default): KV "
              "cache dtype aligns with model dtype (no quantization). "
              "\"int8\": Enables INT8 quantization on MLU, and on NPU for "
              "GLM-5.2 SFA packed KV cache.");

DEFINE_string(indexer_cache_dtype,
              "auto",
              "Indexer cache data type for quantization. \"auto\" (default): "
              "Indexer cache dtype aligns with model dtype (no "
              "quantization). \"int8\": Enables INT8 quantization when "
              "supported. Supported on NPU and MLU backends.");

DEFINE_bool(enable_prefix_cache,
            true,
            "Whether to enable the prefix cache for the block manager.");

DEFINE_string(kv_events_config,
              "",
              "ZMQ KV event publisher JSON; disabled when empty.");

DEFINE_bool(enable_in_batch_prefix_cache,
            false,
            "Whether to cache admitted prefill full blocks into the prefix "
            "cache so that later requests in the same batch can share them.");

DEFINE_int64(max_linear_state_cache_slots,
             0,
             "Maximum active linear-attention state cache slots. 0 derives an "
             "automatic capacity from the available KV cache budget.");

DEFINE_uint32(xxh3_128bits_seed, 1024, "Default XXH3 128-bits hash seed.");

DEFINE_bool(enable_vmm_kvcache,
            false,
            "Whether to enable virtual memory for KV cache "
            "with the physical page pool.");

DEFINE_bool(enable_shared_kv_cache,
            false,
            "Share a fixed physical arena between full-attention KV blocks "
            "and linear-attention states. NPU Torch Qwen3 hybrid models only.");

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
  XLLM_CONFIG_ASSIGN_FROM_FLAG(kv_events_config);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_in_batch_prefix_cache);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_linear_state_cache_slots);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(xxh3_128bits_seed);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_vmm_kvcache);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_shared_kv_cache);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(vmm_elastic);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(vmm_memory_reserve_bytes);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(vmm_startup_timeout_ms);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(phy_page_granularity_size);
}

void KVCacheConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(block_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_cache_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_memory_utilization);
  XLLM_CONFIG_ASSIGN_FROM_JSON(kv_cache_dtype);
  XLLM_CONFIG_ASSIGN_FROM_JSON(indexer_cache_dtype);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_prefix_cache);
  XLLM_CONFIG_ASSIGN_FROM_JSON(kv_events_config);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_in_batch_prefix_cache);
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_linear_state_cache_slots);
  XLLM_CONFIG_ASSIGN_FROM_JSON(xxh3_128bits_seed);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_vmm_kvcache);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_shared_kv_cache);
  XLLM_CONFIG_ASSIGN_FROM_JSON(vmm_elastic);
  XLLM_CONFIG_ASSIGN_FROM_JSON(vmm_memory_reserve_bytes);
  XLLM_CONFIG_ASSIGN_FROM_JSON(vmm_startup_timeout_ms);
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
      config_json, default_config, kv_events_config);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_in_batch_prefix_cache);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_linear_state_cache_slots);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, xxh3_128bits_seed);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_vmm_kvcache);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_shared_kv_cache);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, vmm_elastic);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, vmm_memory_reserve_bytes);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, vmm_startup_timeout_ms);
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
  CHECK(!enable_shared_kv_cache_ || !enable_vmm_kvcache_)
      << "Static shared KV cache and VMM addressing are separate modes.";
  CHECK_GE(vmm_memory_reserve_bytes_, 0);
  CHECK_GT(vmm_startup_timeout_ms_, 0)
      << "vmm_startup_timeout_ms must be positive";
  CHECK(std::isfinite(max_memory_utilization_));
  CHECK_LE(max_memory_utilization_, 1.0);
  CHECK_GE(max_memory_utilization_, 0.0);
  CHECK_GT(phy_page_granularity_size_, 0);
  CHECK_GE(max_cache_size_, 0);
  if (!enable_vmm_kvcache_ || !vmm_elastic_) {
    CHECK_GT(max_memory_utilization_, 0.0)
        << "Physical KV cache and fixed VMM require positive utilization";
  }
  if (!kv_events_config_.empty()) {
    const nlohmann::json config =
        nlohmann::json::parse(kv_events_config_, nullptr, false);
    CHECK(config.is_object()) << "kv_events_config must be a JSON object";
    CHECK(config.contains("endpoint") && config["endpoint"].is_string())
        << "kv_events_config requires endpoint";
    for (const std::string& key :
         {"publisher", "replay_endpoint", "topic", "motor_pool_endpoint"}) {
      CHECK(!config.contains(key) || config[key].is_string())
          << key << " must be a string";
    }
    for (const std::string& key : {"hwm", "snapshot_interval_seconds"}) {
      CHECK(!config.contains(key) ||
            (config[key].is_number_integer() && config[key] > 0 &&
             config[key] <= INT32_MAX))
          << key << " must be a positive int32";
    }
  }
  if (indexer_cache_dtype_ != "auto" && indexer_cache_dtype_ != "int8") {
    LOG(FATAL) << "Invalid indexer_cache_dtype=\"" << indexer_cache_dtype_
               << "\". Supported values are exactly \"auto\" and \"int8\".";
  }
}

}  // namespace xllm
