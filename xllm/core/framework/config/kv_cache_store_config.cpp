/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/config/kv_cache_store_config.h"

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

DEFINE_uint32(prefetch_timeout,
              0,
              "Prefetch timeout for prefetch from kv cache store.");

DEFINE_uint32(prefetch_batch_size,
              2,
              "Prefetch from kvcache store copy batch size.");

DEFINE_uint32(layers_wise_copy_batchs, 4, "Layer wise H2D copy batchs.");

DEFINE_double(host_blocks_factor,
              0.0,
              "Host block factor, e.g. host block num = host_blocks_factor * "
              "hbm block num.");

DEFINE_bool(enable_kvcache_store, false, "Whether to use kvcache store.");

DEFINE_bool(enable_cache_upload,
            false,
            "Whether to upload cache info to service. This feature is only "
            "available when service routing is enabled.");

DEFINE_string(store_protocol,
              "tcp",
              "KV cache store protocol(e.g. tcp, rdma).");

DEFINE_string(store_master_server_address,
              "",
              "The address information of the store master service.");

DEFINE_string(store_metadata_server,
              "",
              "The address of the kv cache store metadata service.");

DEFINE_string(store_local_hostname,
              "",
              "The local host name of the kv cache store client.");

DEFINE_bool(enable_control_h2d_block_num,
            false,
            "Whether to control h2d copy block num.");

namespace xllm {

void KVCacheStoreConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(prefetch_timeout);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(prefetch_batch_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(layers_wise_copy_batchs);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(host_blocks_factor);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_kvcache_store);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_cache_upload);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(store_protocol);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(store_master_server_address);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(store_metadata_server);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(store_local_hostname);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_control_h2d_block_num);
}

void KVCacheStoreConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(prefetch_timeout);
  XLLM_CONFIG_ASSIGN_FROM_JSON(prefetch_batch_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(layers_wise_copy_batchs);
  XLLM_CONFIG_ASSIGN_FROM_JSON(host_blocks_factor);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_kvcache_store);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_cache_upload);
  XLLM_CONFIG_ASSIGN_FROM_JSON(store_protocol);
  XLLM_CONFIG_ASSIGN_FROM_JSON(store_master_server_address);
  XLLM_CONFIG_ASSIGN_FROM_JSON(store_metadata_server);
  XLLM_CONFIG_ASSIGN_FROM_JSON(store_local_hostname);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_control_h2d_block_num);
}

void KVCacheStoreConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const KVCacheStoreConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, prefetch_timeout);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, prefetch_batch_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, layers_wise_copy_batchs);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, host_blocks_factor);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_kvcache_store);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_cache_upload);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, store_protocol);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, store_master_server_address);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, store_metadata_server);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, store_local_hostname);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_control_h2d_block_num);
}

KVCacheStoreConfig& KVCacheStoreConfig::get_instance() {
  static KVCacheStoreConfig config;
  return config;
}

void KVCacheStoreConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
