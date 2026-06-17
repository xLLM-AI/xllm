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

#pragma once

#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

class KVCacheStoreConfig final {
 public:
  KVCacheStoreConfig() = default;
  ~KVCacheStoreConfig() = default;

  static KVCacheStoreConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "KV CACHE STORE OPTIONS",
        {"prefetch_timeout",
         "prefetch_batch_size",
         "layers_wise_copy_batchs",
         "host_blocks_factor",
         "enable_kvcache_store",
         "enable_cache_upload",
         "store_protocol",
         "store_master_server_address",
         "store_metadata_server",
         "store_local_hostname",
         "enable_control_h2d_block_num"}};
    return kOptionCategory;
  }

  PROPERTY(uint32_t, prefetch_timeout) = 0;

  PROPERTY(uint32_t, prefetch_batch_size) = 2;

  PROPERTY(uint32_t, layers_wise_copy_batchs) = 4;

  PROPERTY(double, host_blocks_factor) = 0.0;

  PROPERTY(bool, enable_kvcache_store) = false;

  PROPERTY(bool, enable_cache_upload) = false;

  PROPERTY(std::string, store_protocol) = "tcp";

  PROPERTY(std::string, store_master_server_address);

  PROPERTY(std::string, store_metadata_server);

  PROPERTY(std::string, store_local_hostname);

  PROPERTY(bool, enable_control_h2d_block_num) = false;
};

}  // namespace xllm
