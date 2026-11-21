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
         "store_backend",
         "store_protocol",
         "store_rdma_devices",
         "store_master_server_address",
         "store_metadata_server",
         "store_local_hostname",
         "store_yuanrong_host",
         "store_yuanrong_port",
         "store_yuanrong_timeout_ms",
         "enable_control_h2d_block_num"}};
    return kOptionCategory;
  }

  PROPERTY(uint32_t, prefetch_timeout) = 120000;

  PROPERTY(uint32_t, prefetch_batch_size) = 2;

  PROPERTY(uint32_t, layers_wise_copy_batchs) = 4;

  PROPERTY(double, host_blocks_factor) = 0.0;

  PROPERTY(bool, enable_kvcache_store) = false;

  // Which backend holds offloaded blocks: "mooncake" (peer-to-peer store
  // client) or "yuanrong" (content-addressed host store). The latter
  // requires a build with USE_YUANRONG.
  PROPERTY(std::string, store_backend) = "mooncake";

  PROPERTY(std::string, store_protocol) = "tcp";

  // Comma-separated HCAs for the embedded Store client.
  PROPERTY(std::string, store_rdma_devices) = "";

  PROPERTY(std::string, store_master_server_address);

  PROPERTY(std::string, store_metadata_server);

  PROPERTY(std::string, store_local_hostname);

  // YuanRong worker endpoint used when store_backend=yuanrong.
  PROPERTY(std::string, store_yuanrong_host);

  PROPERTY(int32_t, store_yuanrong_port) = 0;

  PROPERTY(int32_t, store_yuanrong_timeout_ms) = 500;

  PROPERTY(bool, enable_control_h2d_block_num) = false;
};

}  // namespace xllm
