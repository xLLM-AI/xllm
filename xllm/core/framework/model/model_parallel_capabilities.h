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
#include <string_view>

namespace xllm {

// Indexer cache blocks use the DCP shard domain only for model families whose
// indexer cache is physically partitioned with their MLA cache.
enum class DcpIndexerCacheLayout : int8_t {
  STANDARD = 0,
  SHARDED = 1,
};

inline bool is_glm_dsa_model_family(std::string_view model_type) {
  return model_type == "glm_moe_dsa" || model_type == "glm_moe_dsa_mtp";
}

inline bool is_glm_dsa_target_model(std::string_view model_type) {
  return model_type == "glm_moe_dsa";
}

inline bool is_glm_dsa_mtp_draft_model(std::string_view model_type) {
  return model_type == "glm_moe_dsa_mtp";
}

inline DcpIndexerCacheLayout resolve_dcp_indexer_cache_layout(
    std::string_view model_type,
    int32_t kv_split_size) {
  if (is_glm_dsa_model_family(model_type) && kv_split_size > 1) {
    return DcpIndexerCacheLayout::SHARDED;
  }
  return DcpIndexerCacheLayout::STANDARD;
}

inline bool uses_dcp_sharded_indexer_cache(std::string_view model_type,
                                           int32_t kv_split_size) {
  return resolve_dcp_indexer_cache_layout(model_type, kv_split_size) ==
         DcpIndexerCacheLayout::SHARDED;
}

}  // namespace xllm
