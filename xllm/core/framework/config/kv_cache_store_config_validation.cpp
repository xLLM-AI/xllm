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

#include "core/framework/config/kv_cache_store_config_validation.h"

#include <glog/logging.h>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/kv_cache_store_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model_loader/hf_model_args.h"

namespace xllm {

void validate_kv_cache_store_config(const std::string& model_type,
                                    const ModelConfig& model_config,
                                    const KVCacheConfig& kv_cache_config,
                                    const KVCacheStoreConfig& store_config) {
  if (!model_type.empty()) {
    ModelArgs model_args;
    CHECK(load_hf_model_args(
        model_config.model(), model_config.backend(), &model_args))
        << "Failed to load model args from " << model_config.model();
    if (model_args.index_kpool_compress()) {
      CHECK_LE(store_config.host_blocks_factor(), 1.0)
          << "Compressed KPool host offload requires request-state scheduler "
             "support.";
      CHECK(!store_config.enable_kvcache_store())
          << "Compressed KPool external storage is not supported yet.";
    }
  }

  if (store_config.enable_kvcache_store()) {
    CHECK(kv_cache_config.enable_prefix_cache())
        << "KV cache Store requires --enable_prefix_cache=true.";
    CHECK_GT(store_config.host_blocks_factor(), 1.0)
        << "KV cache Store requires --host_blocks_factor > 1.";
  }
}

}  // namespace xllm
