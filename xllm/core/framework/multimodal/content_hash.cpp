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

#include "core/framework/multimodal/content_hash.h"

#include "core/framework/config/kv_cache_config.h"

namespace xllm {

XXH3Key hash_tensor(const torch::Tensor& tensor) {
  const torch::Tensor contiguous_tensor = tensor.contiguous();
  return hash_bytes(
      contiguous_tensor.data_ptr(),
      contiguous_tensor.numel() * contiguous_tensor.element_size(),
      KVCacheConfig::get_instance().xxh3_128bits_seed());
}

XXH3Key hash_string(const std::string& value) {
  return hash_string(std::string_view(value),
                     KVCacheConfig::get_instance().xxh3_128bits_seed());
}

}  // namespace xllm
