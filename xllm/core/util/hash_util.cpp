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

#include "hash_util.h"

#include "core/common/global_flags.h"

namespace xllm {

XXH3Key hash_tensor(const torch::Tensor& tensor) {
  XXH3Key key;
  torch::Tensor contiguous_tensor = tensor.contiguous();
  XXH128_hash_t hash = XXH3_128bits_withSeed(
      reinterpret_cast<const void*>(contiguous_tensor.data_ptr()),
      tensor.numel() * tensor.element_size(),
      FLAGS_xxh3_128bits_seed);
  std::memcpy(key.data, &hash, sizeof(hash));
  return key;
}

XXH3Key hash_string(const std::string& str) {
  XXH3Key key;
  XXH128_hash_t hash =
      XXH3_128bits_withSeed(reinterpret_cast<const void*>(str.data()),
                            str.size(),
                            FLAGS_xxh3_128bits_seed);
  std::memcpy(key.data, &hash, sizeof(hash));
  return key;
}
}  // namespace xllm
