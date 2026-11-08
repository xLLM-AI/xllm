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

#include "core/util/hash_util.h"

namespace xllm {

XXH3Key hash_bytes(const void* data, size_t size, uint64_t seed) {
  const XXH128_hash_t hash = XXH3_128bits_withSeed(data, size, seed);
  XXH3Key key;
  std::memcpy(key.data, &hash, sizeof(hash));
  return key;
}

XXH3Key hash_string(std::string_view value, uint64_t seed) {
  return hash_bytes(value.data(), value.size(), seed);
}

}  // namespace xllm
