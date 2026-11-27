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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace xllm {

// Internal identity, scoped to a model instance. Never used as a wire group ID.
struct CachePartitionId {
  std::string model_id;
  uint32_t ordinal = 0;
};

enum class CacheComponentRole {
  KEY,
  VALUE,
  MLA_LATENT,
  MLA_ROPE,
  KEY_SCALE,
  VALUE_SCALE,
  MLA_SCALE,
  INDEX,
  INDEX_SCALE,
  INDEX_GATE,
  INDEX_VALID,
  KPOOL_TAIL,
  CONV,
  SSM,
};

enum class CacheScalarType {
  FLOAT16,
  BFLOAT16,
  FLOAT32,
  INT8,
  UINT8,
  FLOAT8_E4M3FN,
  FLOAT8_E5M2,
};

// A semantic payload is not necessarily a separate tensor: packed MLA stores
// latent, rotary keys and scales in one row. Packing remains a storage concern.
struct CacheComponent {
  CacheComponentRole role;
  CacheScalarType dtype;
  std::vector<int32_t> layers;
};

}  // namespace xllm
