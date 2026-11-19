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
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(T5SplitQkvAppendCacheTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, heads);
TILING_DATA_FIELD_DEF(uint32_t, length);
TILING_DATA_FIELD_DEF(uint32_t, dim);
TILING_DATA_FIELD_DEF(uint32_t, tile_elements);
TILING_DATA_FIELD_DEF(uint32_t, query_tiles);
TILING_DATA_FIELD_DEF(uint32_t, cache_tiles);
TILING_DATA_FIELD_DEF(uint32_t, total_tiles);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(T5SplitQkvAppendCache,
                           T5SplitQkvAppendCacheTilingData)
}  // namespace optiling
