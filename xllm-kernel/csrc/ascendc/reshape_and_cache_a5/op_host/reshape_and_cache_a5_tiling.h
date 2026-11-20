/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReshapeAndCacheA5TilingData)
TILING_DATA_FIELD_DEF(uint32_t, num_tokens);
TILING_DATA_FIELD_DEF(uint32_t, row_elements);
TILING_DATA_FIELD_DEF(uint32_t, total_slots);
TILING_DATA_FIELD_DEF(uint32_t, tile_elements);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(XllmKernelReshapeAndCacheA5,
                           ReshapeAndCacheA5TilingData)
}  // namespace optiling
