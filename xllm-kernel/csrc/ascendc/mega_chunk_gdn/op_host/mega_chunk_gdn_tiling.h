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

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MegaChunkGdnTilingData)
TILING_DATA_FIELD_DEF(uint32_t, block_dim);
TILING_DATA_FIELD_DEF(uint32_t, num_matrices);
TILING_DATA_FIELD_DEF(uint32_t, num_heads);
TILING_DATA_FIELD_DEF(uint32_t, num_key_heads);
TILING_DATA_FIELD_DEF(int64_t, has_initial_state);
TILING_DATA_FIELD_DEF(int64_t, batch_size);
TILING_DATA_FIELD_DEF(int64_t, seq_len);
TILING_DATA_FIELD_DEF(int64_t, total_tokens);
TILING_DATA_FIELD_DEF(uint64_t, ffts_addr);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(XllmKernelMegaChunkGdn, MegaChunkGdnTilingData)
}  // namespace optiling
