/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MegaGdnPrefillOpTilingData)
TILING_DATA_FIELD_DEF(uint32_t, block_dim);
TILING_DATA_FIELD_DEF(uint32_t, vector_task_count);
TILING_DATA_FIELD_DEF(uint32_t, target_arch);
TILING_DATA_FIELD_DEF(uint32_t, num_matrices);
TILING_DATA_FIELD_DEF(uint32_t, batch_size);
TILING_DATA_FIELD_DEF(uint32_t, num_heads);
TILING_DATA_FIELD_DEF(uint32_t, num_key_heads);
TILING_DATA_FIELD_DEF(uint32_t, token_block_size);
TILING_DATA_FIELD_DEF(uint32_t, token_block_count);
TILING_DATA_FIELD_DEF(uint32_t, base_dim);
TILING_DATA_FIELD_DEF(uint32_t, base_dim_count);
TILING_DATA_FIELD_DEF(uint32_t, conv_dim);
TILING_DATA_FIELD_DEF(uint32_t, conv_state_slots);
TILING_DATA_FIELD_DEF(uint32_t, conv_state_len);
TILING_DATA_FIELD_DEF(uint32_t, ssm_state_slots);
TILING_DATA_FIELD_DEF(uint32_t, checkpoint_stride);
TILING_DATA_FIELD_DEF(int64_t, total_tokens);
TILING_DATA_FIELD_DEF(uint64_t, ffts_addr);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(XllmKernelMegaGdnPrefillOp,
                           MegaGdnPrefillOpTilingData)
}  // namespace optiling
