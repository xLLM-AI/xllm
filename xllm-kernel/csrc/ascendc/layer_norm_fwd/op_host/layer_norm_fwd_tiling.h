/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LayerNormFwdTilingData)
TILING_DATA_FIELD_DEF(uint32_t, m);
TILING_DATA_FIELD_DEF(uint32_t, full_n);
TILING_DATA_FIELD_DEF(uint32_t, group_size);
TILING_DATA_FIELD_DEF(uint32_t, ngroups);
TILING_DATA_FIELD_DEF(uint32_t, stride_x);
TILING_DATA_FIELD_DEF(uint32_t, stride_y);
TILING_DATA_FIELD_DEF(uint32_t, stride_z);
TILING_DATA_FIELD_DEF(uint32_t, group_align);
TILING_DATA_FIELD_DEF(uint32_t, tile_rows);
TILING_DATA_FIELD_DEF(uint32_t, chunk_size);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(uint32_t, has_bias);
TILING_DATA_FIELD_DEF(uint32_t, has_z);
TILING_DATA_FIELD_DEF(uint32_t, norm_before_gate);
TILING_DATA_FIELD_DEF(uint32_t, is_rms_norm);
TILING_DATA_FIELD_DEF(uint32_t, use_a2_scalar_store_fix);
TILING_DATA_FIELD_DEF(uint32_t, kernel_mode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(XllmKernelLayerNormFwd, LayerNormFwdTilingData)
}  // namespace optiling
