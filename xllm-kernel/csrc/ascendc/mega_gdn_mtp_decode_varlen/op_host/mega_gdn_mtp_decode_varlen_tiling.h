/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MegaGdnMtpDecodeVarlenTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch_size);
TILING_DATA_FIELD_DEF(int64_t, sequence_length);
TILING_DATA_FIELD_DEF(int64_t, num_k_heads);
TILING_DATA_FIELD_DEF(int64_t, num_v_heads);
TILING_DATA_FIELD_DEF(int64_t, state_index_width);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(XllmKernelMegaGdnMtpDecodeVarlen,
                           MegaGdnMtpDecodeVarlenTilingData)

}  // namespace optiling
