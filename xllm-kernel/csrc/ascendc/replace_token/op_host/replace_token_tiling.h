/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReplaceTokenTilingData)
TILING_DATA_FIELD_DEF(int32_t, sequenceLength);
TILING_DATA_FIELD_DEF(int32_t, max_tokens);
TILING_DATA_FIELD_DEF(int32_t, blength);
// TILING_DATA_FIELD_DEF_ARR(int64_t,256,groupOffset);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(XllmKernelReplaceToken, ReplaceTokenTilingData)
}  // namespace optiling
