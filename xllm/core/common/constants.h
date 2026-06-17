/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <cstdint>

namespace xllm {

inline constexpr char kInferContentLength[] = "Infer-Content-Length";
inline constexpr char kContentLength[] = "Content-Length";

// Reserved row 0 of the linear / SSM state cache, used as the padding slot for
// padded decode batch rows. This mirrors the KV cache convention, where block
// id 0 is permanently held for padding (see BlockManagerImpl and
// SingleBlockManager). Keeping both caches padded to row 0 makes the padding
// contract uniform across attention and linear-attention layers; real
// sequences are handed ids in [1, num_blocks - 1] only.
inline constexpr int32_t kPaddingLinearStateId = 0;

}  // namespace xllm
