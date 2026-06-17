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

#include <cstdint>

namespace xllm {

// DeepSeek V4 MLU cache index mapping for one DSA layer.
struct DSACacheMapping {
  int64_t cmp_cache_idx = -1;
  int64_t index_cache_idx = -1;
  int64_t ori_cache_idx = -1;
  int64_t kv_state_cache_idx = -1;
  int64_t score_state_cache_idx = -1;
  int64_t index_kv_state_cache_idx = -1;
  int64_t index_score_state_cache_idx = -1;
};

}  // namespace xllm
