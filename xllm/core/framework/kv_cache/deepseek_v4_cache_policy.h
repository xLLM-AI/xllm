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

#include <torch/torch.h>

#include <cstdint>

namespace xllm {

struct DeepSeekV4CachePolicy {
  torch::ScalarType index_dtype = torch::kInt8;
  int64_t index_dtype_size = 1;
  bool has_indexer_cache_scale = true;
  torch::ScalarType scale_dtype = torch::kFloat16;
  int64_t scale_dtype_size = 2;
};

inline DeepSeekV4CachePolicy get_dsv4_cache_policy(
    torch::ScalarType model_dtype) {
  DeepSeekV4CachePolicy policy;
#if defined(USE_MLU)
  policy.index_dtype = model_dtype;
  policy.index_dtype_size =
      static_cast<int64_t>(torch::elementSize(model_dtype));
  policy.has_indexer_cache_scale = false;
  policy.scale_dtype_size = 0;
#else
  policy.index_dtype_size =
      static_cast<int64_t>(torch::elementSize(policy.index_dtype));
  policy.scale_dtype_size =
      static_cast<int64_t>(torch::elementSize(policy.scale_dtype));
#endif
  return policy;
}

}  // namespace xllm
