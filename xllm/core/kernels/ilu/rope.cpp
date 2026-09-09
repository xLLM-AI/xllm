/* Copyright 2025-2026 The xLLM Authors.

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

#include "ilu_ops_api.h"
#include "utils.h"

namespace xllm::kernel::ilu {

void apply_rope_pos_ids_cos_sin_cache(torch::Tensor& query,
                                      torch::Tensor& key,
                                      torch::Tensor& cos_sin_cache,
                                      torch::Tensor& positions,
                                      bool interleave) {
  const int64_t head_size = cos_sin_cache.size(-1);
  infer::xllm_rotary_embedding(
      positions, query, key, head_size, cos_sin_cache, !interleave);
}

}  // namespace xllm::kernel::ilu
