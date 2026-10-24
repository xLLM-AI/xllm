/* Copyright 2026 The xLLM Authors.

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

#pragma once

#include "core/framework/model/model_args.h"

namespace xllm {

// Apply only when selecting the Python CUDA FlashInfer backend, before cache
// estimation/allocation and exporting ModelArgs to Python. Model parsers retain
// logical dimensions; other backends must not apply this policy.
inline void configure_flashinfer_kv_cache_layout(ModelArgs& model_args) {
  // The validated FA2 path requires D=96 padded to 128. Projection, RoPE and
  // attention scale retain the logical width. MLA owns a different geometry;
  // already supported widths do not require additional storage.
  if (!model_args.enable_mla() && model_args.head_dim() == 96 &&
      model_args.kv_cache_storage_head_dim() == 0) {
    model_args.kv_cache_storage_head_dim() = 128;
  }
}

}  // namespace xllm
