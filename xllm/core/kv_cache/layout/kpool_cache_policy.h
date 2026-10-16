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

// This is a cache-layout policy, not a model configuration property. Keep the
// platform/model rollout gate beside the cache layout consumers so Python
// model setup, capacity estimation and KV transfer use one decision.
inline bool uses_compressed_kpool_tail_cache(const ModelArgs& args) {
#if defined(USE_NPU)
  return args.index_kpool() > 1 && args.index_kpool_compress() &&
         args.index_kpool_always_select_tail() &&
         is_glm5_next_target_model_type(args.model_type());
#else
  (void)args;
  return false;
#endif
}

}  // namespace xllm
