/* Copyright 2025-2026 The xLLM Authors.

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

#include "dit_non_cache.h"

namespace xllm {

void DiTNonCache::init(const DiTCacheConfig& /*cfg*/,
                       const ParallelArgs& /*parallel_args*/) {
  // NonCache: nothing to initialize.
}

bool DiTNonCache::on_before_block(const CacheBlockIn& blockin) { return false; }

CacheBlockOut DiTNonCache::on_after_block(const CacheBlockIn& blockin) {
  TensorMap out_map;
  out_map["hidden_states"] =
      get_tensor_or_empty(blockin.tensors, "hidden_states");
  out_map["encoder_hidden_states"] =
      get_tensor_or_empty(blockin.tensors, "encoder_hidden_states");
  return CacheBlockOut(out_map);
}

bool DiTNonCache::on_before_step(const CacheStepIn& stepin) { return false; }

CacheStepOut DiTNonCache::on_after_step(const CacheStepIn& stepin) {
  TensorMap out;
  out["hidden_states"] = get_tensor_or_empty(stepin.tensors, "hidden_states");
  return CacheStepOut(out);
}

}  // namespace xllm
