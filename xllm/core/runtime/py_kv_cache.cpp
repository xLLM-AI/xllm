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

#include "core/runtime/py_kv_cache.h"

#include "core/util/pybind_helper.h"

namespace xllm::runtime {

pybind11::tuple make_python_kv_cache(const KVCache& kv_cache) {
  return pybind11::make_tuple(
      optional_tensor(kv_cache.get_k_cache()),
      optional_tensor(kv_cache.get_v_cache()),
      optional_tensor(kv_cache.get_index_cache()),
      optional_tensor(kv_cache.get_conv_cache()),
      optional_tensor(kv_cache.get_ssm_cache()),
      optional_tensor(kv_cache.get_swa_cache()),
      optional_tensor(kv_cache.get_compress_kv_state()),
      optional_tensor(kv_cache.get_compress_score_state()),
      optional_tensor(kv_cache.get_compress_index_kv_state()),
      optional_tensor(kv_cache.get_compress_index_score_state()),
      optional_tensor(kv_cache.get_indexer_cache_scale()),
      optional_tensor(kv_cache.get_kpool_tail()));
}

}  // namespace xllm::runtime
