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

#include <pybind11/pybind11.h>

#include "core/kv_cache/storage/kv_cache.h"

namespace xllm::runtime {

// Convert the engine-owned cache slots to the named Python LayerCache tuple.
// Keep the order synchronized with python/attention/backend.py.
pybind11::tuple make_python_kv_cache(const KVCache& kv_cache);

}  // namespace xllm::runtime
