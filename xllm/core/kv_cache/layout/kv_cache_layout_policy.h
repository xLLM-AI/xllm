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

#include <cstdint>
#include <string>
#include <vector>

namespace xllm {

// for qwen3.5
bool is_linear_attention_layer(
    int64_t layer_idx,
    int64_t full_attention_interval,
    const std::vector<std::string>& layer_types = {});

// Whether NPU KV cache should use FRACTAL_NZ layout for a model type.
bool use_npu_nz_kv_cache_layout(const std::string& model_type);

}  // namespace xllm
