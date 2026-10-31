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

namespace xllm::layer {

// Spec verification keeps its dense per-sequence layout, even when the
// executor marks it as chunked prefill. Decode never uses packed prefill.
constexpr bool use_packed_gdn_prefill(bool is_prefill,
                                      bool is_chunked_prefill,
                                      bool is_spec_verify) {
  return !is_spec_verify && (is_prefill || is_chunked_prefill);
}

}  // namespace xllm::layer
