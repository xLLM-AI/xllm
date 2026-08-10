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

#include <string>

namespace xllm {

// The multi-SLO priority strategy always drives the chunked prefill scheduling
// path, so a service can run chunked prefill without setting the raw
// enable_chunked_prefill flag. Both the scheduler batch-mode resolution and the
// DCP startup gate must agree on this effective value; sharing this single
// predicate keeps them from drifting apart.
inline bool resolve_effective_chunked_prefill(
    bool raw_chunked_prefill,
    const std::string& priority_strategy) {
  return raw_chunked_prefill || priority_strategy == "multi_slo_and_prio";
}

}  // namespace xllm
