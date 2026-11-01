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

namespace xllm {

class ModelArgs;
namespace runtime {
struct Options;
}

// Zero counts select the existing full-history/grouped cache path.
struct DraftSwaCachePlan {
  int64_t window_size = 0;
  int64_t num_blocks = 0;
};

// Opt in by draft model semantics, independently of deployment configuration.
bool should_use_draft_swa_cache(const ModelArgs& draft_args,
                                const runtime::Options& options);

DraftSwaCachePlan resolve_draft_swa_cache_plan(const ModelArgs& draft_args,
                                               const runtime::Options& options);

}  // namespace xllm
