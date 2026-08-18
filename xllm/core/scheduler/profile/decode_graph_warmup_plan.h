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

#include <cstdint>
#include <vector>

#include "runtime/decode_graph_bucket.h"

namespace xllm {

// A graph warmup schedule for decode. batch_sizes contains global scheduler
// batch sizes, not per-DP-rank sizes; the plan builder accounts for dp_size
// when selecting them. The profile manager executes these entries to capture
// the graph shapes an executor can later replay.
struct DecodeGraphWarmupPlan {
  runtime::DecodeGraphExecutionShape execution_shape;
  std::vector<int32_t> batch_sizes;
};

// Returns the legacy, backend-neutral decode schedule. Keep this schedule for
// a backend until it explicitly advertises a more precise graph warmup
// capability through Platform.
DecodeGraphWarmupPlan get_compatibility_decode_graph_warmup_plan(
    int32_t max_global_batch_size,
    int32_t dp_size);

// Builds the decode graph warmup schedule from the Engine's effective runtime
// shape. Backends that support MTP token-bucket warmup replace the legacy
// schedule only for padded multi-token decode; all other cases preserve the
// compatibility schedule above.
DecodeGraphWarmupPlan build_decode_graph_warmup_plan(
    const runtime::DecodeGraphExecutionShape& execution_shape,
    int32_t max_global_batch_size,
    int32_t dp_size);

}  // namespace xllm
