/* Copyright 2025-2026 The xLLM Authors.

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

namespace xllm {

class Sequence;
struct ModelArgs;

std::string graph_warmup_progress(int32_t completed,
                                  int32_t total,
                                  int32_t token_bucket,
                                  double latency_ms);

// Returns a process-unique request id for synthetic profiling/warmup requests.
// Distinct ids keep these requests separable from each other (and from real
// requests) in the embedding cache, so stale decode state from a recycled
// embedding block cannot be mistaken for a warmup request's own state.
std::string next_warmup_request_id();

void prepare_warmup_decode_sequence(Sequence& sequence,
                                    const ModelArgs& model_args,
                                    int32_t num_speculative_tokens);

}  // namespace xllm
