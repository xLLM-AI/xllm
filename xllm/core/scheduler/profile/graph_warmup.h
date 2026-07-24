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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/types.h"

namespace xllm {

class Sequence;

enum class GraphWarmupPlan : int8_t {
  UNIFIED = 0,
  PREFILL_ONLY = 1,
  DECODE_ONLY = 2,
};

GraphWarmupPlan graph_warmup_plan(InstanceRole role);

std::vector<int32_t> graph_warmup_buckets(int32_t max_seqs_per_batch);

bool skip_graph_bucket(int32_t bucket, int32_t dp_size);

std::vector<int32_t> graph_decode_buckets(int32_t max_seqs_per_batch,
                                          int32_t dp_size);

std::string graph_warmup_progress(int32_t completed,
                                  int32_t total,
                                  int32_t bucket,
                                  double latency_ms);

// Returns a process-unique request id for synthetic profiling/warmup requests.
// Distinct ids keep these requests separable from each other (and from real
// requests) in the embedding cache, so stale decode state from a recycled
// embedding block cannot be mistaken for a warmup request's own state.
std::string next_warmup_request_id();

// Prepares a synthetic decode sequence for graph warmup. When speculative
// decoding is enabled (MTP), the worker's decode path requires a valid decode
// state written through the MTP bootstrap channel before it validates the
// per-token decode state. This injects a placeholder bootstrap embedding of
// shape [1, embedding_width] so the bootstrap path runs during graph capture;
// the embedding values are irrelevant because warmup only captures the graph.
// Does nothing when speculative decoding is disabled.
void prepare_warmup_decode_sequence(Sequence* sequence,
                                    int64_t embedding_width,
                                    int32_t num_speculative_tokens);

}  // namespace xllm
