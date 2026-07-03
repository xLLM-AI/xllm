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

#include <cstddef>
#include <memory>
#include <vector>

#include "framework/batch/batch.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "scheduler/disagg_pd_scheduler.h"

namespace xllm {

struct PDChunkBudget {
  size_t next_tokens = 0;
  size_t max_tokens = 0;
};

PDChunkBudget pick_pd_chunk_budget(size_t kv_tokens,
                                   size_t num_tokens,
                                   size_t max_chunk,
                                   size_t remaining_budget);

// Number of KV-cache blocks a prefill sequence must still allocate from the
// free pool to reach its full prompt length. `held_blocks` is how many blocks
// the sequence already holds (kv_state().num_blocks(BlockType::KV)); because
// that count includes any prefix-cache-shared blocks, a sequence reusing a long
// shared prefix reserves proportionally fewer blocks. Returns 0 once the prompt
// is fully covered or when block_size is 0. Used by the completion-invariant
// admission gate so that a set of concurrently-prefilling requests can never
// over-subscribe the KV cache (hold-and-wait deadlock).
size_t pd_prefill_remaining_blocks(size_t num_prompt_tokens,
                                   size_t held_blocks,
                                   size_t block_size);

// Whether a fresh prefill request's COMPLETE footprint still fits total
// capacity on top of what is already reserved. `reserved_blocks` is the
// complete footprint already reserved for the in-flight set plus fresh requests
// admitted earlier this step (shared across the online and offline queues);
// `request_full_blocks` is this request's own complete footprint
// (ceil(prompt/block_size)); `total_blocks` is the whole KV capacity. Reserving
// each started request's COMPLETE footprint against TOTAL (rather than its
// shrinking remainder against the momentary free count) is what serializes
// near-capacity prompts: it stops new starts from outrunning completions, which
// is the actual cause of the hold-and-wait deadlock (the PD prefill hang). The
// caller bypasses this check for an already in-flight request (must continue)
// and for the sole request in flight (so a lone oversized prompt reaches the
// exceeds_block_capacity failure path instead of hanging).
bool pd_prefill_footprint_fits(size_t reserved_blocks,
                               size_t request_full_blocks,
                               size_t total_blocks);

class DisaggPDChunkedPrefillScheduler final : public DisaggPDScheduler {
 public:
  DisaggPDChunkedPrefillScheduler(Engine* engine, const Options& options);
  ~DisaggPDChunkedPrefillScheduler() override = default;

 protected:
  std::vector<Batch> prepare_batch() override;

 private:
  bool alloc_chunk(Sequence* sequence,
                   size_t token_budget,
                   size_t* actual_tokens);
  void match_prefix_blocks(Sequence* sequence);
  void schedule_waiting_prefill(RequestPriorityQueue& queue,
                                size_t& remaining_token_budget,
                                size_t& remaining_seq_budget,
                                size_t total_blocks,
                                size_t& reserved_blocks,
                                std::vector<std::shared_ptr<Request>>& done);
  void update_metrics();
};

}  // namespace xllm
