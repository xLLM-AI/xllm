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
                                std::vector<std::shared_ptr<Request>>& done);
  void update_metrics();
};

}  // namespace xllm
