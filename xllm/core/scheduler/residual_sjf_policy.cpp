/* Copyright 2026 The xLLM Authors.

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

#include <absl/time/clock.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/framework/config/scheduler_config.h"
#include "core/util/verbose_trace_logger.h"
#include "framework/request/priority_comparator.h"
#include "scheduler/scheduler_policy.h"

namespace xllm {

namespace {

// Residual local prefill cost in KV blocks for a waiting PD-prefill request:
// floor(prompt tokens / block size) minus the locally cached full blocks.
// Without a KV manager the cost degrades to 0 (FCFS by arrival).
size_t get_residual_sjf_cost_blocks(KVCacheManager* kv_cache_manager,
                                    const std::shared_ptr<Request>& request) {
  CHECK(!request->sequences().empty());
  Sequence* sequence = request->sequences()[0].get();
  CHECK(sequence != nullptr);
  const size_t num_cached_blocks =
      kv_cache_manager == nullptr
          ? 0
          : kv_cache_manager->get_num_local_computed_blocks(sequence);
  const int32_t block_size =
      kv_cache_manager == nullptr ? 0 : kv_cache_manager->block_size();
  return residual_sjf_cost_blocks(sequence->num_prompt_tokens(),
                                  num_cached_blocks,
                                  static_cast<size_t>(std::max(block_size, 0)));
}

}  // namespace

void ResidualSJFPolicy::schedule(
    SchedulerState& state,
    ScheduleBudget& budget,
    std::vector<std::shared_ptr<Request>>& finished) {
  const SchedulerConfig& scheduler_config = SchedulerConfig::get_instance();

  // Compute the residual cost once per waiting request per step so the sort
  // comparator stays cheap: the prefix-cache probe walks prompt-length block
  // hashes, and block hashes are memoized on the sequence after the first
  // computation.
  std::unordered_map<const Request*, size_t> residual_cost_blocks;
  residual_cost_blocks.reserve(state.prefill_queue.size());
  for (auto it = state.prefill_queue.begin(); it != state.prefill_queue.end();
       ++it) {
    residual_cost_blocks.emplace(
        (*it).get(), get_residual_sjf_cost_blocks(state.kv_cache_manager, *it));
  }

  const ResidualSJFComparator comparator(
      [&residual_cost_blocks](
          const std::shared_ptr<Request>& request) -> size_t {
        const auto found = residual_cost_blocks.find(request.get());
        return found == residual_cost_blocks.end() ? 0 : found->second;
      },
      scheduler_config.residual_sjf_max_wait_ms(),
      absl::Now());
  state.prefill_queue.sort(comparator);

  // Opt-in dispatch-order trace: one entry per waiting request, in the order
  // this step will drain the prefill queue, with its residual cost in blocks.
  // Enabled only via --enable_verbose_trace_log, so the hot path stays free
  // unless a validation run asks for it.
  int32_t queue_rank = 0;
  for (auto it = state.prefill_queue.begin(); it != state.prefill_queue.end();
       ++it) {
    const auto found = residual_cost_blocks.find((*it).get());
    const size_t cost = found == residual_cost_blocks.end() ? 0 : found->second;
    XLLM_VERBOSE_TRACE() << "event=residual_sjf_order rank=" << queue_rank
                         << " req=" << (*it)->request_id()
                         << " cost_blocks=" << static_cast<int64_t>(cost);
    ++queue_rank;
  }

  PrefillFirstPolicy::schedule(state, budget, finished);
}

}  // namespace xllm
