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

#include "core/framework/batch/batch_builder.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "core/common/metrics.h"

namespace xllm {

BatchBuilder::BatchBuilder(int32_t dp_size) : dp_size_(dp_size) {
  CHECK_GT(dp_size_, 0);
}

BatchGroup BatchBuilder::build(
    const std::vector<std::shared_ptr<Request>>& requests,
    const std::vector<Sequence*>& sequences,
    const std::vector<size_t>& budgets,
    std::vector<std::vector<BlockTransferInfo>>* swap_infos) const {
  CHECK_EQ(sequences.size(), budgets.size())
      << "Each scheduled sequence requires one token budget";
  if (swap_infos != nullptr) {
    CHECK_EQ(swap_infos->size(), static_cast<size_t>(dp_size_));
  }

  std::vector<size_t> sequence_counts(dp_size_);
  std::vector<size_t> group_counts(dp_size_);
  size_t num_prompt_tokens = 0;
  size_t num_generated_tokens = 0;

  // Validate and count before assembling or consuming pending block transfers.
  for (size_t i = 0; i < sequences.size(); ++i) {
    auto* sequence = sequences[i];
    CHECK(sequence != nullptr);
    CHECK(!sequence->finished());
    CHECK_GE(sequence->dp_rank(), 0);
    CHECK_LT(sequence->dp_rank(), dp_size_);
    CHECK_GT(budgets[i], 0);
    CHECK_LE(budgets[i], std::numeric_limits<uint32_t>::max());
    ++sequence_counts[sequence->dp_rank()];
    const size_t cached_tokens = sequence->kv_state().kv_cache_tokens_num();
    const size_t remaining_prompt_tokens =
        sequence->num_prompt_tokens() > cached_tokens
            ? sequence->num_prompt_tokens() - cached_tokens
            : 0;
    const size_t prompt_tokens = std::min(remaining_prompt_tokens, budgets[i]);
    num_prompt_tokens += prompt_tokens;
    num_generated_tokens += budgets[i] - prompt_tokens;
  }

  bool retain_request_groups = false;
  for (const auto& request : requests) {
    CHECK(request != nullptr);
    retain_request_groups |= request->check_beam_search();
  }
  if (retain_request_groups) {
    for (const auto& request : requests) {
      auto* group = request->sequence_group();
      CHECK(group != nullptr);
      CHECK(!group->sequences().empty());
      CHECK_GE(group->dp_rank(), 0);
      CHECK_LT(group->dp_rank(), dp_size_);
      ++group_counts[group->dp_rank()];
    }
  }

  BatchGroup batches(static_cast<size_t>(dp_size_));
  for (int32_t rank = 0; rank < dp_size_; ++rank) {
    batches[rank].reserve(sequence_counts[rank], group_counts[rank]);
  }
  for (size_t i = 0; i < sequences.size(); ++i) {
    batches[sequences[i]->dp_rank()].add(sequences[i],
                                         static_cast<uint32_t>(budgets[i]));
  }
  if (retain_request_groups) {
    for (const auto& request : requests) {
      auto* group = request->sequence_group();
      batches[group->dp_rank()].add(group);
    }
  }
  if (swap_infos != nullptr) {
    for (int32_t rank = 0; rank < dp_size_; ++rank) {
      if (batches[rank].empty()) {
        continue;
      }
      batches[rank].set_swap_block_transfer_infos(
          std::move((*swap_infos)[rank]));
      (*swap_infos)[rank].clear();
    }
  }

  COUNTER_ADD(num_processing_tokens_total_prompt, num_prompt_tokens);
  COUNTER_ADD(num_processing_tokens_total_generated, num_generated_tokens);
  if (!sequences.empty()) {
    HISTOGRAM_OBSERVE(
        num_prompt_tokens_per_request,
        static_cast<int64_t>(num_prompt_tokens / sequences.size()));
    HISTOGRAM_OBSERVE(
        num_generated_tokens_per_request,
        static_cast<int64_t>(num_generated_tokens / sequences.size()));
  }
  return batches;
}

}  // namespace xllm
