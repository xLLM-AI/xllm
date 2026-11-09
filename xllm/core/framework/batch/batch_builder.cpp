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

#include <absl/container/inlined_vector.h>
#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "core/common/metrics.h"

namespace xllm {

BatchBuilder::BatchBuilder(int32_t dp_size) : dp_size_(dp_size) {
  CHECK_GT(dp_size_, 0);
}

BatchGroup BatchBuilder::build(const std::vector<Sequence*>& sequences,
                               const std::vector<size_t>& budgets) const {
  CHECK_EQ(sequences.size(), budgets.size())
      << "Each scheduled sequence requires one token budget";

  absl::InlinedVector<size_t, 8> sequence_counts(dp_size_);
  // Plain LLM sequences do not carry embeddings. Allocate this only when the
  // first embedding is observed, then reserve the exact per-rank count.
  absl::InlinedVector<size_t, 8> embedding_counts;
  size_t num_prompt_tokens = 0;
  size_t num_generated_tokens = 0;

  // Validate ranks and budgets before allocating per-rank batches. Batch::add
  // checks sequence completion before pending block transfers are consumed.
  for (size_t i = 0; i < sequences.size(); ++i) {
    auto* sequence = sequences[i];
    CHECK(sequence != nullptr);
    CHECK_GE(sequence->dp_rank(), 0);
    CHECK_LT(sequence->dp_rank(), dp_size_);
    CHECK_GT(budgets[i], 0);
    CHECK_LE(budgets[i], std::numeric_limits<uint32_t>::max());
    ++sequence_counts[sequence->dp_rank()];
    if (sequence->has_input_embedding()) {
      if (embedding_counts.empty()) {
        embedding_counts.resize(dp_size_);
      }
      ++embedding_counts[sequence->dp_rank()];
    }
    const size_t cached_tokens = sequence->kv_state().kv_cache_tokens_num();
    const size_t remaining_prompt_tokens =
        sequence->num_prompt_tokens() > cached_tokens
            ? sequence->num_prompt_tokens() - cached_tokens
            : 0;
    const size_t prompt_tokens = std::min(remaining_prompt_tokens, budgets[i]);
    num_prompt_tokens += prompt_tokens;
    num_generated_tokens += budgets[i] - prompt_tokens;
  }

  BatchGroup batches(static_cast<size_t>(dp_size_));
  for (int32_t rank = 0; rank < dp_size_; ++rank) {
    batches[rank].reserve(
        sequence_counts[rank],
        /*group_count=*/0,
        embedding_counts.empty() ? 0 : embedding_counts[rank]);
  }
  for (size_t i = 0; i < sequences.size(); ++i) {
    batches[sequences[i]->dp_rank()].add(sequences[i],
                                         static_cast<uint32_t>(budgets[i]));
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
