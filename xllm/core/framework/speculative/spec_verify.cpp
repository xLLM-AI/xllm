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

#include "core/framework/speculative/spec_verify.h"

#include <glog/logging.h>

#include <utility>

#include "framework/sampling/rejection_sampler.h"
#include "runtime/forward_params.h"
#include "util/tensor_helper.h"

namespace xllm {

std::vector<SpeculativeTokenStats> calculate_contiguous_speculative_token_stats(
    const torch::Tensor& tokens,
    const std::vector<int32_t>& proposed_tokens) {
  CHECK(tokens.defined()) << "speculative output tokens are undefined";
  CHECK_EQ(tokens.dim(), 2) << "speculative output tokens should be 2D";
  const int64_t batch_size = tokens.size(0);
  const int64_t token_width = tokens.size(1);
  CHECK_EQ(proposed_tokens.size(), static_cast<size_t>(batch_size))
      << "proposed token count batch mismatch";

  const torch::Tensor int_tokens = to_cpu_contiguous(tokens, torch::kInt64);
  const int64_t* data = int_tokens.const_data_ptr<int64_t>();
  std::vector<SpeculativeTokenStats> sequence_stats(
      static_cast<size_t>(batch_size));
  for (int64_t row = 0; row < batch_size; ++row) {
    const int64_t proposed = proposed_tokens[static_cast<size_t>(row)];
    CHECK_GE(proposed, 0) << "proposed token count should not be negative";
    CHECK_LE(proposed + 1, token_width)
        << "proposed token count exceeds output width";

    SpeculativeTokenStats& stats = sequence_stats[static_cast<size_t>(row)];
    stats.proposed_tokens = proposed;
    const int64_t* row_ptr = data + row * token_width;
    for (int64_t column = 1; column < proposed + 1; ++column) {
      if (row_ptr[column] < 0) {
        break;
      }
      ++stats.accepted_tokens;
    }
  }
  return sequence_stats;
}

ForwardOutput finalize_verify_output(ForwardOutput target_output,
                                     SampleOutput val_output) {
  val_output.embeddings = torch::Tensor();
  target_output.sample_output = std::move(val_output);
  return target_output;
}

SpeculativeOutputStats calculate_speculative_output_stats(
    const torch::Tensor& tokens,
    int64_t num_speculative_tokens) {
  const int64_t batch_size = tokens.size(0);
  const int64_t token_width = tokens.size(1);
  const torch::Tensor int_tokens = to_cpu_contiguous(tokens, torch::kInt64);
  const int64_t* data = int_tokens.const_data_ptr<int64_t>();
  CHECK_LE(token_width, num_speculative_tokens + 1)
      << "next_tokens width exceeds num_speculative_tokens + 1.";
  SpeculativeOutputStats stats;
  stats.accepted_per_position.resize(
      static_cast<size_t>(num_speculative_tokens));
  stats.sequence_stats.resize(static_cast<size_t>(batch_size));
  for (int64_t row = 0; row < batch_size; ++row) {
    const int64_t* row_ptr = data + row * token_width;
    SpeculativeTokenStats& sequence_stats =
        stats.sequence_stats[static_cast<size_t>(row)];
    sequence_stats.proposed_tokens = token_width - 1;
    for (int64_t column = 0; column < token_width; ++column) {
      if (row_ptr[column] < 0) {
        continue;
      }
      ++stats.committed_tokens;
      if (column > 0) {
        ++stats.accepted_per_position[static_cast<size_t>(column - 1)];
        ++sequence_stats.accepted_tokens;
      }
    }
  }
  return stats;
}

namespace spec_verify {

SampleOutput run_rejection_sampling(const SamplerPolicy& policy,
                                    const DraftProposal& draft_proposal,
                                    const torch::Tensor& target_logits,
                                    const ForwardOutput& target_output,
                                    const torch::Tensor& bonus_token_ids,
                                    bool enable_fused_kernel) {
  const RejectionSampler sampler(policy.do_sample,
                                 policy.all_random_sample,
                                 policy.all_greedy_sample,
                                 target_output.logprobs,
                                 target_output.max_top_logprobs,
                                 enable_fused_kernel);

  SampleOutput sample_output =
      sampler.forward(draft_proposal.coerce_to(bonus_token_ids),
                      target_logits,
                      bonus_token_ids,
                      /*mask_out_rejected_tokens=*/true);

  const torch::Tensor& embeddings = target_output.sample_output.embeddings;
  sample_output.embeddings = embeddings.view(
      {target_logits.size(0), target_logits.size(1), embeddings.size(-1)});
  return sample_output;
}

}  // namespace spec_verify

}  // namespace xllm
