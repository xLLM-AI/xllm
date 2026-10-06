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

#include <algorithm>
#include <utility>

#include "framework/model/model_input_params.h"
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

bool has_speculative_verify_block_table_layout(
    const torch::Tensor& block_tables,
    const std::vector<torch::Tensor>& multi_block_tables,
    int64_t num_sequences) {
  if (num_sequences <= 0) {
    return false;
  }
  const auto is_compatible = [num_sequences](const torch::Tensor& table) {
    return table.defined() && table.dim() == 2 &&
           table.size(0) == num_sequences && table.size(1) > 0 &&
           table.scalar_type() == torch::kInt32;
  };
  if (block_tables.defined()) {
    return is_compatible(block_tables);
  }
  return !multi_block_tables.empty() && std::all_of(multi_block_tables.begin(),
                                                    multi_block_tables.end(),
                                                    is_compatible);
}

torch::Tensor make_speculative_verify_control_block_table(
    int64_t num_sequences,
    int64_t block_table_capacity) {
  CHECK_GT(num_sequences, 0);
  CHECK_GT(block_table_capacity, 0);
  return torch::zeros({num_sequences, block_table_capacity},
                      torch::TensorOptions()
                          .dtype(torch::kInt32)
                          .device(torch::kCPU)
                          .pinned_memory(true));
}

int64_t speculative_verify_block_table_capacity(int64_t max_position_embeddings,
                                                int64_t block_size) {
  CHECK_GT(max_position_embeddings, 0);
  CHECK_GT(block_size, 0);
  return (max_position_embeddings + block_size - 1) / block_size + 1;
}

torch::Tensor materialize_speculative_verify_tokens(
    const torch::Tensor& verify_tokens,
    const std::vector<torch::Tensor>& draft_token_sources) {
  if (draft_token_sources.empty()) {
    return verify_tokens;
  }
  CHECK(verify_tokens.defined());
  CHECK_EQ(verify_tokens.dim(), 1);
  const int64_t verify_width =
      static_cast<int64_t>(draft_token_sources.size()) + 1;
  CHECK_EQ(verify_tokens.numel() % verify_width, 0);
  const int64_t batch_size = verify_tokens.numel() / verify_width;
  torch::Tensor verify_rows = verify_tokens.view({batch_size, verify_width});
  for (size_t step = 0; step < draft_token_sources.size(); ++step) {
    const torch::Tensor& source = draft_token_sources[step];
    CHECK(source.defined());
    CHECK_EQ(source.numel(), batch_size);
    verify_rows.select(/*dim=*/1, static_cast<int64_t>(step) + 1)
        .copy_(source.flatten(), /*non_blocking=*/true);
  }
  return verify_tokens;
}

torch::Tensor materialize_graph_speculative_verify_tokens(
    const torch::Tensor& tokens,
    const GraphInputView& graph_input) {
  const torch::Tensor& verify_tokens =
      graph_input.input_tokens_override.defined()
          ? graph_input.input_tokens_override
          : tokens;
  return materialize_speculative_verify_tokens(
      verify_tokens, graph_input.spec_verify_draft_token_sources);
}

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
