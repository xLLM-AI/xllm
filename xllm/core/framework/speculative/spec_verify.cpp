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

#include "framework/model/model_input_params.h"
#include "framework/sampling/rejection_sampler.h"
#include "runtime/forward_params.h"

namespace xllm {

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
