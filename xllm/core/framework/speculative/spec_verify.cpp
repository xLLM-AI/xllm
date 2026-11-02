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

#include "core/framework/model/model_params_view.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/speculative/rejection_sampler.h"

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

SampleOutput run_rejection_sampling(SamplingParameters sampling_params,
                                    const DraftProposal& draft_proposal,
                                    const torch::Tensor& target_logits,
                                    const torch::Tensor& target_hidden_states) {
  SampleOutput sample_output =
      rejection_sample(sampling_params, draft_proposal, target_logits);

  if (target_hidden_states.defined()) {
    // Reuse next_tokens' [batch, num_val_tokens] shape instead of
    // re-deriving the width.
    const torch::Tensor& next_tokens = sample_output.next_tokens;
    sample_output.embeddings =
        target_hidden_states.view({next_tokens.size(0),
                                   next_tokens.size(1),
                                   target_hidden_states.size(-1)});
  }
  return sample_output;
}

torch::Tensor rejection_sample_ragged(
    torch::Tensor& target_logits,
    const SamplingParameters& request_sampling,
    const torch::Tensor& owners,
    const torch::Tensor& columns) {
  CHECK_EQ(target_logits.dim(), 2);
  CHECK_EQ(owners.dim(), 1);
  CHECK_EQ(owners.numel(), target_logits.size(0));
  CHECK_EQ(owners.device(), target_logits.device());
  CHECK(owners.scalar_type() == torch::kInt32 ||
        owners.scalar_type() == torch::kInt64);
  CHECK(request_sampling.selected_token_idxes().defined());
  CHECK_GT(request_sampling.num_sample_seqs(), 0);
  if (request_sampling.sampling_mode() == SamplingMode::MIXED) {
    CHECK(request_sampling.do_sample().defined());
  }
  CHECK(!request_sampling.logprobs()) << "logprobs unsupported";
  const bool minimum = request_sampling.min_token_ids().defined();
  CHECK_EQ(minimum, request_sampling.min_token_positions().defined());
  CHECK_EQ(minimum, request_sampling.min_tokens_remaining().defined());
  // Scalar transforms and the compact minimum-length controls have explicit
  // ragged mappings. Other penalties, seeds and constraints remain unsupported.
  constexpr std::array<SamplingTensorAccessor, 4> kRequestFields = {
      SamplingTensorAccessor{&SamplingParameters::temperatures,
                             &SamplingParameters::temperatures},
      SamplingTensorAccessor{&SamplingParameters::top_k,
                             &SamplingParameters::top_k},
      SamplingTensorAccessor{&SamplingParameters::top_p,
                             &SamplingParameters::top_p},
      SamplingTensorAccessor{&SamplingParameters::min_p,
                             &SamplingParameters::min_p}};
  for (const auto& field : kSamplingFields) {
    const torch::Tensor& values = field.member(request_sampling);
    const bool packed_field =
        std::find(kRequestFields.begin(), kRequestFields.end(), field.member) !=
        kRequestFields.end();
    const bool minimum_field =
        field.member ==
            SamplingTensorAccessor{&SamplingParameters::min_token_positions,
                                   &SamplingParameters::min_token_positions} ||
        field.member ==
            SamplingTensorAccessor{&SamplingParameters::min_tokens_remaining,
                                   &SamplingParameters::min_tokens_remaining} ||
        field.member ==
            SamplingTensorAccessor{&SamplingParameters::min_token_ids,
                                   &SamplingParameters::min_token_ids};
    CHECK(packed_field || minimum_field || !values.defined())
        << "unsupported sampling field for ragged sampling";
  }

  // Padding rows borrow another request's logits; give them a valid
  // distribution without touching real request rows.
  target_logits.masked_fill_(owners.lt(0).unsqueeze(1), 0);
  const torch::Tensor request_rows = owners.clamp_min(0).to(torch::kInt64);
  SamplingParameters packed;
  packed.sampling_mode() = request_sampling.sampling_mode();
  for (const auto member : kRequestFields) {
    const auto& values = member(request_sampling);
    if (values.defined()) {
      CHECK_EQ(values.device(), owners.device());
      CHECK_EQ(values.dim(), 1);
      CHECK_EQ(values.numel(), request_sampling.num_sample_seqs());
      member(packed) = values.index_select(/*dim=*/0, request_rows);
    }
  }
  if (request_sampling.sampling_mode() == SamplingMode::MIXED) {
    packed.do_sample() =
        request_sampling.do_sample().index_select(/*dim=*/0, request_rows);
  }
  if (minimum) {
    CHECK(columns.defined());
    CHECK_EQ(columns.sizes(), owners.sizes());
    CHECK_EQ(columns.device(), owners.device());
    CHECK(columns.scalar_type() == torch::kInt32 ||
          columns.scalar_type() == torch::kInt64);
    CHECK(request_sampling.min_tokens_remaining().defined());
    CHECK_EQ(request_sampling.min_token_ids().device(), owners.device());
    CHECK_EQ(request_sampling.min_tokens_remaining().device(), owners.device());
    packed.min_token_ids() = request_sampling.min_token_ids().index_select(
        /*dim=*/0, request_rows);
    packed.min_tokens_remaining() =
        request_sampling.min_tokens_remaining().index_select(
            /*dim=*/0, request_rows) -
        columns.to(torch::kInt64);
    // Uncommitted padding must retain a finite distribution, even if request
    // zero suppresses token zero until its minimum is reached.
    packed.min_tokens_remaining().masked_fill_(owners.lt(0), 0);
  }
  // Residual X|X!=d ~ p/(1-p(d)); the caller establishes TP consensus
  // before committing KV.
  return sample(target_logits, packed).next_tokens;
}

}  // namespace spec_verify

}  // namespace xllm
