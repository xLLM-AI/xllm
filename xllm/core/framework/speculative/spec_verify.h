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

#pragma once

#include <torch/types.h>

#include "core/framework/sampling/sampling_params.h"

namespace xllm {

class DraftProposal;
class GraphInputView;
class SampleOutput;

namespace spec_verify {

// Expanded verify needs a primary block-table tensor for its generic graph
// metadata. Model-managed cache layouts instead carry every real cache table
// in multi_block_tables, so the primary table may be intentionally absent.
bool has_speculative_verify_block_table_layout(
    const torch::Tensor& block_tables,
    const std::vector<torch::Tensor>& multi_block_tables,
    int64_t num_sequences);

// Build the zero-filled primary control table used only by generic expanded
// metadata when the model owns its real cache layout through
// multi_block_tables.
torch::Tensor make_speculative_verify_control_block_table(
    int64_t num_sequences,
    int64_t block_table_capacity);

// Shared allocation/launch width for target verification block tables. The
// extra entry covers the speculative token that can cross a block boundary.
int64_t speculative_verify_block_table_capacity(int64_t max_position_embeddings,
                                                int64_t block_size);

// Fill deferred draft token columns for eager target verification. The graph
// override, when present, owns the returned tensor storage.
torch::Tensor materialize_speculative_verify_tokens(
    const torch::Tensor& verify_tokens,
    const std::vector<torch::Tensor>& draft_token_sources);
torch::Tensor materialize_graph_speculative_verify_tokens(
    const torch::Tensor& tokens,
    const GraphInputView& graph_input);

// Scores the RAW target logits (sampling transforms are applied inside the
// rejection_sample exactly once); `target_logits` arrives flat as
// [batch * val_tokens, vocab] — callers never pre-shape, and the validation
// width is derived from the draft proposal. Takes the sampling params by
// value so the rejection sampler owns them: params-owning callers (MTP's
// verify_params chain, suffix via validate) std::move through the
// forwarding layers for one copy per verify step; callers reading a const
// input (DFlash's run_validate) copy once at this boundary.
// `target_hidden_states`, when defined, is attached to the output as the
// [batch, val_tokens, hidden] view.
SampleOutput run_rejection_sampling(SamplingParameters sampling_params,
                                    const DraftProposal& draft_proposal,
                                    const torch::Tensor& target_logits,
                                    const torch::Tensor& target_hidden_states);
// Packed verification for a deterministic (point-mass) draft: sample each
// target row, then let the caller commit the prefix matching its proposals.
// owners maps packed rows to requests; -1 padding borrows request zero's
// parameters and must never be committed. Minimum-length controls additionally
// require columns: each packed row's prediction offset within its request.
// Penalties, other constraints, seeded sampling and logprobs are unsupported.
torch::Tensor rejection_sample_ragged(
    torch::Tensor& target_logits,
    const SamplingParameters& request_sampling,
    const torch::Tensor& owners,
    const torch::Tensor& columns = {});

}  // namespace spec_verify

}  // namespace xllm
