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

namespace xllm {

class DraftProposal;
class GraphInputView;
struct SampleOutput;
struct ForwardOutput;

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

struct SamplerPolicy {
  torch::Tensor do_sample;
  bool all_random_sample = false;
  bool all_greedy_sample = false;
};

SampleOutput run_rejection_sampling(const SamplerPolicy& policy,
                                    const DraftProposal& draft_proposal,
                                    const torch::Tensor& target_logits,
                                    const ForwardOutput& target_output,
                                    const torch::Tensor& bonus_token_ids,
                                    bool enable_fused_kernel);

}  // namespace spec_verify

}  // namespace xllm
