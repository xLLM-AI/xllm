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

#include <torch/torch.h>

namespace xllm {

struct SampleOutput;
struct ForwardInput;
struct ForwardOutput;
class EmbeddingCache;

// Selects the target/draft hidden that feeds the next draft-extend step. Target
// prefill exposes the full hidden as `embeddings` and the per-sequence selected
// rows as `selected_embeddings`; draft/decode expose only the selected rows as
// `embeddings`. `gathered_sample_hidden_states` supplies pre-gathered rows for
// the CP draft/decode shard, whose all-gather-space `logits_indices` cannot
// re-index the local hidden.
void output_spec_hidden_states(
    SampleOutput& sample_output,
    const torch::Tensor& hidden_states,
    const torch::Tensor& aux_hidden_states,
    const torch::Tensor& logits_indices,
    const torch::Tensor& gathered_sample_hidden_states,
    bool is_target_prefill,
    bool cp_enabled);

void clear_all_output_embeddings(ForwardOutput& output);

// Writes the target prefill context to the embedding cache and exposes the
// selected hidden as `embeddings` for the PD handoff to the first draft-extend
// step.
void prepare_first_draft_inputs(EmbeddingCache& embedding_cache,
                                const ForwardInput& input,
                                ForwardOutput& output);

}  // namespace xllm
