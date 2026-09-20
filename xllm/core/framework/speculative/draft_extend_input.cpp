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

#include "core/framework/speculative/draft_extend_input.h"

#include <torch/torch.h>

#include "common/macros.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/embedding_cache.h"
#include "runtime/forward_params.h"

namespace xllm {

namespace {

void clear_selected_embeddings(ForwardOutput& output) {
  output.sample_output.selected_embeddings = torch::Tensor();
}

}  // namespace

void output_spec_hidden_states(
    SampleOutput& sample_output,
    const torch::Tensor& hidden_states,
    const torch::Tensor& aux_hidden_states,
    const torch::Tensor& logits_indices,
    const torch::Tensor& gathered_sample_hidden_states,
    bool is_target_prefill,
    bool cp_enabled) {
  const bool has_aux = aux_hidden_states.defined();
  const torch::Tensor& target_hidden_states =
      has_aux ? aux_hidden_states : hidden_states;
  torch::Tensor sample_hidden_states;
  if (logits_indices.defined()) {
    // CP decode must reuse the pre-gathered rows: logits_indices live in
    // all-gather space and cannot index the local hidden shard. Off that path
    // we still reuse them when no aux hidden is requested, since gathered rows
    // are produced from the main hidden only.
    const bool must_use_gathered = cp_enabled && !is_target_prefill;
    if (must_use_gathered) {
      CHECK(gathered_sample_hidden_states.defined())
          << "gathered selected hidden is required on the CP decode path";
    }
    if (gathered_sample_hidden_states.defined() &&
        (must_use_gathered || !has_aux)) {
      sample_hidden_states = gathered_sample_hidden_states;
    } else {
      CHECK(target_hidden_states.defined())
          << "speculative hidden states are undefined";
      sample_hidden_states = target_hidden_states.index_select(
          /*dim=*/0,
          logits_indices.to(torch::dtype(torch::kLong)
                                .device(target_hidden_states.device())));
    }
    CHECK_EQ(sample_hidden_states.dim(), 2);
    CHECK_EQ(sample_hidden_states.size(0), logits_indices.numel())
        << "selected speculative hidden must match logits_indices rows";
  }
  if (is_target_prefill) {
    sample_output.embeddings = target_hidden_states;
    sample_output.selected_embeddings = sample_hidden_states;
  } else {
    sample_output.embeddings = sample_hidden_states;
  }
}

void clear_all_output_embeddings(ForwardOutput& output) {
  output.sample_output.embeddings = torch::Tensor();
  clear_selected_embeddings(output);
}

void prepare_first_draft_inputs(EmbeddingCache& embedding_cache,
                                const ForwardInput& input,
                                ForwardOutput& output) {
  const torch::Tensor& sample_hidden_states =
      output.sample_output.selected_embeddings;
  CHECK(sample_hidden_states.defined())
      << "target worker must populate "
         "selected_embeddings on the speculative "
         "prefill path";
  embedding_cache.write_prefill_target_context(
      input.input_params.embedding.embedding_ids,
      input.input_params.embedding.request_ids,
      output.sample_output.next_tokens,
      sample_hidden_states);
  output.sample_output.embeddings = sample_hidden_states.detach();
  clear_selected_embeddings(output);
}

}  // namespace xllm
