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

#include "core/framework/batch/rec_batch_output_handler.h"

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/framework/batch/batch_output_action.h"
#include "core/framework/model/rec_model_utils.h"
#include "core/framework/request/onerec_sequence.h"
#include "core/framework/request/rec_sequence.h"

namespace xllm {

void RecBatchOutputHandler::prepare(const BatchInputData& data) {
  // Multi-round device execution returns beam results, not per-token rows.
  if (input_type_ == BatchInputType::REC_MULTI_ROUND) {
    clear();
    return;
  }
  if (input_type_ == BatchInputType::SEQUENCE ||
      !use_legacy_onerec_prefill_only_contract()) {
    sequence_handler_.prepare(data);
    return;
  }

  clear();
  sequence_handler_.reserve(data.sequences.size());
  for (size_t seq_index = 0; seq_index < data.sequences.size(); ++seq_index) {
    auto* sequence = data.sequences[seq_index];
    if (sequence == nullptr) {
      continue;
    }
    const auto* onerec_sequence = dynamic_cast<const OneRecSequence*>(sequence);
    const bool needs_context_target =
        onerec_sequence != nullptr && sequence->tokens().empty() &&
        sequence->kv_state().kv_cache_tokens_num() == 0 &&
        onerec_sequence->num_decoder_embeddings() > 0;
    if (needs_context_target) {
      sequence_handler_.add_sequence_target(sequence);
      continue;
    }
    sequence_handler_.add_sequence_targets(sequence,
                                           data.allowed_max_tokens[seq_index]);
  }
}

void RecBatchOutputHandler::process_sample_output(
    const BatchOutputData& data,
    const RawForwardOutput& output) {
  sequence_handler_.process_sample_output(
      data, output, BatchOutputAction::IMMEDIATE);
}

void RecBatchOutputHandler::process_sample_output(
    const BatchOutputData& data,
    const SampleOutput& output,
    bool force_requested_beam_result_size) {
  sequence_handler_.process_sample_output(data,
                                          output,
                                          BatchOutputAction::IMMEDIATE,
                                          force_requested_beam_result_size);
}

void RecBatchOutputHandler::process_beam_search_output(
    const BatchOutputData& data,
    const RawForwardOutput& output) {
  sequence_handler_.process_beam_search_output(
      data, output, BatchOutputAction::IMMEDIATE);
}

void RecBatchOutputHandler::process_beam_sequence_group(
    const BatchOutputData& data,
    const ForwardOutput& output) {
  if (!output.beam_sequence_group.defined() ||
      output.beam_sequence_group.numel() == 0) {
    return;
  }

  // Get sequences from either data.sequences or data.sequence_groups
  const auto& sequences = data.sequences;
  if (sequences.empty()) {
    return;
  }

  const int32_t beam_width = sequences[0]->sampling_param()->beam_width;
  if (beam_width <= 1) {
    return;
  }
  const int32_t result_width =
      static_cast<int32_t>(output.beam_sequence_group.size(1));
  const int32_t total_rounds =
      static_cast<int32_t>(output.beam_sequence_group.size(2));
  size_t num_groups = data.sequence_groups.size();
  if (num_groups == 0) {
    // Sequence-only input has one result group per scheduled sequence.
    num_groups = sequences.size();
  }

  // Tensor should already be on CPU (transferred in get_model_output)
  auto seq_group_accessor = output.beam_sequence_group.accessor<int32_t, 3>();

  // out_logprobs from beam_search_output, shape: [batch * beam_width]
  // Tensor should already be on CPU (transferred in get_model_output)
  const bool has_logprobs = output.beam_search_output.out_logprobs.defined() &&
                            output.beam_search_output.out_logprobs.numel() > 0;

  for (size_t g = 0; g < num_groups; ++g) {
    std::vector<std::vector<int32_t>> group_flat2d;
    std::vector<float> last_logprobs;
    group_flat2d.reserve(static_cast<size_t>(result_width));
    last_logprobs.reserve(static_cast<size_t>(result_width));

    for (int32_t b = 0; b < result_width; ++b) {
      std::vector<int32_t> row_tokens;
      row_tokens.reserve(static_cast<size_t>(total_rounds));
      for (int32_t c = 0; c < total_rounds; ++c) {
        // Access [g][b][c]
        row_tokens.emplace_back(seq_group_accessor[g][b][c]);
      }
      group_flat2d.emplace_back(std::move(row_tokens));
      if (has_logprobs) {
        // logprobs is flattened [batch * result_width] for multi-round widened
        // final output.
        const int32_t logprob_idx = static_cast<int32_t>(g) * result_width + b;
        last_logprobs.emplace_back(
            output.beam_search_output.out_logprobs[logprob_idx].item<float>());
      }
    }
    // Access sequence from data.sequence_groups if available
    Sequence* seq = data.sequence_groups.empty()
                        ? sequences[g]
                        : data.sequence_groups[g]->sequences()[0].get();
    RecSequence::from(*seq).set_beam_search_result(
        RecBeamSearchResult(result_width,
                            total_rounds,
                            std::move(group_flat2d),
                            std::move(last_logprobs)));
  }
}

}  // namespace xllm
