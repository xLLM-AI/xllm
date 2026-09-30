/* Copyright 2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/framework/batch/batch_input_data.h"
#include "core/runtime/forward_params.h"

namespace xllm {

struct BatchOutputData {
  const std::vector<Sequence*>& sequences;
  const std::vector<SequencesGroup*>& sequence_groups;
};

// Composed by BatchState. Captures sampling targets before input building
// advances KV state, and keeps them through both phases of schedule-overlap
// writeback. Requests own the target sequences and must outlive the pending
// forward.
class BatchOutputHandler final {
 public:
  void clear() { output_targets_.clear(); }
  void prepare(const BatchInputData& data,
               bool use_context_embedding_targets = false);
  void process_sample_output(const BatchOutputData& data,
                             const RawForwardOutput& output,
                             bool replace_fake_token);
  void process_sample_output(const BatchOutputData& data,
                             const SampleOutput& output,
                             bool replace_fake_token,
                             bool force_requested_beam_result_size);
  void process_beam_sequence_group(const BatchOutputData& data,
                                   const ForwardOutput& output);
  void process_beam_search_output(const BatchOutputData& data,
                                  const RawForwardOutput& output,
                                  bool replace_fake_token);

 private:
  struct OutputTarget {
    Sequence* sequence = nullptr;
    size_t sample_id = 0;
    bool from_sample_slot = false;
  };

  void refresh_output_targets(const BatchInputData& data);
  void refresh_onerec_prefill_output_targets(const BatchInputData& data);
  bool update_sequence_state(Sequence* sequence, bool replace_fake_token);
  void append_token_for_sequence(Sequence* sequence,
                                 const Token& token,
                                 int32_t token_idx,
                                 bool replace_fake_token);
  void process_beam_search(const BatchOutputData& data,
                           bool force_requested_result_size = false);

  std::vector<OutputTarget> output_targets_;
};

}  // namespace xllm
