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
#include "core/framework/batch/batch_output_action.h"
#include "core/runtime/forward_params.h"

namespace xllm {

struct BatchOutputData {
  const std::vector<Sequence*>& sequences;
  const std::vector<SequencesGroup*>& sequence_groups;
};

// Captures sampling targets before input building advances KV state. These
// targets determine output row ownership through both phases of
// schedule-overlap writeback, independently of later sequence stages. Requests
// own the target sequences and must outlive the pending forward.
class BatchOutputHandler final {
 public:
  void clear() { output_targets_.clear(); }
  void reserve(size_t target_count) { output_targets_.reserve(target_count); }
  void prepare(const BatchInputData& data);
  // Domain handlers may collect token-based and explicitly selected targets
  // in forward row order after clear(), before input building advances KV
  // state.
  void add_sequence_targets(Sequence* sequence, uint32_t token_budget);
  void add_sequence_target(Sequence* sequence);
  void process_sample_output(const BatchOutputData& data,
                             const RawForwardOutput& output,
                             BatchOutputAction action);
  void process_sample_output(const BatchOutputData& data,
                             const SampleOutput& output,
                             BatchOutputAction action);

 private:
  struct OutputTarget {
    Sequence* sequence = nullptr;
    size_t sample_id = 0;
    bool from_sample_slot = false;
  };

  void append_token_for_sequence(Sequence* sequence,
                                 const Token& token,
                                 int32_t token_idx,
                                 bool replace_fake_token);
  void finish_sample_output(BatchOutputAction action);

  std::vector<OutputTarget> output_targets_;
};

}  // namespace xllm
