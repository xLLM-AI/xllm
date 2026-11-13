/* Copyright 2025-2026 The xLLM Authors.
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

#include "core/framework/batch/batch.h"

namespace xllm {

Batch::Batch(Sequence* sequence) { add(sequence); }
Batch::Batch(const std::vector<Sequence*>& sequences) { add(sequences); }

BatchInputType Batch::input_type() const { return BatchInputType::SEQUENCE; }

void Batch::reserve(size_t sequence_count, size_t group_count) {
  state_.reserve(sequence_count, group_count);
}

void Batch::reserve(size_t sequence_count,
                    size_t group_count,
                    size_t input_embedding_count) {
  state_.reserve(sequence_count, group_count, input_embedding_count);
}

void Batch::add(Sequence* sequence, uint32_t allowed_max_token) {
  state_.add(sequence, allowed_max_token);
}

void Batch::add(const std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    add(sequence);
  }
}

void Batch::add(SequencesGroup* group) { state_.add(group); }
void Batch::update_forward_type(Sequence* sequence) {
  state_.update_forward_type(sequence);
}
void Batch::refresh_forward_type() {
  state_.refresh_forward_type(get_sequences());
}

size_t Batch::size() const { return state_.sequence_plan().size(); }

Sequence* Batch::operator[](size_t index) const {
  return state_.sequence_plan()[index].sequence;
}

std::vector<Sequence*> Batch::get_sequences() {
  return static_cast<const Batch&>(*this).get_sequences();
}
std::vector<Sequence*> Batch::get_sequences() const {
  if (!state_.sequence_plan().empty()) {
    return state_.sequence_plan().sequences();
  }
  return state_.group_sequences();
}

template <typename Input>
Input Batch::prepare_forward_input(
    const ModelArgs& args,
    ThreadPool* thread_pool,
    int32_t cp_size,
    SharedKVCacheMappingTracker* mapping_tracker) {
  const auto data = state_.prepare_batch_input();
  output_handler_.prepare(data);
  return state_.build_forward_input<Input>(
      data, args, thread_pool, cp_size, mapping_tracker);
}

template LlmForwardInput Batch::prepare_forward_input<LlmForwardInput>(
    const ModelArgs&,
    ThreadPool*,
    int32_t,
    SharedKVCacheMappingTracker*);
template VlmForwardInput Batch::prepare_forward_input<VlmForwardInput>(
    const ModelArgs&,
    ThreadPool*,
    int32_t,
    SharedKVCacheMappingTracker*);

void Batch::process_sample_output(const RawForwardOutput& output,
                                  BatchOutputAction action) {
  output_handler_.process_sample_output(get_sequences(), output, action);
}

void Batch::process_sample_output(const SampleOutput& output,
                                  BatchOutputAction action) {
  output_handler_.process_sample_output(get_sequences(), output, action);
}
}  // namespace xllm
