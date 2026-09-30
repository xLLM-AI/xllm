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

void Batch::add(Sequence* sequence, uint32_t allowed_max_token) {
  state_.add(sequence, allowed_max_token);
}

void Batch::add(const std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    add(sequence);
  }
}

void Batch::add(SequencesGroup* group) { state_.add(group); }
void Batch::set_batch_id() { state_.set_batch_id(); }
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

void Batch::refresh_sequences_from_groups() {
  state_.refresh_sequences_from_groups();
}

ForwardInput Batch::prepare_forward_input(uint32_t num_decoding_tokens,
                                          uint32_t min_decoding_batch_size,
                                          const ModelArgs& args,
                                          int32_t cp_size) {
  return state_.prepare_sequence_input(
      num_decoding_tokens, min_decoding_batch_size, args, cp_size);
}

ForwardInput Batch::prepare_forward_input(const ModelArgs& args,
                                          ThreadPool* thread_pool,
                                          int32_t cp_size) {
  return state_.prepare_distributed_input(args, thread_pool, cp_size);
}

void Batch::process_sample_output(const RawForwardOutput& output,
                                  bool replace_fake_token) {
  const auto sequences = get_sequences();
  state_.output_handler().process_sample_output(
      {sequences, state_.sequence_groups()}, output, replace_fake_token);
}

void Batch::process_sample_output(const SampleOutput& output,
                                  bool replace_fake_token,
                                  bool force_requested_beam_result_size) {
  const auto sequences = get_sequences();
  state_.output_handler().process_sample_output(
      {sequences, state_.sequence_groups()},
      output,
      replace_fake_token,
      force_requested_beam_result_size);
}

void Batch::process_beam_search_output(const RawForwardOutput& output,
                                       bool replace_fake_token) {
  const auto sequences = get_sequences();
  state_.output_handler().process_beam_search_output(
      {sequences, state_.sequence_groups()}, output, replace_fake_token);
}
}  // namespace xllm
