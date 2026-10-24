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

#include "core/framework/batch/rec_batch.h"

namespace xllm {

RecBatch::RecBatch(BatchInputType input_type) : state_(input_type) {}

bool RecBatch::uses_group_input() const { return state_.uses_group_input(); }

size_t RecBatch::size() const { return state_.size(); }

Sequence* RecBatch::sequence(size_t index) const {
  return state_.sequence(index);
}

std::vector<Sequence*> RecBatch::get_sequences() const {
  return state_.get_sequences();
}

void RecBatch::refresh_sequences_from_groups() {
  state_.refresh_sequences_from_groups();
}

RecForwardInput RecBatch::prepare_forward_input(
    uint32_t num_decoding_tokens,
    uint32_t min_decoding_batch_size,
    const ModelArgs& args,
    int32_t cp_size) {
  return state_.prepare_forward_input(
      num_decoding_tokens, min_decoding_batch_size, args, cp_size);
}

RecForwardInput RecBatch::prepare_forward_input(const ModelArgs& args,
                                                ThreadPool* thread_pool,
                                                int32_t cp_size) {
  return state_.prepare_forward_input(args, thread_pool, cp_size);
}

RecForwardInput RecBatch::prepare_rec_forward_input(
    uint32_t num_decoding_tokens,
    uint32_t min_decoding_batch_size,
    const ModelArgs& args,
    MPMCThreadPool* thread_pool) {
  return state_.prepare_rec_forward_input(
      num_decoding_tokens, min_decoding_batch_size, args, thread_pool);
}

void RecBatch::process_sample_output(const RawForwardOutput& output,
                                     bool replace_fake_token) {
  state_.process_sample_output(output, replace_fake_token);
}

void RecBatch::process_sample_output(const SampleOutput& output,
                                     bool replace_fake_token,
                                     bool force_requested_beam_result_size) {
  state_.process_sample_output(
      output, replace_fake_token, force_requested_beam_result_size);
}

void RecBatch::process_beam_search_output(const RawForwardOutput& output,
                                          bool replace_fake_token) {
  state_.process_beam_search_output(output, replace_fake_token);
}

void RecBatch::process_beam_sequence_group(const ForwardOutput& output) {
  state_.process_beam_sequence_group(output);
}

void RecBatch::finish() { state_.finish(); }

}  // namespace xllm
