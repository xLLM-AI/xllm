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

#include "core/framework/batch/rec_batch_state.h"

#include <glog/logging.h>

#include <limits>

#include "core/framework/batch/rec_forward_input_builder.h"

namespace xllm {

RecBatchState::RecBatchState(BatchInputType input_type)
    : input_type_(input_type), output_handler_(input_type) {
  switch (input_type_) {
    case BatchInputType::SEQUENCE:
    case BatchInputType::ONEREC:
    case BatchInputType::ONEREC_XATTENTION:
    case BatchInputType::REC_MULTI_ROUND:
      return;
  }
  LOG(FATAL) << "Unsupported batch input type: "
             << static_cast<int32_t>(input_type_);
}

bool RecBatchState::uses_group_input() const {
  return input_type_ == BatchInputType::ONEREC ||
         input_type_ == BatchInputType::ONEREC_XATTENTION;
}

size_t RecBatchState::size() const {
  return uses_group_input() ? sequence_state_.num_group_sequences()
                            : sequence_state_.sequence_plan().size();
}

Sequence* RecBatchState::sequence(size_t index) const {
  return uses_group_input() ? sequence_state_.group_sequence(index)
                            : sequence_state_.sequence_plan()[index].sequence;
}

std::vector<Sequence*> RecBatchState::get_sequences() const {
  if (!uses_group_input() && !sequence_state_.sequence_plan().empty()) {
    return sequence_state_.sequence_plan().sequences();
  }
  return sequence_state_.group_sequences();
}

void RecBatchState::refresh_forward_type() {
  sequence_state_.refresh_forward_type(get_sequences());
}

void RecBatchState::refresh_sequences_from_groups() {
  if (!uses_group_input()) {
    sequence_state_.refresh_sequences_from_groups();
  }
}

RecForwardInput RecBatchState::prepare_forward_input(
    uint32_t num_decoding_tokens,
    uint32_t min_decoding_batch_size,
    const ModelArgs& args,
    int32_t cp_size) {
  if (input_type_ == BatchInputType::SEQUENCE) {
    const auto data = sequence_state_.prepare_sequence_input_data();
    output_handler_.prepare(data);
    return sequence_state_.build_rec_sequence_input(
        data, num_decoding_tokens, min_decoding_batch_size, args, cp_size);
  }
  return prepare_rec_forward_input(num_decoding_tokens,
                                   min_decoding_batch_size,
                                   args,
                                   /*thread_pool=*/nullptr);
}

RecForwardInput RecBatchState::prepare_forward_input(const ModelArgs& args,
                                                     ThreadPool* thread_pool,
                                                     int32_t cp_size) {
  CHECK(input_type_ == BatchInputType::SEQUENCE)
      << "Sequence packing requires a sequence batch";
  const auto data = sequence_state_.prepare_distributed_input_data();
  output_handler_.prepare(data);
  return sequence_state_.build_rec_sequence_input(
      data, args, thread_pool, cp_size);
}

RecForwardInput RecBatchState::prepare_rec_forward_input(
    uint32_t num_decoding_tokens,
    uint32_t min_decoding_batch_size,
    const ModelArgs& args,
    MPMCThreadPool* thread_pool) {
  CHECK(input_type_ != BatchInputType::SEQUENCE)
      << "Rec input requires an explicit Rec batch input type";
  output_handler_.clear();
  if (sequence_state_.empty()) {
    return {};
  }
  BatchSequencePlan group_plan;
  const BatchSequencePlan* plan = &sequence_state_.sequence_plan();
  if (uses_group_input()) {
    CHECK(!sequence_state_.sequence_groups().empty())
        << "OneRec input requires request groups";
    group_plan.reserve(size());
    for (auto* sequence : get_sequences()) {
      group_plan.add(sequence, std::numeric_limits<uint32_t>::max());
    }
    plan = &group_plan;
  } else {
    CHECK(sequence_state_.sequence_groups().empty() || !plan->empty())
        << "Sequence input requires scheduled sequences";
  }
  auto data = sequence_state_.input_data(*plan);
  output_handler_.prepare(data);
  auto builder =
      RecForwardInputBuilder::create(input_type_, data, &args, thread_pool);
  return builder->build_rec_forward_input(num_decoding_tokens,
                                          min_decoding_batch_size);
}

void RecBatchState::process_sample_output(const RawForwardOutput& output,
                                          bool replace_fake_token) {
  const auto sequences = get_sequences();
  output_handler_.process_sample_output(
      {sequences, sequence_state_.sequence_groups()},
      output,
      replace_fake_token);
}

void RecBatchState::process_sample_output(
    const SampleOutput& output,
    bool replace_fake_token,
    bool force_requested_beam_result_size) {
  const auto sequences = get_sequences();
  output_handler_.process_sample_output(
      {sequences, sequence_state_.sequence_groups()},
      output,
      replace_fake_token,
      force_requested_beam_result_size);
}

void RecBatchState::process_beam_search_output(const RawForwardOutput& output,
                                               bool replace_fake_token) {
  const auto sequences = get_sequences();
  output_handler_.process_beam_search_output(
      {sequences, sequence_state_.sequence_groups()},
      output,
      replace_fake_token);
}

void RecBatchState::process_beam_sequence_group(const ForwardOutput& output) {
  const auto sequences = get_sequences();
  output_handler_.process_beam_sequence_group(
      {sequences, sequence_state_.sequence_groups()}, output);
}

void RecBatchState::finish() {
  for (auto* group : sequence_state_.sequence_groups()) {
    group->finish();
  }
  for (auto* sequence : get_sequences()) {
    sequence->finish();
  }
}

}  // namespace xllm
