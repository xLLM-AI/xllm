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

#include "core/runtime/json_object_decode_continuation.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>

#include "core/framework/constrained/json_object_constraint_batch.h"
#include "core/platform/stream.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/json_object_output_rows.h"
#include "core/runtime/vlm_forward_params.h"
#include "core/util/tensor_helper.h"

namespace xllm {

template <typename Input>
void JsonObjectDecodeContinuation::advance_constraints(
    Input& input,
    const ForwardOutput& prior_output,
    const std::vector<std::string>& prior_sample_sequence_ids,
    const torch::Device& device,
    Stream& compute_stream) {
  if (input.json_object_states.empty()) {
    return;
  }
  CHECK_EQ(input.sample_sequence_ids.size(), input.json_object_states.size())
      << "JSON grammar states must align with sampled sequence ids";
  CHECK_EQ(input.sample_prior_output_rows.size(),
           input.json_object_states.size())
      << "JSON grammar states must align with prior output rows";

  const bool has_prior_output =
      prior_output.sample_output.next_tokens.defined();
  if (has_prior_output) {
    c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
    if (prior_output.ready_event != nullptr) {
      CHECK(compute_stream.wait_event(prior_output.ready_event))
          << "failed to wait for last-step output before JSON grammar update";
    }
    CHECK(prior_output.sample_output.next_tokens.dim() == 1 ||
          prior_output.sample_output.next_tokens.dim() == 2)
        << "overlap JSON token output must be 1-D or 2-D, got "
        << prior_output.sample_output.next_tokens.sizes();
  }

  bool expects_prior_output = false;
  for (size_t state_idx = 0; state_idx < input.json_object_states.size();
       ++state_idx) {
    if (input.json_object_states[state_idx].initialized() &&
        input.sample_prior_output_rows[state_idx] >= 0) {
      expects_prior_output = true;
      break;
    }
  }
  if (has_prior_output &&
      (!prior_sample_sequence_ids.empty() || expects_prior_output)) {
    CHECK_EQ(
        prior_sample_sequence_ids.size(),
        static_cast<size_t>(prior_output.sample_output.next_tokens.size(0)))
        << "last-step sampled request ids must align with output rows";
  }

  std::vector<int32_t> output_rows;
  std::vector<JsonObjectOutputError> output_errors;
  std::string row_error;
  const std::vector<std::string> empty_sample_sequence_ids;
  CHECK(detail::resolve_json_object_output_rows(
      input.json_object_states,
      input.sample_sequence_ids,
      input.sample_prior_output_rows,
      has_prior_output ? prior_sample_sequence_ids : empty_sample_sequence_ids,
      &output_rows,
      &output_errors,
      &row_error))
      << row_error;

  for (const JsonObjectOutputError& output_error : output_errors) {
    LOG(ERROR) << "JSON overlap row mismatch: sequence_id="
               << output_error.sample_sequence_id
               << ", error=" << output_error.message;
    input.json_object_errors.emplace_back(output_error);
  }

  for (size_t state_idx = 0; state_idx < input.json_object_states.size();
       ++state_idx) {
    if (output_rows[state_idx] == detail::kInvalidJsonObjectOutputRow) {
      input.json_object_states[state_idx] = JsonObjectGrammarState();
      output_rows[state_idx] = detail::kNoPriorJsonObjectOutputRow;
    }
  }

  torch::Tensor next_tokens;
  const bool needs_prior_tokens =
      std::any_of(output_rows.begin(), output_rows.end(), [](int32_t row) {
        return row >= 0;
      });
  if (needs_prior_tokens) {
    c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
    next_tokens =
        safe_to(prior_output.sample_output.next_tokens,
                torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU),
                /*non_blocking=*/false);
  }

  const std::vector<JsonObjectConstraintViolation> violations =
      advance_json_object_constraints(
          input.json_object_states, next_tokens, output_rows);
  for (const JsonObjectConstraintViolation& violation : violations) {
    const std::string& sequence_id =
        input.sample_sequence_ids[violation.sequence_index];
    LOG(ERROR) << "last-step token violates json_object grammar, sequence_id="
               << sequence_id
               << ", output_row=" << output_rows[violation.sequence_index]
               << ", token_offset=" << violation.token_offset
               << ", token_id=" << violation.token_id
               << ", committed_tokens=" << violation.committed_tokens
               << ", state_fingerprint=" << violation.state_fingerprint;
    input.json_object_errors.push_back(
        {sequence_id, "prior token violates json_object grammar"});
  }
  input.sampling_params.filter_bitmask() =
      build_json_object_filter_bitmask(input.json_object_states, device);
  // Prefer compact bitmask; clear dense float mask to avoid duplicate H2D.
  input.sampling_params.filter_mask() = torch::Tensor();
}

template <typename Input>
void JsonObjectDecodeContinuation::sanitize_error_inputs(Input& input) {
  std::string error;
  CHECK(detail::sanitize_json_object_error_token_ids(
      &input.token_ids,
      input.sample_sequence_ids,
      input.sample_prior_output_rows,
      input.json_object_errors,
      &error))
      << error;
}

void JsonObjectDecodeContinuation::prepare(
    LlmForwardInput& input,
    const ForwardOutput& prior_output,
    const std::vector<std::string>& prior_sample_sequence_ids,
    const torch::Device& device,
    Stream& compute_stream) {
  advance_constraints(
      input, prior_output, prior_sample_sequence_ids, device, compute_stream);
  sanitize_error_inputs(input);
}

void JsonObjectDecodeContinuation::prepare(
    VlmForwardInput& input,
    const ForwardOutput& prior_output,
    const std::vector<std::string>& prior_sample_sequence_ids,
    const torch::Device& device,
    Stream& compute_stream) {
  advance_constraints(
      input, prior_output, prior_sample_sequence_ids, device, compute_stream);
  sanitize_error_inputs(input);
}

}  // namespace xllm
