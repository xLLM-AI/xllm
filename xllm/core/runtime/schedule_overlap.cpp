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

#include "core/runtime/schedule_overlap.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>

#include "core/runtime/json_object_decode_adapter.h"
#include "core/runtime/vlm_forward_params.h"
#include "core/runtime/worker_impl.h"
#if defined(USE_NPU)
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#elif defined(USE_MUSA)
#include "core/kernels/musa/musa_ops_api.h"
#endif

namespace xllm {

ScheduleOverlap::ScheduleOverlap(bool is_driver,
                                 bool enable_eplb,
                                 bool retire_unconsumed_inputs)
    : state_(/*publish_to_host=*/is_driver || enable_eplb,
             /*retain_tokenless_output=*/enable_eplb,
             retire_unconsumed_inputs) {}

template <typename Input>
std::optional<ForwardOutput> ScheduleOverlap::execute_impl(WorkerImpl& worker,
                                                           Input& input) {
  std::shared_ptr<const void> prior_tokens_owner;
  if (state_.has_tokens()) {
    prior_tokens_owner = std::make_shared<torch::Tensor>(
        state_.prior_output().sample_output.next_tokens);
  }
  prepare_continuation_impl(worker, input);
  torch::Tensor sampling_rows;
  auto output = worker.step_for_schedule_overlap(
      input, [&]() { sampling_rows = prepare_sampling_impl(worker, input); });
  if (sampling_rows.defined()) {
    if (output.has_value()) {
      // Keep the pinned H2D source alive through this output's completion.
      output->retained_inputs.emplace_back(
          std::make_shared<torch::Tensor>(std::move(sampling_rows)));
    } else {
      CHECK_EQ(worker.compute_stream_->synchronize(), 0);
    }
  }
  // Composite workers may return retained inputs without an output event. The
  // completion boundary must cover the full worker step before ownership is
  // transferred to the overlap state.
  if (output.has_value() && output->ready_event == nullptr &&
      (!output->retained_inputs.empty() || prior_tokens_owner != nullptr)) {
    output->ready_event = worker.compute_stream_->record_event_or_sync();
  }
  if (prior_tokens_owner != nullptr) {
    if (output.has_value() && output->ready_event != nullptr) {
      output->retained_inputs.emplace_back(std::move(prior_tokens_owner));
    } else {
      CHECK_EQ(worker.compute_stream_->synchronize(), 0);
    }
  }
  worker.observe_execution_memory(input.input_params.meta.is_graph_warmup);
  complete_step_impl(input, *worker.compute_stream_, output);
  return output;
}

template <typename Input>
void ScheduleOverlap::prepare_continuation_impl(WorkerImpl& worker,
                                                Input& input) {
  if (input.token_ids.numel() == 0 ||
      !input.input_params.meta.batch_forward_type.has_decode()) {
    return;
  }

  const bool reuse_tokens =
      input.json_object_states.empty()
          ? state_.has_tokens()
          : state_.can_reuse_tokens(input.input_params.embedding.request_ids);
  // Only the continuation runs under this guard. Model execution retains its
  // own stream selection, including composite speculative workers.
  Stream& compute_stream = *worker.compute_stream_;
  auto stream_guard = compute_stream.set_stream_guard();
  CHECK(compute_stream.wait_event(input.runtime.metadata_ready_event))
      << "failed to wait input metadata before overlap continuation";
  const ForwardOutput empty_output;
  const ForwardOutput& prior_output =
      state_.has_tokens() ? state_.prior_output() : empty_output;
  // JSON continuation owns its prior-output dependency. Wait here only for
  // plain token replacement to avoid enqueueing the same event twice.
  if (reuse_tokens && input.json_object_states.empty()) {
    CHECK(compute_stream.wait_event(prior_output.ready_event))
        << "failed to wait last step output before overlap continuation";
  }
  JsonObjectDecodeAdapter::prepare(input,
                                   prior_output,
                                   state_.sample_sequence_ids(),
                                   worker.device_,
                                   compute_stream);
  if (reuse_tokens &&
      !worker.prepare_model_decode_continuation(input, prior_output)) {
    replace_tokens(
        input,
        prior_output,
        worker.can_prepare_npu_graph_decode_input(input.input_params));
  }
}

template <typename Input>
torch::Tensor ScheduleOverlap::prepare_sampling_impl(WorkerImpl& worker,
                                                     Input& input) {
  if (input.token_ids.numel() == 0 ||
      !input.input_params.meta.batch_forward_type.has_decode() ||
      !state_.has_tokens() || input.sample_prior_output_rows.empty()) {
    return {};
  }
  const auto& sampling = input.sampling_params;
  if (!sampling.bad_words().defined() && !sampling.seeds().defined() &&
      !sampling.repetition_penalties().defined() &&
      !sampling.frequency_penalties().defined()) {
    return {};
  }
  for (size_t row = 0; row < input.sample_prior_output_rows.size(); ++row) {
    const int32_t prior_row = input.sample_prior_output_rows[row];
    if (prior_row < 0) {
      continue;
    }
    CHECK_LT(static_cast<size_t>(prior_row),
             state_.sample_sequence_ids().size());
    CHECK_EQ(input.sample_sequence_ids[row],
             state_.sample_sequence_ids()[prior_row]);
  }

  auto stream_guard = worker.compute_stream_->set_stream_guard();
  const auto& prior_tokens = state_.prior_output().sample_output.next_tokens;
  const bool async_copy = !prior_tokens.device().is_cpu();
  const torch::Tensor host_rows = torch::empty(
      {static_cast<int64_t>(input.sample_prior_output_rows.size())},
      torch::TensorOptions()
          .device(torch::kCPU)
          .dtype(torch::kInt64)
          .pinned_memory(async_copy));
  std::copy(input.sample_prior_output_rows.begin(),
            input.sample_prior_output_rows.end(),
            host_rows.data_ptr<int64_t>());
  advance_sampling_history(
      input.sampling_params,
      prior_tokens,
      host_rows.to(prior_tokens.device(), /*non_blocking=*/async_copy));
  return async_copy ? host_rows : torch::Tensor{};
}

template <typename Input>
void ScheduleOverlap::complete_step_impl(const Input& input,
                                         Stream& compute_stream,
                                         std::optional<ForwardOutput>& output) {
  if (output.has_value()) {
    output->is_graph_warmup = input.input_params.meta.is_graph_warmup;
    output->json_object_errors.insert(output->json_object_errors.end(),
                                      input.json_object_errors.begin(),
                                      input.json_object_errors.end());
  }
  record_output(compute_stream,
                output,
                input.input_params.embedding.request_ids,
                input.sample_sequence_ids);
}

void ScheduleOverlap::record_output(
    Stream& compute_stream,
    const std::optional<ForwardOutput>& output,
    const std::vector<std::string>& request_ids,
    const std::vector<std::string>& sample_sequence_ids) {
  if (!state_.record(output, request_ids, sample_sequence_ids)) {
    // Shutdown rejects publication and cache-only updates alike. Drain before
    // the caller releases the output and its asynchronous input owners.
    CHECK_EQ(compute_stream.synchronize(), 0);
  }
}

std::optional<ForwardOutput> ScheduleOverlap::execute(WorkerImpl& worker,
                                                      LlmForwardInput& input) {
  return execute_impl(worker, input);
}

std::optional<ForwardOutput> ScheduleOverlap::execute(WorkerImpl& worker,
                                                      VlmForwardInput& input) {
  return execute_impl(worker, input);
}

ForwardOutput ScheduleOverlap::consume() { return state_.consume(); }

void ScheduleOverlap::shutdown() { state_.shutdown(); }

template <typename Input>
void ScheduleOverlap::replace_tokens(
    Input& input,
    const ForwardOutput& prior_output,
    [[maybe_unused]] bool use_graph_decode_input) {
#if defined(USE_NPU)
  // These APIs take a mutable Tensor handle without changing its storage.
  auto next_tokens = prior_output.sample_output.next_tokens;
  kernel::npu::replace_token(input.token_ids,
                             next_tokens,
                             /*synchronize_stream=*/false);
  if (use_graph_decode_input) {
    input.input_params.graph.input_tokens_override = input.token_ids;
  }
#elif defined(USE_MUSA)
  auto next_tokens = prior_output.sample_output.next_tokens;
  kernel::musa::replace_token(input.token_ids,
                              next_tokens,
                              /*synchronize_stream=*/false);
#else
  auto& flatten_tokens = input.token_ids;
  auto neg_mask = flatten_tokens < 0;
  auto clamped_neg_indices = torch::clamp(-flatten_tokens, 0);
  auto replacement =
      prior_output.sample_output.next_tokens.index({clamped_neg_indices - 1});
  input.token_ids = torch::where(neg_mask, replacement, flatten_tokens);
#endif
}

}  // namespace xllm
