/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/runtime/vlm_worker_impl.h"

#include <c10/core/StreamGuard.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>

#include "common/macros.h"
#include "common/metrics.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/speculative/draft_extend_input.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "models/model_registry.h"
#include "util/timer.h"

namespace xllm {

namespace {

void wait_input_ready_events(const VlmForwardInput& input,
                             const Stream& stream) {
  CHECK(stream.wait_event(input.runtime.metadata_ready_event))
      << "failed to wait VlmForwardInput metadata ready event";
}

}  // namespace

VLMWorkerImpl::VLMWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options,
                             bool is_input_release_deferred)
    : WorkerImpl(parallel_args, device, options, is_input_release_deferred) {
  device_.set_device();
}

bool VLMWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  // initialize model
  context.set_encoder_embedding_mode(false);
  model_ = create_vlm_model(context);
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ = std::make_unique<Executor>(model_.get(), context, options_);
  model_executor_->set_dp_process_group(
      context.get_parallel_args().dp_local_process_group_);
  return true;
}

std::optional<ForwardOutput> VLMWorkerImpl::step(const VlmForwardInput& input) {
  std::unique_ptr<Stream> stream = device_.current_stream();
  wait_input_ready_events(input, *stream);
  return step_internal(input, ForwardSyncPolicy::LEGACY);
}

std::optional<ForwardOutput> VLMWorkerImpl::step_for_schedule_overlap(
    const VlmForwardInput& input,
    const std::function<void()>& prepare_sampling) {
  prepare_schedule_overlap(input);
  return execute_no_sync_on_stream(input,
                                   *compute_stream_,
                                   ForwardCompletion::RECORD_EVENT,
                                   prepare_sampling);
}

std::optional<ForwardOutput> VLMWorkerImpl::execute_no_sync_on_stream(
    const VlmForwardInput& input,
    Stream& compute_stream,
    ForwardCompletion completion,
    const std::function<void()>& prepare_sampling) {
  const ForwardSyncPolicy sync_policy = ForwardSyncPolicy::NO_SYNC;
  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  wait_input_ready_events(input, compute_stream);
  return step_internal(input, sync_policy, completion, prepare_sampling);
}

std::optional<ForwardOutput> VLMWorkerImpl::step_internal(
    const VlmForwardInput& input,
    ForwardSyncPolicy sync_policy,
    ForwardCompletion completion,
    const std::function<void()>& prepare_sampling) {
  Timer timer;
  const bool empty_shard =
      input.input_params.meta.num_sequences == 0 &&
      (!input.token_ids.defined() || input.token_ids.numel() == 0);
  if (empty_shard) {
    return ForwardOutput{};
  }

  const auto& sampling_params = input.sampling_params;
  ModelOutput model_output;
  torch::Tensor logits;
  torch::Tensor lm_head_input_indices;
  torch::Tensor selected_hidden_from_lm_head;
  {
    ModelParamsView params(input.input_params);
    model_output = model_executor_->forward(
        input.token_ids, input.positions, kv_caches_, params);
    if (sampling_params.num_sample_seqs() > 0) {
      lm_head_input_indices =
          remap_logits_indices(sampling_params.logits_indices(),
                               params.parallel,
                               model_output.hidden_states);
    }
  }

  if (sampling_params.num_sample_seqs() > 0) {
    if (options_.enable_speculative_decode()) {
      logits = model_->logits(model_output.hidden_states,
                              lm_head_input_indices,
                              selected_hidden_from_lm_head);
    } else {
      logits =
          model_->logits(model_output.hidden_states, lm_head_input_indices);
    }
  }

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());

  if (suppresses_step_output()) {
    if (sync_policy == ForwardSyncPolicy::LEGACY) {
      auto ret = device_.synchronize_default_stream();
      (void)ret;
    }
    return std::nullopt;
  }

  ForwardOutput output;
  if (sampling_params.num_sample_seqs() > 0) {
    output.logprobs = sampling_params.logprobs();
    output.max_top_logprobs = sampling_params.max_top_logprobs();
    if (!input.skip_sampling_for_logits_only) {
      if (prepare_sampling) {
        prepare_sampling();
      }
      auto sample_output = sample(logits, sampling_params);
      COUNTER_ADD(execution_latency_seconds_sampling, timer.elapsed_seconds());

      output.sample_output = sample_output;
    }
    output.logits = logits;
  }

  if (options_.enable_speculative_decode()) {
    const bool is_target_prefill =
        !input.input_params.meta.batch_forward_type.is_decode() &&
        !options_.is_draft_model();
    output_spec_hidden_states(output.sample_output,
                              model_output,
                              lm_head_input_indices,
                              selected_hidden_from_lm_head,
                              is_target_prefill,
                              /*cp_enabled=*/options_.cp_size() > 1);
  }

  if (sync_policy == ForwardSyncPolicy::NO_SYNC) {
    finalize_no_sync_output(output, input, completion);
    return output;
  }

  auto ret = device_.synchronize_default_stream();
  (void)ret;
  return output;
}

}  // namespace xllm
