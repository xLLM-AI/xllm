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

#include "runtime/dspark_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>

#include <optional>
#include <utility>

#include "common/metrics.h"
#include "core/framework/sampling/dspark_sampling.h"
#include "core/framework/speculative/adaptive_speculative_controller.h"
#include "framework/parallel_state/process_group.h"
#include "runtime/llm_worker_impl.h"
#include "util/timer.h"

namespace xllm {

DSparkWorkerImpl::DSparkWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : DFlashWorkerImpl(parallel_args, device, options),
      sampling_process_group_(parallel_args.tp_group_ != nullptr
                                  ? parallel_args.tp_group_
                                  : parallel_args.process_group_) {}

DSparkWorkerImpl::DraftBlock DSparkWorkerImpl::run_decode_draft(
    const LlmForwardInput& input,
    LlmForwardInput& validate_input) {
  Timer draft_timer;

  // Same input build as DFlash, but sample_from_anchor()==true makes the query
  // block N-wide and every position predicts a draft token.
  LlmForwardInput query_input;
  prepare_query_inputs(input, query_input);

  CHECK(input.token_ids_host.defined())
      << "DSpark requires host token ids for the anchor.";
  const int32_t batch_size = input.input_params.meta.num_sequences;
  CHECK_GE(input.token_ids_host.numel(), batch_size)
      << "DSpark anchor token count must cover the decode batch.";
  torch::Tensor anchor_token_ids =
      input.token_ids_host.slice(/*dim=*/0, /*start=*/0, /*end=*/batch_size)
          .to(draft_impl_->device(), torch::kLong);
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();

  CHECK_GT(num_speculative_tokens, 0)
      << "DSpark requires num_speculative_tokens > 0.";
  LlmForwardInput logits_input = query_input.clone();
  logits_input.skip_sampling_for_logits_only = true;
  // Request pre-lm_head hidden alongside logits so ConfidenceHead can consume
  // the same [num_reqs*num_spec, hidden] rows without a second projection.
  // Only pay the cost when adaptive pruning is actually enabled — static
  // sampling doesn't consume confidence, and asking for selected_hidden
  // triggers a target-side gather and a downstream linear+sigmoid every
  // step.
  const bool want_confidence =
      draft_impl_->has_dspark_confidence_head() && adaptive_enabled();
  logits_input.return_selected_hidden = want_confidence;

  LlmForwardInput processed_input;
  draft_impl_->prepare_work_before_execute_on_stream(
      logits_input, processed_input, *prepare_stream_);
  std::optional<ForwardOutput> draft_output =
      draft_impl_->execute_no_sync_on_stream(processed_input,
                                             *compute_stream_,
                                             /*record_ready_event=*/false);
  CHECK(draft_output.has_value())
      << "DSpark draft forward must return an output.";
  CHECK(draft_output->logits.defined())
      << "DSpark draft forward must return logits.";
  // Match DFlash's host/device overlap: validation input construction only
  // reads the original input, so prepare it after the asynchronous draft
  // launch instead of delaying that launch.
  prepare_validate_inputs(input, validate_input);

  const int64_t num_rows = draft_output->logits.size(/*dim=*/0);
  CHECK_EQ(num_rows % num_speculative_tokens, 0)
      << "DSpark draft logits rows must be divisible by "
         "num_speculative_tokens.";
  CHECK_EQ(num_rows / num_speculative_tokens, batch_size)
      << "DSpark draft logits batch must match the decode batch.";
  torch::Tensor base_logits = draft_output->logits.view(
      {batch_size, num_speculative_tokens, draft_output->logits.size(-1)});
  torch::Tensor last_hidden;
  if (want_confidence) {
    if (draft_output->selected_hidden.defined() &&
        draft_output->selected_hidden.size(0) == num_rows) {
      last_hidden = draft_output->selected_hidden.view(
          {batch_size,
           num_speculative_tokens,
           draft_output->selected_hidden.size(-1)});
    } else {
      LOG_FIRST_N(WARNING, 1)
          << "DSpark: draft did not expose selected_hidden; ConfidenceHead "
          << "disabled for adaptive pruning, using logits-derived probs.";
    }
  }

  DraftBlock draft_block = [&]() {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    SamplingParameters sampling_params_on_device = input.sampling_params.to(
        base_logits.device(), base_logits.scalar_type());
    prepare_draft_sampling(draft_sampling_mode_, sampling_params_on_device);
    return sample_block(
        base_logits, last_hidden, anchor_token_ids, sampling_params_on_device);
  }();

  draft_block.retained_inputs = take_retained_inputs(*draft_output);

  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              draft_timer.elapsed_seconds());
  return draft_block;
}

DSparkWorkerImpl::DraftBlock DSparkWorkerImpl::sample_block(
    const torch::Tensor& base_logits,
    const torch::Tensor& last_hidden,
    const torch::Tensor& anchor_token_ids,
    const SamplingParameters& sampling_params) const {
  CHECK_EQ(base_logits.dim(), 3);
  const int64_t num_reqs = base_logits.size(0);
  const int64_t num_speculative_tokens = base_logits.size(1);
  const bool with_confidence =
      last_hidden.defined() && draft_impl_->has_dspark_confidence_head();
  if (with_confidence) {
    CHECK_EQ(last_hidden.dim(), 3)
        << "DSpark last_hidden must be [num_reqs, n_spec, hidden_size].";
    CHECK_EQ(last_hidden.size(0), num_reqs)
        << "DSpark last_hidden batch mismatch.";
    CHECK_EQ(last_hidden.size(1), num_speculative_tokens)
        << "DSpark last_hidden n_spec mismatch.";
  }
  DraftBlock draft_block;
  draft_block.proposal = sample_dspark_block(
      base_logits,
      anchor_token_ids,
      sampling_params,
      [this](const torch::Tensor& previous) {
        return draft_impl_->dspark_markov_bias(previous);
      },
      [this, &sampling_params](torch::Tensor& tokens) {
        synchronize_sampled_token_ids(
            tokens, sampling_params.sampling_mode == SamplingMode::GREEDY);
      });
  const auto& token_ids = draft_block.proposal.token_ids();

  // ConfidenceHead over the whole block, batched. Equivalent to computing it
  // per step inside the loop on (last_hidden[:, k], prev_k) — prev_k is the
  // anchor for step 0 and the draft token sampled at step k-1 otherwise — but
  // as a single linear+sigmoid instead of gamma small kernel launches. The
  // inputs don't depend on the sampling of the same step, so hoisting is exact.
  if (with_confidence) {
    torch::Tensor prev_matrix = torch::cat(
        {anchor_token_ids.view({num_reqs, 1}),
         token_ids.slice(/*dim=*/1, /*start=*/0, num_speculative_tokens - 1)},
        /*dim=*/1);  // [num_reqs, num_spec]
    draft_block.confidence_probs =
        draft_impl_->dspark_confidence_probs(last_hidden, prev_matrix);
    CHECK_EQ(draft_block.confidence_probs.dim(), 2)
        << "batched confidence probs must be [num_reqs, num_spec]";
    CHECK_EQ(draft_block.confidence_probs.size(0), num_reqs)
        << "batched confidence probs batch mismatch";
    CHECK_EQ(draft_block.confidence_probs.size(1), num_speculative_tokens)
        << "batched confidence probs n_spec mismatch";
  }

  return draft_block;
}

void DSparkWorkerImpl::synchronize_sampled_token_ids(
    torch::Tensor& sampled_token_ids,
    bool all_greedy_sample) const {
  if (all_greedy_sample || sampling_process_group_ == nullptr ||
      sampling_process_group_->world_size() <= 1) {
    return;
  }
  sampled_token_ids = sampled_token_ids.contiguous();
  sampling_process_group_->broadcast(sampled_token_ids, /*root_rank=*/0);
}

}  // namespace xllm
