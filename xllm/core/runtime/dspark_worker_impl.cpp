/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

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
#include "framework/parallel_state/process_group.h"
#include "framework/sampling/sampler.h"
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
    const ForwardInput& input,
    ForwardInput& validate_input) {
  Timer timer;

  // Same input build as DFlash, but sample_from_anchor()==true makes the query
  // block N-wide and every position predicts a draft token.
  ForwardInput query_input;
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
  ForwardInput logits_input = query_input;
  logits_input.skip_sampling_for_logits_only = true;
  // Request pre-lm_head hidden alongside logits so ConfidenceHead can consume
  // the same [num_reqs*num_spec, hidden] rows without a second projection.
  // Only pay the cost when adaptive pruning is actually enabled — static
  // sampling doesn't consume confidence, and asking for selected_hidden
  // triggers a target-side gather and a downstream linear+sigmoid every
  // step.
  const bool want_confidence = draft_impl_->has_dspark_confidence_head() &&
                               adaptive_spec_controller_ != nullptr &&
                               adaptive_spec_controller_->enabled();
  logits_input.return_selected_hidden = want_confidence;

  ForwardInput processed_input;
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
      // The current lm_head backend did not surface selected_hidden. Fall
      // back to sampler-gathered probs as the adaptive signal — ConfidenceHead
      // stays a nice-to-have; adaptive pruning still works (a little coarser).
      LOG_FIRST_N(WARNING, 1)
          << "DSpark: draft did not expose selected_hidden; ConfidenceHead "
          << "disabled for adaptive pruning, using proposal probs.";
    }
  }

  BlockSampleOutput sample_output;
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    SamplingParameters sampling_params_on_device = input.sampling_params.to(
        base_logits.device(), base_logits.scalar_type());
    sample_output = sample_block(
        base_logits, last_hidden, anchor_token_ids, sampling_params_on_device);
  }

  DraftBlock draft_block;
  draft_block.token_ids = std::move(sample_output.token_ids);
  draft_block.probs = std::move(sample_output.probs);
  draft_block.confidence_probs = std::move(sample_output.confidence_probs);
  draft_block.retained_inputs = take_retained_inputs(*draft_output);

  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  return draft_block;
}

DSparkWorkerImpl::BlockSampleOutput DSparkWorkerImpl::sample_block(
    const torch::Tensor& base_logits,
    const torch::Tensor& last_hidden,
    const torch::Tensor& anchor_token_ids,
    const SamplingParameters& sampling_params) const {
  CHECK_EQ(base_logits.dim(), 3)
      << "DSpark base_logits must be [num_reqs, n_spec, draft_vocab].";
  CHECK_EQ(anchor_token_ids.dim(), 1)
      << "DSpark anchor_token_ids must be [num_reqs].";
  CHECK_EQ(anchor_token_ids.size(0), base_logits.size(0))
      << "DSpark anchor token batch must match base_logits.";
  CHECK_EQ(anchor_token_ids.scalar_type(), torch::kLong)
      << "DSpark anchor_token_ids must use int64.";
  CHECK_EQ(anchor_token_ids.device(), base_logits.device())
      << "DSpark anchor_token_ids and base_logits must share a device.";

  const int64_t num_reqs = base_logits.size(/*dim=*/0);
  const int64_t num_speculative_tokens = base_logits.size(/*dim=*/1);
  const int64_t draft_vocab_size = base_logits.size(/*dim=*/2);
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
  SamplingParameters step_sampling_params = sampling_params;
  const torch::TensorOptions index_options =
      torch::TensorOptions().dtype(torch::kInt).device(base_logits.device());
  step_sampling_params.selected_token_idxes = torch::empty({0}, index_options);
  step_sampling_params.sample_idxes = torch::empty({0}, index_options);
  step_sampling_params.return_probs = !step_sampling_params.all_greedy_sample;
  step_sampling_params.logprobs = false;
  step_sampling_params.max_top_logprobs = 0;
  step_sampling_params.use_beam_search = false;

  torch::Tensor token_ids = torch::empty(
      {num_reqs, num_speculative_tokens},
      torch::TensorOptions().dtype(torch::kLong).device(base_logits.device()));
  torch::Tensor proposal_probs =
      torch::empty({num_reqs, num_speculative_tokens},
                   torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(base_logits.device()));
  // Filled after the sample loop by a single batched ConfidenceHead call; stays
  // undefined when confidence is disabled.
  torch::Tensor confidence_probs;

  using ISlice = torch::indexing::Slice;
  Sampler sampler;
  torch::Tensor previous_token_ids = anchor_token_ids;
  for (int64_t token_idx = 0; token_idx < num_speculative_tokens; ++token_idx) {
    torch::Tensor markov_bias =
        draft_impl_->dspark_markov_bias(previous_token_ids);
    CHECK_EQ(markov_bias.dim(), 2)
        << "DSpark Markov bias must be [num_reqs, draft_vocab].";
    CHECK_EQ(markov_bias.size(0), num_reqs)
        << "DSpark Markov bias batch must match base_logits.";
    CHECK_EQ(markov_bias.size(1), draft_vocab_size)
        << "DSpark reduced-vocab drafts need draft-to-target remapping, not "
           "yet implemented.";

    torch::Tensor step_logits =
        base_logits.select(/*dim=*/1, /*index=*/token_idx) + markov_bias;
    SampleOutput sample_output =
        sampler.forward(step_logits, step_sampling_params);
    torch::Tensor sampled_token_ids = sample_output.next_tokens;
    synchronize_sampled_token_ids(sampled_token_ids, step_sampling_params);

    torch::Tensor selected_proposal_probs;
    if (step_sampling_params.all_greedy_sample) {
      selected_proposal_probs = torch::ones({num_reqs},
                                            torch::TensorOptions()
                                                .dtype(torch::kFloat32)
                                                .device(base_logits.device()));
    } else {
      CHECK_EQ(sample_output.probs.dim(), 2)
          << "DSpark random/mixed sampling requires dense proposal probs.";
      selected_proposal_probs =
          sample_output.probs.gather(/*dim=*/1, sampled_token_ids.view({-1, 1}))
              .view({-1})
              .to(torch::kFloat32);
      if (!step_sampling_params.all_random_sample) {
        selected_proposal_probs =
            torch::where(step_sampling_params.do_sample,
                         selected_proposal_probs,
                         torch::ones_like(selected_proposal_probs));
      }
    }

    token_ids.index_put_({ISlice(), token_idx}, sampled_token_ids);
    proposal_probs.index_put_({ISlice(), token_idx}, selected_proposal_probs);
    previous_token_ids = sampled_token_ids;
  }

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
    confidence_probs =
        draft_impl_->dspark_confidence_probs_batched(last_hidden, prev_matrix);
    CHECK_EQ(confidence_probs.dim(), 2)
        << "batched confidence probs must be [num_reqs, num_spec]";
    CHECK_EQ(confidence_probs.size(0), num_reqs)
        << "batched confidence probs batch mismatch";
    CHECK_EQ(confidence_probs.size(1), num_speculative_tokens)
        << "batched confidence probs n_spec mismatch";
  }

  return {.token_ids = std::move(token_ids),
          .probs = std::move(proposal_probs),
          .confidence_probs = std::move(confidence_probs)};
}

void DSparkWorkerImpl::synchronize_sampled_token_ids(
    torch::Tensor& sampled_token_ids,
    const SamplingParameters& sampling_params) const {
  if (sampling_params.all_greedy_sample || sampling_process_group_ == nullptr ||
      sampling_process_group_->world_size() <= 1) {
    return;
  }
  sampled_token_ids = sampled_token_ids.contiguous();
  sampling_process_group_->broadcast(sampled_token_ids, /*root_rank=*/0);
}

}  // namespace xllm
