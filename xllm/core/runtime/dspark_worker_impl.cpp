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
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/speculative/draft_proposal.h"
#include "runtime/llm_worker_impl.h"
#include "util/timer.h"

namespace xllm {

DSparkWorkerImpl::DSparkWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : DFlashWorkerImpl(parallel_args, device, options) {}

DraftProposal DSparkWorkerImpl::run_decode_draft(
    const LlmForwardInput& input,
    LlmForwardInput& validate_input) {
  Timer draft_timer;

  // Same input build as DFlash, but sample_from_anchor()==true makes the
  // query block N-wide and every position predicts a draft token.
  LlmForwardInput query_input;
  prepare_query_inputs(input, query_input);

  const torch::Tensor anchors = anchor_token_ids(input);
  const int32_t batch_size = input.input_params.meta.num_sequences;
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();

  CHECK_GT(num_speculative_tokens, 0)
      << "DSpark requires num_speculative_tokens > 0.";
  LlmForwardInput logits_input = query_input.clone();
  logits_input.skip_sampling_for_logits_only = true;
  logits_input.return_selected_hidden = false;
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
  prepare_validate_inputs(input, validate_input);

  const int64_t num_rows = draft_output->logits.size(/*dim=*/0);
  CHECK_EQ(num_rows % num_speculative_tokens, 0)
      << "DSpark draft logits rows must be divisible by "
         "num_speculative_tokens.";
  CHECK_EQ(num_rows / num_speculative_tokens, batch_size)
      << "DSpark draft logits batch must match the decode batch.";
  torch::Tensor base_logits = draft_output->logits.view(
      {batch_size, num_speculative_tokens, draft_output->logits.size(-1)});
  DraftProposal draft_proposal;
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    SamplingParameters sampling_params_on_device = input.sampling_params;
    prepare_draft_sampling(is_probabilistic_draft_, sampling_params_on_device);
    sampling_params_on_device = sampling_params_on_device.to(
        base_logits.device(), base_logits.scalar_type());
    draft_proposal = sample_dspark_markov_block(
        [this](const torch::Tensor& tokens) {
          return draft_impl_->dspark_markov_bias(tokens);
        },
        base_logits,
        anchors,
        sampling_params_on_device,
        [this](torch::Tensor& tokens) {
          sync_sampling_tensor(tokens, sampling_process_group_);
        },
        is_probabilistic_draft_);
  }

  draft_proposal.keep_alive() = take_retained_inputs(*draft_output);

  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              draft_timer.elapsed_seconds());
  return draft_proposal;
}

DraftProposal sample_dspark_markov_block(const DSparkMarkovBias& markov_bias,
                                         const torch::Tensor& base_logits,
                                         const torch::Tensor& anchors,
                                         const SamplingParameters& sampling,
                                         const SyncSampledTokens& sync_tokens,
                                         bool is_probabilistic_draft,
                                         torch::Tensor token_ids_out) {
  const int64_t rows = base_logits.size(0);
  const int64_t steps = base_logits.size(1);
  const int64_t vocab = base_logits.size(2);
  CHECK_EQ(anchors.size(0), rows);
  // markov_bias embeds anchors and rejects other dtypes with an opaque
  // kernel error; enforce the contract here.
  CHECK_EQ(anchors.scalar_type(), torch::kInt64);
  CHECK_EQ(anchors.device(), base_logits.device());
  DraftProposal proposal =
      token_ids_out.defined()
          ? DraftProposal::from_token_buffer(
                std::move(token_ids_out),
                vocab,
                is_probabilistic_draft,
                base_logits.options().dtype(torch::kFloat32))
          : DraftProposal::preallocate(
                rows,
                steps,
                vocab,
                is_probabilistic_draft,
                anchors.options(),
                base_logits.options().dtype(torch::kFloat32));
  auto previous = anchors;
  torch::Tensor scratch;
  const bool needs_sync = sampling.sampling_mode != SamplingMode::GREEDY;
  for (int64_t step = 0; step < steps; ++step) {
    auto bias = markov_bias(previous);
    CHECK_EQ(bias.sizes(), torch::IntArrayRef({rows, vocab}))
        << "DSpark reduced vocabulary requires draft-to-target remapping.";
    torch::Tensor step_logits;
    if (proposal.logits().has_value()) {
      // add_out into the logits page lets the sampler mutate it in place;
      // draft_logits aliases the page directly.
      torch::Tensor page = proposal.logits_at(step);
      torch::add_out(page, base_logits.select(1, step), bias);
      step_logits = page;
    } else {
      if (!scratch.defined()) {
        scratch = torch::empty(base_logits.select(1, step).sizes(),
                               base_logits.options());
      }
      torch::add_out(scratch, base_logits.select(1, step), bias);
      step_logits = scratch;
    }
    auto sampled = sample(step_logits, sampling);
    if (needs_sync) {
      sync_tokens(sampled.next_tokens);
    }
    proposal.tokens_at(step).copy_(sampled.next_tokens);
    previous = sampled.next_tokens;
  }
  return proposal;
}

}  // namespace xllm
