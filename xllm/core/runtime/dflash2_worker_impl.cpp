/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "runtime/dflash2_worker_impl.h"

#include <glog/logging.h>

#include "common/metrics.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/sampling/dflash2_sampling.h"
#include "framework/sampling/gumbel_sampling.h"
#include "util/timer.h"

namespace xllm {

DFlash2WorkerImpl::DFlash2WorkerImpl(const ParallelArgs& parallel_args,
                                     const torch::Device& device,
                                     const runtime::Options& options)
    : DFlashWorkerImpl(parallel_args, device, options),
      sampling_process_group_(parallel_args.tp_group_ != nullptr
                                  ? parallel_args.tp_group_
                                  : parallel_args.process_group_) {}

DFlashWorkerImpl::DraftBlock DFlash2WorkerImpl::run_decode_draft(
    const ForwardInput& input,
    ForwardInput& validate_input) {
  Timer timer;
  ForwardInput query_input;
  prepare_query_inputs(input, query_input);
  // The target's recurrent state must not leak into the pure full-attention
  // draft; the target validation input is prepared separately and keeps it.
  query_input.input_params.clear_linear_attention_state();

  const int32_t batch_size = input.input_params.meta.num_sequences;
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  CHECK_GT(batch_size, 0);
  CHECK_GT(num_speculative_tokens, 0);
  CHECK(input.token_ids_host.defined());
  CHECK_GE(input.token_ids_host.numel(), batch_size);
  torch::Tensor anchor_token_ids =
      input.token_ids_host.slice(/*dim=*/0, /*start=*/0, /*end=*/batch_size)
          .to(draft_impl_->device(), torch::kLong);

  // Build the Gumbel noise and finish its TP consensus up front: one
  // host-blocking broadcast here replaces the seven per-step index
  // broadcasts inside the previous path walk, so the loop below runs
  // device-side end to end and never blocks the host mid-loop.
  const ModelArgs& draft_args = draft_impl_->context_.get_model_args();
  const int64_t selector_top_k = draft_args.dflash2_selector_top_k();
  c10::StreamGuard noise_guard = compute_stream_->set_stream_guard();
  SamplingParameters draft_sampling_params =
      input.sampling_params.to(draft_impl_->device(), torch::kFloat32);
  if (draft_sampling_mode_ == DraftSamplingMode::GREEDY) {
    force_greedy_draft_sampling(draft_sampling_params);
  }
  const bool need_dense_probs = draft_probs_required(
      draft_sampling_mode_, draft_sampling_params.all_greedy_sample);
  torch::Tensor gumbel_noise = sample_gumbel_noise(batch_size,
                                                   num_speculative_tokens,
                                                   selector_top_k,
                                                   draft_sampling_params,
                                                   draft_impl_->device());
  // Cross-rank RNG divergence must not fork the sampled path: unify the noise
  // once from rank 0.  The edge logits are TP-replicated, so identical noise
  // yields identical argmax on every rank for all steps.
  if (sampling_process_group_ != nullptr &&
      sampling_process_group_->world_size() > 1) {
    gumbel_noise = gumbel_noise.contiguous();
    sampling_process_group_->broadcast(gumbel_noise, /*root_rank=*/0);
  }

  query_input.skip_sampling_for_logits_only = true;
  query_input.return_selected_hidden = true;
  ForwardInput processed_input;
  draft_impl_->prepare_work_before_execute_on_stream(
      query_input,
      processed_input,
      *prepare_stream_,
      /*record_ready_event=*/prepare_stream_.get() != compute_stream_.get());
  draft_impl_->set_hierarchy_layer_synchronizer(processed_input.input_params);
  std::optional<ForwardOutput> draft_output =
      draft_impl_->execute_no_sync_on_stream(processed_input,
                                             *compute_stream_,
                                             /*record_ready_event=*/false);
  CHECK(draft_output.has_value());
  CHECK(draft_output->logits.defined());
  CHECK(draft_output->selected_hidden.defined())
      << "DFlash2 requires selected pre-lm-head hidden states.";
  prepare_validate_inputs(input, validate_input);

  const int64_t num_rows = draft_output->logits.size(0);
  CHECK_EQ(num_rows, static_cast<int64_t>(batch_size) * num_speculative_tokens);
  torch::Tensor unary_logits = draft_output->logits.view(
      {batch_size, num_speculative_tokens, draft_output->logits.size(-1)});
  torch::Tensor hidden_states = draft_output->selected_hidden.view(
      {batch_size,
       num_speculative_tokens,
       draft_output->selected_hidden.size(-1)});

  DFlash2SampleOutput sampled;
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    DFlash2CandidateOutput candidates = draft_impl_->dflash2_candidates(
        hidden_states, unary_logits, anchor_token_ids);
    sampled = sample_dflash2_path(candidates,
                                  draft_sampling_params,
                                  gumbel_noise,
                                  unary_logits.size(/*dim=*/-1),
                                  need_dense_probs);
  }

  DraftBlock draft_block;
  if (sampled.dense_probs.defined()) {
    // Probabilistic DFlash2 paths retain the dense proposal distribution so
    // rejection recovery stays exact.
    draft_block.proposal = DraftProposal(std::move(sampled.token_ids),
                                         std::move(sampled.dense_probs));
  } else {
    // Greedy proposals are delta distributions and need no [B, N, V] tensor.
    draft_block.proposal = DraftProposal(std::move(sampled.token_ids));
  }
  draft_block.retained_inputs = take_retained_inputs(*draft_output);
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  return draft_block;
}

}  // namespace xllm
