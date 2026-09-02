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
  torch::Tensor gumbel_noise = sample_gumbel_noise(batch_size,
                                                   num_speculative_tokens,
                                                   selector_top_k,
                                                   input.sampling_params,
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

  BlockSampleOutput sampled;
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    DFlash2CandidateOutput candidates = draft_impl_->dflash2_candidates(
        hidden_states, unary_logits, anchor_token_ids);
    SamplingParameters sampling_params = input.sampling_params.to(
        unary_logits.device(), unary_logits.scalar_type());
    sampled = sample_path(candidates,
                          sampling_params,
                          gumbel_noise,
                          unary_logits.size(/*dim=*/-1));
  }

  DraftBlock draft_block;
  // DFlash2 samples selector paths from a sparse top-k distribution; the
  // dense per-token proposal must be retained so rejection recovery stays
  // exact. The selected-only probs carry no extra information for the
  // verifier, so only token_ids and the dense proposal feed the DraftProposal.
  draft_block.proposal = DraftProposal(std::move(sampled.token_ids),
                                       std::move(sampled.dense_probs));
  draft_block.retained_inputs = take_retained_inputs(*draft_output);
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  return draft_block;
}

DFlash2WorkerImpl::BlockSampleOutput DFlash2WorkerImpl::sample_path(
    const DFlash2CandidateOutput& candidates,
    const SamplingParameters& sampling_params,
    const torch::Tensor& gumbel_noise,
    int64_t vocab_size) const {
  CHECK_EQ(candidates.candidate_ids.dim(), 3);
  CHECK_EQ(candidates.edge_logits.dim(), 4);
  const int64_t batch_size = candidates.candidate_ids.size(0);
  const int64_t num_steps = candidates.candidate_ids.size(1);
  const int64_t top_k = candidates.candidate_ids.size(2);
  CHECK_EQ(candidates.edge_logits.sizes(),
           torch::IntArrayRef({batch_size, num_steps, top_k, top_k}));
  CHECK_EQ(gumbel_noise.sizes(),
           torch::IntArrayRef({batch_size, num_steps, top_k}));
  const torch::Device device = candidates.edge_logits.device();
  const torch::TensorOptions float_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  // Pre-compute log_softmax(edge_logits / temperature) for every step at
  // once.  The previous implementation re-scaled and re-normalized the
  // gathered row inside each step through the generic sampler, which lowered
  // to an AICPU multinomial plus a per-step consensus broadcast and left the
  // compute stream idle between steps.  Gumbel-max over the same
  // log-probabilities draws from exactly the same distribution:
  //   argmax(log_softmax(logits / T) + g) ~ categorical(softmax(logits / T))
  // and greedy rows (zeroed noise) collapse to plain argmax.  Target-side
  // truncation, penalties, and grammar constraints are applied by
  // verification and must not be applied a second time to the selector's
  // top-k candidate distribution.
  torch::Tensor edge_log_probs = candidates.edge_logits.to(torch::kFloat32);
  if (sampling_params.temperatures.defined()) {
    apply_selector_temperatures(
        edge_log_probs, sampling_params.temperatures, batch_size);
  }
  edge_log_probs = torch::log_softmax(edge_log_probs, /*dim=*/-1);

  torch::Tensor token_ids =
      torch::empty({batch_size, num_steps}, candidates.candidate_ids.options());
  torch::Tensor candidate_probs =
      torch::empty({batch_size, num_steps, top_k}, float_options);
  torch::Tensor previous_indices =
      torch::zeros({batch_size}, candidates.candidate_ids.options());

  using ISlice = torch::indexing::Slice;
  for (int64_t step = 0; step < num_steps; ++step) {
    torch::Tensor edge = edge_log_probs.select(/*dim=*/1, /*index=*/step);
    torch::Tensor gather_indices = previous_indices.view({batch_size, 1, 1})
                                       .expand({batch_size, 1, top_k});
    torch::Tensor row_log_probs =
        edge.gather(/*dim=*/1, gather_indices).squeeze(/*dim=*/1);
    torch::Tensor sampled_indices = gumbel_argmax(
        row_log_probs, gumbel_noise.select(/*dim=*/1, /*index=*/step));

    torch::Tensor step_candidates =
        candidates.candidate_ids.select(/*dim=*/1, /*index=*/step);
    torch::Tensor sampled_tokens =
        step_candidates.gather(/*dim=*/1, sampled_indices.view({-1, 1}))
            .view({-1});
    // Keep the per-step proposal distribution semantics identical to the
    // previous sampler path: random rows expose the full softmax over the
    // top-k candidates, deterministic rows expose a one-hot at the argmax,
    // and mixed batches select per row via do_sample. The exp() lives inside
    // the consuming branches so greedy-only batches skip it.
    torch::Tensor step_probs;
    if (sampling_params.all_random_sample) {
      step_probs = row_log_probs.exp();
    } else {
      torch::Tensor greedy_probs =
          torch::zeros({batch_size, top_k}, float_options);
      greedy_probs.scatter_(/*dim=*/1,
                            sampled_indices.view({-1, 1}),
                            /*value=*/1.0);
      if (sampling_params.all_greedy_sample) {
        step_probs = greedy_probs;
      } else {
        step_probs =
            torch::where(sampling_params.do_sample.view({batch_size, 1}),
                         row_log_probs.exp(),
                         greedy_probs);
      }
    }
    token_ids.index_put_({ISlice(), step}, sampled_tokens);
    candidate_probs.index_put_({ISlice(), step, ISlice()}, step_probs);
    previous_indices = sampled_indices;
  }

  torch::Tensor dense_probs =
      torch::zeros({batch_size, num_steps, vocab_size}, float_options);
  dense_probs.scatter_(
      /*dim=*/-1, candidates.candidate_ids, candidate_probs);
  return {.token_ids = std::move(token_ids),
          .dense_probs = std::move(dense_probs)};
}

}  // namespace xllm
