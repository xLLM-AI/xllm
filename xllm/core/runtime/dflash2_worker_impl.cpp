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

#include "runtime/dflash2_worker_impl.h"

#include <glog/logging.h>

#include "common/metrics.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/sampling/sampler.h"
#include "util/timer.h"

namespace xllm {

namespace {

void clear_target_linear_state_metadata(ModelInputParams& input_params) {
  // Qwen3.8 targets own recurrent GDN state, while the official Qwen3
  // DFlash2 draft is a pure full-attention model.  prepare_query_inputs starts
  // from a copy of the target input, so do not let target-only slot ids make
  // the draft attention metadata builder classify these rows as recurrent.
  // The target validation input is prepared separately and keeps this state.
  input_params.embedding.linear_state_ids.clear();
  input_params.embedding.linear_state_indices = torch::Tensor();
  input_params.linear_state_cache_ops.clear();
  input_params.linear_state_validity_mask.clear();
}

}  // namespace

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
  clear_target_linear_state_metadata(query_input.input_params);

  const int32_t batch_size = input.input_params.meta.num_sequences;
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  CHECK_GT(batch_size, 0);
  CHECK_GT(num_speculative_tokens, 0);
  CHECK(input.token_ids_host.defined());
  CHECK_GE(input.token_ids_host.numel(), batch_size);
  torch::Tensor anchor_token_ids =
      input.token_ids_host.slice(/*dim=*/0, /*start=*/0, /*end=*/batch_size)
          .to(draft_impl_->device(), torch::kLong);

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
    sampled =
        sample_path(candidates, sampling_params, unary_logits.size(/*dim=*/-1));
  }

  DraftBlock draft_block;
  draft_block.token_ids = std::move(sampled.token_ids);
  draft_block.probs = std::move(sampled.selected_probs);
  draft_block.dense_probs = std::move(sampled.dense_probs);
  draft_block.retained_inputs = take_retained_inputs(*draft_output);
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  return draft_block;
}

DFlash2WorkerImpl::BlockSampleOutput DFlash2WorkerImpl::sample_path(
    const DFlash2CandidateOutput& candidates,
    const SamplingParameters& sampling_params,
    int64_t vocab_size) const {
  CHECK_EQ(candidates.candidate_ids.dim(), 3);
  CHECK_EQ(candidates.edge_logits.dim(), 4);
  const int64_t batch_size = candidates.candidate_ids.size(0);
  const int64_t num_steps = candidates.candidate_ids.size(1);
  const int64_t top_k = candidates.candidate_ids.size(2);
  CHECK_EQ(candidates.edge_logits.sizes(),
           torch::IntArrayRef({batch_size, num_steps, top_k, top_k}));

  SamplingParameters step_params = sampling_params;
  const torch::TensorOptions index_options =
      torch::TensorOptions()
          .dtype(torch::kInt)
          .device(candidates.edge_logits.device());
  step_params.selected_token_idxes = torch::empty({0}, index_options);
  step_params.sample_idxes = torch::empty({0}, index_options);
  step_params.return_probs = !step_params.all_greedy_sample;
  step_params.logprobs = false;
  step_params.max_top_logprobs = 0;
  step_params.use_beam_search = false;

  torch::Tensor token_ids =
      torch::empty({batch_size, num_steps}, candidates.candidate_ids.options());
  torch::Tensor selected_probs =
      torch::empty({batch_size, num_steps},
                   torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(candidates.edge_logits.device()));
  torch::Tensor candidate_probs =
      torch::empty({batch_size, num_steps, top_k}, selected_probs.options());
  torch::Tensor previous_indices =
      torch::zeros({batch_size}, candidates.candidate_ids.options());
  Sampler sampler;

  using ISlice = torch::indexing::Slice;
  for (int64_t step = 0; step < num_steps; ++step) {
    torch::Tensor edge =
        candidates.edge_logits.select(/*dim=*/1, /*index=*/step);
    torch::Tensor gather_indices = previous_indices.view({batch_size, 1, 1})
                                       .expand({batch_size, 1, top_k});
    torch::Tensor step_logits =
        edge.gather(/*dim=*/1, gather_indices).squeeze(/*dim=*/1);
    SampleOutput output = sampler.forward(step_logits, step_params);
    torch::Tensor sampled_indices = output.next_tokens.to(torch::kLong);
    synchronize_sampled_indices(sampled_indices, step_params);

    torch::Tensor step_candidates =
        candidates.candidate_ids.select(/*dim=*/1, /*index=*/step);
    torch::Tensor sampled_tokens =
        step_candidates.gather(/*dim=*/1, sampled_indices.view({-1, 1}))
            .view({-1});
    torch::Tensor step_probs;
    if (step_params.all_greedy_sample) {
      step_probs = torch::zeros({batch_size, top_k}, selected_probs.options());
      step_probs.scatter_(
          /*dim=*/1,
          sampled_indices.view({-1, 1}),
          torch::ones({batch_size, 1}, selected_probs.options()));
    } else {
      CHECK_EQ(output.probs.sizes(), step_logits.sizes());
      step_probs = output.probs.to(torch::kFloat32);
      if (!step_params.all_random_sample) {
        torch::Tensor greedy_probs =
            torch::zeros({batch_size, top_k}, selected_probs.options());
        greedy_probs.scatter_(
            /*dim=*/1,
            sampled_indices.view({-1, 1}),
            torch::ones({batch_size, 1}, selected_probs.options()));
        step_probs = torch::where(step_params.do_sample.view({batch_size, 1}),
                                  step_probs,
                                  greedy_probs);
      }
    }
    torch::Tensor chosen_probs =
        step_probs.gather(/*dim=*/1, sampled_indices.view({-1, 1})).view({-1});
    token_ids.index_put_({ISlice(), step}, sampled_tokens);
    selected_probs.index_put_({ISlice(), step}, chosen_probs);
    candidate_probs.index_put_({ISlice(), step, ISlice()}, step_probs);
    previous_indices = sampled_indices;
  }

  torch::Tensor dense_probs = torch::zeros({batch_size, num_steps, vocab_size},
                                           selected_probs.options());
  dense_probs.scatter_(
      /*dim=*/-1, candidates.candidate_ids, candidate_probs);
  return {.token_ids = std::move(token_ids),
          .selected_probs = std::move(selected_probs),
          .dense_probs = std::move(dense_probs)};
}

void DFlash2WorkerImpl::synchronize_sampled_indices(
    torch::Tensor& sampled_indices,
    const SamplingParameters& sampling_params) const {
  if (sampling_params.all_greedy_sample || sampling_process_group_ == nullptr ||
      sampling_process_group_->world_size() <= 1) {
    return;
  }
  sampled_indices = sampled_indices.contiguous();
  sampling_process_group_->broadcast(sampled_indices, /*root_rank=*/0);
}

}  // namespace xllm
