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

#include "common/metrics.h"
#include "core/common/types.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/model/model_type.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/speculative/draft_proposal.h"
#include "runtime/llm_worker_impl.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {

int32_t configure_draft_config(ModelArgs& args,
                               const runtime::Options& options) {
  const bool deepseek_v4 = util::is_deepseek_v4_model_type(args.model_type());
  if (args.block_size() == 0) {
    args.block_size(options.num_speculative_tokens() + 1);
  }
  args.layers_to_capture({});
  if (deepseek_v4) {
#if defined(USE_NPU)
    CHECK_GT(args.dspark_num_layers(), 0)
        << "DeepSeek-V4 DSpark requires at least one draft layer.";
    args.n_layers(args.dspark_num_layers());
    args.n_hash_layers(0);
    args.compress_ratios(std::vector<int32_t>(args.dspark_num_layers(), 1));
    args.dspark_use_native_sas(
        KernelConfig::get_instance().enable_dspark_native_sas());
#endif
    args.model_type(std::string(util::kDeepseekV4DSparkModelType));
  } else {
    args.model_type(args.model_type() == kDSparkGLM5DraftModelType
                        ? std::string(kDSparkGLM5DraftModelType)
                        : "DSparkDraftModel");
  }
  return options.num_speculative_tokens();
}
}  // namespace

DSparkWorkerImpl::DSparkWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : DFlashWorkerImpl(
          parallel_args,
          device,
          options,
          [&] {
            return std::make_unique<LLMWorkerImpl>(
                configure_draft_parallel_args(parallel_args),
                device,
                draft_options(options, options.num_speculative_tokens() - 1),
                parallel_args.dp_size() > 1);
          },
          configure_draft_config) {
  sample_from_anchor_ = true;
}

bool DSparkWorkerImpl::init_model(const std::string& model_weights_path,
                                  int32_t random_seed,
                                  MasterStatus master_status) {
  const bool result = DFlashWorkerImpl::init_model(
      model_weights_path, random_seed, master_status);
  if (draft_impl_->get_status() != WorkerImpl::Status::LOADED) {
    return result;
  }

  const ModelArgs& draft_args = draft_impl_->context_.get_model_args();
  CHECK_EQ(options_.num_speculative_tokens(), draft_args.block_size())
      << "Runtime block size must match the draft model configuration.";
  const ModelArgs& target_args = impl_->context_.get_model_args();
  const int64_t draft_vocab_size = draft_args.draft_vocab_size() > 0
                                       ? draft_args.draft_vocab_size()
                                       : draft_args.vocab_size();
  CHECK(draft_vocab_size == target_args.vocab_size() ||
        draft_vocab_.draft_to_target_ids_.defined())
      << "A reduced draft vocabulary requires a DSpark d2t mapping.";

  if (util::is_deepseek_v4_dspark_model_type(draft_args.model_type())) {
    draft_input_per_request_ = draft_args.dspark_use_native_sas();
    draft_forward_type_ = BatchForwardType::DECODE;
  }
  draft_needs_dummy_tokens_ =
      draft_input_per_request_ &&
      ModelConfig::is_python_model_impl(context_.get_model_impl());
  return result;
}

DraftProposal DSparkWorkerImpl::run_decode_draft(
    const LlmForwardInput& input,
    LlmForwardInput& validate_input) {
  Timer draft_timer;

  LlmForwardInput query_input;
  prepare_query_inputs(input, query_input);

  const int32_t batch_size = input.input_params.meta.num_sequences;
  const torch::Tensor anchors = anchor_token_ids(input);
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();

  LlmForwardInput logits_input = query_input.clone();
  logits_input.skip_sampling_for_logits_only = true;
  logits_input.return_selected_hidden = false;
  std::optional<ForwardOutput> draft_output = run_worker_no_sync_impl(
      *draft_impl_, logits_input, *prepare_stream_, *compute_stream_);
  CHECK(draft_output.has_value())
      << "DSpark draft forward must return an output.";
  CHECK(draft_output->logits.defined())
      << "DSpark draft forward must return logits.";
  prepare_validate_inputs(input, validate_input);

  torch::Tensor base_logits = draft_output->logits.view(
      {batch_size, num_speculative_tokens, draft_output->logits.size(-1)});
  DraftProposal draft_proposal;
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    SamplingParameters sampling_params_on_device = input.sampling_params;
    prepare_draft_sampling(options_.enable_probabilistic_draft(),
                           sampling_params_on_device);
    sampling_params_on_device = sampling_params_on_device.to(
        base_logits.device(), base_logits.scalar_type());
    draft_proposal =
        sample_markov_block(base_logits, anchors, sampling_params_on_device);
  }

  draft_proposal.keep_alive() = take_retained_inputs(*draft_output);

  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              draft_timer.elapsed_seconds());
  return draft_proposal;
}

DraftProposal DSparkWorkerImpl::sample_markov_block(
    const torch::Tensor& base_logits,
    const torch::Tensor& anchors,
    const SamplingParameters& sampling) {
  const int64_t rows = base_logits.size(0);
  const int64_t steps = base_logits.size(1);
  const int64_t draft_model_vocab = base_logits.size(2);
  const bool has_vocab_mapping = draft_vocab_.draft_to_target_ids_.defined();
  const int64_t draft_vocab =
      has_vocab_mapping ? draft_vocab_.target_vocab_size_ : draft_model_vocab;
  DraftProposal proposal =
      DraftProposal::preallocate(rows,
                                 steps,
                                 draft_vocab,
                                 options_.enable_probabilistic_draft(),
                                 anchors.options(),
                                 base_logits.options().dtype(torch::kFloat32));
  auto previous = anchors;
  torch::Tensor scratch;
  for (int64_t step = 0; step < steps; ++step) {
    auto bias = draft_impl_->get_causal_lm()->dspark_markov_bias(previous);
    CHECK_EQ(bias.sizes(), torch::IntArrayRef({rows, draft_model_vocab}))
        << "DSpark markov bias must be [rows, draft_model_vocab].";
    torch::Tensor step_logits;
    if (!has_vocab_mapping && proposal.logits().has_value()) {
      // add_out into the logits page lets the sampler mutate it in place;
      // draft_logits aliases the page directly.
      torch::Tensor page = proposal.logits_at(step);
      torch::add_out(page, base_logits.select(1, step), bias);
      step_logits = page;
    } else {
      if (!scratch.defined()) {
        const bool float32_scratch =
            has_vocab_mapping && proposal.logits().has_value();
        scratch = torch::empty(
            base_logits.select(1, step).sizes(),
            base_logits.options().dtype(
                float32_scratch ? torch::kFloat32 : base_logits.scalar_type()));
      }
      torch::add_out(scratch, base_logits.select(1, step), bias);
      step_logits = scratch;
    }
    auto sampled = sample(step_logits, sampling);
    maybe_broadcast_spec_tokens(sampled.next_tokens, sampling.sampling_mode());
    if (has_vocab_mapping) {
      if (proposal.logits().has_value()) {
        sampled.draft_logits = step_logits;
        draft_vocab_.remap(sampled, proposal.logits_at(step));
      } else {
        sampled.draft_logits = torch::Tensor();
        draft_vocab_.remap(sampled);
      }
    }
    proposal.tokens_at(step).copy_(sampled.next_tokens);
    previous = sampled.next_tokens;
  }
  return proposal;
}

}  // namespace xllm
