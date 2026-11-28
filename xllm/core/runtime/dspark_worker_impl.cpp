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

#include <c10/core/StreamGuard.h>
#include <glog/logging.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "core/common/types.h"
#include "core/framework/config/model_config.h"
#include "core/framework/model/model_type.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/sampling/sampling_params.h"
#include "runtime/llm_worker_impl.h"
#include "util/json_reader.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

void DSparkWorkerImpl::apply_draft_model_overrides(ModelArgs& args) {
  const bool deepseek_v4 = is_deepseek_v4_family(args.model_type());
  args.layers_to_capture({});
  if (deepseek_v4) {
#if defined(USE_NPU)
    CHECK_GT(args.dspark_num_layers(), 0)
        << "DeepSeek-V4 DSpark requires at least one draft layer.";
    const bool is_v41 = args.model_type() == "deepseek_v41";
    if (is_v41) {
      CHECK(ModelConfig::is_python_model_impl(context_.get_model_impl()))
          << "DeepSeek-V4.1 DSpark requires the Python model implementation.";
      JsonReader reader;
      CHECK(reader.parse(options_.draft_model_path().value() + "/config.json"));
      args.n_routed_experts(
          reader.value_or<int32_t>("text_config.dspark_n_routed_experts", 128));
      args.num_experts_per_tok(reader.value_or<int32_t>(
          "text_config.dspark_num_experts_per_tok", 3));
      args.n_activated_experts(args.num_experts_per_tok());
      args.kv_source_layer_ids({});
      args.index_source_layer_ids({});
      args.candidate_source_layer_id(-1);
      args.engram_layer_ids({});
      args.engram_num_embeddings({});
      args.first_k_dense_replace(0);
      args.enable_confidence_head(false);
    }
    args.n_layers(args.dspark_num_layers());
    args.n_hash_layers(0);
    args.compress_ratios(
        std::vector<int32_t>(args.dspark_num_layers(), is_v41 ? 0 : 1));
    args.model_type(is_v41 ? std::string(kDeepseekV41DSparkModelType)
                           : "deepseek_v4_dspark");
#else
    args.model_type("deepseek_v4_dspark");
#endif
  } else {
    args.model_type(args.model_type() == kDSparkGLM5DraftModelType
                        ? std::string(kDSparkGLM5DraftModelType)
                        : "DSparkDraftModel");
  }
}

bool DSparkWorkerImpl::init_model(const std::string& model_weights_path,
                                  int32_t random_seed,
                                  MasterStatus master_status) {
  const bool result = DFlashWorkerImpl::init_model(
      model_weights_path, random_seed, master_status);
  if (draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    draft_needs_dummy_tokens_ =
        prepare_for_draft().per_request &&
        ModelConfig::is_python_model_impl(context_.get_model_impl());
  }
  return result;
}

DecodeExpansionOptions DSparkWorkerImpl::prepare_for_draft() const {
  auto decode_options = DFlashWorkerImpl::prepare_for_draft();
  decode_options.query_width = options_.num_speculative_tokens();
  decode_options.per_request =
      draft_impl_->context_.get_model_args().model_type() !=
      "deepseek_v4_dspark";
  return decode_options;
}

std::pair<torch::Tensor, torch::Tensor> DSparkWorkerImpl::run_decode_draft(
    const LlmForwardInput& input,
    LlmForwardInput& validate_input,
    LlmForwardInput& draft_forward_input) {
  Timer draft_timer;

  LlmForwardInput query_input;
  prepare_query_inputs(input, query_input);

  const int32_t batch_size = input.input_params.meta.num_sequences;
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();

  query_input.skip_sampling_for_logits_only = true;
  std::optional<ForwardOutput> draft_output =
      run_worker_no_sync_impl(*draft_impl_,
                              query_input,
                              draft_forward_input,
                              *prepare_stream_,
                              *compute_stream_);
  CHECK(draft_output.has_value())
      << "DSpark draft forward must return an output.";
  CHECK(draft_output->logits.defined())
      << "DSpark draft forward must return logits.";
  prepare_validate_inputs(input, validate_input);

  torch::Tensor base_logits = draft_output->logits.view(
      {batch_size, num_speculative_tokens, draft_output->logits.size(-1)});
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  // The first draft query of each sequence carries its corrected anchor.
  const torch::Tensor anchors =
      draft_forward_input.token_ids
          .slice(0,
                 0,
                 static_cast<int64_t>(batch_size) * num_speculative_tokens,
                 num_speculative_tokens)
          .to(torch::kLong);
  SamplingParameters draft_sampling_params = input.sampling_params;
  // The draft forward already selected these logits. Markov sampling uses all
  // rows, so it does not need another blocking copy of the selection indices.
  draft_sampling_params.logits_indices() = torch::Tensor();
  prepare_draft_sampling(options_.enable_probabilistic_draft(),
                         draft_sampling_params);
  auto draft_proposal =
      sample_markov_block(base_logits,
                          anchors,
                          draft_sampling_params.to(base_logits.device(),
                                                   base_logits.scalar_type()));

  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              draft_timer.elapsed_seconds());
  return draft_proposal;
}

std::pair<torch::Tensor, torch::Tensor> DSparkWorkerImpl::sample_markov_block(
    const torch::Tensor& base_logits,
    const torch::Tensor& anchors,
    const SamplingParameters& sampling) {
  const int64_t rows = base_logits.size(0);
  const int64_t steps = base_logits.size(1);
  const int64_t draft_model_vocab = base_logits.size(2);
  torch::Tensor token_ids = torch::empty({rows, steps}, anchors.options());
  const auto logits_options = options_.enable_probabilistic_draft()
                                  ? base_logits.options().dtype(torch::kFloat32)
                                  : base_logits.options();
  torch::Tensor draft_logits;
  if (sampling.sampling_mode() != SamplingMode::GREEDY) {
    draft_logits = torch::empty(
        {rows, steps, draft_vocab_.mapped_vocab_size(draft_model_vocab)},
        logits_options);
  }
  const torch::Tensor writable_draft_logits =
      draft_vocab_.draft_logits_view(draft_logits);
  const torch::Tensor workspace =
      writable_draft_logits.defined()
          ? torch::Tensor()
          : torch::empty({rows, draft_model_vocab}, logits_options);
  auto previous = anchors;
  for (int64_t step = 0; step < steps; ++step) {
    auto bias = draft_impl_->get_causal_lm()->dspark_markov_bias(previous);
    CHECK_EQ(bias.sizes(), torch::IntArrayRef({rows, draft_model_vocab}))
        << "DSpark markov bias must be [rows, draft_model_vocab].";
    torch::Tensor step_logits =
        workspace.defined() ? workspace
                            : writable_draft_logits.select(/*dim=*/1, step);
    torch::add_out(step_logits, base_logits.select(/*dim=*/1, step), bias);

    auto sampled = sample(step_logits, sampling);
    maybe_broadcast_spec_tokens(sampled.next_tokens, sampling.sampling_mode());
    torch::Tensor target_logits;
    if (draft_logits.defined() && workspace.defined()) {
      sampled.draft_logits = step_logits;
      target_logits = draft_logits.select(/*dim=*/1, step);
    }
    draft_vocab_.remap(sampled, std::move(target_logits));
    token_ids.select(/*dim=*/1, step).copy_(sampled.next_tokens);
    previous = sampled.next_tokens;
  }
  return {std::move(token_ids), std::move(draft_logits)};
}

}  // namespace xllm
