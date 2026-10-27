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

#include "suffix_worker_impl.h"

#include <algorithm>

#include "common/metrics.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/spec_verify.h"
#include "core/util/tensor_helper.h"
#include "util/slice.h"
#include "util/timer.h"

namespace xllm {

namespace {

runtime::Options SuffixTargetOptions(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false);
  return opts;
}
}  // namespace

SuffixWorkerImpl::SuffixWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : SpeculativeWorkerImpl<LlmForwardInput>(parallel_args,
                                             device,
                                             options,
                                             SuffixTargetOptions(options)),
      suffix_cache_(options_.speculative_suffix_cache_max_depth(),
                    options_.speculative_suffix_max_cached_requests()) {}

std::optional<ForwardOutput> SuffixWorkerImpl::step_empty(
    const LlmForwardInput& input) {
  if (!input.input_params.meta.batch_forward_type.is_decode()) {
    auto output = impl_->step(input);
    output->sample_output.embeddings = torch::Tensor();
    return output;
  } else {
    LlmForwardInput new_input = input.clone();
    scale_speculative_parallel_token_counts(new_input.input_params,
                                            options_.num_speculative_tokens());

    auto future = impl_->step_async(new_input);
    ForwardOutput output = std::move(future).get().value();
    output.sample_output.embeddings = torch::Tensor();
    return output;
  }
}

std::optional<ForwardOutput> SuffixWorkerImpl::step_prefill(
    const LlmForwardInput& input) {
  // run the target model to get first token and hidden states
  ForwardOutput output;
  {
    AUTO_COUNTER(speculative_execution_latency_seconds_target);
    output = impl_->step_async(input).get().value();
  }

  const auto& input_params = input.input_params;
  const int32_t num_sequences = input_params.meta.num_sequences;
  const auto& request_ids = input_params.embedding.request_ids;

  if (request_ids.size() == static_cast<size_t>(num_sequences)) {
    const torch::Tensor& token_ids = input.token_ids_host;
    Slice<int32_t> tokens_ids_slice = tensor_slice(token_ids);

    int32_t start_idx = 0;
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      int32_t q_len =
          input_params.attention.host.q_seq_lens[static_cast<size_t>(seq_id)];
      Slice<int32_t> seq_tokens =
          tokens_ids_slice.slice(start_idx, start_idx + q_len);
      start_idx += q_len;

      const std::string& req_id = request_ids[seq_id];
      if (req_id.empty()) {
        continue;
      }

      suffix_cache_.start_request(req_id, seq_tokens);
    }

    if (output.sample_output.next_tokens.defined() &&
        output.sample_output.next_tokens.numel() ==
            static_cast<int64_t>(num_sequences)) {
      torch::Tensor next_tokens = to_cpu_contiguous(
          output.sample_output.next_tokens.view({-1}), torch::kInt);
      Slice<int32_t> next_tokens_slice = tensor_slice(next_tokens);
      for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
        int32_t token = next_tokens_slice[seq_id];
        if (token < 0) {
          continue;
        }
        const std::string& req_id = request_ids[seq_id];
        if (req_id.empty()) {
          continue;
        }
        suffix_cache_.add_active_response(req_id,
                                          std::span<const int32_t>(&token, 1));
      }
    }
  }

  output.sample_output.embeddings = torch::Tensor();
  if (this->suppresses_non_driver_output()) {
    return std::nullopt;
  }
  return output;
}

std::optional<ForwardOutput> SuffixWorkerImpl::step_decode(
    const LlmForwardInput& input) {
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  const auto& request_ids = input.input_params.embedding.request_ids;

  const bool has_request_ids =
      request_ids.size() == static_cast<size_t>(num_sequences);
  if (has_request_ids) {
    std::unordered_set<std::string> current_req_ids;
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      if (!request_ids[seq_id].empty()) {
        current_req_ids.insert(request_ids[seq_id]);
      }
    }

    for (const auto& req_id : suffix_active_decode_req_ids_) {
      if (current_req_ids.find(req_id) == current_req_ids.end()) {
        suffix_cache_.stop_request(req_id);
      }
    }
    suffix_active_decode_req_ids_ = std::move(current_req_ids);
  }

  const torch::Tensor& input_token_ids = input.token_ids_host;
  Slice<int32_t> input_tokens_slice = tensor_slice(input_token_ids);

  Timer draft_timer;

  std::vector<int32_t> draft_tokens_flat;
  draft_tokens_flat.reserve(num_sequences * num_speculative_tokens);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const int32_t fallback_token = input_tokens_slice[seq_id];
    for (int32_t i = 0; i < num_speculative_tokens; ++i) {
      draft_tokens_flat.emplace_back(fallback_token);
    }

    if (!has_request_ids) {
      continue;
    }

    const std::string& req_id = request_ids[seq_id];
    if (req_id.empty()) {
      continue;
    }

    const Draft draft =
        suffix_cache_.speculate(req_id,
                                std::span<const int32_t>(&fallback_token, 1),
                                num_speculative_tokens);

    const int32_t fill_count =
        std::min<int32_t>(num_speculative_tokens, draft.token_ids.size());
    for (int32_t i = 0; i < fill_count; ++i) {
      draft_tokens_flat[seq_id * num_speculative_tokens + i] =
          draft.token_ids[i];
    }
  }

  LlmForwardInput validate_input;
  prepare_validate_inputs(input, validate_input);
  auto& validate_token_ids = validate_input.token_ids;
  for (int32_t i = 0; i < num_speculative_tokens; ++i) {
    std::vector<int32_t> draft_col;
    draft_col.reserve(num_sequences);
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      draft_col.emplace_back(
          draft_tokens_flat[seq_id * num_speculative_tokens + i]);
    }
    auto draft_col_tensor =
        async_h2d_tensor(draft_col, validate_token_ids.device());
    auto mask = (validate_token_ids == -1 * (i + 1));
    validate_token_ids.masked_scatter_(mask, draft_col_tensor);
  }

  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              draft_timer.elapsed_seconds());

  ForwardOutput target_output;
  {
    AUTO_COUNTER(speculative_execution_latency_seconds_target);
    target_output = impl_->step_async(validate_input).get().value();
  }

  torch::Tensor draft_token_ids =
      torch::tensor(draft_tokens_flat,
                    torch::TensorOptions().dtype(torch::kLong))
          .view({num_sequences, num_speculative_tokens})
          .to(target_output.logits.device());

  DraftProposal draft_proposal = DraftProposal(std::move(draft_token_ids));

  SampleOutput val_output;
  {
    AUTO_COUNTER(speculative_execution_latency_seconds_validation);
    val_output = validate(input.sampling_params, draft_proposal, target_output);
  }

  if (has_request_ids) {
    const torch::Tensor accepted_tokens =
        to_cpu_contiguous(val_output.next_tokens, torch::kInt);
    Slice<int32_t> accepted_tokens_slice =
        tensor_slice<int32_t>(accepted_tokens);

    std::vector<int32_t> accepted;
    accepted.reserve(num_val_tokens);
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      const std::string& req_id = request_ids[seq_id];
      if (req_id.empty()) {
        continue;
      }

      accepted.clear();
      for (int32_t j = 0; j < num_val_tokens; ++j) {
        const int32_t token =
            accepted_tokens_slice[seq_id * num_val_tokens + j];
        if (token < 0) {
          break;
        }
        accepted.emplace_back(token);
      }

      if (!accepted.empty()) {
        suffix_cache_.add_active_response(
            req_id, std::span<const int32_t>(accepted.data(), accepted.size()));
      }
    }
  }

  if (this->suppresses_non_driver_output()) {
    return std::nullopt;
  }
  // Land the verify tokens on host once here, matching DFlash, so the metrics
  // and serialization consumers never issue a second blocking D2H.
  val_output.next_tokens = to_cpu_contiguous(val_output.next_tokens);
  return finalize_verify_output(std::move(target_output),
                                std::move(val_output),
                                input.json_object_states);
}

SampleOutput SuffixWorkerImpl::validate(SamplingParameters sampling_params,
                                        const DraftProposal& draft_proposal,
                                        const ForwardOutput& target_output) {
  const int32_t num_val_tokens = options_.num_speculative_tokens() + 1;
  const int32_t batch_size =
      static_cast<int32_t>(draft_proposal.token_ids().size(/*dim=*/0));
  CHECK_EQ(target_output.logits.size(/*dim=*/0),
           static_cast<int64_t>(batch_size) * num_val_tokens)
      << "suffix validate logits shape mismatch";

  // The suffix draft carries no model distribution (q = 1 deterministic
  // suggestion), so the rejection sampler's no-logits path applies.
  return spec_verify::run_rejection_sampling(
      std::move(sampling_params),
      draft_proposal,
      target_output.logits,
      target_output.sample_output.embeddings);
}

}  // namespace xllm
