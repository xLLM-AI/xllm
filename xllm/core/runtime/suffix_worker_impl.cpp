/* Copyright 2025-2026 The xLLM Authors.

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

#include "suffix_worker_impl.h"

#include <algorithm>

#include "common/metrics.h"
#include "framework/sampling/rejection_sampler.h"
#include "util/slice.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {

void append_tokens_with_limit(std::vector<int32_t>& history,
                              std::span<const int32_t> tokens,
                              size_t max_size) {
  history.insert(history.end(), tokens.begin(), tokens.end());
  if (history.size() > max_size) {
    history.erase(history.begin(), history.end() - max_size);
  }
}

std::string summarize_int32_span(std::span<const int32_t> values,
                                 size_t limit = 8) {
  std::string out = "[";
  const size_t n = std::min(values.size(), limit);
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) {
      out += ",";
    }
    out += std::to_string(values[i]);
  }
  if (values.size() > n) {
    out += ",...";
  }
  out += "]";
  return out;
}

}  // namespace

namespace {
runtime::Options SuffixTargetOptions(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false);
  return opts;
}
}  // namespace

SuffixWorkerImpl::SuffixWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options,
                                   WorkerType worker_type)
    : SpeculativeWorkerImpl(parallel_args,
                            device,
                            options,
                            SuffixTargetOptions(options),
                            worker_type) {
  suffix_cache_ = std::make_unique<SuffixDecodingCache>(
      options_.speculative_suffix_cache_max_depth(),
      options_.speculative_suffix_max_cached_requests());
}

std::optional<ForwardOutput> SuffixWorkerImpl::step_empty(
    const ForwardInput& input) {
  if (!input.input_params.meta.batch_forward_type.is_decode()) {
    auto output = target_impl_->step(input);
    output->sample_output.embeddings = torch::Tensor();
    return output;
  } else {
    ForwardInput new_input = input;
    for (auto& it : new_input.input_params.parallel.dp_global_token_nums) {
      it *= options_.num_speculative_tokens() + 1;
    }

    auto future = target_impl_->step_async(new_input);
    ForwardOutput output = std::move(future).get().value();
    output.sample_output.embeddings = torch::Tensor();
    return output;
  }
}

std::optional<ForwardOutput> SuffixWorkerImpl::step_prefill(
    const ForwardInput& input) {
  Timer timer;
  // run the target model to get first token and hidden states
  auto future = target_impl_->step_async(input);
  ForwardOutput output = std::move(future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  const auto& input_params = input.input_params;
  const int32_t num_sequences = input_params.meta.num_sequences;
  const auto& request_ids = input_params.embedding.request_ids;

  if (suffix_cache_ != nullptr &&
      request_ids.size() == static_cast<size_t>(num_sequences)) {
    const torch::Tensor& token_ids = input.token_ids_host;
    Slice<int32_t> tokens_ids_slice = {token_ids.data_ptr<int32_t>(),
                                       static_cast<size_t>(token_ids.numel())};

    int32_t start_idx = 0;
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      int32_t q_len = input_params.get_q_seq_len(seq_id);
      Slice<int32_t> seq_tokens =
          tokens_ids_slice.slice(start_idx, start_idx + q_len);
      start_idx += q_len;

      const std::string req_id = request_ids[seq_id];
      if (req_id.empty()) {
        continue;
      }

      if (!suffix_cache_->has_active_request(req_id)) {
        suffix_cache_->start_request(req_id, seq_tokens);
        suffix_recent_tokens_[req_id].clear();
      } else {
        suffix_cache_->add_active_prompt(req_id, seq_tokens);
      }
      append_tokens_with_limit(
          suffix_recent_tokens_[req_id],
          seq_tokens,
          static_cast<size_t>(suffix_cache_->max_tree_depth()));
    }

    torch::Tensor next_tokens =
        safe_to(output.sample_output.next_tokens, torch::kCPU);
    if (next_tokens.defined() &&
        next_tokens.numel() == static_cast<int64_t>(num_sequences)) {
      next_tokens = next_tokens.view({-1}).to(torch::kInt);
      Slice<int32_t> next_tokens_slice = {
          next_tokens.data_ptr<int32_t>(),
          static_cast<size_t>(next_tokens.numel())};
      for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
        int32_t token = next_tokens_slice[seq_id];
        if (token < 0) {
          continue;
        }
        const std::string req_id = request_ids[seq_id];
        if (req_id.empty()) {
          continue;
        }
        suffix_cache_->add_active_response(req_id,
                                           std::span<const int32_t>(&token, 1));
        append_tokens_with_limit(
            suffix_recent_tokens_[req_id],
            std::span<const int32_t>(&token, 1),
            static_cast<size_t>(suffix_cache_->max_tree_depth()));
      }
    }
  }

  output.sample_output.embeddings = torch::Tensor();
  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  return output;
}

std::optional<ForwardOutput> SuffixWorkerImpl::step_decode(
    const ForwardInput& input) {
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  const auto& request_ids = input.input_params.embedding.request_ids;

  const bool has_request_ids =
      suffix_cache_ != nullptr &&
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
        if (suffix_cache_->has_active_request(req_id)) {
          suffix_cache_->stop_request(req_id);
        }
        suffix_recent_tokens_.erase(req_id);
      }
    }
    suffix_active_decode_req_ids_ = std::move(current_req_ids);
  }

  const torch::Tensor& input_token_ids = input.token_ids_host;
  Slice<int32_t> input_tokens_slice = {
      input_token_ids.data_ptr<int32_t>(),
      static_cast<size_t>(input_token_ids.numel())};

  Timer timer;

  std::vector<int32_t> draft_tokens_flat;
  draft_tokens_flat.reserve(num_sequences * num_speculative_tokens);
  std::vector<std::string> req_ids(num_sequences);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    int32_t fallback_token = input_tokens_slice[seq_id];
    for (int32_t i = 0; i < num_speculative_tokens; ++i) {
      draft_tokens_flat.emplace_back(fallback_token);
    }

    if (suffix_cache_ == nullptr ||
        request_ids.size() != static_cast<size_t>(num_sequences)) {
      continue;
    }

    const std::string req_id = request_ids[seq_id];
    if (req_id.empty()) {
      continue;
    }
    req_ids[seq_id] = req_id;

    if (!suffix_cache_->has_active_request(req_id)) {
      suffix_cache_->start_request(
          req_id, std::span<const int32_t>(&fallback_token, 1));
      suffix_recent_tokens_[req_id].clear();
      append_tokens_with_limit(
          suffix_recent_tokens_[req_id],
          std::span<const int32_t>(&fallback_token, 1),
          static_cast<size_t>(suffix_cache_->max_tree_depth()));
    }

    auto& history = suffix_recent_tokens_[req_id];
    if (history.empty()) {
      append_tokens_with_limit(
          history,
          std::span<const int32_t>(&fallback_token, 1),
          static_cast<size_t>(suffix_cache_->max_tree_depth()));
    }

    SuffixDecodingDraft draft = suffix_cache_->speculate(
        req_id,
        std::span<const int32_t>(history.data(), history.size()),
        /*max_spec_tokens=*/num_speculative_tokens,
        options_.speculative_suffix_max_spec_factor(),
        options_.speculative_suffix_max_spec_offset(),
        options_.speculative_suffix_min_token_prob(),
        options_.speculative_suffix_use_tree_spec());

    const int32_t fill_count =
        std::min<int32_t>(num_speculative_tokens, draft.token_ids.size());
    for (int32_t i = 0; i < fill_count; ++i) {
      draft_tokens_flat[seq_id * num_speculative_tokens + i] =
          draft.token_ids[i];
    }
  }

  ForwardInput validate_input;
  prepare_validate_inputs(input, validate_input);
  validate_input.skip_sampling_for_logits_only = true;

  auto& validate_token_ids = validate_input.token_ids;
  for (int32_t i = 0; i < num_speculative_tokens; ++i) {
    std::vector<int32_t> draft_col;
    draft_col.reserve(num_sequences);
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      draft_col.emplace_back(
          draft_tokens_flat[seq_id * num_speculative_tokens + i]);
    }
    auto draft_col_tensor =
        torch::tensor(draft_col, validate_token_ids.options());
    auto mask = (validate_token_ids == -1 * (i + 1));
    validate_token_ids.masked_scatter_(mask, draft_col_tensor);
  }

  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  timer.reset();
  auto future = target_impl_->step_async(validate_input);
  ForwardOutput target_output = std::move(future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  torch::Tensor draft_token_ids =
      torch::tensor(draft_tokens_flat,
                    torch::TensorOptions().dtype(torch::kLong))
          .view({num_sequences, num_speculative_tokens})
          .to(target_output.logits.device());

  // RejectionSampler::forward requires draft_probs tensor for interface
  // compatibility, but greedy-only validation does not use its values.
  // Use a minimal placeholder to avoid one-hot scatter over vocab.
  auto draft_probs = torch::empty({num_sequences, num_speculative_tokens, 1},
                                  torch::TensorOptions()
                                      .dtype(torch::kFloat32)
                                      .device(target_output.logits.device()));

  timer.reset();
  SampleOutput val_output = validate(
      input.sampling_params, draft_token_ids, draft_probs, target_output);
  COUNTER_ADD(speculative_execution_latency_seconds_validation,
              timer.elapsed_seconds());

  if (suffix_cache_ != nullptr &&
      request_ids.size() == static_cast<size_t>(num_sequences)) {
    torch::Tensor accepted_tokens =
        safe_to(val_output.next_tokens, torch::kCPU).to(torch::kInt);
    accepted_tokens = accepted_tokens.view({num_sequences, num_val_tokens});
    Slice<int32_t> accepted_tokens_slice = {
        accepted_tokens.data_ptr<int32_t>(),
        static_cast<size_t>(accepted_tokens.numel())};

    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      const std::string& req_id = req_ids[seq_id];
      if (req_id.empty()) {
        continue;
      }

      std::vector<int32_t> accepted;
      accepted.reserve(num_val_tokens);
      int32_t first_reject_idx = -1;
      std::vector<int32_t> row_tokens;
      row_tokens.reserve(num_val_tokens);
      for (int32_t j = 0; j < num_val_tokens; ++j) {
        int32_t token = accepted_tokens_slice[seq_id * num_val_tokens + j];
        row_tokens.emplace_back(token);
        if (token < 0) {
          if (first_reject_idx < 0) {
            first_reject_idx = j;
          }
          break;
        }
        accepted.emplace_back(token);
      }

      if (seq_id < 8) {
        VLOG(3) << "[spec-validate-output] seq=" << seq_id
                << " req_id=" << req_id << " accepted_len=" << accepted.size()
                << " first_reject_idx=" << first_reject_idx << " accepted="
                << summarize_int32_span(std::span<const int32_t>(
                       accepted.data(), accepted.size()))
                << " row_tokens="
                << summarize_int32_span(std::span<const int32_t>(
                       row_tokens.data(), row_tokens.size()));
        VLOG(3) << "[spec-reject-handle] seq=" << seq_id
                << " accepted_prefix_len=" << accepted.size()
                << " first_reject_idx=" << first_reject_idx << " pos_offset = "
                << (static_cast<int32_t>(accepted.size()) - 1) << " row_tokens="
                << summarize_int32_span(std::span<const int32_t>(
                       row_tokens.data(), row_tokens.size()));
      }

      if (!accepted.empty()) {
        suffix_cache_->add_active_response(
            req_id, std::span<const int32_t>(accepted.data(), accepted.size()));
        append_tokens_with_limit(
            suffix_recent_tokens_[req_id],
            std::span<const int32_t>(accepted.data(), accepted.size()),
            static_cast<size_t>(suffix_cache_->max_tree_depth()));
      }
    }
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  val_output.embeddings = torch::Tensor();
  target_output.sample_output = val_output;
  return target_output;
}

SampleOutput SuffixWorkerImpl::validate(
    const SamplingParameters& sampling_params,
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_probs,
    const ForwardOutput& target_output) {
  (void)sampling_params;
  const int32_t num_val_tokens = options_.num_speculative_tokens() + 1;
  const int32_t batch_size =
      static_cast<int32_t>(draft_token_ids.size(/*dim=*/0));
  const int32_t vocab_size =
      static_cast<int32_t>(target_output.logits.size(/*dim=*/-1));
  CHECK_EQ(target_output.logits.size(/*dim=*/0),
           static_cast<int64_t>(batch_size) * num_val_tokens)
      << "suffix validate logits shape mismatch";

  using ISlice = torch::indexing::Slice;
  auto target_logits =
      target_output.logits.view({batch_size, num_val_tokens, vocab_size});
  // Use target greedy token as the bonus token, consistent with greedy verify.
  auto bonus_token_ids =
      target_logits.index({ISlice(), num_val_tokens - 1, ISlice()})
          .argmax(/*dim=*/-1, /*keepdim=*/true);

  // Suffix decoding always uses greedy sampling for validation,
  // regardless of the user's sampling parameters.
  auto greedy_do_sample = torch::zeros({batch_size}, torch::kBool);
  auto rejection_sampler =
      std::make_unique<RejectionSampler>(greedy_do_sample,
                                         /*all_random_sample=*/false,
                                         /*all_greedy_sample=*/true,
                                         target_output.logprobs,
                                         target_output.max_top_logprobs,
                                         enable_fused_kernel_);

  SampleOutput sample_output =
      rejection_sampler->forward(draft_token_ids.to(bonus_token_ids),
                                 draft_probs.to(target_logits.device()),
                                 target_logits,
                                 bonus_token_ids,
                                 /*mask_out_rejected_tokens=*/true);

  auto embeddings = target_output.sample_output.embeddings;
  sample_output.embeddings =
      embeddings.view({batch_size, num_val_tokens, embeddings.size(-1)});

  return sample_output;
}

}  // namespace xllm
