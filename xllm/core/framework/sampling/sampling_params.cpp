/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/framework/sampling/sampling_params.h"

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "core/common/metrics.h"
#include "core/common/types.h"
#include "core/util/tensor_helper.h"

namespace xllm {

namespace {

void check_host_tensor(const torch::Tensor& tensor,
                       torch::ScalarType scalar_type) {
  CHECK(tensor.defined());
  CHECK(tensor.device().is_cpu()) << "token_at requires a CPU tensor";
  CHECK(tensor.is_contiguous())
      << "token_at requires contiguous tensor storage";
  CHECK(tensor.scalar_type() == scalar_type)
      << "token_at tensor dtype does not match its output field";
}

void check_next_tokens(const torch::Tensor& next_tokens, int64_t rank) {
  CHECK(next_tokens.defined());
  CHECK(next_tokens.device().is_cpu()) << "token_at requires CPU next_tokens";
  CHECK(next_tokens.is_contiguous())
      << "token_at requires contiguous next_tokens";
  CHECK(next_tokens.scalar_type() == torch::kInt32 ||
        next_tokens.scalar_type() == torch::kInt64)
      << "token_at requires int32 or int64 next_tokens";
  CHECK_EQ(next_tokens.dim(), rank)
      << "token_at overload must match next_tokens rank";
}

Token token_at_offset(const SampleOutput& output, int64_t offset) {
  const torch::Tensor& next_tokens = output.next_tokens;
  const int64_t id = next_tokens.scalar_type() == torch::kInt32
                         ? next_tokens.const_data_ptr<int32_t>()[offset]
                         : next_tokens.const_data_ptr<int64_t>()[offset];
  Token token(id);
  CHECK_EQ(output.top_tokens.defined(), output.top_logprobs.defined())
      << "token_at requires paired top-k tensors";
  CHECK(!output.top_tokens.defined() || output.logprobs.defined())
      << "token_at requires logprobs when top-k tensors are present";
  if (output.logprobs.defined()) {
    check_host_tensor(output.logprobs, torch::kFloat32);
    CHECK(output.logprobs.sizes() == next_tokens.sizes())
        << "token_at logprobs must match next_tokens shape";
    token.logprob = output.logprobs.const_data_ptr<float>()[offset];
  }
  if (output.top_tokens.defined()) {
    check_host_tensor(output.top_tokens, torch::kInt64);
    check_host_tensor(output.top_logprobs, torch::kFloat32);
    CHECK_EQ(output.top_tokens.dim(), next_tokens.dim() + 1);
    CHECK(output.top_tokens.sizes() == output.top_logprobs.sizes());
    CHECK(output.top_tokens.sizes().slice(0, next_tokens.dim()) ==
          next_tokens.sizes());
    const int64_t top_k = output.top_tokens.size(-1);
    if (top_k > 0) {
      token.top_tokens = {
          output.top_tokens.const_data_ptr<int64_t>() + offset * top_k,
          static_cast<size_t>(top_k)};
      token.top_logprobs = {
          output.top_logprobs.const_data_ptr<float>() + offset * top_k,
          static_cast<size_t>(top_k)};
    }
  }
  return token;
}

}  // namespace

Token SampleOutput::token_at(int64_t output_idx) const {
  check_next_tokens(next_tokens, /*rank=*/1);
  CHECK_GE(output_idx, 0);
  CHECK_LT(output_idx, next_tokens.size(0));
  return token_at_offset(*this, output_idx);
}

Token SampleOutput::token_at(int64_t output_idx, int64_t token_idx) const {
  check_next_tokens(next_tokens, /*rank=*/2);
  CHECK_GE(output_idx, 0);
  CHECK_LT(output_idx, next_tokens.size(0));
  CHECK_GE(token_idx, 0);
  CHECK_LT(token_idx, next_tokens.size(1));
  return token_at_offset(*this, output_idx * next_tokens.size(1) + token_idx);
}

torch::Tensor append_token_suffix(const torch::Tensor& suffix,
                                  const torch::Tensor& tokens) {
  CHECK_EQ(suffix.dim(), 2);
  CHECK_EQ(tokens.dim(), 2);
  CHECK_EQ(suffix.size(0), tokens.size(0));
  if (suffix.size(1) == 0 || tokens.size(1) == 0) {
    return suffix;
  }
  const torch::Tensor lengths = tokens.ge(0).sum(/*dim=*/1, /*keepdim=*/true);
  const torch::Tensor columns =
      torch::arange(suffix.size(1), lengths.options());
  return torch::cat({suffix, tokens.to(suffix.options())}, /*dim=*/1)
      .gather(/*dim=*/1, lengths + columns);
}

void advance_sampling_history(SamplingParameters& params,
                              const torch::Tensor& prior_tokens,
                              const std::vector<int32_t>& output_rows) {
  if (!prior_tokens.defined() || output_rows.empty() ||
      (!params.bad_words().defined() && !params.seeds().defined() &&
       !params.repetition_penalties().defined() &&
       !params.frequency_penalties().defined())) {
    return;
  }
  const torch::Tensor rows =
      torch::tensor(output_rows, torch::kInt64).to(prior_tokens.device());
  advance_sampling_history(params, prior_tokens, rows);
}

void advance_sampling_history(SamplingParameters& params,
                              const torch::Tensor& prior_tokens,
                              const torch::Tensor& output_rows) {
  CHECK_EQ(output_rows.numel(), params.num_sample_seqs());
  const torch::Tensor tokens =
      prior_tokens.dim() == 1 ? prior_tokens.unsqueeze(1) : prior_tokens;
  params.prefix_tokens() =
      tokens.index_select(/*dim=*/0, output_rows.clamp_min(0))
          .to(torch::kInt64);
  params.prefix_tokens().masked_fill_(output_rows.lt(0).unsqueeze(1), -1);
  if (params.bad_word_suffix().defined()) {
    params.bad_word_suffix() =
        append_token_suffix(params.bad_word_suffix(), params.prefix_tokens());
  }
  if (params.seed_offsets().defined()) {
    params.seed_offsets() =
        params.seed_offsets() + params.prefix_tokens().ge(0).sum(/*dim=*/1);
  }
}

void SamplingParameters::init(
    const std::vector<const RequestSamplingParam*>& req_sampling_params,
    const std::vector<int32_t>& selected_token_idxes,
    const std::vector<std::vector<int64_t>>& unique_token_ids_vec,
    const std::vector<std::vector<int32_t>>& unique_token_counts_vec,
    const std::vector<int32_t>& unique_token_lens_vec) {
  *this = SamplingParameters();
  CHECK_EQ(req_sampling_params.size(), selected_token_idxes.size());

  std::vector<float> frequency_penalties;
  std::vector<float> presence_penalties;
  std::vector<float> repetition_penalties;
  std::vector<float> temperatures;
  std::vector<float> top_p;
  std::vector<float> min_p;
  std::vector<int64_t> top_k;
  frequency_penalties.reserve(req_sampling_params.size());
  presence_penalties.reserve(req_sampling_params.size());
  repetition_penalties.reserve(req_sampling_params.size());
  temperatures.reserve(req_sampling_params.size());
  top_p.reserve(req_sampling_params.size());
  min_p.reserve(req_sampling_params.size());
  top_k.reserve(req_sampling_params.size());
  bool logprobs = false;
  int64_t max_top_logprobs = 0;
  bool is_embeddings = false;
  for (const auto* p : req_sampling_params) {
    frequency_penalties.push_back(p->frequency_penalty);
    presence_penalties.push_back(p->presence_penalty);
    repetition_penalties.push_back(p->repetition_penalty);
    temperatures.emplace_back(
        p->temperature > 0 ? std::max(p->temperature, 0.01F) : 0.0F);
    top_p.emplace_back(p->temperature == 0 ? 1.0F : p->top_p);
    top_k.emplace_back(p->temperature == 0 ? 0 : p->top_k);
    min_p.emplace_back(p->temperature == 0 ? 0.0F : p->min_p);
    logprobs = logprobs || p->logprobs;
    is_embeddings = is_embeddings || p->is_embeddings;
    max_top_logprobs = std::max(max_top_logprobs, p->top_logprobs);
  }

  bool need_token_stats = false;

  if (std::any_of(frequency_penalties.begin(),
                  frequency_penalties.end(),
                  [](float t) { return t != 0.0; }) ||
      std::any_of(presence_penalties.begin(),
                  presence_penalties.end(),
                  [](float t) { return t != 0.0; })) {
    this->frequency_penalties() = make_pinned_cpu_tensor(frequency_penalties);
    this->presence_penalties() = make_pinned_cpu_tensor(presence_penalties);
    need_token_stats = true;
  }
  if (std::any_of(repetition_penalties.begin(),
                  repetition_penalties.end(),
                  [](float t) { return t != 1.0; })) {
    this->repetition_penalties() = make_pinned_cpu_tensor(repetition_penalties);
    need_token_stats = true;
  }
  if (std::any_of(temperatures.begin(), temperatures.end(), [](float t) {
        return t != 0.0 && t != 1.0;
      })) {
    this->temperatures() = make_pinned_cpu_tensor(temperatures);
  }
  if (std::any_of(
          top_k.begin(), top_k.end(), [](int64_t t) { return t > 0; })) {
    this->top_k() = make_pinned_cpu_tensor(top_k);
  }
  if (std::any_of(
          top_p.begin(), top_p.end(), [](float t) { return t != 1.0; })) {
    this->top_p() = make_pinned_cpu_tensor(top_p);
  }

  if (std::any_of(
          min_p.begin(), min_p.end(), [](float value) { return value > 0; })) {
    this->min_p() = make_pinned_cpu_tensor(min_p);
  }

  this->selected_token_idxes() = make_pinned_cpu_tensor(selected_token_idxes);
  if (need_token_stats) {
    CHECK_EQ(req_sampling_params.size(), unique_token_ids_vec.size());
    CHECK_EQ(req_sampling_params.size(), unique_token_counts_vec.size());
    CHECK_EQ(req_sampling_params.size(), unique_token_lens_vec.size());
    this->unique_token_ids() =
        create_2d_tensor(unique_token_ids_vec, torch::kInt64);
    this->unique_token_counts() =
        create_2d_tensor(unique_token_counts_vec, torch::kInt);
    this->unique_token_ids_lens() =
        make_pinned_cpu_tensor(unique_token_lens_vec);
  }

  // Only a mixed batch dispatches per sequence; uniform batches (greedy or
  // random, seeded or not) carry no do_sample tensor. The temperatures
  // vector above normalizes temperature: 0.0F is greedy.
  const size_t num_random_seqs = std::count_if(
      temperatures.begin(), temperatures.end(), [](const float temperature) {
        return temperature != 0.0F;
      });
  const bool any_random = num_random_seqs > 0;
  const bool all_random =
      !temperatures.empty() && num_random_seqs == temperatures.size();
  if (any_random && !all_random) {
    // The rejection sampler's torch::where requires a bool condition tensor.
    std::vector<bool> do_sample;
    do_sample.reserve(temperatures.size());
    for (const float temperature : temperatures) {
      do_sample.push_back(temperature != 0.0F);
    }
    this->do_sample() = make_pinned_cpu_tensor(do_sample);
  }
  this->sampling_mode() =
      all_random ? SamplingMode::RANDOM
                 : (any_random ? SamplingMode::MIXED : SamplingMode::GREEDY);
  this->logprobs() = logprobs;
  this->max_top_logprobs() = max_top_logprobs;
  this->is_embeddings() = is_embeddings;
}

SamplingParameters SamplingParameters::to(const torch::Device& device,
                                          torch::ScalarType dtype) const {
  SamplingParameters params;

  // selected indices are tiny control tensors and correctness-critical. Use
  // blocking H2D copies to avoid consuming partially transferred index
  // buffers on NPU runtime paths.
  params.selected_token_idxes() =
      selected_token_idxes().defined()
          ? safe_to(selected_token_idxes(), device).contiguous()
          : selected_token_idxes();
  const torch::TensorOptions options = torch::device(device).dtype(dtype);
  if (filter_bitmask().defined()) {
    if (device.is_cpu()) {
      params.filter_bitmask() =
          safe_to(filter_bitmask(), device, true).contiguous();
    } else {
      Timer transfer_timer;
      params.filter_bitmask() =
          safe_to(filter_bitmask(), device, true).contiguous();
      HISTOGRAM_OBSERVE(
          json_object_mask_transfer_submission_latency_microseconds,
          static_cast<int64_t>(transfer_timer.elapsed_microseconds()));
    }
  }
  params.frequency_penalties() = safe_to(frequency_penalties(), options, true);
  params.presence_penalties() = safe_to(presence_penalties(), options, true);
  params.repetition_penalties() =
      safe_to(repetition_penalties(), options, true);
  params.temperatures() = safe_to(temperatures(), options, true);
  params.top_p() = safe_to(top_p(), options, true);
  params.min_p() = safe_to(min_p(), options, true);
  params.seed_offsets() = safe_to(seed_offsets(), device, true);
  params.bad_words() = safe_to(bad_words(), device, true);
  params.bad_word_suffix() = safe_to(bad_word_suffix(), device, true);
  params.prefix_tokens() = safe_to(prefix_tokens(), device, true);

  params.seeds() = safe_to(seeds(), device, true);
  params.logits_bias() = safe_to(logits_bias(), device, true);
  params.min_token_positions() = safe_to(min_token_positions(), device, true);
  params.min_tokens_remaining() = safe_to(min_tokens_remaining(), device, true);
  params.min_token_ids() = safe_to(min_token_ids(), device, true);
  params.top_k() = safe_to(top_k(), device, true);

  params.unique_token_ids() = safe_to(unique_token_ids(), device, true);
  params.unique_token_counts() = safe_to(unique_token_counts(), device, true);
  params.unique_token_ids_lens() =
      safe_to(unique_token_ids_lens(), device, true);

  params.do_sample() = safe_to(do_sample(), device, true);
  params.sampling_mode() = sampling_mode();
  params.logprobs() = logprobs();
  params.return_logits() = return_logits();
  params.max_top_logprobs() = max_top_logprobs();
  params.is_embeddings() = is_embeddings();
  return params;
}

}  // namespace xllm
