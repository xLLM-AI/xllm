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

void SamplingParameters::init(
    const std::vector<const RequestSamplingParam*>& req_sampling_params,
    const std::vector<int32_t>& selected_token_idxes,
    const std::vector<std::vector<int64_t>>& unique_token_ids_vec,
    const std::vector<std::vector<int32_t>>& unique_token_counts_vec,
    const std::vector<int32_t>& unique_token_lens_vec,
    const std::vector<torch::Tensor>& filter_mask_rows) {
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
  int32_t num_return_sequences = 0;
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
    num_return_sequences =
        std::max(num_return_sequences, p->num_return_sequences);
    if (p->beam_width > 0) {
      use_beam_search = true;
    }
  }

  bool need_token_stats = false;

  if (std::any_of(frequency_penalties.begin(),
                  frequency_penalties.end(),
                  [](float t) { return t != 0.0; }) ||
      std::any_of(presence_penalties.begin(),
                  presence_penalties.end(),
                  [](float t) { return t != 0.0; })) {
    this->frequency_penalties = make_pinned_cpu_tensor(frequency_penalties);
    this->presence_penalties = make_pinned_cpu_tensor(presence_penalties);
    need_token_stats = true;
  }
  if (std::any_of(repetition_penalties.begin(),
                  repetition_penalties.end(),
                  [](float t) { return t != 1.0; })) {
    this->repetition_penalties = make_pinned_cpu_tensor(repetition_penalties);
    need_token_stats = true;
  }
  if (std::any_of(temperatures.begin(), temperatures.end(), [](float t) {
        return t != 0.0 && t != 1.0;
      })) {
    this->temperatures = make_pinned_cpu_tensor(temperatures);
  }
  if (std::any_of(
          top_k.begin(), top_k.end(), [](int64_t t) { return t > 0; })) {
    this->top_k = make_pinned_cpu_tensor(top_k);
  }
  if (std::any_of(
          top_p.begin(), top_p.end(), [](float t) { return t != 1.0; })) {
    this->top_p = make_pinned_cpu_tensor(top_p);
  }

  if (std::any_of(
          min_p.begin(), min_p.end(), [](float value) { return value > 0; })) {
    this->min_p = make_pinned_cpu_tensor(min_p);
  }

  this->selected_token_idxes = make_pinned_cpu_tensor(selected_token_idxes);
  const bool has_filter_mask =
      std::any_of(filter_mask_rows.begin(),
                  filter_mask_rows.end(),
                  [](const torch::Tensor& row) { return row.defined(); });
  if (has_filter_mask) {
    Timer mask_batch_timer;
    CHECK_EQ(filter_mask_rows.size(), req_sampling_params.size());
    int64_t vocab_size = 0;
    for (const auto& row : filter_mask_rows) {
      if (row.defined()) {
        CHECK_EQ(row.dim(), 1) << "filter mask rows must be 1-D";
        vocab_size = row.size(0);
        break;
      }
    }
    CHECK_GT(vocab_size, 0)
        << "a filter mask batch must contain a constrained row";
    std::vector<torch::Tensor> rows;
    rows.reserve(filter_mask_rows.size());
    for (const auto& row : filter_mask_rows) {
      if (row.defined()) {
        CHECK_EQ(row.size(0), vocab_size)
            << "filter mask vocabulary sizes must match";
        rows.push_back(row);
      } else {
        rows.push_back(torch::zeros(
            {vocab_size}, torch::TensorOptions().dtype(torch::kFloat32)));
      }
    }
    this->filter_mask =
        torch::cat(rows, /*dim=*/0)
            .view({static_cast<int64_t>(rows.size()), vocab_size});
    HISTOGRAM_OBSERVE(
        json_object_mask_batch_build_latency_microseconds,
        static_cast<int64_t>(mask_batch_timer.elapsed_microseconds()));
  }
  if (need_token_stats) {
    CHECK_EQ(req_sampling_params.size(), unique_token_ids_vec.size());
    CHECK_EQ(req_sampling_params.size(), unique_token_counts_vec.size());
    CHECK_EQ(req_sampling_params.size(), unique_token_lens_vec.size());
    this->unique_token_ids =
        create_2d_tensor(unique_token_ids_vec, torch::kInt64);
    this->unique_token_counts =
        create_2d_tensor(unique_token_counts_vec, torch::kInt);
    this->unique_token_ids_lens = make_pinned_cpu_tensor(unique_token_lens_vec);
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
    this->do_sample = make_pinned_cpu_tensor(do_sample);
  }
  this->sampling_mode =
      all_random ? SamplingMode::RANDOM
                 : (any_random ? SamplingMode::MIXED : SamplingMode::GREEDY);
  this->logprobs = logprobs;
  this->max_top_logprobs = max_top_logprobs;
  this->is_embeddings = is_embeddings;
  this->num_return_sequences = num_return_sequences;
}

SamplingParameters SamplingParameters::to(const torch::Device& device,
                                          torch::ScalarType dtype) const {
  SamplingParameters params;

  // selected indices are tiny control tensors and correctness-critical. Use
  // blocking H2D copies to avoid consuming partially transferred index
  // buffers on NPU runtime paths.
  params.selected_token_idxes =
      selected_token_idxes.defined()
          ? safe_to(selected_token_idxes, device).contiguous()
          : selected_token_idxes;
  const torch::TensorOptions options = torch::device(device).dtype(dtype);
  if (filter_mask.defined()) {
    if (device.is_cpu()) {
      params.filter_mask = safe_to(filter_mask, options, true).contiguous();
    } else {
      Timer transfer_timer;
      params.filter_mask = safe_to(filter_mask, options, true).contiguous();
      HISTOGRAM_OBSERVE(
          json_object_mask_transfer_submission_latency_microseconds,
          static_cast<int64_t>(transfer_timer.elapsed_microseconds()));
    }
  }
  if (filter_bitmask.defined()) {
    if (device.is_cpu()) {
      params.filter_bitmask =
          safe_to(filter_bitmask, device, true).contiguous();
    } else {
      Timer transfer_timer;
      params.filter_bitmask =
          safe_to(filter_bitmask, device, true).contiguous();
      HISTOGRAM_OBSERVE(
          json_object_mask_transfer_submission_latency_microseconds,
          static_cast<int64_t>(transfer_timer.elapsed_microseconds()));
    }
  }
  params.frequency_penalties = safe_to(frequency_penalties, options, true);
  params.presence_penalties = safe_to(presence_penalties, options, true);
  params.repetition_penalties = safe_to(repetition_penalties, options, true);
  params.temperatures = safe_to(temperatures, options, true);
  params.top_p = safe_to(top_p, options, true);
  params.min_p = safe_to(min_p, options, true);
  params.seed_offsets = safe_to(seed_offsets, device, true);
  params.seeds = safe_to(seeds, device, true);
  params.logits_bias = safe_to(logits_bias, device, true);
  params.top_k = safe_to(top_k, device, true);

  params.unique_token_ids = safe_to(unique_token_ids, device, true);
  params.unique_token_counts = safe_to(unique_token_counts, device, true);
  params.unique_token_ids_lens = safe_to(unique_token_ids_lens, device, true);

  params.do_sample = safe_to(do_sample, device, true);
  params.acc_logprob = safe_to(acc_logprob, device, true);
  params.sampling_mode = sampling_mode;
  params.logprobs = logprobs;
  params.return_logits = return_logits;
  params.max_top_logprobs = max_top_logprobs;
  params.is_embeddings = is_embeddings;
  params.num_return_sequences = num_return_sequences;

  params.use_beam_search = use_beam_search;
  return params;
}

void prepare_draft_sampling(DraftSamplingMode mode,
                            SamplingParameters& sampling_params) {
  sampling_params.logprobs = false;
  sampling_params.max_top_logprobs = 0;
  sampling_params.use_beam_search = false;
  if (mode == DraftSamplingMode::GREEDY) {
    sampling_params.sampling_mode = SamplingMode::GREEDY;
    sampling_params.do_sample = torch::Tensor();
    clear_sampling_transforms(sampling_params);
  }
  // The verifier needs q exactly when the draft actually samples.
  sampling_params.return_logits =
      sampling_params.sampling_mode != SamplingMode::GREEDY;
}

}  // namespace xllm
