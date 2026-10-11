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
#include <array>
#include <cstdint>
#include <vector>

#include "core/common/metrics.h"
#include "core/util/tensor_helper.h"

namespace xllm {

namespace {

torch::Tensor concat_rows(const torch::Tensor& left,
                          const torch::Tensor& right,
                          int64_t left_rows,
                          int64_t right_rows,
                          double fill) {
  if (!left.defined() && !right.defined()) {
    return {};
  }
  const torch::Tensor& reference = left.defined() ? left : right;
  if (reference.dim() == 1) {
    return torch::cat(
        {left.defined() ? left
                        : torch::full({left_rows}, fill, reference.options()),
         right.defined()
             ? right
             : torch::full({right_rows}, fill, reference.options())},
        0);
  }
  const int64_t width = std::max(left.defined() ? left.size(1) : 0,
                                 right.defined() ? right.size(1) : 0);
  torch::Tensor joined =
      torch::full({left_rows + right_rows, width}, fill, reference.options());
  if (left.defined()) {
    joined.narrow(0, 0, left_rows).narrow(1, 0, left.size(1)).copy_(left);
  }
  if (right.defined()) {
    joined.narrow(0, left_rows, right_rows)
        .narrow(1, 0, right.size(1))
        .copy_(right);
  }
  return joined;
}

}  // namespace

void SamplingParameters::init(
    const std::vector<const RequestSamplingParam*>& req_sampling_params,
    const std::vector<int32_t>& selected_token_idxes,
    const std::vector<int32_t>& sample_idxes,
    const std::vector<std::vector<int64_t>>& unique_token_ids_vec,
    const std::vector<std::vector<int32_t>>& unique_token_counts_vec,
    const std::vector<int32_t>& unique_token_lens_vec,
    const std::vector<torch::Tensor>& filter_mask_rows) {
  *this = SamplingParameters();
  CHECK_EQ(req_sampling_params.size(), selected_token_idxes.size());
  CHECK_GE(req_sampling_params.size(), sample_idxes.size());

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
    if (sample_idxes.size() != filter_mask_rows.size()) {
      this->filter_mask = this->filter_mask.index_select(
          0, torch::tensor(sample_idxes, torch::kLong));
    }
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

  // construct do sample tensor
  std::vector<bool> do_sample;
  do_sample.reserve(sample_idxes.size());
  for (const auto idx : sample_idxes) {
    const auto* p = req_sampling_params[idx];
    const bool sample = p->temperature != 0.0;
    do_sample.push_back(sample);
  }
  this->sample_idxes = make_pinned_cpu_tensor(sample_idxes);
  // The rejection sampler's torch::where requires a bool condition tensor.
  this->do_sample = make_pinned_cpu_tensor(do_sample);
  this->logprobs = logprobs;
  this->max_top_logprobs = max_top_logprobs;
  this->is_embeddings = is_embeddings;
  this->num_return_sequences = num_return_sequences;
  if (this->do_sample.defined()) {
    this->all_random_sample = this->do_sample.all().item<bool>();
    this->all_greedy_sample = !this->do_sample.any().item<bool>();
  }
}

SamplingParameters SamplingParameters::to(const torch::Device& device,
                                          torch::ScalarType dtype) const {
  SamplingParameters params;

  // selected/sample indices are tiny control tensors and
  // correctness-critical. Use blocking H2D copies to avoid consuming
  // partially transferred index buffers on NPU runtime paths.
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

  params.sample_idxes = sample_idxes.defined()
                            ? safe_to(sample_idxes, device).contiguous()
                            : sample_idxes;
  params.do_sample = safe_to(do_sample, device, true);
  params.acc_logprob = safe_to(acc_logprob, device, true);
  params.all_random_sample = all_random_sample;
  params.all_greedy_sample = all_greedy_sample;
  params.logprobs = logprobs;
  params.return_probs = return_probs;
  params.max_top_logprobs = max_top_logprobs;
  params.is_embeddings = is_embeddings;
  params.num_return_sequences = num_return_sequences;

  params.use_beam_search = use_beam_search;
  return params;
}

void SamplingParameters::concat(const SamplingParameters& param) {
  if (!param.selected_token_idxes.defined() ||
      param.selected_token_idxes.numel() == 0) {
    return;
  }
  if (!selected_token_idxes.defined() || selected_token_idxes.numel() == 0) {
    *this = param;
    return;
  }
  const int64_t left_rows = selected_token_idxes.numel();
  const int64_t right_rows = param.selected_token_idxes.numel();
  const int64_t left_samples = sample_idxes.numel();
  const int64_t right_samples = param.sample_idxes.numel();
  for (const auto& [member, fill] :
       std::array<std::pair<torch::Tensor SamplingParameters::*, double>, 14>{
           {{&SamplingParameters::frequency_penalties, 0},
            {&SamplingParameters::presence_penalties, 0},
            {&SamplingParameters::repetition_penalties, 1},
            {&SamplingParameters::temperatures, 1},
            {&SamplingParameters::top_p, 1},
            {&SamplingParameters::top_k, 0},
            {&SamplingParameters::min_p, 0},
            {&SamplingParameters::logits_bias, 0},
            {&SamplingParameters::seeds, -1},
            {&SamplingParameters::seed_offsets, 0},
            {&SamplingParameters::unique_token_ids, 0},
            {&SamplingParameters::unique_token_counts, 0},
            {&SamplingParameters::unique_token_ids_lens, 0},
            {&SamplingParameters::filter_bitmask, -1}}}) {
    this->*member =
        concat_rows(this->*member, param.*member, left_rows, right_rows, fill);
  }
  filter_mask = concat_rows(
      filter_mask, param.filter_mask, left_samples, right_samples, 0);
  acc_logprob = concat_rows(
      acc_logprob, param.acc_logprob, left_samples, right_samples, 0);
  selected_token_idxes =
      torch::cat({selected_token_idxes,
                  param.selected_token_idxes + selected_token_idxes[-1] + 1},
                 0);
  sample_idxes = torch::cat({sample_idxes, param.sample_idxes + left_rows}, 0);
  do_sample = torch::cat({do_sample, param.do_sample}, 0);
  all_random_sample = all_random_sample && param.all_random_sample;
  all_greedy_sample = all_greedy_sample && param.all_greedy_sample;
  logprobs = logprobs || param.logprobs;
  return_probs = return_probs || param.return_probs;
  is_embeddings = is_embeddings || param.is_embeddings;
  use_beam_search = use_beam_search || param.use_beam_search;
  max_top_logprobs = std::max(max_top_logprobs, param.max_top_logprobs);
  num_return_sequences =
      std::max(num_return_sequences, param.num_return_sequences);
}

}  // namespace xllm
