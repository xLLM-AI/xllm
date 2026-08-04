/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "sampling_params.h"

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "core/common/metrics.h"
#include "core/util/tensor_helper.h"

namespace xllm {

void SamplingParameters::init(
    const std::vector<const RequestSamplingParam*>& req_sampling_params,
    const std::vector<int32_t>& selected_token_idxes,
    const std::vector<int32_t>& sample_idxes,
    const std::vector<std::vector<int64_t>>& unique_token_ids_vec,
    const std::vector<std::vector<int32_t>>& unique_token_counts_vec,
    const std::vector<int32_t>& unique_token_lens_vec,
    const std::vector<torch::Tensor>& filter_mask_rows) {
  CHECK_EQ(req_sampling_params.size(), selected_token_idxes.size());
  CHECK_GE(req_sampling_params.size(), sample_idxes.size());

  std::vector<float> frequency_penalties;
  std::vector<float> presence_penalties;
  std::vector<float> repetition_penalties;
  std::vector<float> temperatures;
  std::vector<float> top_p;
  std::vector<int64_t> top_k;
  frequency_penalties.reserve(req_sampling_params.size());
  presence_penalties.reserve(req_sampling_params.size());
  repetition_penalties.reserve(req_sampling_params.size());
  temperatures.reserve(req_sampling_params.size());
  top_p.reserve(req_sampling_params.size());
  top_k.reserve(req_sampling_params.size());
  bool logprobs = false;
  int64_t max_top_logprobs = 0;
  bool is_embeddings = false;
  int32_t num_return_sequences = 0;
  for (const auto* p : req_sampling_params) {
    frequency_penalties.push_back(p->frequency_penalty);
    presence_penalties.push_back(p->presence_penalty);
    repetition_penalties.push_back(p->repetition_penalty);
    temperatures.push_back(p->temperature);
    top_p.push_back(p->top_p);
    top_k.push_back(p->top_k);
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

  // Create tensor on cpu pinned memory here
  auto int_tensor_options = torch::TensorOptions()
                                .device(torch::kCPU)
                                .dtype(torch::kInt)
                                .pinned_memory(true);
  auto int64_tensor_options = torch::TensorOptions()
                                  .device(torch::kCPU)
                                  .dtype(torch::kInt64)
                                  .pinned_memory(true);
  auto float32_tensor_options = torch::TensorOptions()
                                    .device(torch::kCPU)
                                    .dtype(torch::kFloat32)
                                    .pinned_memory(true);
  auto bool_tensor_options = torch::TensorOptions()
                                 .device(torch::kCPU)
                                 .dtype(torch::kBool)
                                 .pinned_memory(true);
  if (std::any_of(frequency_penalties.begin(),
                  frequency_penalties.end(),
                  [](float t) { return t != 0.0; }) ||
      std::any_of(presence_penalties.begin(),
                  presence_penalties.end(),
                  [](float t) { return t != 0.0; })) {
    this->frequency_penalties =
        torch::tensor(frequency_penalties, float32_tensor_options);
    this->presence_penalties =
        torch::tensor(presence_penalties, float32_tensor_options);
    need_token_stats = true;
  }
  if (std::any_of(repetition_penalties.begin(),
                  repetition_penalties.end(),
                  [](float t) { return t != 1.0; })) {
    this->repetition_penalties =
        torch::tensor(repetition_penalties, float32_tensor_options);
    need_token_stats = true;
  }
  if (std::any_of(temperatures.begin(), temperatures.end(), [](float t) {
        return t != 0.0 && t != 1.0;
      })) {
    this->temperatures = torch::tensor(temperatures, float32_tensor_options);
  }
  if (std::any_of(
          top_k.begin(), top_k.end(), [](int64_t t) { return t > 0; })) {
    this->top_k = torch::tensor(top_k, int64_tensor_options);
  }
  if (std::any_of(
          top_p.begin(), top_p.end(), [](float t) { return t != 1.0; })) {
    this->top_p = torch::tensor(top_p, float32_tensor_options);
  }

  this->selected_token_idxes =
      torch::tensor(selected_token_idxes, int_tensor_options);
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
    this->unique_token_ids_lens =
        torch::tensor(unique_token_lens_vec, int_tensor_options);
  }

  // construct do sample tensor
  std::vector<int32_t> do_sample;
  do_sample.reserve(sample_idxes.size());
  for (const auto idx : sample_idxes) {
    const auto* p = req_sampling_params[idx];
    // need to do sample if any of following is true
    const bool sample = p->do_sample || p->temperature != 0.0 ||
                        p->top_p != 1.0 || p->top_k > 0;
    do_sample.push_back(sample ? 1 : 0);
  }
  this->sample_idxes = torch::tensor(sample_idxes, int_tensor_options);
  this->do_sample = torch::tensor(do_sample, bool_tensor_options);
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
  // selected_token_idxes and sample_idxes are accumulated variable across
  // all sequences in the batch, so the offset of first
  // SamplingParameters is added to the second SamplingParameters
  this->selected_token_idxes =
      safe_concat(this->selected_token_idxes,
                  (param.selected_token_idxes.defined()
                       ? (param.selected_token_idxes +
                          this->selected_token_idxes[-1] + torch::tensor(1))
                       : param.selected_token_idxes),
                  0);
  this->sample_idxes = safe_concat(
      this->sample_idxes,
      (param.sample_idxes.defined()
           ? (param.sample_idxes + this->sample_idxes[-1] + torch::tensor(1))
           : param.sample_idxes),
      0);
  this->frequency_penalties =
      safe_concat(this->frequency_penalties, param.frequency_penalties, 0);
  this->repetition_penalties =
      safe_concat(this->repetition_penalties, param.repetition_penalties, 0);
  this->temperatures = safe_concat(this->temperatures, param.temperatures, 0);
  this->top_p = safe_concat(this->top_p, param.top_p, 0);
  this->top_k = safe_concat(this->top_k, param.top_k, 0);
  this->unique_token_ids =
      safe_concat(this->unique_token_ids, param.unique_token_ids, 0);
  this->unique_token_counts =
      safe_concat(this->unique_token_counts, param.unique_token_counts, 0);
  this->unique_token_ids_lens =
      safe_concat(this->unique_token_ids_lens, param.unique_token_ids_lens, 0);
  this->do_sample = safe_concat(this->do_sample, param.do_sample, 0);
  this->acc_logprob = safe_concat(this->acc_logprob, param.acc_logprob, 0);
  if (this->filter_mask.defined() && param.filter_mask.defined()) {
    this->filter_mask = torch::cat({this->filter_mask, param.filter_mask}, 0);
  } else if (this->filter_mask.defined() || param.filter_mask.defined()) {
    const auto row_count = [](const SamplingParameters& value) {
      if (value.filter_mask.defined()) {
        return value.filter_mask.size(0);
      }
      return value.sample_idxes.defined() ? value.sample_idxes.numel() : 0;
    };
    const torch::Tensor& defined_mask =
        this->filter_mask.defined() ? this->filter_mask : param.filter_mask;
    const int64_t missing_rows =
        this->filter_mask.defined() ? row_count(param) : row_count(*this);
    torch::Tensor unconstrained_rows = torch::zeros(
        {missing_rows, defined_mask.size(1)}, defined_mask.options());
    this->filter_mask =
        this->filter_mask.defined()
            ? torch::cat({this->filter_mask, unconstrained_rows}, 0)
            : torch::cat({unconstrained_rows, param.filter_mask}, 0);
  }
  if (this->filter_bitmask.defined() && param.filter_bitmask.defined()) {
    this->filter_bitmask =
        torch::cat({this->filter_bitmask, param.filter_bitmask}, 0);
  } else if (this->filter_bitmask.defined() || param.filter_bitmask.defined()) {
    const auto row_count = [](const SamplingParameters& value) {
      if (value.filter_bitmask.defined()) {
        return value.filter_bitmask.size(0);
      }
      return value.sample_idxes.defined() ? value.sample_idxes.numel() : 0;
    };
    const torch::Tensor& defined_mask = this->filter_bitmask.defined()
                                            ? this->filter_bitmask
                                            : param.filter_bitmask;
    const int64_t missing_rows =
        this->filter_bitmask.defined() ? row_count(param) : row_count(*this);
    // All-ones words => allow all tokens (unconstrained).
    torch::Tensor unconstrained_rows =
        torch::full({missing_rows, defined_mask.size(1)},
                    /*fill_value=*/static_cast<int32_t>(-1),
                    defined_mask.options());
    this->filter_bitmask =
        this->filter_bitmask.defined()
            ? torch::cat({this->filter_bitmask, unconstrained_rows}, 0)
            : torch::cat({unconstrained_rows, param.filter_bitmask}, 0);
  }
  this->logprobs = this->logprobs || param.logprobs;
  this->return_probs = this->return_probs || param.return_probs;
  this->is_embeddings = this->is_embeddings || param.is_embeddings;
  this->use_beam_search = this->use_beam_search || param.use_beam_search;
  this->max_top_logprobs =
      std::max(this->max_top_logprobs, param.max_top_logprobs);
  this->num_return_sequences =
      std::max(this->num_return_sequences, param.num_return_sequences);
  return;
}

}  // namespace xllm
