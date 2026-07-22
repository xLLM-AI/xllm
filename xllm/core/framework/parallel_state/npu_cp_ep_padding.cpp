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

#include "framework/parallel_state/npu_cp_ep_padding.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "common/global_flags.h"
#include "core/framework/config/eplb_config.h"
#include "util/tensor_helper.h"

namespace xllm {

void CpEpPaddingData::set_placeholder(const torch::Tensor& placeholder) {
  attn_padding_idx_ = placeholder;
  attn_unpadding_idx_ = placeholder;
  ffn_padding_idx_ = placeholder;
  ffn_unpadding_idx_ = placeholder;
  lm_head_skip_padding_token_indices_ = placeholder;
  gather_prenorm_idx_ = placeholder;
  padding_idx_ = placeholder;
  un_padding_idx_ = placeholder;
  dynamic_ep_idx_ = placeholder;
  moe_idx_ = placeholder;
  expert_array_ = placeholder;
}

CpEpPadding::CpEpPadding(int64_t local_padded_token_num,
                         int32_t num_experts_per_tok,
                         const nlohmann::json& mapping_npu,
                         at::Device device,
                         torch::ScalarType dtype,
                         bool is_prefill)
    : num_experts_per_tok_(num_experts_per_tok),
      mapping_npu_(mapping_npu),
      device_(device),
      dtype_(dtype) {
  attn_tp_size_ = mapping_npu["attnTpSize"].get<int64_t>();
  attn_tp_rank_ = mapping_npu["attnTp"]["rank"].get<int64_t>();
  attn_cp_size_ = mapping_npu["attnCp"]["rankIds"].size();
  input_length_ = std::max<int64_t>(local_padded_token_num, 1);

  const bool has_moe_ep = mapping_npu_.contains("moeEpSize") &&
                          mapping_npu_["moeEpSize"].get<int64_t>() > 1;
  const int32_t expert_parallel_degree =
      ::xllm::EPLBConfig::get_instance().expert_parallel_degree();
  is_dynamic_ep_ = has_moe_ep && (expert_parallel_degree == 2 ||
                                  (expert_parallel_degree == 3 && is_prefill));
}

CpEpPaddingData CpEpPadding::build() {
  prepare_indices();
  handle_expert_parallel();
  return assemble_result();
}

void CpEpPadding::prepare_indices() {
  CHECK_GT(attn_tp_size_, 0) << "attnTpSize must be positive.";
  CHECK_GT(attn_cp_size_, 0) << "attnCp group size must be positive.";
  CHECK_LT(attn_tp_rank_, attn_tp_size_)
      << "attnTp rank must be smaller than attnTpSize.";

  const int64_t padding_length =
      (attn_tp_size_ - input_length_ % attn_tp_size_) % attn_tp_size_;
  const int64_t input_len_padding_per_group = input_length_ + padding_length;
  const int64_t input_len_padding_per_rank =
      input_len_padding_per_group / attn_tp_size_;

  attn_padding_idx_ =
      torch::cat({torch::arange(input_length_, torch::kInt32),
                  torch::zeros({padding_length}, torch::kInt32)});

  gather_prenorm_idx_ =
      attn_padding_idx_.slice(0,
                              attn_tp_rank_ * input_len_padding_per_rank,
                              (attn_tp_rank_ + 1) * input_len_padding_per_rank);

  std::vector<torch::Tensor> all_gather_skip_components;
  all_gather_skip_components.reserve(attn_cp_size_);
  for (int64_t cp_rank = 0; cp_rank < attn_cp_size_; ++cp_rank) {
    all_gather_skip_components.emplace_back(
        torch::arange(input_length_, torch::kInt32) +
        cp_rank * input_len_padding_per_group);
  }
  const auto all_gather_skip_padding_token_indices =
      torch::cat(all_gather_skip_components, 0);

  if (is_dynamic_ep_) {
    attn_unpadding_idx_ =
        torch::arange(input_len_padding_per_rank, torch::kInt32);
    ffn_padding_idx_ = attn_unpadding_idx_;
  } else {
    attn_unpadding_idx_ = all_gather_skip_padding_token_indices;

    std::vector<torch::Tensor> ffn_padding_components;
    ffn_padding_components.reserve(attn_cp_size_);
    for (int64_t cp_rank = 0; cp_rank < attn_cp_size_; ++cp_rank) {
      ffn_padding_components.emplace_back(
          torch::cat({torch::arange(input_length_ * cp_rank,
                                    input_length_ * (cp_rank + 1),
                                    torch::kInt32),
                      torch::zeros({padding_length}, torch::kInt32)}));
    }
    ffn_padding_idx_ = torch::cat(ffn_padding_components, 0);
  }

  padding_idx_ = attn_padding_idx_;
  un_padding_idx_ = torch::zeros({1}, torch::kInt32);
  ffn_unpadding_idx_ = torch::arange(input_length_, torch::kInt32);
  lm_head_skip_padding_token_indices_ = all_gather_skip_padding_token_indices;
}

void CpEpPadding::handle_expert_parallel() {
  if (!is_dynamic_ep_) {
    dynamic_ep_idx_ = torch::zeros({1}, torch::kInt32);
    moe_idx_ = torch::zeros({1}, torch::kInt32);
    expert_array_ = torch::tensor({0});
    return;
  }

  torch::Tensor dynamic_ep_idx_padding;
  if (attn_tp_size_ == 1) {
    dynamic_ep_idx_ =
        torch::arange(input_length_ * num_experts_per_tok_, torch::kInt32);
    dynamic_ep_idx_padding = torch::arange(
        attn_unpadding_idx_.size(0) * num_experts_per_tok_, torch::kInt32);
  } else {
    dynamic_ep_idx_ = torch::arange(
        attn_unpadding_idx_.size(0) * num_experts_per_tok_, torch::kInt32);
    dynamic_ep_idx_padding = dynamic_ep_idx_;
  }

  const int64_t base_length = dynamic_ep_idx_padding.size(0);
  const float buffer_factor =
      get_all2all_buffer_factor(static_cast<int>(base_length));
  int ep_input_length = static_cast<int>(base_length * buffer_factor);

  int64_t group_size = mapping_npu_["moeEpSize"].get<int64_t>();
  const int all2all_padding = ep_input_length % group_size;
  const int padding =
      (all2all_padding != 0) ? (group_size - all2all_padding) : 0;

  int ep_input_length_padding = ep_input_length + padding;
  std::vector<int32_t> moe_idx_data;
  moe_idx_data.reserve(ep_input_length_padding);
  for (int i = 1; i <= ep_input_length_padding; ++i) {
    moe_idx_data.push_back(i);
  }
  moe_idx_ = torch::tensor(moe_idx_data, torch::dtype(torch::kInt32));
  expert_array_ = safe_to(
      torch::ones({moe_idx_.sizes()[0]}, dtype_).view({-1, 1}), device_, true);
}

float CpEpPadding::get_all2all_buffer_factor(int length) const {
  float all2all_buffer_factor =
      static_cast<float>(mapping_npu_["moeEpSize"].get<int64_t>());
  length *= mapping_npu_["attnCpSize"].get<int>();

  const std::vector<std::pair<int, float>> length_thresholds = {
      {1048576, 1.32f},
      {524288, 1.4f},
      {262144, 1.53f},
      {131072, 1.8f},
      {32768, 3.0f},
      {8192, 5.2f},
      {0, 8.0f}};
  for (const auto& threshold : length_thresholds) {
    if (length >= threshold.first) {
      all2all_buffer_factor = threshold.second;
      break;
    }
  }
  return all2all_buffer_factor;
}

CpEpPaddingData CpEpPadding::assemble_result() const {
  CpEpPaddingData result;
  result.attn_padding_idx(safe_to(attn_padding_idx_, device_, true))
      .attn_unpadding_idx(safe_to(attn_unpadding_idx_, device_, true))
      .ffn_padding_idx(safe_to(ffn_padding_idx_, device_, true))
      .ffn_unpadding_idx(safe_to(ffn_unpadding_idx_, device_, true))
      .lm_head_skip_padding_token_indices(
          safe_to(lm_head_skip_padding_token_indices_, device_, true))
      .gather_prenorm_idx(safe_to(gather_prenorm_idx_, device_, true))
      .padding_idx(safe_to(padding_idx_, device_, true))
      .un_padding_idx(safe_to(un_padding_idx_, device_, true))
      .dynamic_ep_idx(safe_to(dynamic_ep_idx_, device_, true))
      .moe_idx(safe_to(moe_idx_, device_, true))
      .expert_array(safe_to(expert_array_, device_, true));
  return result;
}

}  // namespace xllm
