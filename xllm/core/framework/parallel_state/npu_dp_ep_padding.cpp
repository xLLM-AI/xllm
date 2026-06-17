/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "framework/parallel_state/npu_dp_ep_padding.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/global_flags.h"
#include "core/framework/config/eplb_config.h"
#include "util/tensor_helper.h"

namespace xllm {

void DpEpPaddingData::set_placeholder(const torch::Tensor& placeholder) {
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

DpEpPadding::DpEpPadding(torch::Tensor token_size_per_dp_group,
                         int32_t num_experts_per_tok,
                         const nlohmann::json& mapping_npu,
                         at::Device device,
                         torch::ScalarType dtype,
                         bool is_prefill)
    : token_size_per_dp_group_(token_size_per_dp_group.contiguous()),
      num_experts_per_tok_(num_experts_per_tok),
      mapping_npu_(mapping_npu),
      device_(device),
      dtype_(dtype),
      is_prefill_(is_prefill),
      expert_parallel_degree_(0) {
  // Validate input tensor
  if (token_size_per_dp_group_.dim() != 1) {
    LOG(FATAL)
        << "token_size_per_dp_group must be 1-dimensional, current dim is "
        << token_size_per_dp_group_.dim();
  }
  token_size_per_dp_group_ = torch::where(token_size_per_dp_group_ == 0,
                                          torch::tensor(1).to(torch::kInt32),
                                          token_size_per_dp_group)
                                 .to(torch::kInt32);

  int64_t attn_tp_rank = mapping_npu_["attnTp"]["rank"].get<int64_t>();
  int64_t attn_dp_rank = mapping_npu_["attnDp"]["rank"].get<int64_t>();
  int64_t attn_tp_size = mapping_npu_["attnTpSize"].get<int64_t>();

  if (attn_tp_rank >= attn_tp_size || attn_tp_rank < 0) {
    LOG(FATAL) << "Invalid attnTp rank, attn_tp_rank = " << attn_tp_rank
               << ", attn_tp_size = " << attn_tp_size;
  }

  rank_ = attn_tp_rank + attn_dp_rank * attn_tp_size;

  // Set expert parallel degree
  if (mapping_npu_.contains("moeEpSize") &&
      mapping_npu_["moeEpSize"].get<int64_t>() > 1) {
    expert_parallel_degree_ =
        ::xllm::EPLBConfig::get_instance().expert_parallel_degree();
  }
  input_ids_len_ = token_size_per_dp_group_[attn_dp_rank].item<int64_t>();
  max_dp_batch_size_ = token_size_per_dp_group_.max().item<int64_t>();

  attn_padding_idx_ = torch::tensor({0});
  attn_unpadding_idx_ = torch::tensor({0});
  ffn_padding_idx_ = torch::tensor({0});
  ffn_unpadding_idx_ = torch::tensor({0});
  lm_head_skip_padding_token_indices_ = torch::tensor({0});
  gather_prenorm_idx_ = torch::tensor({0});
  padding_idx_ = torch::tensor({0});
  un_padding_idx_ = torch::tensor({0});
  dynamic_ep_idx_ = torch::tensor({0});
  moe_idx_ = torch::tensor({0});
  expert_array_ = torch::tensor({0});
}

DpEpPaddingData DpEpPadding::build() {
  calculate_max_token_size();
  prepare_indices();
  handle_expert_parallel();
  return assemble_result();
}

void DpEpPadding::calculate_max_token_size() {
  max_token_size_per_dp_group_ = token_size_per_dp_group_.max().item<int64_t>();
  const auto remainder =
      max_token_size_per_dp_group_ % mapping_npu_["attnTpSize"].get<int64_t>();
  if (remainder != 0) {
    max_token_size_per_dp_group_ +=
        mapping_npu_["attnTpSize"].get<int64_t>() - remainder;
  }
}

void DpEpPadding::prepare_indices() {
  prepare_cumulative_sum();

  ffn_padding_idx_ = build_ffn_padding_idx();

  lm_head_skip_padding_token_indices_ = build_lm_head_indices();

  dp_padding_idx_ = build_dp_padding_indices();
  // padding_idx_ =
  // dp_padding_idx_[mapping_npu_["attnDp"]["rank"].get<int64_t>()];
  padding_idx_ = dp_padding_idx_[mapping_npu_["attnDpSize"].get<int64_t>() - 1];
  attn_padding_idx_ =
      dp_padding_idx_[mapping_npu_["attnDp"]["rank"].get<int64_t>()];
  un_padding_idx_ = torch::zeros({1}, torch::kInt32);

  prepare_gather_indices();
  new_dp_size_.clear();

  const int64_t max_token_size = token_size_per_dp_group_.max().item<int64_t>();
  const int64_t tp_group_size = mapping_npu_["attnTpSize"].get<int64_t>();

  for (int64_t i = 0;
       i < static_cast<int64_t>(token_size_per_dp_group_.size(0));
       ++i) {
    const int64_t input_length = token_size_per_dp_group_[i].item<int64_t>();
    int64_t input_length_padding = max_token_size - input_length;
    const int64_t reduce_scatter_padding = max_token_size % tp_group_size;

    if (reduce_scatter_padding != 0) {
      input_length_padding += tp_group_size - reduce_scatter_padding;
    }

    const int64_t padded_len = input_length + input_length_padding;
    for (int64_t j = 0; j < tp_group_size; ++j) {
      new_dp_size_.push_back(padded_len / tp_group_size);
    }
  }

  if (expert_parallel_degree_ == 2 ||
      (expert_parallel_degree_ == 3 && is_prefill_)) {
    attn_unpadding_idx_ = torch::arange(new_dp_size_[rank_], torch::kInt32);
    ffn_padding_idx_ = attn_unpadding_idx_;
  } else {
    // Build all_gather_unpadding tensor
    torch::Tensor all_gather_unpadding = build_all_gather_unpadding();
    torch::Tensor reduce_scatter_unpadding = build_reduce_scatter_unpadding();
    attn_unpadding_idx_ =
        all_gather_unpadding.index({reduce_scatter_unpadding}).reshape({-1});
  }
  ffn_unpadding_idx_ = build_ffn_unpadding_idx();
}

std::vector<torch::Tensor> DpEpPadding::build_attn_padding_indices() {
  std::vector<torch::Tensor> indices;
  indices.reserve(token_size_per_dp_group_.size(0));
  const int64_t tp_size = mapping_npu_["attnTpSize"].get<int64_t>();
  const int64_t max_token_size = max_token_size_per_dp_group_;

  for (int64_t i = 0; i < token_size_per_dp_group_.size(0); ++i) {
    const auto input_length = token_size_per_dp_group_[i].item<int64_t>();
    const int64_t reduce_scatter_padding = max_token_size % tp_size;
    int64_t input_length_padding = max_token_size - input_length;

    if (reduce_scatter_padding != 0) {
      input_length_padding += tp_size - reduce_scatter_padding;
    }

    auto valid = torch::arange(input_length, torch::kInt32);

    auto padding = torch::zeros(input_length_padding, torch::kInt32);

    indices.push_back(torch::cat({valid, padding}));
  }
  return indices;
}

std::vector<torch::Tensor> DpEpPadding::build_dp_padding_indices() {
  std::vector<torch::Tensor> indices;
  indices.reserve(token_size_per_dp_group_.size(0));
  const int64_t max_token_size = max_token_size_per_dp_group_;

  for (int64_t i = 0; i < token_size_per_dp_group_.size(0); ++i) {
    int32_t current_size = token_size_per_dp_group_[i].item<int32_t>();
    indices.push_back(torch::cat(
        {torch::arange(current_size, torch::kInt32),
         torch::zeros((max_token_size - current_size), torch::kInt32)}));
  }
  return indices;
}

torch::Tensor DpEpPadding::build_all_gather_unpadding() {
  const int64_t tp_group_size = mapping_npu_["attnTpSize"].get<int64_t>();
  const int64_t max_dp_size =
      *std::max_element(new_dp_size_.begin(), new_dp_size_.end());
  auto actual_indices = torch::arange(new_dp_size_[rank_], torch::kInt32);
  const int64_t padding_length = max_dp_size - new_dp_size_[rank_];
  auto padding_indices = torch::zeros(padding_length, torch::kInt32);

  const int64_t tp_rank = mapping_npu_["attnTp"]["rank"].get<int64_t>();
  const int64_t token_size_per_tp_rank =
      max_token_size_per_dp_group_ / tp_group_size;
  const int64_t start_idx = tp_rank * token_size_per_tp_rank;
  const int64_t end_idx = (tp_rank + 1) * token_size_per_tp_rank;
  auto gather_prenorm_idx = attn_padding_idx_.slice(0, start_idx, end_idx);
  auto all_gather_padding = torch::cat({actual_indices, padding_indices});

  std::vector<torch::Tensor> components;
  components.reserve(new_dp_size_.size());
  for (int64_t i = 0; i < new_dp_size_.size(); ++i) {
    auto partial = torch::arange(new_dp_size_[i], torch::kInt32) +
                   i * all_gather_padding.size(0);
    components.push_back(partial);
  }
  return torch::cat(components);
}

torch::Tensor DpEpPadding::build_reduce_scatter_unpadding() {
  const int64_t tp_group_size = mapping_npu_["attnTpSize"].get<int64_t>();

  std::vector<torch::Tensor> components;
  components.reserve(token_size_per_dp_group_.size(0));
  for (int64_t i = 0; i < token_size_per_dp_group_.size(0); ++i) {
    int64_t offset = 0;
    for (int64_t j = 0; j < (i * tp_group_size) && j < new_dp_size_.size();
         ++j) {
      offset += new_dp_size_[j];
    }
    auto partial = torch::arange(token_size_per_dp_group_[i].item<int64_t>(),
                                 torch::kInt32) +
                   offset;
    components.push_back(partial);
  }
  return torch::cat(components);
}

torch::Tensor DpEpPadding::build_ffn_unpadding_idx() {
  const auto& rank = mapping_npu_["attnDp"]["rank"].get<int64_t>();
  int64_t token_size = token_size_per_dp_group_[rank].item<int64_t>();
  token_size = (token_size == 0) ? 1 : token_size;  // Ensure at least 1 element

  // Add proper offset based on DP rank
  int64_t offset = 0;
  for (int64_t i = 0; i < rank; ++i) {
    offset += token_size_per_dp_group_[i].item<int64_t>();
  }

  return torch::arange(token_size, torch::kInt32);
}

void DpEpPadding::prepare_cumulative_sum() {
  token_size_per_dp_group_startid_ = torch::cumsum(token_size_per_dp_group_, 0);
  if (token_size_per_dp_group_startid_.size(0) > 0) {
    token_size_per_dp_group_startid_[-1] = 0;
  }
}

torch::Tensor DpEpPadding::build_ffn_padding_idx() {
  const int64_t dp_group_count = token_size_per_dp_group_.size(0);

  std::vector<torch::Tensor> components;
  components.reserve(dp_group_count);
  for (int64_t dp_group_id = 0; dp_group_id < dp_group_count; ++dp_group_id) {
    const int64_t token_size =
        token_size_per_dp_group_[dp_group_id].item<int64_t>();

    int64_t start = 0;
    if (dp_group_id > 0) {
      start = (dp_group_id == dp_group_count - 1)
                  ? token_size_per_dp_group_startid_[dp_group_id - 1]
                        .item<int64_t>()
                  : token_size_per_dp_group_startid_[dp_group_id - 1]
                        .item<int64_t>();
    }

    torch::Tensor valid;
    if (token_size > 0) {
      valid = torch::arange(token_size, torch::kInt32) + start;
    } else {
      valid = torch::tensor({}, torch::kInt32);
    }

    const int64_t padding_size = max_token_size_per_dp_group_ - token_size;
    torch::Tensor padding = torch::zeros(padding_size, torch::kInt32);
    if (padding_size > 0 && token_size == 0) {
      padding[0] = start;
    }

    components.push_back(torch::cat({valid, padding}));
  }

  return torch::cat(components);
}

torch::Tensor DpEpPadding::build_lm_head_indices() {
  std::vector<torch::Tensor> components;
  components.reserve(token_size_per_dp_group_.size(0));
  for (int64_t rank_id = 0; rank_id < token_size_per_dp_group_.size(0);
       ++rank_id) {
    const auto j = token_size_per_dp_group_[rank_id].item<int64_t>();
    const int64_t offset = rank_id * max_token_size_per_dp_group_;
    components.push_back(torch::arange(j, torch::kInt32) + offset);
  }
  return torch::cat(components);
}

void DpEpPadding::prepare_gather_indices() {
  const int64_t tp_rank = mapping_npu_["attnTp"]["rank"].get<int64_t>();
  const int64_t token_size_per_tp_rank =
      max_token_size_per_dp_group_ / mapping_npu_["attnTpSize"].get<int64_t>();
  const int64_t start = tp_rank * token_size_per_tp_rank;
  const int64_t end = start + token_size_per_tp_rank;

  const int64_t safe_end = std::min(end, max_token_size_per_dp_group_);

  if (start < safe_end) {
    gather_prenorm_idx_ = attn_padding_idx_.slice(0, start, safe_end);
  } else {
    gather_prenorm_idx_ = torch::zeros({0}, torch::kInt32);
  }
}

void DpEpPadding::handle_expert_parallel() {
  torch::Tensor dynamic_ep_idx_padding;
  if (expert_parallel_degree_ == 2 ||
      (expert_parallel_degree_ == 3 && is_prefill_)) {
    if (mapping_npu_["attnTpSize"] == 1) {
      dynamic_ep_idx_ =
          torch::arange(input_ids_len_ * num_experts_per_tok_, torch::kInt32);
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
    expert_array_ =
        safe_to(torch::ones({moe_idx_.sizes()[0]}, dtype_).view({-1, 1}),
                device_,
                true);
  } else {
    dynamic_ep_idx_ = torch::zeros({1}, torch::kInt32);
    moe_idx_ = torch::zeros({1}, torch::kInt32);
  }
}

float DpEpPadding::get_all2all_buffer_factor(int length) {
  float all2all_buffer_factor = mapping_npu_["moeEpSize"].get<float>();
  length *= mapping_npu_["attnDpSize"].get<int>();

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

DpEpPaddingData DpEpPadding::assemble_result() const {
  DpEpPaddingData result;
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
