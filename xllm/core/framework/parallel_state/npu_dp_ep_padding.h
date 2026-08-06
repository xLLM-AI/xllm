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

#pragma once

#include <torch/torch.h>

#include <vector>

#include "common/macros.h"
#include "framework/parallel_state/mapping_npu.h"
#include "util/json_reader.h"

namespace xllm {
struct DpEpPaddingData {
  void set_placeholder(const torch::Tensor& placeholder);

  PROPERTY(torch::Tensor, attn_padding_idx);

  PROPERTY(torch::Tensor, attn_unpadding_idx);

  PROPERTY(torch::Tensor, ffn_padding_idx);

  PROPERTY(torch::Tensor, ffn_unpadding_idx);

  PROPERTY(torch::Tensor, lm_head_skip_padding_token_indices);

  PROPERTY(torch::Tensor, gather_prenorm_idx);

  PROPERTY(torch::Tensor, padding_idx);

  PROPERTY(torch::Tensor, un_padding_idx);

  PROPERTY(torch::Tensor, dynamic_ep_idx);

  PROPERTY(torch::Tensor, moe_idx);

  PROPERTY(torch::Tensor, expert_array);

  PROPERTY(torch::Tensor, post_lmhead_gather_indices);
};

class DpEpPadding {
 public:
  DpEpPadding(torch::Tensor token_size_per_dp_group,
              torch::Tensor raw_token_size_per_dp_group,
              int32_t num_experts_per_tok,
              const nlohmann::json& mapping_npu,
              at::Device device,
              torch::ScalarType dtype,
              bool is_prefill);

  DpEpPaddingData build();

 private:
  void calculate_max_token_size();
  void prepare_indices();
  void prepare_cumulative_sum();
  torch::Tensor build_ffn_padding_idx();
  torch::Tensor build_ffn_unpadding_idx();
  torch::Tensor build_reduce_scatter_unpadding();
  torch::Tensor build_lm_head_indices();
  torch::Tensor build_all_gather_unpadding();
  std::vector<torch::Tensor> build_dp_padding_indices();
  std::vector<torch::Tensor> build_attn_padding_indices();
  void prepare_gather_indices();
  void handle_expert_parallel();
  float get_all2all_buffer_factor(int length);
  DpEpPaddingData assemble_result() const;

  bool is_prefill_ = true;
  int32_t num_experts_per_tok_;
  torch::Tensor token_size_per_dp_group_;
  const nlohmann::json mapping_npu_;
  int expert_parallel_degree_;
  int64_t rank_;
  int64_t input_ids_len_;
  int64_t max_dp_batch_size_;
  int64_t max_token_size_per_dp_group_;
  torch::Tensor raw_token_size_per_dp_group_;
  torch::Tensor token_size_per_dp_group_startid_;

  torch::Tensor ffn_padding_idx_;
  torch::Tensor lm_head_skip_padding_token_indices_;
  std::vector<torch::Tensor> dp_padding_idx_;
  torch::Tensor attn_padding_idx_;
  torch::Tensor gather_prenorm_idx_;
  torch::Tensor dynamic_ep_idx_ = torch::zeros(1, torch::kInt32);
  torch::Tensor moe_idx_ = torch::zeros(1, torch::kInt32);

  torch::Tensor attn_unpadding_idx_;
  torch::Tensor ffn_unpadding_idx_;
  torch::Tensor padding_idx_;
  torch::Tensor un_padding_idx_;
  torch::Tensor post_lmhead_gather_indices_;
  torch::Tensor all_gather_padding_;
  torch::Tensor expert_array_;
  std::vector<int64_t> new_dp_size_;
  at::Device device_;
  torch::ScalarType dtype_;
};
}  // namespace xllm
