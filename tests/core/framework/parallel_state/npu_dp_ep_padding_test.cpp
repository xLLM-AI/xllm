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

#include "framework/parallel_state/npu_dp_ep_padding.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "framework/parallel_state/mapping_npu.h"

namespace xllm {

MappingNPU::Options get_mapping_options() {
  MappingNPU::Options options;
  options.dp_size(2)
      .tp_size(8)
      .moe_tp_size(16)
      .moe_ep_size(1)
      .pp_size(1)
      .sp_size(1);
  return options;
}

TEST(DpEpPaddingTest, Build) {
  std::string rank_table_file;
  MappingNPU::Options options = get_mapping_options();
  MappingNPU mapping(rank_table_file, 16, 0, options);
  nlohmann::json data = mapping.to_json();
  torch::Tensor token_size_per_dp_group = torch::tensor({10, 10});
  DpEpPadding dp_ep_padding(token_size_per_dp_group,
                            torch::Tensor(),
                            8,
                            data,
                            torch::Device(torch::kCPU),
                            torch::Dtype(torch::kInt32),
                            true);
  DpEpPaddingData dp_ep_padding_data = dp_ep_padding.build();
  LOG(INFO) << "attn_padding_idx:" << dp_ep_padding_data.attn_padding_idx();
  LOG(INFO) << "attn_unpadding_idx:" << dp_ep_padding_data.attn_unpadding_idx();
  LOG(INFO) << "ffn_padding_idx:" << dp_ep_padding_data.ffn_padding_idx();
  LOG(INFO) << "ffn_unpadding_idx:" << dp_ep_padding_data.ffn_unpadding_idx();
  LOG(INFO) << "lm_head_skip_padding_token_indices:"
            << dp_ep_padding_data.lm_head_skip_padding_token_indices();
  LOG(INFO) << "gather_prenorm_idx:" << dp_ep_padding_data.gather_prenorm_idx();
  LOG(INFO) << "padding_idx:" << dp_ep_padding_data.padding_idx();
  LOG(INFO) << "un_padding_idx:" << dp_ep_padding_data.un_padding_idx();
  LOG(INFO) << "dynamic_ep_idx:" << dp_ep_padding_data.dynamic_ep_idx();
  LOG(INFO) << "moe_idx:" << dp_ep_padding_data.moe_idx();
}

TEST(DpEpPaddingTest, BuildFfnPaddingWithEmptyDpGroup) {
  std::string rank_table_file;
  MappingNPU::Options options = get_mapping_options();
  MappingNPU mapping(rank_table_file, 16, 8, options);
  nlohmann::json data = mapping.to_json();

  torch::Tensor token_size_per_dp_group = torch::tensor({0, 2});
  DpEpPadding dp_ep_padding(token_size_per_dp_group,
                            token_size_per_dp_group,
                            8,
                            data,
                            torch::Device(torch::kCPU),
                            torch::Dtype(torch::kInt32),
                            false);
  DpEpPaddingData dp_ep_padding_data = dp_ep_padding.build();

  EXPECT_LE(dp_ep_padding_data.ffn_padding_idx().max().item<int32_t>(), 1);
  EXPECT_TRUE(
      torch::equal(dp_ep_padding_data.lm_head_skip_padding_token_indices(),
                   torch::tensor({8, 9}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(dp_ep_padding_data.attn_unpadding_idx(),
                           torch::tensor({8, 9}, torch::kInt32)));
}

TEST(DpEpPaddingTest, BuildFfnPaddingWithTrailingEmptyDpGroup) {
  std::string rank_table_file;
  MappingNPU::Options options = get_mapping_options();
  MappingNPU mapping(rank_table_file, 16, 0, options);
  nlohmann::json data = mapping.to_json();

  torch::Tensor token_size_per_dp_group = torch::tensor({1, 0});
  DpEpPadding dp_ep_padding(token_size_per_dp_group,
                            token_size_per_dp_group,
                            8,
                            data,
                            torch::Device(torch::kCPU),
                            torch::Dtype(torch::kInt32),
                            false);
  DpEpPaddingData dp_ep_padding_data = dp_ep_padding.build();

  EXPECT_LE(dp_ep_padding_data.ffn_padding_idx().max().item<int32_t>(), 0);
  EXPECT_TRUE(
      torch::equal(dp_ep_padding_data.lm_head_skip_padding_token_indices(),
                   torch::tensor({0}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(dp_ep_padding_data.attn_unpadding_idx(),
                           torch::tensor({0}, torch::kInt32)));
}

TEST(DpEpPaddingTest, BuildAttnUnpaddingWithTrailingEmptyDpGroup) {
  std::string rank_table_file;
  MappingNPU::Options options = get_mapping_options();
  MappingNPU mapping(rank_table_file, 16, 0, options);
  nlohmann::json data = mapping.to_json();

  torch::Tensor token_size_per_dp_group = torch::tensor({2, 0});
  DpEpPadding dp_ep_padding(token_size_per_dp_group,
                            token_size_per_dp_group,
                            8,
                            data,
                            torch::Device(torch::kCPU),
                            torch::Dtype(torch::kInt32),
                            false);
  DpEpPaddingData dp_ep_padding_data = dp_ep_padding.build();

  EXPECT_TRUE(torch::equal(dp_ep_padding_data.attn_unpadding_idx(),
                           torch::tensor({0, 1}, torch::kInt32)));
}

}  // namespace xllm
