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

#include "layers/mlu/deepseek_v4/dsa_metadata_builder_mlu.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "framework/model/model_input_params.h"
#include "layers/common/attention_metadata.h"
#include "layers/mlu/deepseek_v4/dsa_empty_dp_input.h"

namespace xllm {
namespace layer {
namespace {

ModelInputParams make_params(BatchForwardType forward_type,
                             const std::vector<int32_t>& q_cu_lens,
                             const std::vector<int32_t>& kv_cu_lens) {
  ModelInputParams params;
  params.meta.batch_forward_type = forward_type;
  params.meta.num_sequences = static_cast<int32_t>(q_cu_lens.size()) - 1;
  params.meta.actual_num_sequences = params.meta.num_sequences;
  params.meta.q_max_seq_len = 0;
  params.meta.kv_max_seq_len = 0;
  std::vector<int32_t> q_lens;
  std::vector<int32_t> kv_lens;
  q_lens.reserve(q_cu_lens.size() - 1);
  kv_lens.reserve(kv_cu_lens.size() - 1);
  for (int32_t idx = 1; idx < static_cast<int32_t>(q_cu_lens.size()); ++idx) {
    const int32_t q_len = q_cu_lens[idx] - q_cu_lens[idx - 1];
    const int32_t kv_len = kv_cu_lens[idx] - kv_cu_lens[idx - 1];
    q_lens.emplace_back(q_len);
    kv_lens.emplace_back(kv_len);
    params.meta.q_max_seq_len = std::max(params.meta.q_max_seq_len, q_len);
    params.meta.kv_max_seq_len = std::max(params.meta.kv_max_seq_len, kv_len);
  }
  params.attention.host.q_seq_lens = q_cu_lens;
  params.attention.host.kv_seq_lens = kv_cu_lens;
  params.attention.host.q_cu_seq_lens.assign(q_cu_lens.begin() + 1,
                                             q_cu_lens.end());
  params.attention.device.q_seq_lens = torch::tensor(q_cu_lens, torch::kInt32);
  params.attention.device.kv_seq_lens =
      torch::tensor(kv_cu_lens, torch::kInt32);
  params.attention.device.q_cu_seq_lens =
      torch::tensor(params.attention.host.q_cu_seq_lens, torch::kInt32);
  params.attention.device.new_cache_slots = torch::arange(
      q_cu_lens.back(), torch::TensorOptions().dtype(torch::kInt32));
  params.attention.device.block_tables =
      torch::empty({0, 0}, torch::TensorOptions().dtype(torch::kInt32));
  return params;
}

std::vector<int64_t> tensor_vec(const torch::Tensor& tensor) {
  torch::Tensor cpu_tensor = tensor.to(torch::kCPU).to(torch::kInt64);
  std::vector<int64_t> values;
  values.reserve(static_cast<size_t>(cpu_tensor.numel()));
  for (int64_t idx = 0; idx < cpu_tensor.numel(); ++idx) {
    values.emplace_back(cpu_tensor[idx].item<int64_t>());
  }
  return values;
}

AttentionMetadata build_metadata(ModelInputParams params,
                                 const torch::Tensor& positions,
                                 int64_t window_size = 0) {
  return DSAMetadataBuilderMlu::build(params, positions, {}, {}, window_size);
}

TEST(DSAMetadataBuilderMluTest, BuildsCanonicalSeqMetadataFromCuLens) {
  ModelInputParams params =
      make_params(BatchForwardType::PREFILL, {0, 2, 3}, {0, 5, 9});
  torch::Tensor positions = torch::tensor({3, 4, 8}, torch::kInt32);
  AttentionMetadata metadata = build_metadata(params, positions);

  ASSERT_NE(metadata.dsa_metadata, nullptr);
  const DSAMetadata& dsa = *metadata.dsa_metadata;
  EXPECT_EQ(tensor_vec(metadata.q_cu_seq_lens),
            std::vector<int64_t>({0, 2, 3}));
  EXPECT_EQ(tensor_vec(metadata.kv_cu_seq_lens),
            std::vector<int64_t>({0, 5, 9}));
  EXPECT_EQ(tensor_vec(metadata.q_seq_lens), std::vector<int64_t>({2, 1}));
  EXPECT_EQ(tensor_vec(metadata.kv_seq_lens), std::vector<int64_t>({5, 4}));
  EXPECT_EQ(tensor_vec(dsa.q_seq_lens), std::vector<int64_t>({2, 1}));
  EXPECT_EQ(tensor_vec(dsa.kv_seq_lens), std::vector<int64_t>({5, 4}));
  EXPECT_EQ(dsa.query_start_offsets, std::vector<int64_t>({0, 2, 3}));
  EXPECT_EQ(dsa.start_pos_vec, std::vector<int64_t>({3, 3}));
  EXPECT_TRUE(torch::equal(dsa.seq_lens, dsa.kv_seq_lens));
}

TEST(DSAMetadataBuilderMluTest, EmptyDpRankParamsBuildOneTokenDsaMetadata) {
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = 0;
  params.meta.actual_num_sequences = 0;
  params.meta.q_max_seq_len = 0;
  params.meta.kv_max_seq_len = 0;
  params.parallel.dp_global_token_nums = {1, 0, 1, 0};
  std::vector<DSAGroupInfo> group_infos = {{DSACacheType::SLIDING_WINDOW, 1, 1},
                                           {DSACacheType::TOKEN, 4, 1},
                                           {DSACacheType::TOKEN, 128, 1}};

  fill_dsv4_empty_dp_params(params, group_infos, /*window_size=*/128);

  ASSERT_EQ(params.meta.num_sequences, 1);
  ASSERT_EQ(params.meta.actual_num_sequences, 1);
  ASSERT_EQ(params.meta.q_max_seq_len, 1);
  ASSERT_GE(params.meta.kv_max_seq_len, 128);
  ASSERT_TRUE(params.attention.device.new_cache_slots.defined());
  ASSERT_TRUE(params.attention.device.q_seq_lens.defined());
  ASSERT_TRUE(params.attention.device.kv_seq_lens.defined());
  ASSERT_EQ(params.multi_block_tables.size(), group_infos.size());

  std::vector<std::vector<DSACacheInfo>> caches_info = {
      {{0, DSACacheType::SLIDING_WINDOW, 1, 1},
       {1, DSACacheType::TOKEN, 4, 1},
       {2, DSACacheType::TOKEN, 128, 1}}};
  AttentionMetadata metadata =
      DSAMetadataBuilderMlu::build(params,
                                   torch::tensor({0}, torch::kInt32),
                                   caches_info,
                                   group_infos,
                                   /*window_size=*/128);

  ASSERT_NE(metadata.dsa_metadata, nullptr);
  EXPECT_FALSE(metadata.is_dummy);
  EXPECT_EQ(tensor_vec(metadata.q_seq_lens), std::vector<int64_t>({1}));
  EXPECT_EQ(tensor_vec(metadata.kv_seq_lens), std::vector<int64_t>({128}));
  EXPECT_TRUE(
      torch::equal(metadata.slot_mapping, torch::tensor({0}, torch::kInt32)));
}

TEST(DSAMetadataBuilderMluTest, UsesAttentionMetadataSeqLens) {
  ModelInputParams params =
      make_params(BatchForwardType::PREFILL, {0, 2, 3}, {0, 5, 9});
  params.attention.host.q_cu_seq_lens.clear();
  params.attention.host.kv_cu_seq_lens.clear();
  params.attention.host.q_seq_lens = {99};
  params.attention.host.kv_seq_lens = {199};

  AttentionMetadata metadata =
      build_metadata(params, torch::tensor({3, 4, 8}, torch::kInt32));

  ASSERT_NE(metadata.dsa_metadata, nullptr);
  EXPECT_EQ(tensor_vec(metadata.q_cu_seq_lens),
            std::vector<int64_t>({0, 2, 3}));
  EXPECT_EQ(tensor_vec(metadata.kv_cu_seq_lens),
            std::vector<int64_t>({0, 5, 9}));
  EXPECT_EQ(tensor_vec(metadata.q_seq_lens), std::vector<int64_t>({2, 1}));
  EXPECT_EQ(tensor_vec(metadata.kv_seq_lens), std::vector<int64_t>({5, 4}));
}

TEST(DSAMetadataBuilderMluTest, PreservesAttentionFlags) {
  ModelInputParams prefill =
      make_params(BatchForwardType::PREFILL, {0, 1}, {0, 1});
  prefill.graph.attn_mask = torch::ones({1, 1}, torch::kFloat32);
  AttentionMetadata prefill_metadata =
      build_metadata(prefill, torch::tensor({0}, torch::kInt32));
  ASSERT_NE(prefill_metadata.dsa_metadata, nullptr);
  EXPECT_TRUE(prefill_metadata.is_prefill);
  EXPECT_TRUE(prefill_metadata.attn_mask.defined());
  EXPECT_FALSE(prefill_metadata.dsa_metadata->attn_mask.defined());

  ModelInputParams mixed = make_params(BatchForwardType::MIXED, {0, 1}, {0, 4});
  AttentionMetadata mixed_metadata =
      build_metadata(mixed, torch::tensor({3}, torch::kInt32));
  EXPECT_TRUE(mixed_metadata.is_chunked_prefill);

  ModelInputParams dummy = make_params(BatchForwardType::DECODE, {0}, {0});
  dummy.meta.q_max_seq_len = 0;
  AttentionMetadata dummy_metadata =
      build_metadata(dummy, torch::empty({0}, torch::kInt32));
  EXPECT_TRUE(dummy_metadata.is_dummy);
}

TEST(DSAMetadataBuilderMluTest, BuildsCompressedPositionMetadata) {
  ModelInputParams params =
      make_params(BatchForwardType::PREFILL, {0, 3, 6}, {0, 3, 6});
  torch::Tensor positions =
      torch::tensor({2, 3, 4, 126, 127, 128}, torch::kInt32);
  AttentionMetadata metadata = build_metadata(params, positions);
  const DSAMetadata& dsa = *metadata.dsa_metadata;

  EXPECT_TRUE(torch::equal(dsa.input_positions, positions));
  EXPECT_EQ(tensor_vec(dsa.c4_pad_positions),
            std::vector<int64_t>({0, 124, 0}));
  EXPECT_EQ(tensor_vec(dsa.c128_pad_positions), std::vector<int64_t>({0, 0}));
  EXPECT_FALSE(dsa.cos_table.defined());
  EXPECT_FALSE(dsa.sin_table.defined());
  EXPECT_FALSE(dsa.compressed_cos_table.defined());
  EXPECT_FALSE(dsa.compressed_sin_table.defined());
}

TEST(DSAMetadataBuilderMluTest, ExpandsBlockTablesAndSlotMappings) {
  ModelInputParams params =
      make_params(BatchForwardType::CHUNKED_PREFILL, {0, 2, 3}, {0, 5, 9});
  params.multi_block_tables = {
      torch::tensor({10, 11, 20, 21}, torch::kInt32).view({2, 2}),
      torch::tensor({30, 31, 40, 41, 42, 43}, torch::kInt32).view({2, 3})};
  std::vector<DSAGroupInfo> group_infos = {
      {DSACacheType::TOKEN, 4, 16}, {DSACacheType::SLIDING_WINDOW, 1, 4}};
  std::vector<std::vector<DSACacheInfo>> caches_info = {
      {{0, DSACacheType::TOKEN, 4, 16},
       {0, DSACacheType::TOKEN, 4, 16},
       {1, DSACacheType::SLIDING_WINDOW, 1, 4}}};

  AttentionMetadata metadata =
      DSAMetadataBuilderMlu::build(params,
                                   torch::tensor({3, 4, 8}, torch::kInt32),
                                   caches_info,
                                   group_infos,
                                   /*window_size=*/4);
  const DSAMetadata& dsa = *metadata.dsa_metadata;

  ASSERT_EQ(dsa.block_tables.size(), 1);
  ASSERT_EQ(dsa.block_tables[0].size(), 3);
  EXPECT_TRUE(torch::equal(dsa.block_tables[0][0], dsa.block_tables[0][1]));
  EXPECT_EQ(tensor_vec(dsa.slot_mappings[0][0]),
            std::vector<int64_t>({160, 320}));
  EXPECT_EQ(tensor_vec(dsa.slot_mappings[0][2]),
            std::vector<int64_t>({123, 124, 167}));
}

TEST(DSAMetadataBuilderMluTest, SwaSlotsUseAbsoluteBlockColumns) {
  ModelInputParams params =
      make_params(BatchForwardType::CHUNKED_PREFILL, {0, 1}, {0, 24});
  params.multi_block_tables = {
      torch::tensor({-1, -1, -1, -1, -1, 11}, torch::kInt32).view({1, 6})};
  std::vector<DSAGroupInfo> group_infos = {
      {DSACacheType::SLIDING_WINDOW, 1, 4}};
  std::vector<std::vector<DSACacheInfo>> caches_info = {
      {{0, DSACacheType::SLIDING_WINDOW, 1, 4}}};

  AttentionMetadata metadata =
      DSAMetadataBuilderMlu::build(params,
                                   torch::tensor({23}, torch::kInt32),
                                   caches_info,
                                   group_infos,
                                   /*window_size=*/4);
  const DSAMetadata& dsa = *metadata.dsa_metadata;

  ASSERT_EQ(dsa.block_tables.size(), 1);
  ASSERT_EQ(dsa.block_tables[0].size(), 1);
  EXPECT_EQ(dsa.block_tables[0][0].size(1), 32);
  EXPECT_EQ(dsa.block_tables[0][0][0][4].item<int32_t>(), 0);
  EXPECT_EQ(dsa.block_tables[0][0][0][5].item<int32_t>(), 11);
  EXPECT_EQ(tensor_vec(dsa.slot_mappings[0][0]), std::vector<int64_t>({47}));
}

TEST(DSAMetadataBuilderMluTest, SwaKeepsSparseAbsoluteReadTable) {
  ModelInputParams params =
      make_params(BatchForwardType::CHUNKED_PREFILL, {0, 2, 5}, {0, 24, 43});
  torch::Tensor swa_table = torch::full({2, 11}, -1, torch::kInt32);
  auto swa_table_acc = swa_table.accessor<int32_t, 2>();
  swa_table_acc[0][4] = 49;
  swa_table_acc[0][5] = 50;
  swa_table_acc[1][3] = 59;
  swa_table_acc[1][4] = 60;
  params.multi_block_tables = {swa_table};
  std::vector<DSAGroupInfo> group_infos = {
      {DSACacheType::SLIDING_WINDOW, 1, 4}};
  std::vector<std::vector<DSACacheInfo>> caches_info = {
      {{0, DSACacheType::SLIDING_WINDOW, 1, 4}}};

  AttentionMetadata metadata = DSAMetadataBuilderMlu::build(
      params,
      torch::tensor({22, 23, 16, 17, 18}, torch::kInt32),
      caches_info,
      group_infos,
      /*window_size=*/4);
  const DSAMetadata& dsa = *metadata.dsa_metadata;

  ASSERT_EQ(dsa.block_tables.size(), 1);
  ASSERT_EQ(dsa.block_tables[0].size(), 1);
  EXPECT_EQ(dsa.block_tables[0][0].size(0), 2);
  EXPECT_EQ(dsa.block_tables[0][0].size(1), 32);
  EXPECT_EQ(dsa.block_tables[0][0][0][3].item<int32_t>(), 0);
  EXPECT_EQ(dsa.block_tables[0][0][0][4].item<int32_t>(), 49);
  EXPECT_EQ(dsa.block_tables[0][0][0][5].item<int32_t>(), 50);
  EXPECT_EQ(dsa.block_tables[0][0][1][2].item<int32_t>(), 0);
  EXPECT_EQ(dsa.block_tables[0][0][1][3].item<int32_t>(), 59);
  EXPECT_EQ(dsa.block_tables[0][0][1][4].item<int32_t>(), 60);
  EXPECT_EQ(tensor_vec(dsa.slot_mappings[0][0]),
            std::vector<int64_t>({202, 203, 240, 241, 242}));
}

TEST(DSAMetadataBuilderMluTest, RejectsMissingSwaLiveBlock) {
  ModelInputParams params =
      make_params(BatchForwardType::CHUNKED_PREFILL, {0, 1}, {0, 24});
  params.multi_block_tables = {
      torch::tensor({-1, -1, -1, -1, -1, -1}, torch::kInt32).view({1, 6})};
  std::vector<DSAGroupInfo> group_infos = {
      {DSACacheType::SLIDING_WINDOW, 1, 4}};
  std::vector<std::vector<DSACacheInfo>> caches_info = {
      {{0, DSACacheType::SLIDING_WINDOW, 1, 4}}};

  EXPECT_DEATH(DSAMetadataBuilderMlu::build(params,
                                            torch::tensor({23}, torch::kInt32),
                                            caches_info,
                                            group_infos,
                                            /*window_size=*/4),
               "SWA live window has an invalid absolute block");
}

TEST(DSAMetadataBuilderMluTest, BuildsC128AttentionMetadata) {
  ModelInputParams params =
      make_params(BatchForwardType::DECODE, {0, 1}, {0, 128});
  params.multi_block_tables = {
      torch::tensor({7}, torch::TensorOptions().dtype(torch::kInt32))
          .view({1, 1})};
  std::vector<DSAGroupInfo> group_infos = {{DSACacheType::TOKEN, 128, 16}};
  std::vector<std::vector<DSACacheInfo>> caches_info = {
      {{0, DSACacheType::TOKEN, 128, 16}}};

  AttentionMetadata metadata =
      DSAMetadataBuilderMlu::build(params,
                                   torch::tensor({127}, torch::kInt32),
                                   caches_info,
                                   group_infos,
                                   /*window_size=*/0);
  const DSAMetadata& dsa = *metadata.dsa_metadata;

  EXPECT_EQ(tensor_vec(dsa.c128_attn_metadata.context_lens),
            std::vector<int64_t>({1}));
  EXPECT_EQ(tensor_vec(dsa.c128_attn_metadata.block_table_for_attn),
            std::vector<int64_t>({7}));
  EXPECT_EQ(dsa.c128_attn_metadata.max_context_len, 1);
  EXPECT_FALSE(dsa.c128_metadata.defined());
}

TEST(DSAMetadataBuilderMluTest, BuildsDecodeCompressedSlotMapping) {
  ModelInputParams params =
      make_params(BatchForwardType::DECODE, {0, 1}, {0, 129});
  params.multi_block_tables = {
      torch::tensor({10, 11, 12}, torch::kInt32).view({1, 3})};
  std::vector<DSAGroupInfo> group_infos = {{DSACacheType::TOKEN, 4, 16}};
  std::vector<std::vector<DSACacheInfo>> caches_info = {
      {{0, DSACacheType::TOKEN, 4, 16}}};

  AttentionMetadata metadata =
      DSAMetadataBuilderMlu::build(params,
                                   torch::tensor({128}, torch::kInt32),
                                   caches_info,
                                   group_infos,
                                   /*window_size=*/0);
  ASSERT_EQ(metadata.dsa_metadata->slot_mappings.size(), 1);
  ASSERT_EQ(metadata.dsa_metadata->slot_mappings[0].size(), 1);
  EXPECT_EQ(tensor_vec(metadata.dsa_metadata->slot_mappings[0][0]),
            std::vector<int64_t>({192}));
  EXPECT_FALSE(metadata.dsa_metadata->cmp_slots_dict.contains(4));

  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  AttentionMetadata prefill_metadata =
      DSAMetadataBuilderMlu::build(params,
                                   torch::tensor({128}, torch::kInt32),
                                   caches_info,
                                   group_infos,
                                   /*window_size=*/0);
  EXPECT_FALSE(prefill_metadata.dsa_metadata->cmp_slots_dict.contains(4));
}

TEST(DSAMetadataBuilderMluTest, IgnoresLegacyCuLensFields) {
  ModelInputParams params =
      make_params(BatchForwardType::PREFILL, {0, 2, 3}, {0, 5, 9});
  params.attention.host.q_cu_seq_lens = {99, 100};
  params.attention.host.kv_cu_seq_lens = {88, 89};

  AttentionMetadata metadata =
      build_metadata(params, torch::tensor({3, 4, 8}, torch::kInt32));

  ASSERT_NE(metadata.dsa_metadata, nullptr);
  EXPECT_EQ(tensor_vec(metadata.q_cu_seq_lens),
            std::vector<int64_t>({0, 2, 3}));
  EXPECT_EQ(tensor_vec(metadata.kv_cu_seq_lens),
            std::vector<int64_t>({0, 5, 9}));
  EXPECT_EQ(tensor_vec(metadata.dsa_metadata->q_seq_lens),
            std::vector<int64_t>({2, 1}));
  EXPECT_EQ(tensor_vec(metadata.dsa_metadata->kv_seq_lens),
            std::vector<int64_t>({5, 4}));
}

}  // namespace
}  // namespace layer
}  // namespace xllm
