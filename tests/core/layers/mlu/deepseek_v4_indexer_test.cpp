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

#include "layers/mlu/deepseek_v4/deepseek_v4_indexer.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "common/global_flags.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/deepseek_v4/dsa_metadata_builder_mlu.h"
#include "layers/mlu/deepseek_v4_ref_utils.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "util/linalg.h"

namespace xllm {
namespace layer {
namespace {

constexpr int64_t kCompressRatio = 4;

struct TestConfig {
  int64_t dim = 32;
  int64_t q_lora_rank = 32;
  int64_t index_n_heads = 64;
  int64_t index_head_dim = 128;
  int64_t rope_head_dim = 64;
  int64_t index_topk = 512;
  double norm_eps = 1e-6;
};

struct RefOut {
  torch::Tensor q;
  torch::Tensor kv;
  torch::Tensor weights;
  torch::Tensor topk;
  torch::Tensor context_lens;
};

struct TopkRefOut {
  torch::Tensor topk;
  torch::Tensor context_lens;
};

int64_t q_len_at(const std::vector<int64_t>& offsets, int64_t seq_idx) {
  return offsets[seq_idx + 1] - offsets[seq_idx];
}

torch::Tensor seeded(const std::string& key,
                     torch::IntArrayRef shape,
                     torch::ScalarType dtype) {
  return (test::seeded_tensor(key, shape, dtype, torch::Device(torch::kCPU)) -
          0.5) *
         0.2;
}

std::vector<int64_t> compressed_positions(
    const std::vector<int64_t>& q_lens,
    const std::vector<int64_t>& start_pos) {
  std::vector<int64_t> positions;
  for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(q_lens.size());
       ++seq_idx) {
    for (int64_t token_idx = 0; token_idx < q_lens[seq_idx]; ++token_idx) {
      const int64_t pos = start_pos[seq_idx] + token_idx;
      if ((pos + 1) % kCompressRatio != 0) {
        continue;
      }
      positions.emplace_back(pos + 1 - kCompressRatio);
    }
  }
  return positions;
}

TopkRefOut topk_ref(const torch::Tensor& q,
                    const torch::Tensor& weights,
                    const torch::Tensor& full_kv,
                    const std::vector<int64_t>& start_pos,
                    const std::vector<int64_t>& q_offsets,
                    int64_t index_topk) {
  std::vector<torch::Tensor> rows;
  std::vector<int32_t> context_lens;
  int64_t kv_offset = 0;
  for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(start_pos.size());
       ++seq_idx) {
    const int64_t begin = q_offsets[seq_idx];
    const int64_t end = q_offsets[seq_idx + 1];
    const int64_t final_kv_len = start_pos[seq_idx] + end - begin;
    const int64_t compressed_len = final_kv_len / kCompressRatio;
    torch::Tensor kv_seq =
        full_kv.slice(0, kv_offset, kv_offset + compressed_len);
    for (int64_t token_idx = 0; token_idx < end - begin; ++token_idx) {
      const int64_t flat_idx = begin + token_idx;
      const int64_t valid_count =
          (start_pos[seq_idx] + token_idx + 1) / kCompressRatio;
      const int64_t selected_count = std::min(index_topk, valid_count);
      context_lens.emplace_back(static_cast<int32_t>(selected_count));
      torch::Tensor out = torch::full(
          {index_topk}, -1, torch::TensorOptions().dtype(torch::kInt32));
      if (valid_count > 0) {
        torch::Tensor q_row = q[flat_idx];
        torch::Tensor score =
            torch::matmul(q_row.reshape({q_row.size(0), q_row.size(1)}),
                          kv_seq.slice(0, 0, valid_count).t());
        score = (score.relu() * weights[flat_idx].unsqueeze(-1)).sum(0);
        torch::Tensor idx =
            std::get<1>(score.topk(selected_count, -1)).to(torch::kInt32);
        out.slice(0, 0, selected_count)
            .copy_(idx + static_cast<int32_t>(kv_offset));
      }
      rows.emplace_back(out);
    }
    kv_offset += compressed_len;
  }
  return {torch::stack(rows, 0), torch::tensor(context_lens, torch::kInt32)};
}

torch::Tensor make_block_table(const std::vector<std::vector<int32_t>>& rows) {
  std::vector<int32_t> values;
  int64_t cols = 0;
  for (const auto& row : rows) {
    cols = std::max(cols, static_cast<int64_t>(row.size()));
  }
  values.reserve(rows.size() * cols);
  for (const auto& row : rows) {
    for (int64_t col = 0; col < cols; ++col) {
      values.emplace_back(col < static_cast<int64_t>(row.size()) ? row[col]
                                                                 : 0);
    }
  }
  return torch::tensor(values, torch::kInt32)
      .view({static_cast<int64_t>(rows.size()), cols});
}

int32_t max_seq_len(const std::vector<int>& cu_lens) {
  int32_t max_len = 0;
  for (size_t idx = 1; idx < cu_lens.size(); ++idx) {
    max_len = std::max(max_len, cu_lens[idx] - cu_lens[idx - 1]);
  }
  return max_len;
}

ModelInputParams make_c128_params(BatchForwardType batch_forward_type,
                                  int32_t num_sequences,
                                  const std::vector<int>& q_cu_lens,
                                  const std::vector<int>& kv_cu_lens,
                                  const torch::Tensor& block_table) {
  ModelInputParams params;
  params.meta.batch_forward_type = batch_forward_type;
  params.meta.num_sequences = num_sequences;
  params.meta.actual_num_sequences = num_sequences;
  params.meta.q_max_seq_len = max_seq_len(q_cu_lens);
  params.meta.kv_max_seq_len = max_seq_len(kv_cu_lens);
  params.attention.host.q_seq_lens.assign(q_cu_lens.begin(), q_cu_lens.end());
  params.attention.host.kv_seq_lens.assign(kv_cu_lens.begin(),
                                           kv_cu_lens.end());
  params.attention.host.q_cu_seq_lens.assign(q_cu_lens.begin() + 1,
                                             q_cu_lens.end());
  params.attention.device.q_seq_lens = torch::tensor(q_cu_lens, torch::kInt32);
  params.attention.device.kv_seq_lens =
      torch::tensor(kv_cu_lens, torch::kInt32);
  params.attention.device.q_cu_seq_lens =
      torch::tensor(params.attention.host.q_cu_seq_lens, torch::kInt32);
  params.attention.device.new_cache_slots = torch::empty(
      {q_cu_lens.back()}, torch::TensorOptions().dtype(torch::kInt32));
  params.attention.device.block_tables =
      torch::empty({0, 0}, torch::TensorOptions().dtype(torch::kInt32));
  params.multi_block_tables = {block_table};
  return params;
}

AttentionMetadata build_c128_metadata(BatchForwardType batch_forward_type,
                                      int32_t num_sequences,
                                      const std::vector<int>& q_cu_lens,
                                      const std::vector<int>& kv_cu_lens,
                                      const torch::Tensor& block_table,
                                      int32_t block_size) {
  ModelInputParams params = make_c128_params(
      batch_forward_type, num_sequences, q_cu_lens, kv_cu_lens, block_table);
  torch::Tensor positions = torch::arange(q_cu_lens.back(), torch::kInt32);
  std::vector<std::vector<DSACacheInfo>> caches_info = {
      {{0, DSACacheType::TOKEN, 128, block_size}}};
  std::vector<DSAGroupInfo> group_infos = {
      {DSACacheType::TOKEN, 128, block_size}};
  return DSAMetadataBuilderMlu::build(params,
                                      positions,
                                      caches_info,
                                      group_infos,
                                      /*window_size=*/0);
}

torch::Tensor sparse_slot_ref(const torch::Tensor& logical_topk,
                              const torch::Tensor& context_lens,
                              const AttentionMetadata& metadata,
                              const torch::Tensor& block_table) {
  const DSAMetadata& dsa = *metadata.dsa_metadata;
  const int64_t block_size = FLAGS_block_size;
  CHECK_GT(block_size, 0);
  CHECK_EQ(logical_topk.dim(), 2);
  CHECK_EQ(context_lens.dim(), 1);
  CHECK_EQ(logical_topk.size(0), context_lens.size(0));
  CHECK(block_table.defined());

  torch::Tensor out = logical_topk.cpu().to(torch::kInt32).contiguous().clone();
  torch::Tensor context_lens_cpu =
      context_lens.cpu().to(torch::kInt32).contiguous();
  torch::Tensor block_table_cpu =
      block_table.cpu().to(torch::kInt64).contiguous();
  auto out_acc = out.accessor<int32_t, 2>();
  auto context_acc = context_lens_cpu.accessor<int32_t, 1>();
  auto block_acc = block_table_cpu.accessor<int64_t, 2>();

  std::vector<int64_t> c4_offsets;
  std::vector<int64_t> c4_lens;
  c4_offsets.reserve(dsa.start_pos_vec.size());
  c4_lens.reserve(dsa.start_pos_vec.size());
  int64_t c4_offset = 0;
  for (int64_t seq_idx = 0;
       seq_idx < static_cast<int64_t>(dsa.start_pos_vec.size());
       ++seq_idx) {
    const int64_t q_len = q_len_at(dsa.query_start_offsets, seq_idx);
    const int64_t c4_len =
        (dsa.start_pos_vec[seq_idx] + q_len) / kCompressRatio;
    c4_offsets.emplace_back(c4_offset);
    c4_lens.emplace_back(c4_len);
    c4_offset += c4_len;
  }

  for (int64_t row = 0; row < out.size(0); ++row) {
    const int64_t selected_count = context_acc[row];
    CHECK_LE(selected_count, out.size(1));
    for (int64_t col = 0; col < selected_count; ++col) {
      const int64_t logical_row = out_acc[row][col];
      if (logical_row < 0) {
        continue;
      }
      int64_t seq_idx = 0;
      while (seq_idx < static_cast<int64_t>(c4_offsets.size()) &&
             logical_row >= c4_offsets[seq_idx] + c4_lens[seq_idx]) {
        ++seq_idx;
      }
      CHECK_LT(seq_idx, static_cast<int64_t>(c4_offsets.size()));
      const int64_t seq_row = logical_row - c4_offsets[seq_idx];
      const int64_t block_col = seq_row / block_size;
      const int64_t block_offset = seq_row % block_size;
      CHECK_LT(block_col, block_table_cpu.size(1));
      out_acc[row][col] = static_cast<int32_t>(
          block_acc[seq_idx][block_col] * block_size + block_offset);
    }
  }
  return out;
}

void expect_context_lens_shape(const torch::Tensor& context_lens,
                               int64_t row_count) {
  EXPECT_EQ(context_lens.dim(), 1);
  EXPECT_EQ(context_lens.size(0), row_count);
  EXPECT_EQ(context_lens.scalar_type(), torch::kInt32);
}

void expect_topk_prefix_equal(const torch::Tensor& actual,
                              const torch::Tensor& expected,
                              const torch::Tensor& actual_context_lens,
                              const torch::Tensor& expected_context_lens) {
  torch::Tensor actual_cpu = actual.cpu().to(torch::kInt32);
  torch::Tensor expected_cpu = expected.cpu().to(torch::kInt32);
  torch::Tensor actual_context_lens_cpu =
      actual_context_lens.cpu().to(torch::kInt32);
  torch::Tensor context_lens_cpu =
      expected_context_lens.cpu().to(torch::kInt32);
  ASSERT_EQ(actual_cpu.sizes(), expected_cpu.sizes());
  ASSERT_EQ(actual_cpu.size(0), context_lens_cpu.size(0));
  EXPECT_TRUE(torch::equal(actual_context_lens_cpu, context_lens_cpu))
      << "actual context lens=" << actual_context_lens_cpu
      << "\nexpected context lens=" << context_lens_cpu;
  for (int64_t row = 0; row < actual_cpu.size(0); ++row) {
    const int64_t selected_count = context_lens_cpu[row].item<int32_t>();
    ASSERT_LE(selected_count, actual_cpu.size(1));
    if (selected_count == 0) {
      continue;
    }
    torch::Tensor actual_prefix = actual_cpu[row].slice(0, 0, selected_count);
    torch::Tensor expected_prefix =
        expected_cpu[row].slice(0, 0, selected_count);
    if (torch::equal(actual_prefix, expected_prefix)) {
      continue;
    }
    torch::Tensor mismatch = actual_prefix.ne(expected_prefix).nonzero();
    const int64_t col = mismatch[0][0].item<int64_t>();
    ADD_FAILURE() << "topk prefix mismatch at row " << row << ", col " << col
                  << ", actual=" << actual_prefix[col].item<int32_t>()
                  << ", expected=" << expected_prefix[col].item<int32_t>()
                  << "\nactual first column=" << actual_cpu.select(1, 0)
                  << "\nexpected first column=" << expected_cpu.select(1, 0)
                  << "\nactual context lens=" << actual_context_lens_cpu
                  << "\nexpected context lens=" << context_lens_cpu
                  << "\nactual prefix=" << actual_prefix
                  << "\nexpected prefix=" << expected_prefix;
    return;
  }
}

void expect_topk_prefix_set_equal(const torch::Tensor& actual,
                                  const torch::Tensor& expected,
                                  const torch::Tensor& actual_context_lens,
                                  const torch::Tensor& expected_context_lens) {
  torch::Tensor actual_cpu = actual.cpu().to(torch::kInt32);
  torch::Tensor expected_cpu = expected.cpu().to(torch::kInt32);
  torch::Tensor actual_context_lens_cpu =
      actual_context_lens.cpu().to(torch::kInt32);
  torch::Tensor context_lens_cpu =
      expected_context_lens.cpu().to(torch::kInt32);
  ASSERT_EQ(actual_cpu.sizes(), expected_cpu.sizes());
  EXPECT_TRUE(torch::equal(actual_context_lens_cpu, context_lens_cpu));
  for (int64_t row = 0; row < actual_cpu.size(0); ++row) {
    const int64_t selected_count = context_lens_cpu[row].item<int32_t>();
    if (selected_count == 0) {
      continue;
    }
    torch::Tensor actual_prefix =
        actual_cpu[row].slice(0, 0, selected_count).contiguous();
    torch::Tensor expected_prefix =
        expected_cpu[row].slice(0, 0, selected_count).contiguous();
    actual_prefix = std::get<0>(actual_prefix.sort(/*dim=*/0));
    expected_prefix = std::get<0>(expected_prefix.sort(/*dim=*/0));
    EXPECT_TRUE(torch::equal(actual_prefix, expected_prefix))
        << "topk prefix set mismatch at row " << row
        << "\nactual prefix=" << actual_prefix
        << "\nexpected prefix=" << expected_prefix;
  }
}

}  // namespace

TEST(DeepseekV4DSAMetadataBuilderTest, SyncsPrefillSeqLensToAttentionMetadata) {
  ModelInputParams params =
      make_c128_params(BatchForwardType::PREFILL,
                       /*num_sequences=*/2,
                       /*q_cu_lens=*/{0, 9, 17},
                       /*kv_cu_lens=*/{0, 9, 17},
                       torch::empty({0, 0}, torch::kInt32));

  torch::Tensor positions = torch::arange(17, torch::kInt32);
  AttentionMetadata metadata = DSAMetadataBuilderMlu::build(
      params, positions, {}, {}, /*window_size=*/0);

  ASSERT_NE(metadata.dsa_metadata, nullptr);
  EXPECT_TRUE(torch::equal(metadata.q_cu_seq_lens,
                           torch::tensor({0, 9, 17}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(metadata.kv_cu_seq_lens,
                           torch::tensor({0, 9, 17}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(metadata.q_seq_lens, torch::tensor({9, 8}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(metadata.kv_seq_lens, torch::tensor({9, 8}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(metadata.dsa_metadata->q_seq_lens, metadata.q_seq_lens));
  EXPECT_TRUE(
      torch::equal(metadata.dsa_metadata->kv_seq_lens, metadata.kv_seq_lens));
}

TEST(DeepseekV4DSAMetadataBuilderTest, BuildsC128MetadataForChunkedPrefill) {
  AttentionMetadata metadata =
      build_c128_metadata(BatchForwardType::CHUNKED_PREFILL,
                          /*num_sequences=*/2,
                          /*q_cu_lens=*/{0, 4, 7},
                          /*kv_cu_lens=*/{0, 130, 390},
                          make_block_table({{11, 12}, {21, 22}}),
                          /*block_size=*/2);

  ASSERT_NE(metadata.dsa_metadata, nullptr);
  const DSACompressedAttentionMetadata& c128 =
      metadata.dsa_metadata->c128_attn_metadata;
  torch::Tensor expected_lens =
      torch::tensor({0, 1, 1, 1, 2, 2, 2}, torch::kInt32);
  torch::Tensor expected_table =
      torch::tensor({-1, 11, 11, 11, 21, 21, 21}, torch::kInt32).view({7, 1});
  EXPECT_TRUE(torch::equal(c128.context_lens, expected_lens));
  EXPECT_TRUE(torch::equal(c128.block_table_for_attn, expected_table));
  EXPECT_EQ(c128.max_context_len, 2);
}

TEST(DeepseekV4DSAMetadataBuilderTest, BuildsC128TableForDecode) {
  AttentionMetadata metadata =
      build_c128_metadata(BatchForwardType::DECODE,
                          /*num_sequences=*/2,
                          /*q_cu_lens=*/{0, 1, 2},
                          /*kv_cu_lens=*/{0, 257, 770},
                          make_block_table({{31, 32, 33}, {41, 42, 43}}),
                          /*block_size=*/2);

  ASSERT_NE(metadata.dsa_metadata, nullptr);
  const DSACompressedAttentionMetadata& c128 =
      metadata.dsa_metadata->c128_attn_metadata;
  torch::Tensor expected_lens = torch::tensor({2, 4}, torch::kInt32);
  torch::Tensor expected_table =
      torch::tensor({31, -1, 41, 42}, torch::kInt32).view({2, 2});
  EXPECT_TRUE(torch::equal(c128.context_lens, expected_lens));
  EXPECT_TRUE(torch::equal(c128.block_table_for_attn, expected_table));
  EXPECT_EQ(c128.max_context_len, 4);
}

TEST(DeepseekV4DSAMetadataBuilderTest, SkipsC128MetadataForEmptyBatch) {
  AttentionMetadata metadata = build_c128_metadata(
      BatchForwardType::DECODE,
      /*num_sequences=*/0,
      /*q_cu_lens=*/{0},
      /*kv_cu_lens=*/{0},
      torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kInt32)),
      /*block_size=*/2);

  ASSERT_NE(metadata.dsa_metadata, nullptr);
  const DSACompressedAttentionMetadata& c128 =
      metadata.dsa_metadata->c128_attn_metadata;
  EXPECT_FALSE(c128.context_lens.defined());
  EXPECT_FALSE(c128.block_table_for_attn.defined());
  EXPECT_EQ(c128.max_context_len, 0);
}

class DeepseekV4IndexerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    FLAGS_block_size = 1;
    torch::Device torch_device(Platform::type_torch(), 0);
    Device device(torch_device);
    device.set_seed();
    device_ = torch_device;
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(torch_device)
                   .requires_grad(false);
    cpu_options_ =
        torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
  }

  std::unordered_map<std::string, torch::Tensor> make_weights() {
    std::unordered_map<std::string, torch::Tensor> weights;
    weights["wq_b.weight"] = seeded(
        "dsv4.indexer.wq_b",
        {config_.index_n_heads * config_.index_head_dim, config_.q_lora_rank},
        torch::kBFloat16);
    weights["weights_proj.weight"] =
        seeded("dsv4.indexer.weights",
               {config_.index_n_heads, config_.dim},
               torch::kBFloat16) +
        0.25;
    weights["compressor.wkv.weight"] =
        seeded("dsv4.indexer.wkv",
               {2 * config_.index_head_dim, config_.dim},
               torch::kBFloat16);
    weights["compressor.wgate.weight"] =
        seeded("dsv4.indexer.wgate",
               {2 * config_.index_head_dim, config_.dim},
               torch::kBFloat16);
    weights["compressor.norm.weight"] =
        test::seeded_tensor("dsv4.indexer.norm",
                            {config_.index_head_dim},
                            torch::kFloat32,
                            torch::Device(torch::kCPU)) +
        0.5;
    weights["compressor.ape"] =
        seeded("dsv4.indexer.ape",
               {kCompressRatio, 2 * config_.index_head_dim},
               torch::kFloat32);
    return weights;
  }

  StateDict make_state_dict(
      const std::unordered_map<std::string, torch::Tensor>& weights) {
    std::unordered_map<std::string, torch::Tensor> device_weights;
    for (const auto& item : weights) {
      device_weights[item.first] = item.second.to(device_);
    }
    return StateDict(device_weights);
  }

  AttentionMetadata make_metadata(
      const std::vector<int64_t>& q_lens,
      const std::vector<int64_t>& start_pos,
      const std::vector<std::vector<int32_t>>& c4_blocks,
      const std::vector<int32_t>& c4_slots,
      bool chunked) {
    AttentionMetadata metadata;
    auto dsa = std::make_shared<DSAMetadata>();
    dsa->layer_id = 0;
    dsa->start_pos_vec = start_pos;
    dsa->query_start_offsets = {0};
    int64_t total_tokens = 0;
    int64_t total_c4_rows = 0;
    std::vector<int32_t> q_cu = {0};
    std::vector<int32_t> kv_cu = {0};
    std::vector<int32_t> q_seq_lens;
    std::vector<int32_t> kv_lens;
    std::vector<int32_t> c4_lens;
    std::vector<int32_t> input_positions;
    for (size_t i = 0; i < q_lens.size(); ++i) {
      total_tokens += q_lens[i];
      q_cu.emplace_back(static_cast<int32_t>(total_tokens));
      dsa->query_start_offsets.emplace_back(total_tokens);
      const int64_t kv_len = start_pos[i] + q_lens[i];
      kv_cu.emplace_back(kv_cu.back() + static_cast<int32_t>(kv_len));
      q_seq_lens.emplace_back(static_cast<int32_t>(q_lens[i]));
      kv_lens.emplace_back(static_cast<int32_t>(kv_len));
      const int64_t c4_len = kv_len / kCompressRatio;
      c4_lens.emplace_back(static_cast<int32_t>(c4_len));
      total_c4_rows += c4_len;
      dsa->index_max_c4_len = std::max(dsa->index_max_c4_len, c4_len);
      for (int64_t token_idx = 0; token_idx < q_lens[i]; ++token_idx) {
        input_positions.emplace_back(
            static_cast<int32_t>(start_pos[i] + token_idx));
      }
    }
    dsa->index_total_c4_len = total_c4_rows;
    std::vector<int64_t> c4_positions = compressed_positions(q_lens, start_pos);
    std::vector<int32_t> c4_position_values;
    c4_position_values.reserve(c4_positions.size());
    for (int64_t position : c4_positions) {
      c4_position_values.emplace_back(static_cast<int32_t>(position));
    }
    torch::TensorOptions int_options =
        torch::TensorOptions().dtype(torch::kInt32).device(device_);
    dsa->input_positions = torch::tensor(input_positions, int_options);
    dsa->c4_pad_positions = torch::tensor(c4_position_values, int_options);
    dsa->q_cu_seq_lens = torch::tensor(q_cu, int_options);
    dsa->kv_cu_seq_lens = torch::tensor(kv_cu, int_options);
    dsa->q_seq_lens = torch::tensor(q_seq_lens, int_options);
    dsa->kv_seq_lens = torch::tensor(kv_lens, int_options);
    dsa->index_c4_seq_lens = torch::tensor(c4_lens, int_options);
    dsa->actual_seq_lengths_query = dsa->q_cu_seq_lens;
    dsa->actual_seq_lengths_kv = dsa->kv_seq_lens;
    dsa->seq_lens_q = dsa->q_seq_lens;
    dsa->seq_lens = dsa->kv_seq_lens;

    torch::Tensor c4_bt = make_block_table(c4_blocks).to(device_);
    torch::Tensor c4_slot_mapping =
        torch::tensor(c4_slots, torch::TensorOptions().dtype(torch::kInt32))
            .to(device_);
    metadata.dsa_metadata = dsa;
    metadata.is_prefill = true;
    metadata.is_chunked_prefill = chunked;
    metadata.max_query_len = *std::max_element(q_lens.begin(), q_lens.end());
    metadata.max_seq_len = static_cast<int64_t>(total_c4_rows);
    metadata.total_kv_len = total_c4_rows;
    metadata.q_cu_seq_lens = dsa->q_cu_seq_lens;
    metadata.kv_cu_seq_lens = dsa->kv_cu_seq_lens;
    metadata.q_seq_lens = dsa->q_seq_lens;
    metadata.kv_seq_lens = dsa->kv_seq_lens;
    metadata.block_table = c4_bt;
    metadata.slot_mapping = c4_slot_mapping;
    return metadata;
  }

  DeepseekV4IndexerCacheRefs make_cache_refs(
      const torch::Tensor& index_block_table,
      const torch::Tensor& index_slot_mapping,
      int64_t batch_size) {
    std::vector<std::vector<int32_t>> rows;
    rows.reserve(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
      rows.push_back({static_cast<int32_t>(i)});
    }
    torch::Tensor state_block_table = make_block_table(rows).to(device_);
    // kv and score share the single merged compress_index_state block table.
    return {index_block_table, index_slot_mapping, state_block_table};
  }

  DeepseekV4IndexerCacheRefs make_cache_refs(const AttentionMetadata& metadata,
                                             int64_t batch_size) {
    return make_cache_refs(
        metadata.block_table, metadata.slot_mapping, batch_size);
  }

  std::tuple<torch::Tensor, torch::Tensor> make_compressed_tables(
      const DSAMetadata& dsa,
      const torch::Device& device) {
    int64_t table_rows = 1;
    if (dsa.input_positions.defined() && dsa.input_positions.numel() > 0) {
      table_rows = dsa.input_positions.max().item<int64_t>() + 1;
    }
    if (dsa.c4_pad_positions.defined() && dsa.c4_pad_positions.numel() > 0) {
      table_rows =
          std::max(table_rows, dsa.c4_pad_positions.max().item<int64_t>() + 1);
    }
    torch::TensorOptions rope_options =
        torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    return test::make_dsv4_rope_ref(
        table_rows, config_.rope_head_dim, rope_options);
  }

  RefOut reference(
      const torch::Tensor& x,
      const torch::Tensor& qr,
      const std::unordered_map<std::string, torch::Tensor>& weights,
      torch::Tensor& kv_state,
      torch::Tensor& score_state,
      const AttentionMetadata& metadata,
      const std::vector<torch::Tensor>& history_kv = {}) {
    const DSAMetadata& dsa = *metadata.dsa_metadata;
    torch::Tensor hadamard =
        util::create_hadamard_matrix(config_.index_head_dim,
                                     torch::kFloat32,
                                     torch::Device(torch::kCPU),
                                     /*normalize=*/true)
            .to(torch::kBFloat16);
    torch::Tensor q =
        torch::nn::functional::linear(qr, weights.at("wq_b.weight"))
            .view({qr.size(0), config_.index_n_heads, config_.index_head_dim});
    auto [input_sin_table, input_cos_table] =
        make_compressed_tables(dsa, torch::Device(torch::kCPU));
    torch::Tensor input_positions = dsa.input_positions.cpu().to(torch::kLong);
    torch::Tensor input_sin =
        input_positions.numel() == 0
            ? torch::empty({0, config_.rope_head_dim}, cpu_options_)
            : input_sin_table.index_select(/*dim=*/0, input_positions);
    torch::Tensor input_cos =
        input_positions.numel() == 0
            ? torch::empty({0, config_.rope_head_dim}, cpu_options_)
            : input_cos_table.index_select(/*dim=*/0, input_positions);
    test::apply_dsv4_rotary_ref(q,
                                input_sin.to(torch::kBFloat16),
                                input_cos.to(torch::kBFloat16),
                                config_.rope_head_dim);
    q = util::rotate_activation(q.to(torch::kBFloat16), hadamard);
    torch::Tensor c4_positions = dsa.c4_pad_positions.cpu().to(torch::kLong);
    torch::Tensor c4_sin =
        c4_positions.numel() == 0
            ? torch::empty({0, config_.rope_head_dim}, cpu_options_)
            : input_sin_table.index_select(/*dim=*/0, c4_positions);
    torch::Tensor c4_cos =
        c4_positions.numel() == 0
            ? torch::empty({0, config_.rope_head_dim}, cpu_options_)
            : input_cos_table.index_select(/*dim=*/0, c4_positions);
    test::Dsv4CompressorRefWeights ref_weights{
        weights.at("compressor.wkv.weight"),
        weights.at("compressor.wgate.weight"),
        weights.at("compressor.norm.weight"),
        weights.at("compressor.ape")};
    test::Dsv4CompressorRefConfig ref_config{kCompressRatio,
                                             config_.index_head_dim,
                                             config_.rope_head_dim,
                                             true,
                                             config_.norm_eps};
    test::Dsv4CompressorRefResult kv_ref =
        test::dsv4_compressor_ref(x,
                                  ref_weights,
                                  kv_state,
                                  score_state,
                                  dsa.start_pos_vec,
                                  dsa.query_start_offsets,
                                  c4_sin,
                                  c4_cos,
                                  hadamard,
                                  ref_config);
    torch::Tensor kv = kv_ref.output;
    std::vector<torch::Tensor> all_kv = history_kv;
    if (kv.numel() > 0) {
      all_kv.emplace_back(kv);
    }
    torch::Tensor full_kv =
        all_kv.empty() ? torch::empty({0, config_.index_head_dim}, cpu_options_)
                       : torch::cat(all_kv, 0);
    torch::Tensor weights_proj =
        torch::nn::functional::linear(x, weights.at("weights_proj.weight"));
    TopkRefOut topk = topk_ref(q,
                               weights_proj,
                               full_kv,
                               dsa.start_pos_vec,
                               dsa.query_start_offsets,
                               config_.index_topk);
    return {q, kv, weights_proj, topk.topk, topk.context_lens};
  }

  // Returns [index_cache, compress_index_state] where compress_index_state is
  // the owning merged tensor of shape [batch, 8, 4*index_head_dim] with kv in
  // the [0, 2*index_head_dim) lanes and score in the [2*index_head_dim,
  // 4*index_head_dim) lanes (idx_coff_dim = 2*index_head_dim). The indexer
  // forward consumes the owning tensor directly; the reference path uses
  // split_index_state() to obtain narrow views of the same storage.
  std::tuple<torch::Tensor, torch::Tensor> make_states(int64_t batch_size) {
    torch::Tensor index_cache =
        torch::zeros({16, 1, 1, config_.index_head_dim}, options_);
    const int64_t idx_coff_dim = 2 * config_.index_head_dim;
    torch::Tensor compress_index_state = torch::zeros(
        {batch_size, 8, 2 * idx_coff_dim}, options_.dtype(torch::kFloat32));
    compress_index_state
        .narrow(/*dim=*/2,
                /*start=*/idx_coff_dim,
                /*length=*/idx_coff_dim)
        .fill_(-std::numeric_limits<float>::infinity());
    return {index_cache, compress_index_state};
  }

  // Returns [kv, score] narrow views of the owning compress_index_state,
  // sharing its storage. Used to feed the CPU reference, which expects split
  // pools.
  std::tuple<torch::Tensor, torch::Tensor> split_index_state(
      torch::Tensor& compress_index_state) const {
    const int64_t idx_coff_dim = 2 * config_.index_head_dim;
    torch::Tensor kv = compress_index_state.narrow(
        /*dim=*/2, /*start=*/0, /*length=*/idx_coff_dim);
    torch::Tensor score = compress_index_state.narrow(
        /*dim=*/2, /*start=*/idx_coff_dim, /*length=*/idx_coff_dim);
    return {kv, score};
  }

  TestConfig config_;
  torch::Device device_{torch::kCPU};
  torch::TensorOptions options_;
  torch::TensorOptions cpu_options_;
  std::vector<std::vector<DSACacheInfo>> caches_info_;
};

TEST_F(DeepseekV4IndexerTest, PrefillReturnsKernelSparseSlots) {
  const std::vector<int64_t> q_lens = {8, 8};
  const std::vector<int64_t> start_pos = {0, 0};
  AttentionMetadata metadata =
      make_metadata(q_lens, start_pos, {{1, 2}, {3, 4}}, {1, 2, 3, 4}, false);
  std::unordered_map<std::string, torch::Tensor> weights = make_weights();
  torch::Tensor x =
      seeded("dsv4.indexer.prefill.x", {16, config_.dim}, torch::kBFloat16);
  // Keep the two-token topk rows away from ReLU/tie-boundary ordering.
  x[15].add_(0.5);
  torch::Tensor qr = seeded(
      "dsv4.indexer.prefill.qr", {16, config_.q_lora_rank}, torch::kBFloat16);
  qr[7].add_(0.5);
  qr[15].add_(0.5);
  torch::Tensor ref_kv_state =
      torch::zeros({2, 8, 2 * config_.index_head_dim}, cpu_options_);
  torch::Tensor ref_score_state =
      torch::full({2, 8, 2 * config_.index_head_dim},
                  -std::numeric_limits<float>::infinity(),
                  torch::TensorOptions().dtype(torch::kFloat32));
  RefOut expected =
      reference(x, qr, weights, ref_kv_state, ref_score_state, metadata);

  auto [index_cache, compress_index_state] = make_states(2);
  DeepseekV4IndexerCacheRefs cache_refs = make_cache_refs(metadata, 2);
  auto [compressed_sin, compressed_cos] =
      make_compressed_tables(*metadata.dsa_metadata, device_);
  DeepseekV4Indexer indexer =
      DeepseekV4Indexer(DeepseekV4IndexerImpl(config_.dim,
                                              config_.index_n_heads,
                                              config_.index_head_dim,
                                              config_.rope_head_dim,
                                              config_.index_topk,
                                              config_.q_lora_rank,
                                              config_.norm_eps,
                                              options_));
  indexer->load_state_dict(make_state_dict(weights));
  auto [actual_topk, context_lens] = indexer->forward(x.to(device_),
                                                      qr.to(device_),
                                                      index_cache,
                                                      compress_index_state,
                                                      metadata,
                                                      cache_refs,
                                                      true,
                                                      compressed_sin,
                                                      compressed_cos);

  test::verify_tensor_close(
      index_cache.cpu().slice(0, 1, 5).squeeze(1).squeeze(1).to(
          torch::kFloat32),
      expected.kv.to(torch::kFloat32),
      /*rtol=*/2e-2,
      /*atol=*/2e-2);
  EXPECT_EQ(actual_topk.dim(), 2);
  EXPECT_EQ(actual_topk.size(0), 16);
  EXPECT_EQ(actual_topk.size(1), config_.index_topk);
  expect_context_lens_shape(context_lens, 16);
  torch::Tensor expected_sparse = sparse_slot_ref(expected.topk,
                                                  expected.context_lens,
                                                  metadata,
                                                  cache_refs.index_block_table);
  expect_topk_prefix_set_equal(
      actual_topk, expected_sparse, context_lens, expected.context_lens);
}

TEST_F(DeepseekV4IndexerTest, UsesExplicitC4RefsWithDuplicateTokenCaches) {
  AttentionMetadata metadata = make_metadata({8}, {0}, {{0, 1}}, {0, 1}, false);
  torch::Tensor selected_block_table = make_block_table({{3, 4}}).to(device_);
  torch::Tensor selected_slot_mapping =
      torch::tensor({3, 4}, torch::TensorOptions().dtype(torch::kInt32))
          .to(device_);
  DeepseekV4IndexerCacheRefs cache_refs =
      make_cache_refs(selected_block_table, selected_slot_mapping, 1);

  metadata.dsa_metadata->block_tables = {{metadata.block_table,
                                          selected_block_table,
                                          cache_refs.index_state_block_table}};
  metadata.dsa_metadata->slot_mappings = {
      {metadata.slot_mapping, selected_slot_mapping, torch::Tensor()}};
  caches_info_ = {{{0, DSACacheType::TOKEN, 4, 1},
                   {0, DSACacheType::TOKEN, 4, 1},
                   {1, DSACacheType::SLIDING_WINDOW, 1, 8}}};
  metadata.dsa_metadata->caches_info = &caches_info_;

  std::unordered_map<std::string, torch::Tensor> weights = make_weights();
  torch::Tensor x =
      seeded("dsv4.indexer.dup_token.x", {8, config_.dim}, torch::kBFloat16);
  x[7].add_(0.5);
  torch::Tensor qr = seeded(
      "dsv4.indexer.dup_token.qr", {8, config_.q_lora_rank}, torch::kBFloat16);
  qr[7].add_(0.5);
  torch::Tensor ref_kv_state =
      torch::zeros({1, 8, 2 * config_.index_head_dim}, cpu_options_);
  torch::Tensor ref_score_state =
      torch::full({1, 8, 2 * config_.index_head_dim},
                  -std::numeric_limits<float>::infinity(),
                  torch::TensorOptions().dtype(torch::kFloat32));
  RefOut expected =
      reference(x, qr, weights, ref_kv_state, ref_score_state, metadata);

  auto [index_cache, compress_index_state] = make_states(1);
  auto [compressed_sin, compressed_cos] =
      make_compressed_tables(*metadata.dsa_metadata, device_);
  DeepseekV4Indexer indexer =
      DeepseekV4Indexer(DeepseekV4IndexerImpl(config_.dim,
                                              config_.index_n_heads,
                                              config_.index_head_dim,
                                              config_.rope_head_dim,
                                              config_.index_topk,
                                              config_.q_lora_rank,
                                              config_.norm_eps,
                                              options_));
  indexer->load_state_dict(make_state_dict(weights));
  auto [actual_topk, context_lens] = indexer->forward(x.to(device_),
                                                      qr.to(device_),
                                                      index_cache,
                                                      compress_index_state,
                                                      metadata,
                                                      cache_refs,
                                                      true,
                                                      compressed_sin,
                                                      compressed_cos);

  torch::Tensor expected_sparse = sparse_slot_ref(
      expected.topk, expected.context_lens, metadata, selected_block_table);
  expect_topk_prefix_set_equal(
      actual_topk, expected_sparse, context_lens, expected.context_lens);
  test::verify_tensor_close(index_cache.cpu()[3][0][0].to(torch::kFloat32),
                            expected.kv[0].to(torch::kFloat32),
                            /*rtol=*/2e-2,
                            /*atol=*/2e-2);
  torch::Tensor wrong_sparse = sparse_slot_ref(
      expected.topk, expected.context_lens, metadata, metadata.block_table);
  EXPECT_EQ(context_lens.cpu()[3].item<int32_t>(), 1);
  EXPECT_EQ(actual_topk.cpu()[3][0].item<int32_t>(),
            expected_sparse[3][0].item<int32_t>());
  EXPECT_NE(actual_topk.cpu()[3][0].item<int32_t>(),
            wrong_sparse[3][0].item<int32_t>());
}

TEST_F(DeepseekV4IndexerTest, DecodeReturnsKernelSparseSlots) {
  std::unordered_map<std::string, torch::Tensor> weights = make_weights();
  auto [index_cache, compress_index_state] = make_states(1);
  DeepseekV4IndexerCacheRefs warmup_refs = make_cache_refs(
      make_block_table({{1}}).to(device_),
      torch::empty({0},
                   torch::TensorOptions().dtype(torch::kInt32).device(device_)),
      1);
  DeepseekV4Indexer indexer =
      DeepseekV4Indexer(DeepseekV4IndexerImpl(config_.dim,
                                              config_.index_n_heads,
                                              config_.index_head_dim,
                                              config_.rope_head_dim,
                                              config_.index_topk,
                                              config_.q_lora_rank,
                                              config_.norm_eps,
                                              options_));
  indexer->load_state_dict(make_state_dict(weights));

  AttentionMetadata warmup_metadata = make_metadata({3}, {0}, {{1}}, {}, false);
  auto [warmup_sin, warmup_cos] =
      make_compressed_tables(*warmup_metadata.dsa_metadata, device_);
  torch::Tensor warmup_x =
      seeded("dsv4.indexer.decode.warm.x", {3, config_.dim}, torch::kBFloat16);
  torch::Tensor warmup_qr = seeded("dsv4.indexer.decode.warm.qr",
                                   {3, config_.q_lora_rank},
                                   torch::kBFloat16);
  indexer->forward(warmup_x.to(device_),
                   warmup_qr.to(device_),
                   index_cache,
                   compress_index_state,
                   warmup_metadata,
                   warmup_refs,
                   true,
                   warmup_sin,
                   warmup_cos);

  AttentionMetadata metadata = make_metadata({1}, {3}, {{1}}, {1}, false);
  const int64_t batch_size = 1;
  DeepseekV4IndexerCacheRefs cache_refs = make_cache_refs(metadata, batch_size);
  auto [compressed_sin, compressed_cos] =
      make_compressed_tables(*metadata.dsa_metadata, device_);
  metadata.is_prefill = false;
  torch::Tensor x =
      seeded("dsv4.indexer.decode.x", {1, config_.dim}, torch::kBFloat16);
  torch::Tensor qr = seeded(
      "dsv4.indexer.decode.qr", {1, config_.q_lora_rank}, torch::kBFloat16);
  auto [kv_view, score_view] = split_index_state(compress_index_state);
  torch::Tensor ref_kv_state = kv_view.cpu().clone();
  torch::Tensor ref_score_state = score_view.cpu().clone();
  int64_t ratio = 4;
  ref_kv_state.slice(/*dim=*/1, /*start=*/ratio)
      .copy_(ref_kv_state.slice(/*dim=*/1, /*start=*/0, /*length=*/ratio));
  ref_score_state.slice(/*dim=*/1, /*start=*/ratio)
      .copy_(ref_score_state.slice(
          /*dim=*/1, /*start=*/0, /*length=*/ratio));
  RefOut expected =
      reference(x, qr, weights, ref_kv_state, ref_score_state, metadata);

  auto [actual_topk, context_lens] = indexer->forward(x.to(device_),
                                                      qr.to(device_),
                                                      index_cache,
                                                      compress_index_state,
                                                      metadata,
                                                      cache_refs,
                                                      false,
                                                      compressed_sin,
                                                      compressed_cos);
  test::verify_tensor_close(index_cache.cpu()[1][0][0].to(torch::kFloat32),
                            expected.kv[0].to(torch::kFloat32),
                            /*rtol=*/2e-2,
                            /*atol=*/2e-2);
  EXPECT_EQ(actual_topk.dim(), 2);
  EXPECT_EQ(actual_topk.size(0), 1);
  EXPECT_EQ(actual_topk.size(1), config_.index_topk);
  expect_context_lens_shape(context_lens, 1);
  torch::Tensor expected_sparse = sparse_slot_ref(expected.topk,
                                                  expected.context_lens,
                                                  metadata,
                                                  cache_refs.index_block_table);
  expect_topk_prefix_set_equal(
      actual_topk, expected_sparse, context_lens, expected.context_lens);
}

TEST_F(DeepseekV4IndexerTest, ChunkedPrefillReturnsKernelSparseSlots) {
  std::unordered_map<std::string, torch::Tensor> weights = make_weights();
  auto [index_cache, compress_index_state] = make_states(1);
  DeepseekV4IndexerCacheRefs history_refs = make_cache_refs(
      make_block_table({{1, 2}}).to(device_),
      torch::tensor(
          {1}, torch::TensorOptions().dtype(torch::kInt32).device(device_)),
      1);
  DeepseekV4Indexer indexer =
      DeepseekV4Indexer(DeepseekV4IndexerImpl(config_.dim,
                                              config_.index_n_heads,
                                              config_.index_head_dim,
                                              config_.rope_head_dim,
                                              config_.index_topk,
                                              config_.q_lora_rank,
                                              config_.norm_eps,
                                              options_));
  indexer->load_state_dict(make_state_dict(weights));

  AttentionMetadata history_metadata =
      make_metadata({5}, {0}, {{1, 2}}, {1}, false);
  auto [history_sin, history_cos] =
      make_compressed_tables(*history_metadata.dsa_metadata, device_);
  torch::Tensor history_x = seeded(
      "dsv4.indexer.chunk.history.x", {5, config_.dim}, torch::kBFloat16);
  torch::Tensor history_qr = seeded("dsv4.indexer.chunk.history.qr",
                                    {5, config_.q_lora_rank},
                                    torch::kBFloat16);
  torch::Tensor ref_kv_state =
      torch::zeros({1, 8, 2 * config_.index_head_dim}, cpu_options_);
  torch::Tensor ref_score_state =
      torch::full({1, 8, 2 * config_.index_head_dim},
                  -std::numeric_limits<float>::infinity(),
                  torch::TensorOptions().dtype(torch::kFloat32));
  RefOut history_ref = reference(history_x,
                                 history_qr,
                                 weights,
                                 ref_kv_state,
                                 ref_score_state,
                                 history_metadata);
  indexer->forward(history_x.to(device_),
                   history_qr.to(device_),
                   index_cache,
                   compress_index_state,
                   history_metadata,
                   history_refs,
                   true,
                   history_sin,
                   history_cos);

  AttentionMetadata metadata = make_metadata({3}, {5}, {{1, 2}}, {2}, true);
  DeepseekV4IndexerCacheRefs cache_refs = make_cache_refs(metadata, 1);
  auto [compressed_sin, compressed_cos] =
      make_compressed_tables(*metadata.dsa_metadata, device_);
  torch::Tensor x =
      seeded("dsv4.indexer.chunk.x", {3, config_.dim}, torch::kBFloat16);
  torch::Tensor qr = seeded(
      "dsv4.indexer.chunk.qr", {3, config_.q_lora_rank}, torch::kBFloat16);
  RefOut expected = reference(x,
                              qr,
                              weights,
                              ref_kv_state,
                              ref_score_state,
                              metadata,
                              {history_ref.kv});
  auto [actual_topk, context_lens] = indexer->forward(x.to(device_),
                                                      qr.to(device_),
                                                      index_cache,
                                                      compress_index_state,
                                                      metadata,
                                                      cache_refs,
                                                      true,
                                                      compressed_sin,
                                                      compressed_cos);

  test::verify_tensor_close(index_cache.cpu()[2][0][0].to(torch::kFloat32),
                            expected.kv[0].to(torch::kFloat32),
                            /*rtol=*/2e-2,
                            /*atol=*/2e-2);
  EXPECT_EQ(actual_topk.dim(), 2);
  EXPECT_EQ(actual_topk.size(0), 3);
  EXPECT_EQ(actual_topk.size(1), config_.index_topk);
  expect_context_lens_shape(context_lens, 3);
  torch::Tensor expected_sparse = sparse_slot_ref(expected.topk,
                                                  expected.context_lens,
                                                  metadata,
                                                  cache_refs.index_block_table);
  expect_topk_prefix_set_equal(
      actual_topk, expected_sparse, context_lens, expected.context_lens);
}

}  // namespace layer
}  // namespace xllm
