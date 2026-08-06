/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

PagedAttentionTilingLayout atb_v1_layout(int64_t buffer_words = 262144,
                                         uint32_t tiling_key = 0) {
  return {/*buffer_words=*/buffer_words,
          /*header_words=*/44,
          /*row_stride_words=*/17,
          /*max_kv_seq_len_offset=*/22,
          /*kv_split_length_offset=*/23,
          /*kv_split_core_count_offset=*/24,
          /*row_kv_seq_len_offset=*/45,
          /*tiling_key=*/tiling_key};
}

std::string read_source(const std::filesystem::path& path) {
  std::ifstream input(path);
  EXPECT_TRUE(input.is_open()) << path;
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

class TileLangSpecVerifyAttentionTilingUpdateTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }
  static void TearDownTestSuite() { torch_npu::finalize_npu(); }

  torch::Tensor run_update(
      const std::vector<int32_t>& kv_seq_lens,
      int64_t spec_width,
      int64_t block_size,
      int64_t max_kv_seq_len,
      int64_t kv_split_core_count,
      const PagedAttentionTilingLayout& layout = atb_v1_layout()) const {
    torch::Tensor tiling_data = torch::full({layout.buffer_words}, -1, i32_);
    tiling_data[layout.kv_split_core_count_offset] = kv_split_core_count;
    const torch::Tensor kv_seq_lens_tensor = torch::tensor(kv_seq_lens, i32_);
    spec_verify_attention_tiling_update(kv_seq_lens_tensor,
                                        tiling_data,
                                        layout,
                                        spec_width,
                                        block_size,
                                        max_kv_seq_len,
                                        kv_split_core_count);
    return tiling_data.cpu();
  }

  void expect_rows(const torch::Tensor& result,
                   const std::vector<int32_t>& kv_seq_lens,
                   const PagedAttentionTilingLayout& layout) const {
    for (size_t row = 0; row < kv_seq_lens.size(); ++row) {
      const int64_t offset =
          layout.row_kv_seq_len_offset +
          static_cast<int64_t>(row) * layout.row_stride_words;
      EXPECT_EQ(result[offset].item<int32_t>(), kv_seq_lens[row]);
    }
  }

  const torch::Device device_{"npu:0"};
  const torch::TensorOptions i32_ =
      torch::TensorOptions().dtype(torch::kInt32).device(device_);
};

TEST_F(TileLangSpecVerifyAttentionTilingUpdateTest,
       ReportsCompiledWidthsAndRuntimeBlockSizes) {
  for (const int64_t spec_width : {4, 5, 6}) {
    for (const int64_t block_size : {7, 16, 32, 48, 64, 128, 256}) {
      EXPECT_TRUE(has_spec_verify_attention_tiling_update_specialization(
          spec_width, block_size));
      EXPECT_EQ(
          has_spec_verify_graph_update_specialization(spec_width, block_size),
          block_size % custom_paged_attention_block_alignment() == 0);
    }
  }
  for (const int64_t spec_width : {3, 7}) {
    EXPECT_FALSE(has_spec_verify_attention_tiling_update_specialization(
        spec_width, /*block_size=*/128));
    EXPECT_FALSE(has_spec_verify_graph_update_specialization(
        spec_width, /*block_size=*/128));
  }
  EXPECT_FALSE(has_spec_verify_attention_tiling_update_specialization(
      /*spec_width=*/0, /*block_size=*/128));
  EXPECT_FALSE(has_spec_verify_attention_tiling_update_specialization(
      /*spec_width=*/4, /*block_size=*/0));
  EXPECT_FALSE(has_spec_verify_attention_tiling_update_specialization(
      std::numeric_limits<int64_t>::max(), /*block_size=*/128));
}

TEST_F(TileLangSpecVerifyAttentionTilingUpdateTest, UpdatesDynamicKvLengths) {
  const std::vector<int32_t> kv_seq_lens = {125, 126, 127, 128, 129, 190};
  const torch::Tensor result = run_update(kv_seq_lens,
                                          /*spec_width=*/6,
                                          /*block_size=*/128,
                                          /*max_kv_seq_len=*/190,
                                          /*kv_split_core_count=*/2);

  EXPECT_EQ(result[22].item<int32_t>(), 190);
  EXPECT_EQ(result[23].item<int32_t>(), 128);
  expect_rows(result, kv_seq_lens, atb_v1_layout());
  EXPECT_EQ(result[21].item<int32_t>(), -1);
  EXPECT_EQ(result[24].item<int32_t>(), 2);
}

TEST_F(TileLangSpecVerifyAttentionTilingUpdateTest,
       SupportsMtp3ThroughMtp5Widths) {
  struct WidthCase {
    const char* name;
    int64_t spec_width;
    std::vector<int32_t> kv_seq_lens;
    int32_t expected_max_kv_seq_len;
    int32_t expected_split_length;
  };
  const std::vector<WidthCase> test_cases = {
      {"mtp3", 4, {125, 126, 127, 128}, 128, 128},
      {"mtp4", 5, {125, 126, 127, 128, 129}, 129, 256},
      {"mtp5", 6, {125, 126, 127, 128, 129, 130}, 130, 256},
  };

  for (const WidthCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    const torch::Tensor result = run_update(test_case.kv_seq_lens,
                                            test_case.spec_width,
                                            /*block_size=*/128,
                                            test_case.expected_max_kv_seq_len,
                                            /*kv_split_core_count=*/1);
    EXPECT_EQ(result[22].item<int32_t>(), test_case.expected_max_kv_seq_len);
    EXPECT_EQ(result[23].item<int32_t>(), test_case.expected_split_length);
    expect_rows(result, test_case.kv_seq_lens, atb_v1_layout());
  }
}

TEST_F(TileLangSpecVerifyAttentionTilingUpdateTest,
       SupportsRuntimeKvCacheBlockSizes) {
  struct BlockSizeCase {
    int64_t block_size;
    int32_t expected_split_length;
  };
  const std::vector<BlockSizeCase> test_cases = {
      {7, 98},
      {16, 96},
      {32, 96},
      {48, 96},
      {64, 128},
      {128, 128},
      {256, 256},
  };
  const std::vector<int32_t> kv_seq_lens = {125, 126, 127, 190};

  for (const BlockSizeCase& test_case : test_cases) {
    SCOPED_TRACE(::testing::Message() << "block_size=" << test_case.block_size);
    const torch::Tensor result = run_update(kv_seq_lens,
                                            /*spec_width=*/4,
                                            test_case.block_size,
                                            /*max_kv_seq_len=*/190,
                                            /*kv_split_core_count=*/2);
    EXPECT_EQ(result[22].item<int32_t>(), 190);
    EXPECT_EQ(result[23].item<int32_t>(), test_case.expected_split_length);
  }
}

TEST_F(TileLangSpecVerifyAttentionTilingUpdateTest,
       PreservesTemplateKvCorePartitioning) {
  const torch::Tensor result = run_update({5022, 5023, 5024, 5025},
                                          /*spec_width=*/4,
                                          /*block_size=*/128,
                                          /*max_kv_seq_len=*/5025,
                                          /*kv_split_core_count=*/6);

  EXPECT_EQ(result[22].item<int32_t>(), 5025);
  EXPECT_EQ(result[23].item<int32_t>(), 896);
  EXPECT_EQ(result[24].item<int32_t>(), 6);
}

TEST_F(TileLangSpecVerifyAttentionTilingUpdateTest,
       UpdatesMultipleSequenceRows) {
  const std::vector<int32_t> kv_seq_lens = {
      125, 126, 127, 128, 205, 206, 207, 208};
  const torch::Tensor result = run_update(kv_seq_lens,
                                          /*spec_width=*/4,
                                          /*block_size=*/128,
                                          /*max_kv_seq_len=*/208,
                                          /*kv_split_core_count=*/2);

  EXPECT_EQ(result[22].item<int32_t>(), 208);
  EXPECT_EQ(result[23].item<int32_t>(), 128);
  expect_rows(result, kv_seq_lens, atb_v1_layout());
}

TEST_F(TileLangSpecVerifyAttentionTilingUpdateTest,
       UsesBackendProvidedOffsetsAndRowStride) {
  const PagedAttentionTilingLayout layout{/*buffer_words=*/512,
                                          /*header_words=*/50,
                                          /*row_stride_words=*/19,
                                          /*max_kv_seq_len_offset=*/30,
                                          /*kv_split_length_offset=*/31,
                                          /*kv_split_core_count_offset=*/32,
                                          /*row_kv_seq_len_offset=*/53,
                                          /*tiling_key=*/7};
  const std::vector<int32_t> kv_seq_lens = {301, 302, 303, 350};
  const torch::Tensor result = run_update(kv_seq_lens,
                                          /*spec_width=*/4,
                                          /*block_size=*/48,
                                          /*max_kv_seq_len=*/350,
                                          /*kv_split_core_count=*/2,
                                          layout);

  EXPECT_EQ(result[30].item<int32_t>(), 350);
  EXPECT_EQ(result[31].item<int32_t>(), 192);
  expect_rows(result, kv_seq_lens, layout);
  EXPECT_EQ(result[22].item<int32_t>(), -1);
  EXPECT_EQ(result[45].item<int32_t>(), -1);
}

TEST(PagedAttentionTilingLayoutTest, RejectsUnknownOrTruncatedAtbLayouts) {
  EXPECT_FALSE(parse_custom_paged_attention_tiling_layout({}).has_value());
  EXPECT_FALSE(
      parse_custom_paged_attention_tiling_layout(std::vector<uint32_t>(18, 0))
          .has_value());

  std::vector<uint32_t> words(128, 0);
  words[16] = 9;
  words[17] = 44;
  words[18] = 17;
  const auto layout = parse_custom_paged_attention_tiling_layout(words);
  ASSERT_TRUE(layout.has_value());
  EXPECT_EQ(*layout, atb_v1_layout(/*buffer_words=*/128, /*tiling_key=*/9));
  EXPECT_TRUE(paged_attention_tiling_required_words(*layout, /*num_rows=*/4)
                  .has_value());
  EXPECT_FALSE(paged_attention_tiling_required_words(*layout, /*num_rows=*/6)
                   .has_value());

  words[17] = 50;
  EXPECT_FALSE(parse_custom_paged_attention_tiling_layout(words).has_value());
  words[17] = 44;
  words[18] = 19;
  EXPECT_FALSE(parse_custom_paged_attention_tiling_layout(words).has_value());
}

TEST(PagedAttentionTilingLayoutTest, MatchesPinnedXllmOpsSourceContract) {
  const std::filesystem::path tiling_dir =
      std::filesystem::weakly_canonical(
          std::filesystem::path(__FILE__).parent_path() / "../../../../..") /
      "third_party/xllm_ops/atb_customize/ops/custom_paged_attention/"
      "kernel_implement/tiling";
  const std::string header =
      read_source(tiling_dir / "custom_paged_attention_tiling_dependency.h");
  const std::string source =
      read_source(tiling_dir / "custom_paged_attention_tiling_dependency.cpp");
  const std::string tiling_source =
      read_source(tiling_dir / "custom_paged_attention_tiling.cpp");
  EXPECT_NE(header.find(R"(constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t BLOCK_SIZE_32 = 32;
constexpr int32_t TILING_PARA_SIZE = 17;
constexpr int32_t TILING_HEAD_SIZE = 44;)"),
            std::string::npos);
  EXPECT_NE(source.find(R"(const int32_t TILING_KEY = 16;
const int32_t TILING_HEADSIZE = 17;
const int32_t TILING_PARASIZE = 18;)"),
            std::string::npos);
  EXPECT_NE(source.find(R"(const int32_t TILING_MAX_KVSEQLEN = 22;
const int32_t TILING_KVSPLIT = 23;
const int32_t TILING_KVCORENUM = 24;)"),
            std::string::npos);
  EXPECT_NE(source.find("tilingParam[tilingOffset + 1] = "
                        "static_cast<uint32_t>(kvSeqlen);"),
            std::string::npos);
  EXPECT_NE(tiling_source.find("mmInfo.blockSize % BLOCK_SIZE == 0"),
            std::string::npos);
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
