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

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

class TileLangSpecVerifyAttentionTilingUpdateTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }
  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

TEST_F(TileLangSpecVerifyAttentionTilingUpdateTest, UpdatesDynamicKvLengths) {
  const torch::Device device("npu:0");
  const torch::TensorOptions i32 =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  const torch::Tensor kv_seq_lens =
      torch::tensor({125, 126, 127, 128, 129, 190}, i32);
  torch::Tensor tiling_data = torch::full({262144}, -1, i32);

  spec_verify_attention_tiling_update(
      kv_seq_lens, tiling_data, /*block_size=*/128);

  const torch::Tensor result = tiling_data.cpu();
  EXPECT_EQ(result[22].item<int32_t>(), 190);
  EXPECT_EQ(result[23].item<int32_t>(), 256);
  for (int64_t row = 0; row < 6; ++row) {
    const int64_t offset = 45 + row * 17;
    EXPECT_EQ(result[offset].item<int32_t>(), 125 + row + (row == 5 ? 60 : 0));
  }
  EXPECT_EQ(result[21].item<int32_t>(), -1);
  EXPECT_EQ(result[24].item<int32_t>(), -1);
}

TEST_F(TileLangSpecVerifyAttentionTilingUpdateTest, SupportsMtp3AndMtp4Widths) {
  const torch::Device device("npu:0");
  const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
  for (const int64_t spec_width : {4, 5}) {
    const auto kv_seq_lens = torch::arange(125, 125 + spec_width, i32);
    auto tiling_data = torch::full({262144}, -1, i32);

    spec_verify_attention_tiling_update(
        kv_seq_lens, tiling_data, /*block_size=*/128);

    const auto result = tiling_data.cpu();
    const int32_t max_kv = 124 + static_cast<int32_t>(spec_width);
    EXPECT_EQ(result[22].item<int32_t>(), max_kv);
    EXPECT_EQ(result[23].item<int32_t>(), ((max_kv + 127) / 128) * 128);
    for (int64_t row = 0; row < spec_width; ++row) {
      EXPECT_EQ(result[45 + row * 17].item<int32_t>(), 125 + row);
    }
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
