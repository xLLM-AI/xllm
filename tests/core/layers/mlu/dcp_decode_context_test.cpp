/* Copyright 2026 The xLLM Authors.

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

#include "layers/mlu/dcp_decode_context.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/kv_cache/layout/kv_shard_layout.h"
#include "layers/mlu/dsa_topk_state.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm::layer {
namespace {

TEST(DcpDecodeContextTest, LocalizesOnlyOwnedCacheWriteSlots) {
  const DcpDecodeContext context(
      KVShardLayout(/*physical_block_size=*/4, /*dcp_size=*/2, /*dcp_rank=*/1),
      /*dcp_group=*/nullptr);
  torch::Tensor global_slots = torch::tensor({-1, 0, 3, 4, 7, 8, 12});

  torch::Tensor local_slots = context.localize_slots(global_slots);

  EXPECT_TRUE(
      torch::equal(local_slots, torch::tensor({-1, -1, -1, 0, 3, -1, 4})));
}

TEST(DcpDecodeContextTest, PacksOwnedTopkAndUpdatesEachContextLength) {
  const torch::Device device(Platform::type_torch(), 0);
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  const DcpDecodeContext context(
      KVShardLayout(/*physical_block_size=*/4, /*dcp_size=*/2, /*dcp_rank=*/0),
      /*dcp_group=*/nullptr);
  DsaTopkState global_state(
      torch::tensor({{4, 0, 8, 1, 12}, {5, 6, 7, 2, 3}}, options),
      torch::tensor({4, 3}, options));

  DsaTopkState local_state = context.localize_topk(global_state);
  Device(device).synchronize_default_stream();

  EXPECT_TRUE(torch::equal(
      local_state.block_tables().cpu(),
      torch::tensor({{0, 4, 1, 0, 0}, {0, 0, 0, 0, 0}}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(local_state.context_lens().cpu(),
                           torch::tensor({3, 0}, torch::kInt32)));
}

TEST(DcpDecodeContextTest, DcpOnePreservesValidTopkEntries) {
  const torch::Device device(Platform::type_torch(), 0);
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  const DcpDecodeContext context(
      KVShardLayout(/*physical_block_size=*/4, /*dcp_size=*/1, /*dcp_rank=*/0),
      /*dcp_group=*/nullptr);
  DsaTopkState global_state(torch::tensor({{7, 3, 9, 11}}, options),
                            torch::tensor({2}, options));

  DsaTopkState local_state = context.localize_topk(global_state);
  Device(device).synchronize_default_stream();

  EXPECT_TRUE(torch::equal(local_state.block_tables().cpu(),
                           torch::tensor({{7, 3, 0, 0}}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(local_state.context_lens().cpu(),
                           torch::tensor({2}, torch::kInt32)));
}

TEST(DcpDecodeContextTest, FourRanksLocalizeBlockBoundariesAndEmptyRows) {
  Device dev(0);
  dev.set_device();
  const torch::Device device = dev.unwrap();
  torch::Tensor table = torch::full({3, 2048}, -1, torch::kInt32);
  table[0].slice(0, 0, 9).copy_(
      torch::tensor({0, 15, 16, 31, 32, 47, 48, 63, 64}, torch::kInt32));
  table[1][0] = 1000000;  // Ignored because this row has zero valid entries.
  table[2][0] = -1;       // Padding inside a nonempty row.
  const torch::Tensor lens = torch::tensor({9, 0, 1}, torch::kInt32);
  for (int32_t rank = 0; rank < 4; ++rank) {
    DcpDecodeContext context(KVShardLayout(16, 4, rank), nullptr);
    const DsaTopkState result =
        context.localize_topk(DsaTopkState(table.to(device), lens.to(device)));
    torch::Tensor expected = torch::zeros_like(table);
    torch::Tensor expected_lens = torch::zeros_like(lens);
    auto out = expected.accessor<int32_t, 2>();
    auto counts = expected_lens.accessor<int32_t, 1>();
    const auto input = table.accessor<int32_t, 2>();
    for (int32_t col = 0; col < 9; ++col) {
      const int32_t slot = input[0][col];
      if ((slot / 16) % 4 != rank) {
        continue;
      }
      out[0][counts[0]++] = slot / 64 * 16 + slot % 16;
    }
    dev.synchronize_default_stream();
    EXPECT_TRUE(torch::equal(result.block_tables().cpu(), expected));
    EXPECT_TRUE(torch::equal(result.context_lens().cpu(), expected_lens));
  }
}

}  // namespace
}  // namespace xllm::layer
