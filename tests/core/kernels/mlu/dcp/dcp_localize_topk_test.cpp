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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include "core/kernels/mlu/dcp/dcp_topk_test_utils.h"
#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {

using test::GuardedTopk;
using test::kSentinel;
using test::local_reference;

using kernel::mlu::dcp_localize_topk;

class DcpLocalizeTopkTest : public ::testing::Test {
 protected:
  const torch::Device device{torch::kPrivateUse1, 0};

  void check_case(int64_t q,
                  int64_t width,
                  int32_t size,
                  int32_t physical_block_size,
                  bool strided = false) {
    SCOPED_TRACE(::testing::Message()
                 << "Q=" << q << " width=" << width << " size=" << size
                 << " physical_block_size=" << physical_block_size
                 << " strided=" << strided);
    const torch::DeviceGuard guard(device);
    auto cpu_slots = torch::empty({9, width}, torch::kInt32);
    auto cpu_lens = torch::empty({9}, torch::kInt32);
    auto values = cpu_slots.accessor<int32_t, 2>();
    auto lens = cpu_lens.accessor<int32_t, 1>();
    const std::vector<int32_t> edges{
        0, 15, 16, 63, 64, 127, 128, -1, std::numeric_limits<int32_t>::max()};
    for (int64_t row = 0; row < 9; ++row) {
      lens[row] =
          static_cast<int32_t>(row % 3 == 0 ? width : (row * 73) % (width + 1));
      for (int64_t col = 0; col < width; ++col) {
        const int64_t owned =
            col / physical_block_size * size * physical_block_size +
            col % physical_block_size;
        switch (row % 5) {
          case 0:
            values[row][col] = static_cast<int32_t>(owned);
            break;
          case 1:
            values[row][col] = edges[col % edges.size()];
            break;
          case 2:
            values[row][col] = col % 7 == 0 ? -1 : static_cast<int32_t>(col);
            break;
          case 3:
            values[row][col] = 0;
            break;
          default:
            values[row][col] = static_cast<int32_t>((col * 7919) % 131072);
            break;
        }
      }
    }
    const auto selector = torch::arange(q, torch::kInt64).remainder(9);
    const auto expanded_slots = cpu_slots.index_select(0, selector);
    auto slots = expanded_slots.to(device);
    if (strided) {
      slots = torch::full({width * 2, q * 2 + 3}, kSentinel, slots.options())
                  .slice(0, 0, width * 2, 2)
                  .slice(1, 1, q * 2, 2)
                  .transpose(0, 1);
      slots.copy_(expanded_slots);
    }
    const auto input_before = slots.cpu();
    GuardedTopk out(q, width, device, strided);
    for (int32_t rank = 0; rank < size; ++rank) {
      // Reusing allocations with full/empty/mixed lengths exposes stale tails.
      for (int32_t pass : {0, 1, 2, 0}) {
        const auto active = pass == 1   ? torch::zeros_like(cpu_lens)
                            : pass == 2 ? torch::full_like(cpu_lens, width)
                                        : cpu_lens;
        const auto expanded_lengths = active.index_select(0, selector);
        const auto lengths = expanded_lengths.to(device);
        dcp_localize_topk(slots,
                          lengths,
                          rank,
                          size,
                          physical_block_size,
                          out.output,
                          out.lengths);
        const auto expected =
            local_reference(cpu_slots, active, rank, size, physical_block_size);
        out.check({expected.first.index_select(0, selector),
                   expected.second.index_select(0, selector)});
        EXPECT_TRUE(torch::equal(lengths.cpu(), expanded_lengths));
      }
    }
    EXPECT_TRUE(torch::equal(slots.cpu(), input_before));
  }
};

TEST_F(DcpLocalizeTopkTest, OwnershipEdgesAndGenericShapes) {
  for (const auto& [size, physical_block_size, width] :
       std::vector<std::tuple<int32_t, int32_t, int64_t>>{
           {1, 16, 1}, {4, 16, 2048}, {8, 16, 2048}, {4, 7, 259}, {3, 5, 17}}) {
    for (bool strided : {false, true}) {
      check_case(/*q=*/9, width, size, physical_block_size, strided);
    }
  }
  check_case(/*q=*/0,
             /*width=*/17,
             /*size=*/4,
             /*physical_block_size=*/16);
}

TEST_F(DcpLocalizeTopkTest, BatchAndStageBoundaries) {
  for (int64_t q :
       {127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025}) {
    check_case(q,
               /*width=*/2048,
               /*size=*/4,
               /*physical_block_size=*/16);
  }
}

TEST_F(DcpLocalizeTopkTest, LargeQueriesAndStridedTails) {
  for (int64_t q : {2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193}) {
    check_case(q,
               /*width=*/2048,
               /*size=*/4,
               /*physical_block_size=*/16);
  }
  check_case(/*q=*/2049,
             /*width=*/2048,
             /*size=*/4,
             /*physical_block_size=*/16,
             /*strided=*/true);
  check_case(/*q=*/8193,
             /*width=*/2048,
             /*size=*/8,
             /*physical_block_size=*/16,
             /*strided=*/true);
}

TEST_F(DcpLocalizeTopkTest, LargeTopologyProductUsesWideArithmetic) {
  const torch::DeviceGuard guard(device);
  const auto cpu_slots = torch::tensor({{0, 1, 2147483647, -1}}, torch::kInt32);
  const auto cpu_lens = torch::tensor({4}, torch::kInt32);
  const auto slots = cpu_slots.to(device);
  const auto lengths = cpu_lens.to(device);
  GuardedTopk out(/*q=*/1, /*k=*/4, device);
  for (int32_t rank : {0, 1, 2147483646}) {
    dcp_localize_topk(slots,
                      lengths,
                      rank,
                      /*size=*/2147483647,
                      /*physical_block_size=*/2147483647,
                      out.output,
                      out.lengths);
    out.check(local_reference(cpu_slots,
                              cpu_lens,
                              rank,
                              /*size=*/2147483647,
                              /*physical_block_size=*/2147483647));
  }
}

TEST(DcpLocalizeTopkDeathTest, RejectsInvalidContracts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const torch::Device device{torch::kPrivateUse1, 0};
  const auto opts = torch::TensorOptions().device(device).dtype(torch::kInt32);
  auto slots = torch::zeros({2, 4}, opts);
  auto lengths = torch::zeros({2}, opts);
  auto output = torch::empty_like(slots);
  auto output_lengths = torch::empty_like(lengths);
  EXPECT_DEATH(
      dcp_localize_topk(slots, lengths, 4, 4, 16, output, output_lengths),
      "rank");
  EXPECT_DEATH(
      dcp_localize_topk(slots, lengths, 0, 4, 0, output, output_lengths),
      "physical_block_size");
  EXPECT_DEATH(
      dcp_localize_topk(
          slots.to(torch::kInt64), lengths, 0, 4, 16, output, output_lengths),
      "scalar_type");
  auto strided = torch::zeros({4}, opts).slice(0, 0, 4, 2);
  EXPECT_DEATH(
      dcp_localize_topk(slots, strided, 0, 4, 16, output, output_lengths),
      "contiguous");
  EXPECT_DEATH(
      dcp_localize_topk(slots, lengths, 0, 4, 16, slots, output_lengths),
      "alias");
  auto overlap = output.as_strided({2, 4}, {1, 1});
  EXPECT_DEATH(
      dcp_localize_topk(slots, lengths, 0, 4, 16, overlap, output_lengths),
      "stride");
}

}  // namespace
}  // namespace xllm
