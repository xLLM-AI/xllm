/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <torch_npu/torch_npu.h>

#include <array>
#include <cstdint>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kInvalidSlot = -1;

class SfaDcpRemapWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

torch::Tensor remap_reference(const torch::Tensor& slots,
                              int64_t physical_block_size,
                              int64_t shard_size,
                              int64_t shard_rank) {
  const auto cpu = slots.to(torch::kCPU).contiguous();
  const int64_t num_tokens = cpu.size(0);
  const int64_t topk = cpu.size(1);
  const int64_t logical_block_size = physical_block_size * shard_size;
  auto out = torch::empty_like(cpu);
  auto out_a = out.accessor<int32_t, 2>();
  const auto in_a = cpu.accessor<int32_t, 2>();
  for (int64_t token = 0; token < num_tokens; ++token) {
    int64_t packed = 0;
    for (int64_t k = 0; k < topk; ++k) {
      const int32_t slot = in_a[token][k];
      int32_t local = kInvalidSlot;
      if (slot >= 0) {
        const int64_t block = static_cast<int64_t>(slot) / logical_block_size;
        const int64_t off = static_cast<int64_t>(slot) % logical_block_size;
        const int64_t owner = off / physical_block_size;
        const int64_t local_off = off % physical_block_size;
        if (owner == shard_rank) {
          local = static_cast<int32_t>(block * physical_block_size + local_off);
        }
      }
      if (local >= 0) {
        out_a[token][packed] = local;
        ++packed;
      }
    }
    for (int64_t k = packed; k < topk; ++k) {
      out_a[token][k] = kInvalidSlot;
    }
  }
  return out.to(slots.device());
}

TEST_F(SfaDcpRemapWrapperTest, MatchesNaiveCompact) {
  const auto npu = torch::Device("npu:0");
  const int64_t num_tokens = 8;
  const int64_t topk = 2048;
  const std::array<std::array<int64_t, 3>, 3> cases = {{
      {128, 4, 2},
      {64, 2, 1},
      {128, 32, 7},
  }};
  for (const auto& spec : cases) {
    const int64_t physical_block_size = spec[0];
    const int64_t shard_size = spec[1];
    const int64_t shard_rank = spec[2];
    torch::manual_seed(physical_block_size + shard_size);
    auto slots =
        torch::randint(0,
                       8 * physical_block_size * shard_size,
                       {num_tokens, topk},
                       torch::TensorOptions().dtype(torch::kInt32).device(npu));
    auto mask = torch::rand({num_tokens, topk}, npu) < 0.25;
    slots = torch::where(mask, torch::full_like(slots, kInvalidSlot), slots);

    auto out = torch::empty_like(slots);
    auto scratch =
        torch::empty({num_tokens * topk},
                     torch::TensorOptions().dtype(torch::kInt32).device(npu));
    sfa_dcp_remap_out(
        slots, physical_block_size, shard_size, shard_rank, out, scratch);
    auto ref =
        remap_reference(slots, physical_block_size, shard_size, shard_rank);
    EXPECT_TRUE(torch::equal(out, ref))
        << "pb=" << physical_block_size << " dcp=" << shard_size;
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
