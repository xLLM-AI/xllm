/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <string>
#include <vector>

#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace {

class Dsv4ScatterCacheWriteTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

struct CacheWriteCase {
  std::string name;
  int64_t num_tokens;
  int64_t num_blocks;
  int64_t block_size;
  int64_t num_kv_heads;
  int64_t head_dim;
  bool keep_update_head_dim;
};

torch::Tensor build_slot_mapping(int64_t num_tokens,
                                 int64_t num_blocks,
                                 int64_t block_size,
                                 const torch::Device& device) {
  const int64_t cache_rows = num_blocks * block_size;
  torch::Tensor slots = torch::arange(
      num_tokens, torch::TensorOptions().dtype(torch::kLong).device(device));
  return torch::remainder(slots * 37 + 11, cache_rows);
}

torch::Tensor flatten_updates_for_case(const torch::Tensor& update) {
  return update.reshape({-1, update.size(update.dim() - 1)});
}

void xllm_scatter_by_slot_mirror(torch::Tensor& cache,
                                 const torch::Tensor& slot_mapping,
                                 const torch::Tensor& value) {
  if (!cache.defined() || !slot_mapping.defined() || !value.defined()) {
    return;
  }
  if (slot_mapping.numel() == 0 || value.numel() == 0) {
    return;
  }

  torch::Tensor value_2d = flatten_updates_for_case(value);
  torch::Tensor cache_2d = cache.view({-1, value_2d.size(1)});
  torch::Tensor slots =
      slot_mapping.reshape({-1}).to(torch::kLong).to(cache.device());
  const int64_t update_rows = std::min(slots.size(0), value_2d.size(0));
  if (update_rows <= 0) {
    return;
  }

  torch::Tensor slots_slice =
      slots.slice(/*dim=*/0, /*start=*/0, /*end=*/update_rows);
  torch::Tensor value_slice =
      value_2d.slice(/*dim=*/0, /*start=*/0, /*end=*/update_rows);
  torch::Tensor safe_slots = slots_slice.clamp_min(0);
  torch::Tensor valid_mask = slots_slice.ge(0).unsqueeze(1);
  torch::Tensor old_values = cache_2d.index_select(/*dim=*/0, safe_slots);
  torch::Tensor safe_values = torch::where(valid_mask, value_slice, old_values);
  cache_2d.index_copy_(/*dim=*/0, safe_slots, safe_values);
}

void verify_cache_write_case(const CacheWriteCase& test_case) {
  ASSERT_GT(test_case.num_tokens, 0);
  ASSERT_GT(test_case.num_blocks, 0);
  ASSERT_GT(test_case.block_size, 0);
  ASSERT_GT(test_case.num_kv_heads, 0);
  ASSERT_GT(test_case.head_dim, 0);

  const torch::Device npu_device("npu:0");
  const torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(npu_device);

  torch::manual_seed(20260518);
  torch::Tensor base_cache = torch::randn({test_case.num_blocks,
                                           test_case.block_size,
                                           test_case.num_kv_heads,
                                           test_case.head_dim},
                                          bf16_options);
  torch::Tensor update =
      test_case.keep_update_head_dim
          ? torch::randn({test_case.num_tokens,
                          test_case.num_kv_heads,
                          test_case.head_dim},
                         bf16_options)
          : torch::randn({test_case.num_tokens, test_case.head_dim},
                         bf16_options);
  torch::Tensor slot_mapping = build_slot_mapping(test_case.num_tokens,
                                                  test_case.num_blocks,
                                                  test_case.block_size,
                                                  npu_device);
  torch::Tensor indices = slot_mapping.unsqueeze(-1);

  torch::Tensor xllm_cache = base_cache.clone();
  torch::Tensor npu_cache = base_cache.clone();
  xllm_scatter_by_slot_mirror(xllm_cache, slot_mapping, update);
  torch::Tensor npu_cache_2d =
      npu_cache.view({-1, update.size(update.dim() - 1)});
  xllm::kernel::npu::scatter_nd_update(npu_cache_2d, indices, update);
  EXPECT_TRUE(torch::allclose(xllm_cache, npu_cache, /*rtol=*/0, /*atol=*/0))
      << "cache write mismatch for " << test_case.name;
}

void verify_cache_write_case_with_padding_slot(
    const CacheWriteCase& test_case) {
  ASSERT_GT(test_case.num_tokens, 1);
  ASSERT_GT(test_case.num_blocks, 0);
  ASSERT_GT(test_case.block_size, 0);
  ASSERT_GT(test_case.num_kv_heads, 0);
  ASSERT_GT(test_case.head_dim, 0);

  const torch::Device npu_device("npu:0");
  const torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(npu_device);

  torch::manual_seed(20260520);
  torch::Tensor base_cache = torch::randn({test_case.num_blocks,
                                           test_case.block_size,
                                           test_case.num_kv_heads,
                                           test_case.head_dim},
                                          bf16_options);
  torch::Tensor update =
      test_case.keep_update_head_dim
          ? torch::randn({test_case.num_tokens,
                          test_case.num_kv_heads,
                          test_case.head_dim},
                         bf16_options)
          : torch::randn({test_case.num_tokens, test_case.head_dim},
                         bf16_options);
  torch::Tensor slot_mapping = build_slot_mapping(test_case.num_tokens,
                                                  test_case.num_blocks,
                                                  test_case.block_size,
                                                  npu_device);
  slot_mapping.index_put_({0}, -1);

  torch::Tensor expected_cache = base_cache.clone();
  xllm_scatter_by_slot_mirror(expected_cache, slot_mapping, update);

  torch::Tensor actual_cache = base_cache.clone();
  torch::Tensor update_2d = flatten_updates_for_case(update);
  torch::Tensor actual_cache_2d = actual_cache.view({-1, update_2d.size(1)});
  torch::Tensor slots = slot_mapping.reshape({-1}).to(torch::kLong);
  const int64_t update_rows = std::min(slots.size(0), update_2d.size(0));
  torch::Tensor slots_slice =
      slots.slice(/*dim=*/0, /*start=*/0, /*end=*/update_rows);
  torch::Tensor value_slice =
      update_2d.slice(/*dim=*/0, /*start=*/0, /*end=*/update_rows);
  torch::Tensor safe_slots = slots_slice.clamp_min(0);
  torch::Tensor valid_mask = slots_slice.ge(0).unsqueeze(1);
  torch::Tensor old_values =
      actual_cache_2d.index_select(/*dim=*/0, safe_slots);
  torch::Tensor safe_values = torch::where(valid_mask, value_slice, old_values);
  xllm::kernel::npu::scatter_nd_update(
      actual_cache_2d, safe_slots.reshape({-1, 1}), safe_values);

  EXPECT_TRUE(
      torch::allclose(expected_cache, actual_cache, /*rtol=*/0, /*atol=*/0))
      << "padding slot cache write mismatch for " << test_case.name;
}

}  // namespace

TEST_F(Dsv4ScatterCacheWriteTest, DISABLED_OriginalKvCacheWrite) {
  const std::vector<CacheWriteCase> cases = {
      {.name = "ori_decode_1_token",
       .num_tokens = 1,
       .num_blocks = 1024,
       .block_size = 128,
       .num_kv_heads = 1,
       .head_dim = 576,
       .keep_update_head_dim = true},
      {.name = "ori_decode_8_tokens",
       .num_tokens = 8,
       .num_blocks = 1024,
       .block_size = 128,
       .num_kv_heads = 1,
       .head_dim = 576,
       .keep_update_head_dim = true},
      {.name = "ori_decode_64_tokens",
       .num_tokens = 64,
       .num_blocks = 1024,
       .block_size = 128,
       .num_kv_heads = 1,
       .head_dim = 576,
       .keep_update_head_dim = true},
  };
  for (const CacheWriteCase& test_case : cases) {
    verify_cache_write_case(test_case);
  }
}

TEST_F(Dsv4ScatterCacheWriteTest, DISABLED_CompressedKvCacheWrite) {
  const std::vector<CacheWriteCase> cases = {
      {.name = "cmp_decode_1_token",
       .num_tokens = 1,
       .num_blocks = 256,
       .block_size = 128,
       .num_kv_heads = 1,
       .head_dim = 576,
       .keep_update_head_dim = false},
      {.name = "cmp_decode_8_tokens",
       .num_tokens = 8,
       .num_blocks = 256,
       .block_size = 128,
       .num_kv_heads = 1,
       .head_dim = 576,
       .keep_update_head_dim = false},
      {.name = "cmp_decode_64_tokens",
       .num_tokens = 64,
       .num_blocks = 256,
       .block_size = 128,
       .num_kv_heads = 1,
       .head_dim = 576,
       .keep_update_head_dim = false},
  };
  for (const CacheWriteCase& test_case : cases) {
    verify_cache_write_case(test_case);
  }
}

TEST_F(Dsv4ScatterCacheWriteTest, DISABLED_PaddingSlotIsIgnored) {
  const CacheWriteCase test_case = {.name = "cmp_decode_padding_slot",
                                    .num_tokens = 8,
                                    .num_blocks = 16,
                                    .block_size = 128,
                                    .num_kv_heads = 1,
                                    .head_dim = 576,
                                    .keep_update_head_dim = false};
  verify_cache_write_case_with_padding_slot(test_case);
}
