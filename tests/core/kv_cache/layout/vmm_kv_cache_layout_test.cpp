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

#include "core/kv_cache/layout/vmm_kv_cache_layout.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <utility>

#include "core/framework/model/model_args.h"
#include "core/kv_cache/layout/kv_cache_capacity.h"

namespace xllm {
namespace {

constexpr size_t kPageSize = 2 * 1024 * 1024;

TEST(VmmKVCacheLayoutTest, CombinesDifferentTargetAndDraftTensorWidths) {
  ModelArgs target_args;
  target_args.n_layers(8).head_dim(128).n_heads(4).n_kv_heads(4);
  KVCacheCapacity target_capacity;
  target_capacity.n_blocks(128).block_size(128).n_layers(8);
  VmmKVCacheLayout layout(
      target_capacity, target_args, /*world_size=*/1, torch::kBFloat16);

  ModelArgs draft_args;
  draft_args.n_layers(1).head_dim(128).n_heads(1).n_kv_heads(1);
  KVCacheCapacity draft_capacity;
  draft_capacity.n_blocks(128).block_size(128).n_layers(1);
  VmmKVCacheLayout draft_layout(draft_capacity,
                                draft_args,
                                /*world_size=*/1,
                                torch::kBFloat16,
                                /*is_draft_engine=*/true);
  layout.append(std::move(draft_layout));

  // One group covers 64 shared block IDs: four pages per target tensor
  // and one page per draft tensor, across eight target layers and one draft.
  EXPECT_EQ(layout.physical_pages(kPageSize), 66U);
  EXPECT_EQ(layout.block_capacity(65, kPageSize), 0U);
  EXPECT_EQ(layout.block_capacity(66, kPageSize), 64U);
  EXPECT_EQ(layout.block_capacity(131, kPageSize), 64U);
  EXPECT_EQ(layout.block_capacity(132, kPageSize), 128U);
}

TEST(VmmKVCacheLayoutTest, CountsHybridTargetAndFullAttentionDraftLayers) {
  ModelArgs target_args;
  target_args.n_layers(8)
      .head_dim(128)
      .n_heads(4)
      .n_kv_heads(4)
      .full_attention_interval(4)
      .linear_num_key_heads(4)
      .linear_num_value_heads(4)
      .linear_key_head_dim(128)
      .linear_value_head_dim(128)
      .linear_conv_kernel_dim(4);
  KVCacheCapacity target_capacity;
  target_capacity.n_blocks(128)
      .block_size(128)
      .n_layers(8)
      .num_linear_state_blocks(8);
  VmmKVCacheLayout layout(
      target_capacity, target_args, /*world_size=*/1, torch::kBFloat16);
  EXPECT_EQ(layout.physical_pages(kPageSize), 4U);

  // MTP's sole layer uses paged attention despite the target's interval.
  ModelArgs draft_args = target_args;
  draft_args.n_layers(1).n_heads(1).n_kv_heads(1).layer_types(
      {"full_attention"});
  KVCacheCapacity draft_capacity;
  draft_capacity.n_blocks(128).block_size(128).n_layers(1);
  VmmKVCacheLayout draft_layout(draft_capacity,
                                draft_args,
                                /*world_size=*/1,
                                torch::kBFloat16,
                                /*is_draft_engine=*/true);
  layout.append(std::move(draft_layout));

  EXPECT_EQ(layout.physical_pages(kPageSize), 18U);
  EXPECT_EQ(layout.block_capacity(17, kPageSize), 0U);
  EXPECT_EQ(layout.block_capacity(18, kPageSize), 64U);
}

}  // namespace
}  // namespace xllm
