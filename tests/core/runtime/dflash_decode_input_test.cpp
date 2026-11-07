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

#include <vector>

#include "core/framework/batch/decode_input_builder.h"
#include "core/kv_cache/layout/seq_lens_layout.h"
#include "core/runtime/dflash_worker_impl.h"
#include "core/runtime/forward_params.h"
#include "core/util/tensor_helper.h"

namespace xllm {
namespace {

TEST(DflashEmptyBlockDraftTest, EmptyBlockDraftMaterializesWholeQuery) {
  LlmForwardInput input;
  input.input_params.parallel.dp_global_token_nums = {0, 21};
  input.input_params.parallel.raw_dp_global_token_nums = {0, 21};
  input.input_params.parallel.dp_global_sequence_nums = {0, 3};
  input.runtime.input_host_buffer_has_layout = true;
  dflash_detail::pad_empty_attention_dp_batch(
      input,
      ExpandDecodePlan{{7, QueryLayout::kChunked, /*leading_zero_cu=*/false},
                       /*block_size=*/1,
                       /*mask_token_id=*/42});

  const auto& params = input.input_params;
  EXPECT_EQ(params.meta.num_sequences, 1);
  EXPECT_EQ(params.meta.actual_num_sequences, 0);
  EXPECT_EQ(params.meta.q_max_seq_len, 7);
  EXPECT_EQ(params.meta.kv_max_seq_len, 1);
  EXPECT_EQ(tensor_to_vector<int32_t>(input.token_ids_host),
            std::vector<int32_t>(7, 42));
  EXPECT_EQ(tensor_to_vector<int32_t>(input.positions_host),
            std::vector<int32_t>({0, 1, 2, 3, 4, 5, 6}));
  EXPECT_EQ(params.attention.host.q_seq_lens, CuSeqLens::from_lengths({7}));
  EXPECT_EQ(params.attention.host.q_cu_seq_lens, std::vector<int32_t>({0, 7}));
  EXPECT_EQ(params.attention.host.kv_seq_lens, CuSeqLens::from_lengths({1}));
  EXPECT_EQ(params.attention.host.new_cache_slots, std::vector<int32_t>(7, -1));
  EXPECT_EQ(tensor_to_vector<int32_t>(params.attention.host.block_tables),
            std::vector<int32_t>({0}));
  EXPECT_EQ(tensor_to_vector<int32_t>(params.attention.device.paged_kv_indptr),
            std::vector<int32_t>({0, 1}));
  EXPECT_EQ(tensor_to_vector<int32_t>(params.attention.device.paged_kv_indices),
            std::vector<int32_t>({0}));
  EXPECT_EQ(
      tensor_to_vector<int32_t>(params.attention.device.paged_kv_last_page_len),
      std::vector<int32_t>({1}));
  EXPECT_EQ(params.parallel.dp_global_token_nums,
            std::vector<int32_t>({7, 21}));
  EXPECT_EQ(params.parallel.raw_dp_global_token_nums,
            std::vector<int32_t>({0, 21}));
  EXPECT_EQ(params.parallel.dp_global_sequence_nums,
            std::vector<int32_t>({0, 3}));
  EXPECT_FALSE(input.runtime.device_tensors_ready);
  EXPECT_FALSE(input.runtime.input_host_buffer_has_layout);
}

TEST(DflashEmptyBlockDraftTest, ActiveBlockDraftOnlyFillsIdlePeerCount) {
  LlmForwardInput input;
  input.input_params.meta.num_sequences = 3;
  input.input_params.parallel.dp_global_token_nums = {21, 0};
  input.token_ids = torch::tensor({11, 12}, torch::kInt32);
  dflash_detail::pad_empty_attention_dp_batch(
      input,
      ExpandDecodePlan{{7, QueryLayout::kChunked, /*leading_zero_cu=*/false},
                       /*block_size=*/1,
                       /*mask_token_id=*/42});

  EXPECT_EQ(input.input_params.meta.num_sequences, 3);
  EXPECT_EQ(input.input_params.parallel.dp_global_token_nums,
            std::vector<int32_t>({21, 7}));
  EXPECT_EQ(tensor_to_vector<int32_t>(input.token_ids),
            std::vector<int32_t>({11, 12}));
}

}  // namespace

}  // namespace xllm
