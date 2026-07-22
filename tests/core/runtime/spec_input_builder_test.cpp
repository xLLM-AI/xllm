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

#include "runtime/spec_input_builder.h"

#include <gtest/gtest.h>

#include <vector>

#include "framework/model/model_input_params.h"
#include "models/llm/mlu/mtp_topk_state.h"
#include "models/llm/npu/mtp_topk_state.h"
#include "runtime/forward_params.h"

namespace xllm {
namespace specBuilder {
namespace {

Slice<int32_t> to_slice(std::vector<int32_t>& vec) {
  return {vec.data(), static_cast<size_t>(vec.size())};
}

std::vector<int32_t> to_layout_seq_lens(const std::vector<int32_t>& lens) {
#if defined(USE_NPU)
  return lens;
#else
  std::vector<int32_t> out;
  out.reserve(lens.size() + 1);
  out.emplace_back(0);
  int32_t sum = 0;
  for (int32_t len : lens) {
    sum += len;
    out.emplace_back(sum);
  }
  return out;
#endif
}

std::vector<int32_t> tensor_to_vec_int32(const torch::Tensor& tensor) {
  torch::Tensor cpu_tensor =
      tensor.to(torch::kCPU).to(torch::kInt).contiguous();
  const int32_t* data = cpu_tensor.data_ptr<int32_t>();
  return {data, data + cpu_tensor.numel()};
}

ForwardInput make_forward_input(const torch::Tensor& token_ids,
                                const torch::Tensor& positions,
                                const torch::Tensor& block_tables,
                                const std::vector<int32_t>& kv_seq_lens) {
  ForwardInput input;
  input.input_params.meta.num_sequences =
      static_cast<int32_t>(positions.numel());
  input.token_ids_host = token_ids;
  input.positions_host = positions;
  input.input_params.attention.host.block_tables = block_tables;
  input.input_params.attention.host.kv_seq_lens = kv_seq_lens;
  return input;
}

ForwardInput make_multiblock_forward_input(
    const torch::Tensor& token_ids,
    const torch::Tensor& positions,
    const std::vector<torch::Tensor>& multi_block_tables,
    const std::vector<int32_t>& kv_seq_lens) {
  ForwardInput input;
  input.input_params.meta.num_sequences =
      static_cast<int32_t>(positions.numel());
  input.token_ids_host = token_ids;
  input.positions_host = positions;
  input.input_params.multi_block_tables = multi_block_tables;
  input.input_params.attention.host.kv_seq_lens = kv_seq_lens;
  return input;
}

TEST(SpecDecodeInputBuilderTest, DraftInputsSingleRowPerSeq) {
  ModelInputParams params;
  params.meta.num_sequences = 2;
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({5, 9});

  torch::Tensor positions = torch::tensor({4, 8}, torch::kInt);
  torch::Tensor block_tables =
      torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt);
  ForwardInput input =
      make_forward_input(torch::Tensor(), positions, block_tables, kv_seq_lens);
  DecodeRowContext ctx = make_decode_row_context(input);

  DecodeBuildBuffers buf;
  for (int32_t seq_id = 0; seq_id < params.meta.num_sequences; ++seq_id) {
    RowSpec row;
    row.seq_id = seq_id;
    row.position_offset = 1;
    row.append_token = false;
    append_decode_row(ctx, row, /*block_size=*/4, buf);
  }

  EXPECT_TRUE(buf.out_token_ids.empty());
  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({5, 9}));
  EXPECT_EQ(buf.out_new_cache_slots, std::vector<int32_t>({5, 21}));
  EXPECT_EQ(buf.out_kv_seq_lens, to_layout_seq_lens({6, 10}));
}

TEST(SpecDecodeInputBuilderTest, ValidateInputsNonAtbExpansion) {
  ModelInputParams params;
  params.meta.num_sequences = 2;
  const int32_t num_speculative_tokens = 2;
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({5, 9});

  torch::Tensor token_ids = torch::tensor({10, 20}, torch::kInt);
  torch::Tensor positions = torch::tensor({4, 8}, torch::kInt);
  torch::Tensor block_tables =
      torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt);
  ForwardInput input =
      make_forward_input(token_ids, positions, block_tables, kv_seq_lens);
  DecodeRowContext ctx = make_decode_row_context(input);

  DecodeBuildBuffers buf;
  for (int32_t seq_id = 0; seq_id < params.meta.num_sequences; ++seq_id) {
    for (int32_t val_idx = 0; val_idx < num_val_tokens; ++val_idx) {
      RowSpec row;
      row.seq_id = seq_id;
      if (val_idx == 0) {
        row.use_input_token = true;
      } else {
        row.token_id = -1 * val_idx;
      }
      row.position_offset = 1 + val_idx;
      row.append_q_len_one = true;
      row.append_block_table = true;
      append_decode_row(ctx, row, /*block_size=*/4, buf);
    }
  }

  EXPECT_EQ(buf.out_token_ids, std::vector<int32_t>({10, -1, -2, 20, -1, -2}));
  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({5, 6, 7, 9, 10, 11}));
  EXPECT_EQ(buf.out_new_cache_slots,
            std::vector<int32_t>({5, 6, 7, 21, 22, 23}));
  EXPECT_EQ(buf.out_kv_seq_lens, to_layout_seq_lens({6, 7, 8, 10, 11, 12}));
  EXPECT_EQ(buf.out_q_seq_lens, to_layout_seq_lens({1, 1, 1, 1, 1, 1}));
  ASSERT_EQ(buf.out_block_table_rows, 6);
  ASSERT_EQ(buf.out_block_tables.size(), 18);
}

TEST(SpecDecodeInputBuilderTest, AppendDecodeRowTokenKinds) {
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({5, 9});
  torch::Tensor token_ids = torch::tensor({10, 20}, torch::kInt);
  torch::Tensor positions = torch::tensor({4, 8}, torch::kInt);
  torch::Tensor block_tables =
      torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt);
  ForwardInput input =
      make_forward_input(token_ids, positions, block_tables, kv_seq_lens);
  DecodeRowContext ctx = make_decode_row_context(input);

  DecodeBuildBuffers buf;
  append_decode_row(
      ctx,
      {.seq_id = 1, .use_input_token = true, .position_offset = 0},
      /*block_size=*/4,
      buf);
  append_decode_row(ctx,
                    {.seq_id = 0, .token_id = 123, .position_offset = 0},
                    /*block_size=*/4,
                    buf);
  append_decode_row(ctx,
                    {.seq_id = 0, .token_id = -2, .position_offset = 0},
                    /*block_size=*/4,
                    buf);

  EXPECT_EQ(buf.out_token_ids, std::vector<int32_t>({20, 123, -2}));
}

TEST(SpecDecodeInputBuilderTest, AppendDecodeRowUsesInputBlockTableLayout) {
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({5, 9});
  torch::Tensor token_ids = torch::tensor({10, 20}, torch::kInt);
  torch::Tensor positions = torch::tensor({4, 8}, torch::kInt);
  torch::Tensor block_tables =
      torch::tensor({{0, 1, 2, 0}, {3, 4, 5, 0}}, torch::kInt);
  ForwardInput input =
      make_forward_input(token_ids, positions, block_tables, kv_seq_lens);
  DecodeRowContext ctx = make_decode_row_context(input);

  DecodeBuildBuffers buf;
  append_decode_row(ctx,
                    {.seq_id = 1, .token_id = 99, .position_offset = 2},
                    /*block_size=*/4,
                    buf);

  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({10}));
  EXPECT_EQ(buf.out_new_cache_slots, std::vector<int32_t>({22}));
  ASSERT_EQ(buf.out_block_tables.size(), 0);
}

TEST(SpecDecodeInputBuilderTest, ValidateRowsStartFromCorrectedCurrentView) {
  ModelInputParams params;
  params.meta.num_sequences = 2;
  std::vector<int32_t> token_ids = {31, 41};
  std::vector<int32_t> positions = {6, 9};
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({7, 10});

  ForwardInput input = make_forward_input(
      torch::tensor(token_ids, torch::kInt),
      torch::tensor(positions, torch::kInt),
      torch::tensor({{0, 1, 2, 0}, {3, 4, 5, 0}}, torch::kInt),
      kv_seq_lens);
  DecodeRowContext ctx = make_decode_row_context(input);

  DecodeBuildBuffers buf;
  append_decode_row(
      ctx,
      {.seq_id = 0, .token_id = token_ids[0], .position_offset = 0},
      /*block_size=*/4,
      buf);
  append_decode_row(ctx,
                    {.seq_id = 0, .token_id = -1, .position_offset = 1},
                    /*block_size=*/4,
                    buf);
  append_decode_row(
      ctx,
      {.seq_id = 1, .token_id = token_ids[1], .position_offset = 0},
      /*block_size=*/4,
      buf);

  EXPECT_EQ(buf.out_token_ids, std::vector<int32_t>({31, -1, 41}));
  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({6, 7, 9}));
  EXPECT_EQ(buf.out_new_cache_slots, std::vector<int32_t>({6, 7, 21}));
  EXPECT_EQ(buf.out_kv_seq_lens, to_layout_seq_lens({7, 8, 10}));
}

TEST(SpecDecodeInputBuilderTest, ValidateInputsAtbChunkedPrefillShape) {
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({5, 9});
  std::vector<int32_t> atb_kv_seq_lens;
  std::vector<int32_t> atb_q_seq_lens;
  int32_t atb_kv_max_seq_len = 0;
  const int32_t num_val_tokens = 3;

  auto kv_slice = to_slice(kv_seq_lens);
  for (int32_t seq_id = 0; seq_id < 2; ++seq_id) {
    int32_t kv_len = calc_kv_len(kv_slice, seq_id, /*offset=*/0);
    int32_t kv_len_after_validation = kv_len + num_val_tokens;
    update_kv_seq_lens_and_max(
        atb_kv_seq_lens, kv_len_after_validation, atb_kv_max_seq_len);
    append_seq_len_by_layout(atb_q_seq_lens, num_val_tokens);
  }

  EXPECT_EQ(atb_kv_seq_lens, to_layout_seq_lens({8, 12}));
  EXPECT_EQ(atb_q_seq_lens, to_layout_seq_lens({3, 3}));
  EXPECT_EQ(atb_kv_max_seq_len, 12);
}

TEST(SpecDecodeInputBuilderTest, FirstDecodeInputsFixAndNonFixMix) {
  ModelInputParams params;
  params.meta.num_sequences = 2;
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({6, 9});

  torch::Tensor token_ids = torch::tensor({100, 200}, torch::kInt);
  torch::Tensor positions = torch::tensor({5, 8}, torch::kInt);
  torch::Tensor block_tables =
      torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt);
  ForwardInput input =
      make_forward_input(token_ids, positions, block_tables, kv_seq_lens);
  DecodeRowContext ctx = make_decode_row_context(input);

  DecodeBuildBuffers buf;
  std::vector<int32_t> select_row_idx(2, 0);
  auto emit_row =
      [&](int32_t seq_id, int32_t token_id, int32_t position_offset) {
        RowSpec row;
        row.seq_id = seq_id;
        row.token_id = token_id;
        row.position_offset = position_offset;
        row.append_q_len_one = true;
        row.append_block_table = true;
        append_decode_row(ctx, row, /*block_size=*/4, buf);
      };

  emit_row(/*seq_id=*/0, /*token_id=*/90, /*position_offset=*/-1);
  emit_row(/*seq_id=*/0, /*token_id=*/100, /*position_offset=*/0);
  select_row_idx[0] = static_cast<int32_t>(buf.out_token_ids.size()) - 1;

  emit_row(/*seq_id=*/1, /*token_id=*/200, /*position_offset=*/0);
  select_row_idx[1] = static_cast<int32_t>(buf.out_token_ids.size()) - 1;

  EXPECT_EQ(buf.out_token_ids, std::vector<int32_t>({90, 100, 200}));
  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({4, 5, 8}));
  EXPECT_EQ(buf.out_new_cache_slots, std::vector<int32_t>({4, 5, 20}));
  EXPECT_EQ(buf.out_q_seq_lens, to_layout_seq_lens({1, 1, 1}));
  EXPECT_EQ(buf.out_kv_seq_lens, to_layout_seq_lens({5, 6, 9}));
  EXPECT_EQ(select_row_idx, std::vector<int32_t>({1, 2}));
  ASSERT_EQ(buf.out_block_table_rows, 3);
  ASSERT_EQ(buf.out_block_tables.size(), 9);
}

TEST(SpecDecodeInputBuilderTest, AppendDecodeRowWithInputTokenSource) {
  ModelInputParams params;
  params.meta.num_sequences = 2;
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({5, 9});

  torch::Tensor token_ids = torch::tensor({10, 20}, torch::kInt);
  torch::Tensor positions = torch::tensor({4, 8}, torch::kInt);
  torch::Tensor block_tables =
      torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt);
  ForwardInput input =
      make_forward_input(token_ids, positions, block_tables, kv_seq_lens);
  DecodeRowContext ctx = make_decode_row_context(input);

  DecodeBuildBuffers buf;
  append_decode_row(ctx,
                    {.seq_id = 0,
                     .use_input_token = true,
                     .position_offset = 1,
                     .append_q_len_one = true,
                     .append_block_table = true},
                    /*block_size=*/4,
                    buf);
  append_decode_row(ctx,
                    {.seq_id = 1,
                     .token_id = -2,
                     .position_offset = 2,
                     .append_q_len_one = true,
                     .append_block_table = true},
                    /*block_size=*/4,
                    buf);

  EXPECT_EQ(buf.out_token_ids, std::vector<int32_t>({10, -2}));
  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({5, 10}));
  EXPECT_EQ(buf.out_new_cache_slots, std::vector<int32_t>({5, 22}));
  EXPECT_EQ(buf.out_kv_seq_lens, to_layout_seq_lens({6, 11}));
  EXPECT_EQ(buf.out_q_seq_lens, to_layout_seq_lens({1, 1}));
  ASSERT_EQ(buf.out_block_table_rows, 2);
  ASSERT_EQ(buf.out_block_tables.size(), 6);
}

TEST(SpecDecodeInputBuilderTest, ResolveTokenWithPositionOffset) {
  std::vector<int64_t> last_step_tokens = {11, -1, 13, -1, -1, -1};
  Slice<int64_t> last_step_slice = {
      last_step_tokens.data(), static_cast<size_t>(last_step_tokens.size())};

  TokenWithOffset direct =
      resolve_token_with_position_offset(/*input_token_id=*/20,
                                         /*seq_id=*/0,
                                         last_step_slice,
                                         /*last_step_decode_num=*/3);
  EXPECT_EQ(direct.token_id, 20);
  EXPECT_EQ(direct.position_offset, 0);

  TokenWithOffset resolved =
      resolve_token_with_position_offset(/*input_token_id=*/-1,
                                         /*seq_id=*/0,
                                         last_step_slice,
                                         /*last_step_decode_num=*/3);
  EXPECT_EQ(resolved.token_id, 13);
  EXPECT_EQ(resolved.position_offset, 1);

  TokenWithOffset no_accept =
      resolve_token_with_position_offset(/*input_token_id=*/-2,
                                         /*seq_id=*/1,
                                         last_step_slice,
                                         /*last_step_decode_num=*/3);
  EXPECT_EQ(no_accept.token_id, 0);
  EXPECT_EQ(no_accept.position_offset, -1);
}

TEST(SpecDecodeInputBuilderTest, AppendDecodeRowFromLastStep) {
  ModelInputParams params;
  params.meta.num_sequences = 2;
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({6, 9});

  torch::Tensor token_ids = torch::tensor({100, -1}, torch::kInt);
  torch::Tensor positions = torch::tensor({5, 8}, torch::kInt);
  torch::Tensor block_tables =
      torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt);
  ForwardInput input =
      make_forward_input(token_ids, positions, block_tables, kv_seq_lens);
  DecodeRowContext ctx = make_decode_row_context(input);

  std::vector<int64_t> last_step_tokens = {201, 202};
  Slice<int64_t> last_step_slice = {
      last_step_tokens.data(), static_cast<size_t>(last_step_tokens.size())};

  DecodeBuildBuffers buf;
  append_decode_row_from_last_step(ctx,
                                   /*seq_id=*/0,
                                   /*input_token_id=*/100,
                                   last_step_slice,
                                   /*last_step_decode_num=*/2,
                                   /*block_size=*/4,
                                   buf);
  append_decode_row_from_last_step(ctx,
                                   /*seq_id=*/1,
                                   /*input_token_id=*/-1,
                                   last_step_slice,
                                   /*last_step_decode_num=*/2,
                                   /*block_size=*/4,
                                   buf);

  EXPECT_EQ(buf.out_token_ids, std::vector<int32_t>({100, 202}));
  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({5, 9}));
  EXPECT_EQ(buf.out_new_cache_slots, std::vector<int32_t>({5, 21}));
  EXPECT_EQ(buf.out_kv_seq_lens, to_layout_seq_lens({6, 10}));
}

TEST(SpecDecodeInputBuilderTest, QCuSeqLensConsistency) {
  ModelInputParams params;
  params.meta.num_sequences = 3;
  params.attention.host.q_seq_lens = to_layout_seq_lens({1, 2, 3});
  params.attention.host.q_cu_seq_lens = {1, 3, 6};

  torch::Tensor q_cu_seq_lens = build_q_cu_seq_lens_tensor(params);
  EXPECT_EQ(tensor_to_vec_int32(q_cu_seq_lens),
            std::vector<int32_t>({1, 3, 6}));
}

TEST(SpecDecodeInputBuilderTest, QCuSeqLensWithLeadingZero) {
  ModelInputParams params;
  params.meta.num_sequences = 3;
  params.attention.host.q_seq_lens = to_layout_seq_lens({1, 2, 3});
  params.attention.host.q_cu_seq_lens = {1, 3, 6};

  torch::Tensor q_cu_seq_lens =
      build_q_cu_seq_lens_tensor(params, torch::kCPU, true);
  EXPECT_EQ(tensor_to_vec_int32(q_cu_seq_lens),
            std::vector<int32_t>({0, 1, 3, 6}));
}

TEST(SpecDecodeInputBuilderTest, CalcSlotIdOutOfRangeDeath) {
  std::vector<int32_t> block_table = {0};
  EXPECT_DEATH(calc_slot_id(/*position=*/4,
                            to_slice(block_table),
                            /*block_size=*/4),
               "block table index out of range");
}

TEST(DraftProbsBuilderTest, CompressForCacheDense) {
  auto draft_probs =
      torch::tensor({{0.1f, 0.2f, 0.7f}, {0.6f, 0.1f, 0.3f}}, torch::kFloat32);
  auto token_ids = torch::tensor({1, 0}, torch::kInt64);
  auto compressed = draftProbs::compress_for_cache(draft_probs, token_ids);
  auto expected = torch::tensor({0.2f, 0.6f}, torch::kFloat32);
  EXPECT_TRUE(torch::allclose(compressed, expected));
}

TEST(DraftProbsBuilderTest, BuildValidateTensorsSelectedOnly) {
  std::vector<torch::Tensor> token_steps = {
      torch::tensor({3, 4}, torch::kInt64),
      torch::tensor({5, 6}, torch::kInt64)};
  std::vector<torch::Tensor> probs_steps = {
      torch::tensor({0.3f, 0.4f}, torch::kFloat32),
      torch::tensor({0.5f, 0.6f}, torch::kFloat32)};

  auto [draft_token_ids, draft_probs] =
      draftProbs::build_validate_tensors(token_steps,
                                         probs_steps,
                                         /*batch_size=*/2,
                                         /*vocab_size=*/8,
                                         /*enable_opt_validate_probs=*/true);

  EXPECT_EQ(draft_token_ids.dim(), 2);
  EXPECT_EQ(draft_probs.dim(), 2);
  EXPECT_EQ(draft_token_ids.size(0), 2);
  EXPECT_EQ(draft_token_ids.size(1), 2);
  EXPECT_EQ(draft_probs.size(0), 2);
  EXPECT_EQ(draft_probs.size(1), 2);
  EXPECT_TRUE(torch::allclose(
      draft_probs,
      torch::tensor({{0.3f, 0.5f}, {0.4f, 0.6f}}, torch::kFloat32)));
}

TEST(DraftProbsBuilderTest, BuildValidateTensorsSkipsGreedyProbs) {
  std::vector<torch::Tensor> token_steps = {
      torch::tensor({3, 4}, torch::kInt64),
      torch::tensor({5, 6}, torch::kInt64)};
  std::vector<torch::Tensor> probs_steps(token_steps.size());

  auto [draft_token_ids, draft_probs] =
      draftProbs::build_validate_tensors(token_steps,
                                         probs_steps,
                                         /*batch_size=*/2,
                                         /*vocab_size=*/8,
                                         /*enable_opt_validate_probs=*/true,
                                         /*draft_probs_required=*/false);

  EXPECT_TRUE(torch::equal(draft_token_ids,
                           torch::tensor({{3, 5}, {4, 6}}, torch::kInt64)));
  EXPECT_FALSE(draft_probs.defined());
}

TEST(DraftProbsBuilderTest, BuildValidateTensorsRecoveredDense) {
  std::vector<torch::Tensor> token_steps = {
      torch::tensor({1, 2}, torch::kInt64),
      torch::tensor({0, 3}, torch::kInt64)};
  std::vector<torch::Tensor> probs_steps = {
      torch::tensor({0.2f, 0.7f}, torch::kFloat32),
      torch::tensor({0.9f, 0.1f}, torch::kFloat32)};

  auto [draft_token_ids, draft_probs] =
      draftProbs::build_validate_tensors(token_steps,
                                         probs_steps,
                                         /*batch_size=*/2,
                                         /*vocab_size=*/5,
                                         /*enable_opt_validate_probs=*/false);

  EXPECT_EQ(draft_token_ids.dim(), 2);
  EXPECT_EQ(draft_probs.dim(), 3);
  EXPECT_EQ(draft_probs.size(0), 2);
  EXPECT_EQ(draft_probs.size(1), 2);
  EXPECT_EQ(draft_probs.size(2), 5);

  auto selected =
      draft_probs.gather(/*dim=*/-1, draft_token_ids.unsqueeze(-1)).squeeze(-1);
  auto expected_selected =
      torch::tensor({{0.2f, 0.9f}, {0.7f, 0.1f}}, torch::kFloat32);
  EXPECT_TRUE(torch::allclose(selected, expected_selected));

  auto row_sums = draft_probs.sum(/*dim=*/-1);
  EXPECT_TRUE(torch::allclose(row_sums, expected_selected));
}

TEST(DraftProbsBuilderTest, BuildValidateTensorsDenseInputFallback) {
  std::vector<torch::Tensor> token_steps = {
      torch::tensor({2, 1}, torch::kInt64)};
  std::vector<torch::Tensor> probs_steps = {
      torch::tensor({{0.1f, 0.2f, 0.7f}, {0.3f, 0.6f, 0.1f}}, torch::kFloat32)};

  auto [draft_token_ids, draft_probs] =
      draftProbs::build_validate_tensors(token_steps,
                                         probs_steps,
                                         /*batch_size=*/2,
                                         /*vocab_size=*/3,
                                         /*enable_opt_validate_probs=*/true);

  EXPECT_EQ(draft_token_ids.dim(), 2);
  EXPECT_EQ(draft_token_ids.size(0), 2);
  EXPECT_EQ(draft_token_ids.size(1), 1);
  EXPECT_EQ(draft_probs.dim(), 2);
  EXPECT_EQ(draft_probs.size(0), 2);
  EXPECT_EQ(draft_probs.size(1), 1);
  EXPECT_TRUE(torch::allclose(
      draft_probs, torch::tensor({{0.7f}, {0.6f}}, torch::kFloat32)));
}

TEST(SpecDecodeInputBuilderTest, MultiBlockDraftSingleRowPerSeq) {
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({5, 9});
  torch::Tensor positions = torch::tensor({4, 8}, torch::kInt);
  std::vector<torch::Tensor> multi_block_tables = {
      torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt),
      torch::tensor({{10, 11, 12}, {13, 14, 15}}, torch::kInt)};
  ForwardInput input = make_multiblock_forward_input(
      torch::Tensor(), positions, multi_block_tables, kv_seq_lens);
  DecodeRowContext ctx = make_decode_row_context(input);

  EXPECT_TRUE(ctx.model_managed_multiblock);
  EXPECT_EQ(ctx.multi_block_tables.size(), 2);
  EXPECT_EQ(ctx.multi_block_tables[0].size(), 2);
  EXPECT_EQ(ctx.multi_block_tables[1].size(), 2);

  DecodeBuildBuffers buf;
  for (int32_t seq_id = 0; seq_id < input.input_params.meta.num_sequences;
       ++seq_id) {
    RowSpec row;
    row.seq_id = seq_id;
    row.position_offset = 1;
    row.append_token = false;
    row.append_block_table = true;
    append_decode_row(ctx, row, /*block_size=*/4, buf);
  }

  EXPECT_TRUE(buf.out_token_ids.empty());
  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({5, 9}));
  EXPECT_EQ(buf.out_new_cache_slots, std::vector<int32_t>({0, 0}));
  EXPECT_TRUE(buf.out_block_tables.empty());

  ASSERT_EQ(buf.out_multi_block_tables.size(), 2);
  ASSERT_EQ(buf.out_multi_block_tables[0].size(), 2);
  EXPECT_EQ(buf.out_multi_block_tables[0][0], std::vector<int32_t>({0, 1, 2}));
  EXPECT_EQ(buf.out_multi_block_tables[0][1], std::vector<int32_t>({3, 4, 5}));
  ASSERT_EQ(buf.out_multi_block_tables[1].size(), 2);
  EXPECT_EQ(buf.out_multi_block_tables[1][0],
            std::vector<int32_t>({10, 11, 12}));
  EXPECT_EQ(buf.out_multi_block_tables[1][1],
            std::vector<int32_t>({13, 14, 15}));
}

TEST(SpecDecodeInputBuilderTest, MultiBlockKeepsSparseAbsoluteRows) {
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({24, 20});
  torch::Tensor positions = torch::tensor({23, 19}, torch::kInt);
  std::vector<torch::Tensor> multi_block_tables = {torch::tensor(
      {{-1, -1, -1, -1, -1, 50}, {-1, -1, -1, -1, 60, -1}}, torch::kInt)};
  ForwardInput input = make_multiblock_forward_input(
      torch::Tensor(), positions, multi_block_tables, kv_seq_lens);
  DecodeRowContext ctx = make_decode_row_context(input);

  DecodeBuildBuffers buf;
  for (int32_t seq_id = 0; seq_id < input.input_params.meta.num_sequences;
       ++seq_id) {
    RowSpec row;
    row.seq_id = seq_id;
    row.position_offset = 0;
    row.append_token = false;
    row.append_block_table = true;
    append_decode_row(ctx, row, /*block_size=*/4, buf);
  }

  ASSERT_EQ(buf.out_multi_block_tables.size(), 1);
  ASSERT_EQ(buf.out_multi_block_tables[0].size(), 2);
  EXPECT_EQ(buf.out_multi_block_tables[0][0],
            std::vector<int32_t>({-1, -1, -1, -1, -1, 50}));
  EXPECT_EQ(buf.out_multi_block_tables[0][1],
            std::vector<int32_t>({-1, -1, -1, -1, 60, -1}));
}

TEST(SpecDecodeInputBuilderTest, MultiBlockValidateExpansion) {
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({5, 9});
  torch::Tensor token_ids = torch::tensor({10, 20}, torch::kInt);
  torch::Tensor positions = torch::tensor({4, 8}, torch::kInt);
  std::vector<torch::Tensor> multi_block_tables = {
      torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt)};
  ForwardInput input = make_multiblock_forward_input(
      token_ids, positions, multi_block_tables, kv_seq_lens);
  DecodeRowContext ctx = make_decode_row_context(input);

  EXPECT_TRUE(ctx.model_managed_multiblock);
  const int32_t num_val_tokens = 3;

  DecodeBuildBuffers buf;
  for (int32_t seq_id = 0; seq_id < input.input_params.meta.num_sequences;
       ++seq_id) {
    for (int32_t val_idx = 0; val_idx < num_val_tokens; ++val_idx) {
      RowSpec row;
      row.seq_id = seq_id;
      if (val_idx == 0) {
        row.use_input_token = true;
      } else {
        row.token_id = -1 * val_idx;
      }
      row.position_offset = 1 + val_idx;
      row.append_q_len_one = true;
      row.append_block_table = true;
      row.append_kv_len = true;
      append_decode_row(ctx, row, /*block_size=*/4, buf);
    }
  }

  EXPECT_EQ(buf.out_token_ids, std::vector<int32_t>({10, -1, -2, 20, -1, -2}));
  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({5, 6, 7, 9, 10, 11}));
  EXPECT_EQ(buf.out_new_cache_slots, std::vector<int32_t>({0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(buf.out_kv_seq_lens, to_layout_seq_lens({6, 7, 8, 10, 11, 12}));
  EXPECT_EQ(buf.out_q_seq_lens, to_layout_seq_lens({1, 1, 1, 1, 1, 1}));
  EXPECT_TRUE(buf.out_block_tables.empty());

  ASSERT_EQ(buf.out_multi_block_tables.size(), 1);
  ASSERT_EQ(buf.out_multi_block_tables[0].size(), 6);
  EXPECT_EQ(buf.out_multi_block_tables[0][0], std::vector<int32_t>({0, 1, 2}));
  EXPECT_EQ(buf.out_multi_block_tables[0][3], std::vector<int32_t>({3, 4, 5}));
}

TEST(SpecDecodeInputBuilderTest, MakeDecodeRowContextRejectsEmptyBlockTables) {
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({5, 9});
  torch::Tensor positions = torch::tensor({4, 8}, torch::kInt);
  ForwardInput input;
  input.input_params.meta.num_sequences =
      static_cast<int32_t>(positions.numel());
  input.positions_host = positions;
  input.input_params.attention.host.kv_seq_lens = kv_seq_lens;

  EXPECT_DEATH(make_decode_row_context(input),
               "host block_tables must be defined");
}

TEST(SpecMtpTopkInputBuilderTest, SelectsSampledRowsForNextDraftStep) {
  const torch::Tensor topk_indices =
      torch::tensor({{0, 1}, {2, 3}, {4, 5}, {6, 7}}, torch::kInt32);
  const MtpTopkStatePtr state =
      std::make_shared<npu::model::NpuMtpTopkState>(topk_indices);
  SamplingParameters sampling_params;
  sampling_params.selected_token_idxes = torch::tensor({1, 3}, torch::kInt32);

  const MtpTopkStatePtr selected =
      select_mtp_topk_state_for_next_step(state, sampling_params);

  const auto npu_state =
      std::dynamic_pointer_cast<const npu::model::NpuMtpTopkState>(selected);
  ASSERT_NE(npu_state, nullptr);
  const torch::Tensor expected = torch::tensor({{2, 3}, {6, 7}}, torch::kInt32);
  EXPECT_TRUE(torch::equal(npu_state->topk_indices(), expected));
}

TEST(SpecMtpTopkInputBuilderTest, KeepsRowsWhenAlreadyMatchedToDraftBatch) {
  const torch::Tensor topk_indices =
      torch::tensor({{0, 1}, {2, 3}}, torch::kInt32);
  const MtpTopkStatePtr state =
      std::make_shared<npu::model::NpuMtpTopkState>(topk_indices);
  SamplingParameters sampling_params;
  sampling_params.selected_token_idxes = torch::tensor({0, 1}, torch::kInt32);

  const MtpTopkStatePtr selected =
      select_mtp_topk_state_for_next_step(state, sampling_params);

  EXPECT_EQ(selected.get(), state.get());
}

TEST(SpecMtpTopkInputBuilderTest, KeepsUndefinedStateUndefined) {
  SamplingParameters sampling_params;
  sampling_params.selected_token_idxes = torch::tensor({0}, torch::kInt32);

  const MtpTopkStatePtr selected =
      select_mtp_topk_state_for_next_step(nullptr, sampling_params);

  EXPECT_EQ(selected, nullptr);
}

TEST(SpecMtpTopkInputBuilderTest, SelectsMluRowsWithoutMixingLayerState) {
  mlu::model::MluMtpTopkState::LayerStates states;
  states.emplace_back(layer::DsaTopkState(
      torch::tensor({{0, 1}, {2, 3}, {4, 5}}, torch::kInt32),
      torch::tensor({10, 20, 30}, torch::kInt32)));
  states.emplace_back(std::nullopt);
  states.emplace_back(layer::DsaTopkState(
      torch::tensor({{6, 7}, {8, 9}, {10, 11}}, torch::kInt32),
      torch::tensor({40, 50, 60}, torch::kInt32)));
  const MtpTopkStatePtr state =
      std::make_shared<mlu::model::MluMtpTopkState>(std::move(states));
  SamplingParameters sampling_params;
  sampling_params.selected_token_idxes = torch::tensor({2, 0}, torch::kInt32);

  const MtpTopkStatePtr selected =
      select_mtp_topk_state_for_next_step(state, sampling_params);

  const auto mlu_state =
      std::dynamic_pointer_cast<const mlu::model::MluMtpTopkState>(selected);
  ASSERT_NE(mlu_state, nullptr);
  const auto& selected_layers = mlu_state->layer_states();
  ASSERT_EQ(selected_layers.size(), 3);
  ASSERT_TRUE(selected_layers[0].has_value());
  EXPECT_TRUE(torch::equal(selected_layers[0]->block_tables(),
                           torch::tensor({{4, 5}, {0, 1}}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(selected_layers[0]->context_lens(),
                           torch::tensor({30, 10}, torch::kInt32)));
  EXPECT_FALSE(selected_layers[1].has_value());
  ASSERT_TRUE(selected_layers[2].has_value());
  EXPECT_TRUE(torch::equal(selected_layers[2]->block_tables(),
                           torch::tensor({{10, 11}, {6, 7}}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(selected_layers[2]->context_lens(),
                           torch::tensor({60, 40}, torch::kInt32)));
}

TEST(SpecMtpTopkInputBuilderTest, KeepsMluStateWhenRowsAlreadyMatchDraftBatch) {
  mlu::model::MluMtpTopkState::LayerStates states;
  states.emplace_back(
      layer::DsaTopkState(torch::tensor({{0, 1}, {2, 3}}, torch::kInt32),
                          torch::tensor({10, 20}, torch::kInt32)));
  const MtpTopkStatePtr state =
      std::make_shared<mlu::model::MluMtpTopkState>(std::move(states));
  SamplingParameters sampling_params;
  sampling_params.selected_token_idxes = torch::tensor({0, 1}, torch::kInt32);

  const MtpTopkStatePtr selected =
      select_mtp_topk_state_for_next_step(state, sampling_params);

  EXPECT_EQ(selected.get(), state.get());
}

TEST(SpecMtpTopkInputBuilderTest, MovesNpuStateWithModelInputParams) {
  ModelInputParams params;
  const torch::Tensor topk_indices =
      torch::tensor({{0, 1}, {2, 3}}, torch::kInt32);
  params.mtp_topk_state =
      std::make_shared<npu::model::NpuMtpTopkState>(topk_indices);

  const torch::Device target_device("meta");
  const ModelInputParams converted = params.to(target_device);

  const auto converted_state =
      std::dynamic_pointer_cast<const npu::model::NpuMtpTopkState>(
          converted.mtp_topk_state);
  ASSERT_NE(converted_state, nullptr);
  EXPECT_EQ(converted_state->device(), target_device);
  EXPECT_EQ(converted_state->topk_indices().sizes(), topk_indices.sizes());
  EXPECT_EQ(converted_state->topk_indices().scalar_type(),
            topk_indices.scalar_type());
  EXPECT_TRUE(topk_indices.device().is_cpu());
}

TEST(SpecMtpTopkInputBuilderTest, MovesMluStateWithModelInputParams) {
  ModelInputParams params;
  mlu::model::MluMtpTopkState::LayerStates states;
  states.emplace_back(
      layer::DsaTopkState(torch::tensor({{0, 1}}, torch::kInt32),
                          torch::tensor({10}, torch::kInt32)));
  states.emplace_back(std::nullopt);
  params.mtp_topk_state =
      std::make_shared<mlu::model::MluMtpTopkState>(std::move(states));

  const torch::Device target_device("meta");
  const ModelInputParams converted = params.to(target_device);

  const auto converted_state =
      std::dynamic_pointer_cast<const mlu::model::MluMtpTopkState>(
          converted.mtp_topk_state);
  ASSERT_NE(converted_state, nullptr);
  EXPECT_EQ(converted_state->device(), target_device);
  const auto& converted_layers = converted_state->layer_states();
  ASSERT_EQ(converted_layers.size(), 2);
  ASSERT_TRUE(converted_layers[0].has_value());
  EXPECT_EQ(converted_layers[0]->block_tables().device(), target_device);
  EXPECT_EQ(converted_layers[0]->context_lens().device(), target_device);
  EXPECT_FALSE(converted_layers[1].has_value());
}

}  // namespace
}  // namespace specBuilder
}  // namespace xllm
