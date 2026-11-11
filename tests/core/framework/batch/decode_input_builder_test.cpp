/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/framework/batch/decode_input_builder.h"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "core/common/constants.h"
#include "core/framework/config/model_config.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/vlm_forward_params.h"
#include "core/util/scope_guard.h"
#include "core/util/tensor_helper.h"

namespace xllm {
namespace {

LlmForwardInput make_forward_input(const torch::Tensor& token_ids,
                                   const torch::Tensor& positions,
                                   const torch::Tensor& block_tables,
                                   const std::vector<int32_t>& kv_seq_lens) {
  LlmForwardInput input;
  input.input_params.meta.num_sequences =
      static_cast<int32_t>(positions.numel());
  input.token_ids_host = token_ids;
  input.positions_host = positions;
  input.input_params.attention.host.block_tables = block_tables;
  input.input_params.attention.host.kv_seq_lens = kv_seq_lens;
  return input;
}

LlmForwardInput make_multiblock_forward_input(
    const torch::Tensor& token_ids,
    const torch::Tensor& positions,
    const std::vector<torch::Tensor>& multi_block_tables,
    const std::vector<int32_t>& kv_seq_lens) {
  LlmForwardInput input = make_forward_input(
      token_ids, positions, /*block_tables=*/torch::Tensor(), kv_seq_lens);
  input.input_params.multi_block_tables = multi_block_tables;
  return input;
}

std::pair<LlmForwardInput, DecodeBatchContext> make_two_request_ctx() {
  LlmForwardInput input =
      make_forward_input(torch::tensor({10, 20}, torch::kInt),
                         torch::tensor({4, 8}, torch::kInt),
                         torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt),
                         {5, 9});
  DecodeBatchContext ctx = make_decode_batch_context(input);
  return {std::move(input), std::move(ctx)};
}

LlmForwardInput make_shared_forward_input() {
  LlmForwardInput input =
      make_forward_input(torch::tensor({10, 20}, torch::kInt),
                         torch::tensor({3, 7}, torch::kInt),
                         torch::tensor({{1, 2, 3}, {4, 5, 6}}, torch::kInt),
                         {4, 8});
  input.input_params.attention.host.shared_full_block_table =
      torch::tensor({{120, 122, 124}, {180, 182, 184}}, torch::kInt);
  return input;
}

TEST(DecodeInputBuilderTest, AppendDecodeRowBuildsSlotsAndKvGeometry) {
  auto [input, ctx] = make_two_request_ctx();

  DecodeBuildBuffers buf;
  prepare_decode_buffers(buf,
                         ctx,
                         2,
                         /*emit_kv_len=*/true,
                         /*emit_block_tables=*/true);
  append_decode_token(
      ctx, {.req_id = 0, .token_id = 10, .position_offset = 1}, 4, buf);
  append_decode_token(
      ctx, {.req_id = 1, .token_id = -2, .position_offset = 2}, 4, buf);

  EXPECT_EQ(buf.out_token_ids, std::vector<int32_t>({10, -2}));
  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({5, 10}));
  EXPECT_EQ(buf.out_new_cache_slots, std::vector<int32_t>({5, 22}));
  EXPECT_EQ(buf.out_kv_seq_lens, std::vector<int32_t>({6, 11}));
  EXPECT_EQ(buf.out_paged_kv_indptr, std::vector<int32_t>({0, 2, 5}));
  EXPECT_EQ(buf.out_paged_kv_indices, std::vector<int32_t>({0, 1, 3, 4, 5}));
  EXPECT_EQ(buf.out_paged_kv_last_page_len, std::vector<int32_t>({2, 3}));

  LlmModelParams params;
  build_decode_metadata(params, buf, DecodeInputOptions{});
  EXPECT_TRUE(params.meta.batch_forward_type.is_decode());
  EXPECT_EQ(params.meta.q_max_seq_len, 1);
}

TEST(DecodeInputBuilderTest, PreservesRequestGroupsAndAbsoluteHistory) {
  auto& model_config = ModelConfig::get_instance();
  const auto model_impl = model_config.model_impl();
  ScopeGuard restore_model_impl([&] { model_config.model_impl(model_impl); });
  model_config.model_impl("python");
  LlmForwardInput input =
      make_forward_input(torch::Tensor(),
                         torch::tensor({40, 80}, torch::kInt),
                         torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt),
                         {5, 9});
  const DecodeBatchContext context = make_decode_batch_context(input);

  DecodeBuildBuffers buffers;
  prepare_decode_buffers(buffers,
                         context,
                         3,
                         /*emit_kv_len=*/true,
                         /*emit_block_tables=*/true);
  for (const auto& [sequence, offset] :
       {std::pair{0, 0}, std::pair{0, 1}, std::pair{1, 0}}) {
    append_decode_token(
        context,
        {.req_id = sequence, .position_offset = offset, .append_token = false},
        4,
        buffers);
  }
  LlmModelParams params;
  params.meta.actual_num_sequences = 2;
  build_decode_metadata(params, buffers, DecodeInputOptions{});
  const auto& batch = params.meta.execution_batch;
  EXPECT_EQ(batch.num_reqs, 2);
  EXPECT_EQ(batch.num_tokens, 3);
  EXPECT_EQ(batch.num_scheduled_tokens, std::vector<int32_t>({2, 1}));
  EXPECT_EQ(batch.query_start_loc, std::vector<int32_t>({0, 2, 3}));
}

TEST(DecodeInputBuilderTest,
     VerifyRowsPreserveFinalTargetAndLogicalDraftPages) {
  for (const bool chunked : {false, true}) {
    SCOPED_TRACE(chunked);
    LlmForwardInput input = make_shared_forward_input();
    const DecodeBatchContext context = make_decode_batch_context(input);
    const DecodeExpansionOptions options = make_verify_layout(
        /*num_val_tokens=*/2, /*block_size=*/4, chunked);
    DecodeBuildBuffers buffers = expand_decode_inputs(context, options);
    EXPECT_EQ(buffers.out_new_cache_slots,
              std::vector<int32_t>({7, 8, 23, 24}));
    EXPECT_EQ(buffers.out_shared_full_slot_mapping,
              std::vector<int32_t>({483, 488, 731, 736}));

    build_decode_metadata(input.input_params, buffers, options);
    const torch::Tensor expected_pages =
        chunked ? torch::tensor({{120, 122, 124}, {180, 182, 184}}, torch::kInt)
                : torch::tensor({{120, 122, 124},
                                 {120, 122, 124},
                                 {180, 182, 184},
                                 {180, 182, 184}},
                                torch::kInt);
    EXPECT_TRUE(
        torch::equal(input.input_params.attention.host.shared_full_block_table,
                     expected_pages));
    EXPECT_EQ(tensor_to_vector<int32_t>(
                  input.input_params.attention.host.shared_full_slot_mapping),
              std::vector<int32_t>({483, 488, 731, 736}));
  }
}

TEST(DecodeInputBuilderTest, VerifyPaddingNeverWritesFiniteReadPage) {
  LlmForwardInput input =
      make_forward_input(torch::tensor({0}, torch::kInt),
                         torch::tensor({0}, torch::kInt),
                         torch::tensor({{0, 0}}, torch::kInt),
                         {1});
  input.input_params.attention.host.shared_full_block_table =
      torch::tensor({{60, 60}}, torch::kInt);
  input.input_params.attention.host.shared_full_slot_mapping =
      torch::tensor({kPadSlotId}, torch::kInt);
  const DecodeBatchContext context = make_decode_batch_context(input);
  const DecodeExpansionOptions options = make_verify_layout(
      /*num_val_tokens=*/1, /*block_size=*/4, /*chunked=*/false);
  DecodeBuildBuffers buffers = expand_decode_inputs(context, options);
  build_decode_metadata(input.input_params, buffers, options);

  EXPECT_TRUE(
      torch::equal(input.input_params.attention.host.shared_full_block_table,
                   torch::tensor({{60, 60}}, torch::kInt)));
  EXPECT_EQ(tensor_to_vector<int32_t>(
                input.input_params.attention.host.shared_full_slot_mapping),
            std::vector<int32_t>({kPadSlotId}));
}

TEST(DecodeInputBuilderTest, PaddingOverridesSuppressFinalWrites) {
  for (const int32_t padding_slot : {kPaddingBlockId, kPadSlotId}) {
    SCOPED_TRACE(padding_slot);
    LlmForwardInput input = make_shared_forward_input();
    const DecodeBatchContext context = make_decode_batch_context(input);
    DecodeBuildBuffers buffers;
    prepare_decode_buffers(buffers,
                           context,
                           /*num_tokens=*/1,
                           /*emit_kv_len=*/true,
                           /*emit_block_tables=*/true);
    append_decode_token(context,
                        {.req_id = 0,
                         .position_offset = -1,
                         .cache_slot_override = padding_slot},
                        /*block_size=*/4,
                        buffers);
    build_decode_metadata(input.input_params, buffers, DecodeInputOptions{});

    EXPECT_EQ(input.input_params.attention.host.new_cache_slots,
              std::vector<int32_t>({padding_slot}));
    EXPECT_EQ(tensor_to_vector<int32_t>(
                  input.input_params.attention.host.shared_full_slot_mapping),
              std::vector<int32_t>({kPadSlotId}));
  }
}

TEST(DecodeInputBuilderTest, DraftSlotsUpdateWithoutRebuildingPageTables) {
  LlmForwardInput input = make_shared_forward_input();
  const torch::Tensor original_pages =
      input.input_params.attention.host.shared_full_block_table;
  input.input_params.attention.host.shared_full_slot_mapping =
      torch::tensor({483, 731}, torch::kInt);
  const DecodeBatchContext context = make_decode_batch_context(input);
  DecodeBuildBuffers buffers;
  prepare_decode_buffers(buffers,
                         context,
                         /*num_tokens=*/2,
                         /*emit_kv_len=*/true,
                         /*emit_block_tables=*/false);
  for (int32_t request = 0; request < 2; ++request) {
    append_decode_token(
        context,
        {.req_id = request, .position_offset = 1, .append_token = false},
        /*block_size=*/4,
        buffers);
  }
  update_kv_lengths_and_slots(input.input_params, buffers);

  EXPECT_EQ(input.input_params.attention.host.new_cache_slots,
            std::vector<int32_t>({8, 24}));
  EXPECT_EQ(tensor_to_vector<int32_t>(
                input.input_params.attention.host.shared_full_slot_mapping),
            std::vector<int32_t>({488, 736}));
  EXPECT_EQ(
      input.input_params.attention.host.shared_full_block_table.data_ptr(),
      original_pages.data_ptr());
  EXPECT_TRUE(buffers.out_shared_full_block_tables.empty());
}

TEST(DecodeInputBuilderTest, VlmVerifyRowsPreserveVisionState) {
  VlmForwardInput input;
  input.input_params.meta.num_sequences = 2;
  input.input_params.is_spec_verify = true;
  input.token_ids_host = torch::tensor({10, 20}, torch::kInt);
  input.positions_host = torch::tensor({4, 8}, torch::kInt);
  input.input_params.attention.host.block_tables =
      torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt);
  input.input_params.attention.host.kv_seq_lens = {5, 9};
  const torch::Tensor pixels = torch::tensor({{1.0f, 2.0f}});
  input.input_params.multimodal.mm_data.batch(
      {MMData(MMType::IMAGE, MMDict{{"pixel_values", pixels}})});
  const torch::Tensor deep_stack = torch::tensor({{3.0f, 4.0f}});
  input.input_params.multimodal.deep_stacks = {deep_stack};

  const DecodeBatchContext ctx = make_decode_batch_context(input);
  DecodeExpansionOptions decode_options;
  decode_options.block_size = 4;
  DecodeBuildBuffers buf = expand_decode_inputs(ctx, decode_options);
  build_decode_metadata(input.input_params, buf, decode_options);

  EXPECT_EQ(input.input_params.attention.host.q_seq_lens,
            std::vector<int32_t>({1, 1}));
  EXPECT_TRUE(input.input_params.is_spec_verify);
  const auto retained_pixels =
      input.input_params.multimodal.mm_data.get<torch::Tensor>("pixel_values");
  ASSERT_TRUE(retained_pixels.has_value());
  EXPECT_TRUE(torch::equal(*retained_pixels, pixels));
  ASSERT_EQ(input.input_params.multimodal.deep_stacks.size(), 1u);
  EXPECT_EQ(input.input_params.multimodal.deep_stacks.front().data_ptr(),
            deep_stack.data_ptr());
}

TEST(DecodeInputBuilderTest, ChunkedExpansionCommitsPerRequestLengths) {
  auto [input, ctx] = make_two_request_ctx();

  const DecodeExpansionOptions decode_options = make_verify_layout(
      /*num_val_tokens=*/3, /*block_size=*/4, /*chunked=*/true);
  DecodeBuildBuffers buf = expand_decode_inputs(ctx, decode_options);
  build_decode_metadata(input.input_params, buf, decode_options);

  EXPECT_EQ(input.input_params.attention.host.kv_seq_lens,
            std::vector<int32_t>({7, 11}));
  EXPECT_EQ(input.input_params.attention.host.q_seq_lens,
            std::vector<int32_t>({3, 3}));
  EXPECT_EQ(input.input_params.meta.num_sequences, 2);
  EXPECT_TRUE(input.input_params.meta.batch_forward_type.is_chunked_prefill());
}

TEST(DecodeInputBuilderTest, FlatExpansionCommitsOneRowPerToken) {
  auto [input, ctx] = make_two_request_ctx();

  const DecodeExpansionOptions decode_options = make_verify_layout(
      /*num_val_tokens=*/2, /*block_size=*/4, /*chunked=*/false);
  DecodeBuildBuffers buf = expand_decode_inputs(ctx, decode_options);

  EXPECT_EQ(buf.out_token_ids, std::vector<int32_t>({10, -1, 20, -1}));
  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({4, 5, 8, 9}));
  EXPECT_EQ(buf.out_new_cache_slots, std::vector<int32_t>({4, 5, 20, 21}));

  build_decode_metadata(input.input_params, buf, decode_options);
  EXPECT_EQ(tensor_to_vector<int32_t>(
                input.input_params.attention.device.paged_kv_indptr),
            std::vector<int32_t>({0, 2, 4, 7, 10}));
  EXPECT_EQ(input.input_params.meta.num_sequences, 4);
}

TEST(DecodeInputBuilderTest, NonCausalTokensShareFullQueryKvLength) {
  const std::vector<int32_t> kv_seq_lens = {5, 9};
  const std::vector<torch::Tensor> multi_block_tables = {
      torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt)};
  LlmForwardInput input =
      make_multiblock_forward_input(torch::tensor({10, 20}, torch::kInt),
                                    torch::tensor({4, 8}, torch::kInt),
                                    multi_block_tables,
                                    kv_seq_lens);
  const DecodeBatchContext ctx = make_decode_batch_context(input);

  const DecodeExpansionOptions decode_options{{3,
                                               false,
                                               BatchForwardType::DECODE,
                                               /*leading_zero_cu=*/false,
                                               /*causal=*/false},
                                              /*block_size=*/4};
  DecodeBuildBuffers buf = expand_decode_inputs(ctx, decode_options);
  EXPECT_EQ(buf.out_kv_seq_lens, std::vector<int32_t>({7, 7, 7, 11, 11, 11}));
  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({4, 5, 6, 8, 9, 10}));
  ASSERT_EQ(buf.out_multi_block_tables.size(), 1);
  ASSERT_EQ(buf.out_multi_block_tables[0].size(), 6);
  EXPECT_EQ(buf.out_multi_block_tables[0][0], std::vector<int32_t>({0, 1, 2}));
  EXPECT_EQ(buf.out_multi_block_tables[0][3], std::vector<int32_t>({3, 4, 5}));

  input.input_params.meta.actual_num_sequences = 2;
  build_decode_metadata(input.input_params, buf, decode_options);
  EXPECT_EQ(input.input_params.meta.num_sequences, 6);
  EXPECT_EQ(input.input_params.meta.actual_num_sequences, 6);
  EXPECT_TRUE(input.input_params.meta.batch_forward_type.is_decode());
}

TEST(DecodeInputBuilderTest, ResolveTokenWithPositionOffset) {
  std::vector<int64_t> last_step_tokens = {11, -1, 13, -1, -1, -1};
  const Slice<int64_t> last_step_slice = {last_step_tokens.data(),
                                          last_step_tokens.size()};

  const auto [token_id, position_offset] = resolve_token_with_position_offset(
      /*input_token_id=*/-1, last_step_slice, /*last_step_decode_num=*/3);
  EXPECT_EQ(token_id, 13);
  EXPECT_EQ(position_offset, 1);

  const auto [no_accept_token_id, no_accept_position_offset] =
      resolve_token_with_position_offset(
          /*input_token_id=*/-2, last_step_slice, /*last_step_decode_num=*/3);
  EXPECT_EQ(no_accept_token_id, 0);
  EXPECT_EQ(no_accept_position_offset, -1);
}

}  // namespace

}  // namespace xllm
