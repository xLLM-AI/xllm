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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "framework/model/model_input_params.h"
#include "framework/parallel_state/npu/cp/plan.h"
#include "framework/parallel_state/npu_cp_closure.h"

namespace xllm::npu::cp {
namespace {

bool tensor_equal(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  if (!lhs.defined() && !rhs.defined()) {
    return true;
  }
  if (!lhs.defined() || !rhs.defined() || lhs.sizes() != rhs.sizes()) {
    return false;
  }
  return lhs.to(torch::kCPU).eq(rhs.to(torch::kCPU)).all().item<bool>();
}

nlohmann::json mapping_data() {
  nlohmann::json mapping;
  mapping["attnTpSize"] = 1;
  mapping["attnTp"]["rank"] = 0;
  mapping["attnCp"]["rankIds"] = {0, 1};
  mapping["attnCpSize"] = 2;
  mapping["moeEpSize"] = 1;
  return mapping;
}

SourceLayout source_layout() {
  SourceLayout source;
  source.q_seq_lens = {7, 5, 3};
  source.positions = torch::arange(15, torch::kInt32);
  source.kv_cache_tokens_per_seq = {0, 0, 0};
  return source;
}

ParallelConfig parallel_config() {
  ParallelConfig config;
  config.size = 2;
  config.rank = 0;
  config.block_size = 128;
  config.kv_split_size = 2;
  config.num_experts_per_tok = 8;
  config.mapping_data = mapping_data();
  config.device = torch::kCPU;
  config.dtype = torch::kBFloat16;
  return config;
}

void expect_prefill_inputs_equal(const CpPrefillInputs& lhs,
                                 const CpPrefillInputs& rhs) {
  EXPECT_TRUE(tensor_equal(lhs.cp_load_balance_idx, rhs.cp_load_balance_idx));
  EXPECT_TRUE(tensor_equal(lhs.cp_o_recover_idx, rhs.cp_o_recover_idx));
  EXPECT_TRUE(tensor_equal(lhs.cp_kv_recover_idx, rhs.cp_kv_recover_idx));
  EXPECT_TRUE(tensor_equal(lhs.k_gather_index_prev, rhs.k_gather_index_prev));
  EXPECT_TRUE(tensor_equal(lhs.k_gather_index_next, rhs.k_gather_index_next));
  EXPECT_TRUE(tensor_equal(lhs.actual_seq_lengths_query_prev,
                           rhs.actual_seq_lengths_query_prev));
  EXPECT_TRUE(tensor_equal(lhs.actual_seq_lengths_query_next,
                           rhs.actual_seq_lengths_query_next));
  EXPECT_TRUE(tensor_equal(lhs.actual_seq_lengths_key_prev,
                           rhs.actual_seq_lengths_key_prev));
  EXPECT_TRUE(tensor_equal(lhs.actual_seq_lengths_key_next,
                           rhs.actual_seq_lengths_key_next));
}

void expect_padding_equal(const CpEpPaddingData& lhs,
                          const CpEpPaddingData& rhs) {
  EXPECT_TRUE(tensor_equal(lhs.attn_padding_idx(), rhs.attn_padding_idx()));
  EXPECT_TRUE(tensor_equal(lhs.attn_unpadding_idx(), rhs.attn_unpadding_idx()));
  EXPECT_TRUE(tensor_equal(lhs.ffn_padding_idx(), rhs.ffn_padding_idx()));
  EXPECT_TRUE(tensor_equal(lhs.ffn_unpadding_idx(), rhs.ffn_unpadding_idx()));
  EXPECT_TRUE(tensor_equal(lhs.lm_head_skip_padding_token_indices(),
                           rhs.lm_head_skip_padding_token_indices()));
  EXPECT_TRUE(tensor_equal(lhs.gather_prenorm_idx(), rhs.gather_prenorm_idx()));
  EXPECT_TRUE(tensor_equal(lhs.padding_idx(), rhs.padding_idx()));
  EXPECT_TRUE(tensor_equal(lhs.un_padding_idx(), rhs.un_padding_idx()));
  EXPECT_TRUE(tensor_equal(lhs.dynamic_ep_idx(), rhs.dynamic_ep_idx()));
  EXPECT_TRUE(tensor_equal(lhs.moe_idx(), rhs.moe_idx()));
  EXPECT_TRUE(tensor_equal(lhs.expert_array(), rhs.expert_array()));
}

TEST(CpPlanTest, BuildMatchesLegacyComponents) {
  const SourceLayout source = source_layout();
  const ParallelConfig config = parallel_config();
  const Plan plan = Plan::build(source, config);

  const NpuCpPrefillPlan legacy_layout =
      build_npu_cp_prefill_plan(config.size,
                                config.rank,
                                source.q_seq_lens,
                                source.positions,
                                source.have_prefix_slots,
                                source.kv_cache_tokens_per_seq,
                                config.block_size,
                                config.kv_split_size);
  const CpPrefillInputs legacy_inputs =
      prepare_cp_prefill_inputs_from_plan(legacy_layout,
                                          source.have_prefix_slots,
                                          source.kv_cache_tokens_per_seq,
                                          config.block_size,
                                          config.kv_split_size);
  const CpEpPaddingData legacy_padding =
      CpEpPadding(legacy_layout.local_padded_token_num,
                  config.num_experts_per_tok,
                  config.mapping_data,
                  config.device,
                  config.dtype,
                  config.is_prefill)
          .build();

  EXPECT_TRUE(plan.enabled());
  EXPECT_EQ(plan.size(), legacy_layout.cp_size);
  EXPECT_EQ(plan.rank(), legacy_layout.cp_rank);
  EXPECT_EQ(plan.global_real_token_num(), legacy_layout.global_real_token_num);
  EXPECT_EQ(plan.local_padded_token_num(),
            legacy_layout.local_padded_token_num);
  expect_prefill_inputs_equal(plan.prefill_inputs(), legacy_inputs);
  expect_padding_equal(plan.ep_padding_data(), legacy_padding);
}

TEST(CpPlanTest, PreprocessMatchesLegacyClosure) {
  const SourceLayout source = source_layout();
  const ParallelConfig config = parallel_config();
  const Plan plan = Plan::build(source, config);
  const NpuCpPrefillPlan legacy_layout =
      build_npu_cp_prefill_plan(config.size,
                                config.rank,
                                source.q_seq_lens,
                                source.positions,
                                source.have_prefix_slots,
                                source.kv_cache_tokens_per_seq,
                                config.block_size,
                                config.kv_split_size);

  torch::Tensor hidden = torch::arange(60, torch::kFloat).view({15, 4});
  torch::Tensor positions = source.positions;
  ModelInputParams params;
  params.meta.num_sequences = 3;
  params.meta.q_max_seq_len = 7;
  params.meta.kv_max_seq_len = 7;
  params.attention.host.q_seq_lens = source.q_seq_lens;
  params.attention.host.kv_seq_lens = source.q_seq_lens;

  torch::Tensor expected_hidden = npu_cp::localize(hidden, legacy_layout);
  torch::Tensor expected_positions =
      npu_cp::localize_positions(legacy_layout, positions);
  ModelInputParams expected_params = params;
  apply_cp_local_metadata_from_plan(
      expected_params, legacy_layout, torch::kCPU);

  plan.preprocess(hidden, positions, params);

  EXPECT_TRUE(tensor_equal(hidden, expected_hidden));
  EXPECT_TRUE(tensor_equal(positions, expected_positions));
  EXPECT_EQ(params.attention.host.q_seq_lens,
            expected_params.attention.host.q_seq_lens);
  EXPECT_EQ(params.attention.host.kv_seq_lens,
            expected_params.attention.host.kv_seq_lens);
  EXPECT_EQ(params.attention.host.q_cu_seq_lens,
            expected_params.attention.host.q_cu_seq_lens);
  EXPECT_TRUE(tensor_equal(params.attention.device.q_seq_lens,
                           expected_params.attention.device.q_seq_lens));
  EXPECT_TRUE(tensor_equal(params.attention.device.kv_seq_lens,
                           expected_params.attention.device.kv_seq_lens));
  EXPECT_EQ(params.meta.q_max_seq_len, expected_params.meta.q_max_seq_len);
  EXPECT_EQ(params.meta.kv_max_seq_len, expected_params.meta.kv_max_seq_len);
}

TEST(CpPlanTest, RecoveredSlotsMatchLegacyAndAreIdempotent) {
  const SourceLayout source = source_layout();
  const ParallelConfig config = parallel_config();
  const Plan plan = Plan::build(source, config);
  const NpuCpPrefillPlan legacy_layout =
      build_npu_cp_prefill_plan(config.size,
                                config.rank,
                                source.q_seq_lens,
                                source.positions,
                                source.have_prefix_slots,
                                source.kv_cache_tokens_per_seq,
                                config.block_size,
                                config.kv_split_size);
  torch::Tensor slots = torch::arange(15, torch::kInt32) + 1000;
  torch::Tensor expected = npu_cp::localize_slots_recovered(
      slots, legacy_layout, plan.prefill_inputs().cp_kv_recover_idx);

  torch::Tensor recovered = plan.localize_slots_recovered(slots);
  EXPECT_TRUE(tensor_equal(recovered, expected));
  EXPECT_EQ(recovered.numel(), plan.recovered_token_num());
  EXPECT_TRUE(plan.localize_slots_recovered(recovered).is_same(recovered));
}

TEST(CpPlanTest, DisabledPlanIsStrictNoOp) {
  const Plan plan;
  torch::Tensor hidden = torch::randn({4, 8});
  torch::Tensor positions = torch::arange(4, torch::kInt32);
  torch::Tensor slots = torch::arange(4, torch::kInt32);
  ModelInputParams params;
  params.attention.host.q_seq_lens = {4};

  torch::Tensor original_hidden = hidden;
  torch::Tensor original_positions = positions;
  plan.preprocess(hidden, positions, params);

  EXPECT_FALSE(plan.enabled());
  EXPECT_TRUE(hidden.is_same(original_hidden));
  EXPECT_TRUE(positions.is_same(original_positions));
  EXPECT_EQ(params.attention.host.q_seq_lens, std::vector<int32_t>({4}));
  EXPECT_TRUE(plan.postprocess(hidden, nullptr).is_same(hidden));
  EXPECT_TRUE(plan.localize_slots_recovered(slots).is_same(slots));
}

}  // namespace
}  // namespace xllm::npu::cp
