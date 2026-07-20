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

#include "layers/mlu/deepseek_v2_sparse_moe_block.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/fused_moe.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {

class DeepseekV2SparseMoEBlockTestPeer {
 public:
  static FusedMoE moe(DeepseekV2SparseMoEBlockImpl& block) {
    return block.moe_;
  }

  static ProcessGroup* routed_pg(DeepseekV2SparseMoEBlockImpl& block) {
    return block.routed_pg();
  }

  static void set_enable_deep_ep(DeepseekV2SparseMoEBlockImpl& block,
                                 bool enable) {
    block.enable_deep_ep_ = enable;
  }
};

class DeepseekV2SparseMoEBlockTest : public ::testing::Test {
 protected:
  void SetUp() override {
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(Platform::type_torch(), 0)
                   .requires_grad(false);
    model_args_ = test::create_default_model_args();
    model_args_.hidden_size() = 256;
    model_args_.moe_intermediate_size() = 256;
    model_args_.n_routed_experts() = 4;
    model_args_.num_experts_per_tok() = 2;
    model_args_.n_group() = 1;
    model_args_.topk_group() = 2;
    model_args_.n_shared_experts() = 1;
    model_args_.routed_scaling_factor() = 1.0f;
    model_args_.norm_topk_prob() = true;
    model_args_.hidden_act() = "silu";
    model_args_.scoring_func() = "softmax";
    model_args_.topk_method() = "greedy";
    set_pg_ctx();
  }

  void set_pg_ctx() {
    global_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, /*world_size=*/2);
    tp_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, /*world_size=*/2);
    single_rank_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, /*world_size=*/1);

    parallel_args_ =
        ParallelArgs(/*rank=*/0, /*world_size=*/2, global_pg_.get());
    parallel_args_.process_group_ = global_pg_.get();
    parallel_args_.tp_group_ = tp_pg_.get();
    parallel_args_.single_rank_group_ = single_rank_pg_.get();
    parallel_args_.cp_group_ = tp_pg_.get();
    parallel_args_.ep_size_ = 1;
  }

  void set_tp_ctx(int64_t world_size, int64_t ep_size) {
    global_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, world_size);
    tp_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, world_size);
    single_rank_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, /*world_size=*/1);
    dp_pg_.reset();

    parallel_args_ =
        ParallelArgs(/*rank=*/0, world_size, /*dp_size=*/1, global_pg_.get());
    parallel_args_.process_group_ = global_pg_.get();
    parallel_args_.tp_group_ = tp_pg_.get();
    parallel_args_.single_rank_group_ = single_rank_pg_.get();
    parallel_args_.cp_group_ = tp_pg_.get();
    parallel_args_.dp_local_process_group_ = nullptr;
    parallel_args_.ep_size_ = ep_size;
    parallel_args_.moe_ep_group_ = global_pg_.get();
    parallel_args_.moe_tp_group_ = tp_pg_.get();
  }

  void set_tp_dp_ctx(int64_t world_size,
                     int64_t dp_size,
                     int64_t tp_size,
                     int64_t ep_size) {
    global_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, world_size);
    dp_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, dp_size);
    tp_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, tp_size);
    single_rank_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, /*world_size=*/1);

    parallel_args_ =
        ParallelArgs(/*rank=*/0, world_size, dp_size, global_pg_.get());
    parallel_args_.process_group_ = global_pg_.get();
    parallel_args_.tp_group_ = tp_pg_.get();
    parallel_args_.single_rank_group_ = single_rank_pg_.get();
    parallel_args_.cp_group_ = tp_pg_.get();
    parallel_args_.dp_local_process_group_ = dp_pg_.get();
    parallel_args_.ep_size_ = ep_size;
    parallel_args_.moe_ep_group_ = global_pg_.get();
    parallel_args_.moe_tp_group_ = tp_pg_.get();
  }

  torch::Tensor mat(int64_t rows, const std::vector<float>& vals) const {
    return torch::tensor(vals, fp32_opts()).reshape({rows, 2});
  }

  torch::TensorOptions fp32_opts() const {
    return torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(options_.device());
  }

  std::unordered_map<std::string, torch::Tensor> create_fp_weights(
      int64_t n_shared_experts) const {
    std::unordered_map<std::string, torch::Tensor> weight_dict;
    const int64_t num_experts = model_args_.n_routed_experts();
    const int64_t hidden_size = model_args_.hidden_size();
    const int64_t intermediate_size = model_args_.moe_intermediate_size();

    for (int64_t expert_id = 0; expert_id < num_experts; ++expert_id) {
      const std::string expert_prefix =
          "experts." + std::to_string(expert_id) + ".";
      const std::string seed_prefix =
          "deepseek_v2_sparse_moe_block.expert_" + std::to_string(expert_id);
      weight_dict[expert_prefix + "gate_proj.weight"] =
          test::seeded_tensor(seed_prefix + ".gate_proj",
                              {intermediate_size, hidden_size},
                              torch::kBFloat16,
                              options_.device());
      weight_dict[expert_prefix + "up_proj.weight"] =
          test::seeded_tensor(seed_prefix + ".up_proj",
                              {intermediate_size, hidden_size},
                              torch::kBFloat16,
                              options_.device());
      weight_dict[expert_prefix + "down_proj.weight"] =
          test::seeded_tensor(seed_prefix + ".down_proj",
                              {hidden_size, intermediate_size},
                              torch::kBFloat16,
                              options_.device());
    }

    weight_dict["gate.weight"] =
        test::seeded_tensor("deepseek_v2_sparse_moe_block.gate",
                            {num_experts, hidden_size},
                            torch::kBFloat16,
                            options_.device());

    if (n_shared_experts > 0) {
      const int64_t shared_size = intermediate_size * n_shared_experts;
      weight_dict["shared_experts.gate_proj.weight"] =
          test::seeded_tensor("deepseek_v2_sparse_moe_block.shared.gate_proj",
                              {shared_size, hidden_size},
                              torch::kBFloat16,
                              options_.device());
      weight_dict["shared_experts.up_proj.weight"] =
          test::seeded_tensor("deepseek_v2_sparse_moe_block.shared.up_proj",
                              {shared_size, hidden_size},
                              torch::kBFloat16,
                              options_.device());
      weight_dict["shared_experts.down_proj.weight"] =
          test::seeded_tensor("deepseek_v2_sparse_moe_block.shared.down_proj",
                              {hidden_size, shared_size},
                              torch::kBFloat16,
                              options_.device());
    }

    return weight_dict;
  }

  DeepseekV2SparseMoEBlock create_block() const {
    return DeepseekV2SparseMoEBlock(
        model_args_, quant_args_, parallel_args_, options_);
  }

  FusedMoE create_raw_moe() const {
    const FusedMoEArgs moe_args{.is_gated = true,
                                .enable_result_reduction = false};
    return FusedMoE(
        model_args_, moe_args, quant_args_, parallel_args_, options_);
  }

  torch::Tensor run_comm(torch::Tensor x, ProcessGroup* pg) const {
    return x + (pg == tp_pg_.get() ? 10.0f : 20.0f);
  }

  torch::Tensor run_reduce(torch::Tensor x, ProcessGroup* pg) const {
    return x + (pg == tp_pg_.get() ? 100.0f : 200.0f);
  }

  void sync_dev() const {
    xllm::Device(options_.device()).synchronize_default_stream();
  }

  ModelArgs model_args_;
  QuantArgs quant_args_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  torch::TensorOptions options_;
  std::unique_ptr<test::MockProcessGroup> global_pg_;
  std::unique_ptr<test::MockProcessGroup> dp_pg_;
  std::unique_ptr<test::MockProcessGroup> tp_pg_;
  std::unique_ptr<test::MockProcessGroup> single_rank_pg_;

  v32_cp::DeepseekV32CPContext make_sp_ctx(
      std::vector<int32_t> tokens_per_rank = {2, 2}) const {
    v32_cp::DeepseekV32CPContext sp_ctx;
    sp_ctx.rank = 0;
    sp_ctx.process_group = tp_pg_.get();
    sp_ctx.comm_plan.tokens_per_rank = std::move(tokens_per_rank);
    sp_ctx.comm_plan.padded_tokens_per_rank = sp_ctx.comm_plan.tokens_per_rank;
    sp_ctx.comm_plan.token_num_offset = 0;
    sp_ctx.comm_plan.ffn_can_rs =
        v32_sp::can_ffn_rs(sp_ctx.comm_plan.tokens_per_rank);
    return sp_ctx;
  }
};

TEST_F(DeepseekV2SparseMoEBlockTest, PlanExecEnablesAll2AllOnlyForDecode) {
  set_tp_ctx(/*world_size=*/2, /*ep_size=*/2);
  auto block = create_block();
  DeepseekV2SparseMoEBlockTestPeer::set_enable_deep_ep(*block, true);

  ModelInputParams decode_params;
  decode_params.parallel.dp_global_token_nums = {1, 1};
  decode_params.parallel.dp_is_decode = {1, 1};
  auto decode_cfg = block->plan_exec(decode_params);
  EXPECT_TRUE(decode_cfg.enable_all2all);
  EXPECT_FALSE(decode_cfg.need_dp_gather);

  ModelInputParams mixed_params;
  mixed_params.parallel.dp_global_token_nums = {2, 1};
  mixed_params.parallel.dp_is_decode = {0, 1};
  auto mixed_cfg = block->plan_exec(mixed_params);
  EXPECT_FALSE(mixed_cfg.enable_all2all);
  EXPECT_FALSE(mixed_cfg.need_dp_gather);
}

TEST_F(DeepseekV2SparseMoEBlockTest, PlanExecSetsDpGatherWhenAll2AllOff) {
  set_tp_dp_ctx(/*world_size=*/4, /*dp_size=*/2, /*tp_size=*/2, /*ep_size=*/4);
  auto block = create_block();

  ModelInputParams input_params;
  input_params.parallel.dp_global_token_nums = {3, 1};
  input_params.parallel.dp_is_decode = {0, 0};

  auto cfg = block->plan_exec(input_params);
  EXPECT_FALSE(cfg.enable_all2all);
  EXPECT_TRUE(cfg.need_dp_gather);
}

TEST_F(DeepseekV2SparseMoEBlockTest, PrepInDpGatherBuildsLocalSkip) {
  set_tp_dp_ctx(/*world_size=*/4, /*dp_size=*/2, /*tp_size=*/2, /*ep_size=*/4);
  auto block = create_block();

  ModelInputParams input_params;
  input_params.parallel.dp_global_token_nums = {3, 1};
  auto attn_out =
      mat(/*rows=*/4, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  auto residual =
      mat(/*rows=*/4, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f});
  auto full_tokens =
      mat(/*rows=*/4, {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f, 77.0f, 88.0f});
  tp_pg_->set_allgather_outputs(
      {full_tokens.slice(0, 0, 2), full_tokens.slice(0, 2, 4)});

  auto prep = block->prep_in(attn_out,
                             residual,
                             input_params,
                             block->plan_exec(input_params),
                             DeepseekV2AttentionImpl::PostAttnLayout::kTpShard);

  EXPECT_TRUE(prep.need_dp_gather);
  EXPECT_FALSE(prep.need_tp_pad);
  EXPECT_FALSE(prep.pad_info.active);
  test::verify_tensor_close(prep.ffn_in, full_tokens.slice(0, 0, 2));
  test::verify_tensor_close(
      prep.skip_local,
      mat(/*rows=*/3, {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f}));
}

TEST_F(DeepseekV2SparseMoEBlockTest, GatherInDpGatherRebuildsGlobalTokens) {
  set_tp_dp_ctx(/*world_size=*/4, /*dp_size=*/2, /*tp_size=*/2, /*ep_size=*/4);
  auto block = create_block();

  ModelInputParams input_params;
  input_params.parallel.dp_global_token_nums = {3, 1};
  DeepseekV2SparseMoEBlockImpl::PrepOut prep;
  prep.ffn_in = mat(/*rows=*/2, {11.0f, 22.0f, 33.0f, 44.0f});
  prep.need_dp_gather = true;
  auto dp0_tp1 = mat(/*rows=*/2, {55.0f, 66.0f, 0.0f, 0.0f});
  auto dp1_tp0 = mat(/*rows=*/2, {101.0f, 202.0f, 0.0f, 0.0f});
  auto dp1_tp1 = torch::zeros_like(dp1_tp0);
  global_pg_->set_allgather_outputs({prep.ffn_in, dp0_tp1, dp1_tp0, dp1_tp1});

  auto gathered = block->gather_in(prep, input_params);

  test::verify_tensor_close(
      gathered,
      mat(/*rows=*/4,
          {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f, 101.0f, 202.0f}));
}

TEST_F(DeepseekV2SparseMoEBlockTest, PrepInAll2AllPadsTpShardInput) {
  set_tp_ctx(/*world_size=*/2, /*ep_size=*/2);
  auto block = create_block();
  DeepseekV2SparseMoEBlockTestPeer::set_enable_deep_ep(*block, true);

  ModelInputParams input_params;
  input_params.parallel.dp_global_token_nums = {1, 1};
  input_params.parallel.dp_is_decode = {1, 1};
  auto attn_out = mat(/*rows=*/3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto residual = mat(/*rows=*/3, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});

  auto prep = block->prep_in(attn_out,
                             residual,
                             input_params,
                             block->plan_exec(input_params),
                             DeepseekV2AttentionImpl::PostAttnLayout::kTpShard);

  EXPECT_FALSE(prep.need_dp_gather);
  EXPECT_TRUE(prep.need_tp_pad);
  EXPECT_TRUE(prep.pad_info.active);
  EXPECT_EQ(prep.pad_info.original_tokens, 3);
  EXPECT_EQ(prep.pad_info.padded_tokens, 4);
  test::verify_tensor_close(prep.ffn_in,
                            mat(/*rows=*/2, {11.0f, 22.0f, 33.0f, 44.0f}));
  test::verify_tensor_close(prep.skip_local, prep.ffn_in);
}

TEST_F(DeepseekV2SparseMoEBlockTest, PrepInUsesProvidedExecCfg) {
  set_tp_dp_ctx(/*world_size=*/4, /*dp_size=*/2, /*tp_size=*/2, /*ep_size=*/4);
  auto block = create_block();

  ModelInputParams input_params;
  input_params.parallel.dp_global_token_nums = {3, 1};
  input_params.parallel.dp_is_decode = {0, 0};
  auto planned_cfg = block->plan_exec(input_params);
  EXPECT_FALSE(planned_cfg.enable_all2all);
  EXPECT_TRUE(planned_cfg.need_dp_gather);

  DeepseekV2SparseMoEBlockImpl::ExecCfg forced_cfg;
  forced_cfg.enable_all2all = true;
  forced_cfg.need_dp_gather = false;
  auto attn_out = mat(/*rows=*/3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto residual = mat(/*rows=*/3, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});

  auto prep = block->prep_in(attn_out,
                             residual,
                             input_params,
                             forced_cfg,
                             DeepseekV2AttentionImpl::PostAttnLayout::kTpShard);

  EXPECT_FALSE(prep.need_dp_gather);
  EXPECT_TRUE(prep.need_tp_pad);
  EXPECT_TRUE(prep.pad_info.active);
  test::verify_tensor_close(prep.skip_local, prep.ffn_in);
}

TEST_F(DeepseekV2SparseMoEBlockTest, MergeOutTpPadGathersAndUnpads) {
  set_tp_ctx(/*world_size=*/2, /*ep_size=*/2);
  auto block = create_block();

  ModelInputParams input_params;
  DeepseekV2SparseMoEBlockImpl::PrepOut prep;
  prep.need_tp_pad = true;
  prep.pad_info = {.original_tokens = 3, .padded_tokens = 4, .active = true};
  auto shard0 = mat(/*rows=*/2, {1.0f, 2.0f, 3.0f, 4.0f});
  auto shard1 = mat(/*rows=*/2, {5.0f, 6.0f, 0.0f, 0.0f});
  prep.skip_local = shard0;
  tp_pg_->set_allgather_outputs({shard0, shard1});

  auto merged = block->merge_out(shard0, prep, input_params);

  test::verify_tensor_close(
      merged, mat(/*rows=*/3, {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f}));
}

TEST_F(DeepseekV2SparseMoEBlockTest, MergeOutDpGatherSlicesLocalTokens) {
  set_tp_dp_ctx(/*world_size=*/4, /*dp_size=*/2, /*tp_size=*/2, /*ep_size=*/4);
  auto block = create_block();

  ModelInputParams input_params;
  input_params.parallel.dp_global_token_nums = {3, 1};
  DeepseekV2SparseMoEBlockImpl::PrepOut prep;
  prep.skip_local = mat(/*rows=*/3, {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f});
  prep.need_dp_gather = true;
  auto ffn_out =
      mat(/*rows=*/4,
          {101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f});

  auto merged = block->merge_out(ffn_out, prep, input_params);

  test::verify_tensor_close(
      merged,
      mat(/*rows=*/3, {112.0f, 124.0f, 136.0f, 148.0f, 160.0f, 172.0f}));
}

TEST_F(DeepseekV2SparseMoEBlockTest, ForwardReducePathCombinesSharedAndRouted) {
  auto block = create_block();
  auto raw_moe = create_raw_moe();
  StateDict state_dict(create_fp_weights(/*n_shared_experts=*/1));
  block->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  auto hidden_states = test::seeded_tensor("deepseek_v2_sparse_moe_block.input",
                                           {4, model_args_.hidden_size()},
                                           torch::kBFloat16,
                                           options_.device());
  auto routed = raw_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false);
  auto shared = raw_moe->forward_shared(hidden_states);
  auto* shared_pg = raw_moe->shared_pg();
  ASSERT_TRUE(shared.defined());
  ASSERT_EQ(shared_pg, single_rank_pg_.get());

  int comm_calls = 0;
  int reduce_calls = 0;
  auto result = block->forward(
      hidden_states,
      /*enable_moe_all2all=*/false,
      DeepseekV2SparseMoEBlockImpl::CommFns{
          .can_keep_local = std::function<bool(ProcessGroup*)>(
              [](ProcessGroup*) { return false; }),
          .comm = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++comm_calls;
                return run_comm(std::move(x), pg);
              }),
          .reduce = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++reduce_calls;
                return run_reduce(std::move(x), pg);
              }),
      });

  sync_dev();

  auto expected = run_reduce(routed, tp_pg_.get()) + shared;
  EXPECT_FALSE(result.keep_local_output);
  EXPECT_EQ(comm_calls, 0);
  EXPECT_EQ(reduce_calls, 1);
  test::verify_tensor_close(result.output, expected, 1e-3, 1e-4);
}

TEST_F(DeepseekV2SparseMoEBlockTest, ForwardReduceOverlapUsesAsyncReduceFns) {
  auto block = create_block();
  auto raw_moe = create_raw_moe();
  StateDict state_dict(create_fp_weights(/*n_shared_experts=*/1));
  block->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  auto hidden_states =
      test::seeded_tensor("deepseek_v2_sparse_moe_block.async_reduce",
                          {4, model_args_.hidden_size()},
                          torch::kBFloat16,
                          options_.device());
  auto routed = raw_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false);
  auto shared = raw_moe->forward_shared(hidden_states);
  ASSERT_TRUE(shared.defined());

  int comm_calls = 0;
  int reduce_calls = 0;
  int launch_reduce_calls = 0;
  int finish_reduce_calls = 0;
  auto result = block->forward(
      hidden_states,
      /*enable_moe_all2all=*/false,
      DeepseekV2SparseMoEBlockImpl::CommFns{
          .can_keep_local = std::function<bool(ProcessGroup*)>(
              [](ProcessGroup*) { return false; }),
          .comm = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++comm_calls;
                return run_comm(std::move(x), pg);
              }),
          .reduce = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++reduce_calls;
                return run_reduce(std::move(x), pg);
              }),
          .launch_reduce = std::function<parallel_state::ReduceAsyncCtx(
              torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++launch_reduce_calls;
                return parallel_state::ReduceAsyncCtx{
                    .tensor = run_reduce(std::move(x), pg),
                };
              }),
          .finish_reduce =
              std::function<torch::Tensor(parallel_state::ReduceAsyncCtx)>(
                  [&](parallel_state::ReduceAsyncCtx ctx) {
                    ++finish_reduce_calls;
                    return std::move(ctx.tensor);
                  }),
      });

  sync_dev();

  auto expected = run_reduce(routed, tp_pg_.get()) + shared;
  EXPECT_FALSE(result.keep_local_output);
  EXPECT_EQ(comm_calls, 0);
  EXPECT_EQ(reduce_calls, 0);
  EXPECT_EQ(launch_reduce_calls, 1);
  EXPECT_EQ(finish_reduce_calls, 1);
  test::verify_tensor_close(result.output, expected, 1e-3, 1e-4);
}

TEST_F(DeepseekV2SparseMoEBlockTest, ForwardKeepLocalUsesCommPath) {
  auto block = create_block();
  auto raw_moe = create_raw_moe();
  StateDict state_dict(create_fp_weights(/*n_shared_experts=*/1));
  block->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  auto hidden_states = test::seeded_tensor("deepseek_v2_sparse_moe_block.comm",
                                           {4, model_args_.hidden_size()},
                                           torch::kBFloat16,
                                           options_.device());
  auto routed = raw_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false);
  auto shared = raw_moe->forward_shared(hidden_states);
  auto* shared_pg = raw_moe->shared_pg();
  ASSERT_TRUE(shared.defined());
  ASSERT_EQ(shared_pg, single_rank_pg_.get());

  int comm_calls = 0;
  int reduce_calls = 0;
  auto result = block->forward(
      hidden_states,
      /*enable_moe_all2all=*/false,
      DeepseekV2SparseMoEBlockImpl::CommFns{
          .can_keep_local = std::function<bool(ProcessGroup*)>(
              [](ProcessGroup*) { return true; }),
          .comm = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++comm_calls;
                return run_comm(std::move(x), pg);
              }),
          .reduce = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++reduce_calls;
                return run_reduce(std::move(x), pg);
              }),
      });

  sync_dev();

  auto expected = run_comm(routed, tp_pg_.get()) + run_comm(shared, shared_pg);
  EXPECT_TRUE(result.keep_local_output);
  EXPECT_EQ(comm_calls, 2);
  EXPECT_EQ(reduce_calls, 0);
  test::verify_tensor_close(result.output, expected, 1e-3, 1e-4);
}

TEST_F(DeepseekV2SparseMoEBlockTest, ForwardIgnoresSharedLocalGate) {
  auto block = create_block();
  auto raw_moe = create_raw_moe();
  StateDict state_dict(create_fp_weights(/*n_shared_experts=*/1));
  block->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  auto hidden_states =
      test::seeded_tensor("deepseek_v2_sparse_moe_block.shared_fallback",
                          {4, model_args_.hidden_size()},
                          torch::kBFloat16,
                          options_.device());
  auto routed = raw_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false);
  auto shared = raw_moe->forward_shared(hidden_states);
  ASSERT_TRUE(shared.defined());

  auto* routed_pg = DeepseekV2SparseMoEBlockTestPeer::routed_pg(*block);
  auto* shared_pg = DeepseekV2SparseMoEBlockTestPeer::moe(*block)->shared_pg();
  ASSERT_NE(routed_pg, shared_pg);
  ASSERT_EQ(shared_pg, single_rank_pg_.get());

  int comm_calls = 0;
  int reduce_calls = 0;
  auto result = block->forward(
      hidden_states,
      /*enable_moe_all2all=*/false,
      DeepseekV2SparseMoEBlockImpl::CommFns{
          .can_keep_local = std::function<bool(ProcessGroup*)>(
              [&](ProcessGroup* pg) { return pg == routed_pg; }),
          .comm = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++comm_calls;
                return run_comm(std::move(x), pg);
              }),
          .reduce = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++reduce_calls;
                return run_reduce(std::move(x), pg);
              }),
      });

  sync_dev();

  auto expected = run_comm(routed, routed_pg) + run_comm(shared, shared_pg);
  EXPECT_TRUE(result.keep_local_output);
  EXPECT_EQ(comm_calls, 2);
  EXPECT_EQ(reduce_calls, 0);
  test::verify_tensor_close(result.output, expected, 1e-3, 1e-4);
}

TEST_F(DeepseekV2SparseMoEBlockTest, ForwardWithoutSharedIgnoresNullSharedPg) {
  model_args_.n_shared_experts() = 0;
  auto block = create_block();
  auto raw_moe = create_raw_moe();
  StateDict state_dict(create_fp_weights(/*n_shared_experts=*/0));
  block->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  auto hidden_states =
      test::seeded_tensor("deepseek_v2_sparse_moe_block.no_shared",
                          {4, model_args_.hidden_size()},
                          torch::kBFloat16,
                          options_.device());
  auto routed = raw_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false);
  ASSERT_FALSE(raw_moe->forward_shared(hidden_states).defined());

  auto* routed_pg = DeepseekV2SparseMoEBlockTestPeer::routed_pg(*block);

  int comm_calls = 0;
  int reduce_calls = 0;
  auto result = block->forward(
      hidden_states,
      /*enable_moe_all2all=*/false,
      DeepseekV2SparseMoEBlockImpl::CommFns{
          .can_keep_local = std::function<bool(ProcessGroup*)>(
              [&](ProcessGroup* pg) { return pg == routed_pg; }),
          .comm = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++comm_calls;
                return run_comm(std::move(x), pg);
              }),
          .reduce = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++reduce_calls;
                return run_reduce(std::move(x), pg);
              }),
      });

  sync_dev();

  auto expected = run_comm(routed, routed_pg);
  EXPECT_TRUE(result.keep_local_output);
  EXPECT_EQ(comm_calls, 1);
  EXPECT_EQ(reduce_calls, 0);
  test::verify_tensor_close(result.output, expected, 1e-3, 1e-4);
}

TEST_F(DeepseekV2SparseMoEBlockTest, ForwardSPAddsLocalSharedAfterRoutedComm) {
  auto block = create_block();
  auto raw_moe = create_raw_moe();
  StateDict state_dict(create_fp_weights(/*n_shared_experts=*/1));
  block->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  auto local = test::seeded_tensor("deepseek_v2_sparse_moe_block.sp_local",
                                   {2, model_args_.hidden_size()},
                                   torch::kBFloat16,
                                   options_.device());
  auto remote = test::seeded_tensor("deepseek_v2_sparse_moe_block.sp_remote",
                                    {2, model_args_.hidden_size()},
                                    torch::kBFloat16,
                                    options_.device());
  tp_pg_->set_allgather_outputs({local, remote});
  auto sp_ctx = make_sp_ctx();

  auto gathered = torch::cat({local, remote}, 0);
  auto routed = raw_moe->forward_experts(
      gathered, /*enable_all2all_communication=*/false);
  auto shared = raw_moe->forward_shared(local);
  auto* routed_pg = DeepseekV2SparseMoEBlockTestPeer::routed_pg(*block);
  const int64_t local_token_num = local.size(0);

  int comm_calls = 0;
  int reduce_calls = 0;
  auto result = block->forward_sp(
      local,
      sp_ctx,
      DeepseekV2SparseMoEBlockImpl::CommFns{
          .can_keep_local = std::function<bool(ProcessGroup*)>(
              [&](ProcessGroup* pg) { return pg == routed_pg; }),
          .comm = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++comm_calls;
                if (pg == routed_pg) {
                  x = x.slice(0, 0, local_token_num);
                }
                return run_comm(std::move(x), pg);
              }),
          .reduce = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++reduce_calls;
                return run_reduce(std::move(x), pg);
              }),
      });

  sync_dev();

  auto expected =
      run_comm(routed.slice(0, 0, local_token_num), routed_pg) + shared;
  EXPECT_TRUE(result.keep_local_output);
  EXPECT_EQ(comm_calls, 1);
  EXPECT_EQ(reduce_calls, 0);
  test::verify_tensor_close(result.output, expected, 1e-3, 1e-4);
}

TEST_F(DeepseekV2SparseMoEBlockTest, ForwardSPFallsBackWhenLocalKeepOff) {
  auto block = create_block();
  auto raw_moe = create_raw_moe();
  StateDict state_dict(create_fp_weights(/*n_shared_experts=*/1));
  block->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  auto local =
      test::seeded_tensor("deepseek_v2_sparse_moe_block.sp_fallback_local",
                          {2, model_args_.hidden_size()},
                          torch::kBFloat16,
                          options_.device());
  auto remote =
      test::seeded_tensor("deepseek_v2_sparse_moe_block.sp_fallback_remote",
                          {2, model_args_.hidden_size()},
                          torch::kBFloat16,
                          options_.device());
  tp_pg_->set_allgather_outputs({local, remote});
  auto sp_ctx = make_sp_ctx();

  auto gathered = torch::cat({local, remote}, 0);
  auto routed = raw_moe->forward_experts(
      gathered, /*enable_all2all_communication=*/false);
  auto shared = raw_moe->forward_shared(gathered);
  auto* routed_pg = DeepseekV2SparseMoEBlockTestPeer::routed_pg(*block);

  int comm_calls = 0;
  int reduce_calls = 0;
  auto result = block->forward_sp(
      local,
      sp_ctx,
      DeepseekV2SparseMoEBlockImpl::CommFns{
          .can_keep_local = std::function<bool(ProcessGroup*)>(
              [&](ProcessGroup* /*pg*/) { return false; }),
          .comm = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++comm_calls;
                return run_comm(std::move(x), pg);
              }),
          .reduce = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++reduce_calls;
                return run_reduce(std::move(x), pg);
              }),
      });

  sync_dev();

  auto expected = run_reduce(routed, routed_pg) + shared;
  EXPECT_FALSE(result.keep_local_output);
  EXPECT_EQ(comm_calls, 0);
  EXPECT_EQ(reduce_calls, 1);
  test::verify_tensor_close(result.output, expected, 1e-3, 1e-4);
}

}  // namespace layer
}  // namespace xllm
