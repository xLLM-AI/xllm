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

#include "layers/mlu/deepseek_v4/deepseek_v4_sparse_moe_block.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/fused_moe.h"
#include "layers/mlu/tests_utils.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {

class DeepseekV4SparseMoEBlockTestPeer {
 public:
  static FusedMoE moe(DeepseekV4SparseMoEBlockImpl& block) {
    return block.moe_;
  }

  static ProcessGroup* routed_pg(DeepseekV4SparseMoEBlockImpl& block) {
    return block.routed_pg();
  }

  static bool need_gather(DeepseekV4SparseMoEBlockImpl& block) {
    return block.need_gather();
  }

  static void set_enable_deep_ep(DeepseekV4SparseMoEBlockImpl& block,
                                 bool enable_deep_ep) {
    block.enable_deep_ep_ = enable_deep_ep;
  }

  static std::vector<int32_t> row_dp_tokens(
      DeepseekV4SparseMoEBlockImpl& block,
      int64_t hidden_rows,
      const ModelInputParams& input_params) {
    return block.get_row_dp_tokens(hidden_rows, input_params);
  }
};

class DeepseekV4SparseMoEBlockTest : public ::testing::Test {
 protected:
  void SetUp() override {
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(Platform::type_torch(), 0)
                   .requires_grad(false);
    model_args_ = test::create_default_model_args();
    model_args_.model_type() = "deepseek_v4";
    model_args_.hidden_size() = 256;
    model_args_.moe_intermediate_size() = 128;
    model_args_.n_routed_experts() = 4;
    model_args_.num_experts_per_tok() = 2;
    model_args_.n_group() = 2;
    model_args_.topk_group() = 2;
    model_args_.routed_scaling_factor() = 1.0f;
    model_args_.norm_topk_prob() = true;
    model_args_.n_shared_experts() = 1;
    model_args_.hidden_act() = "silu";
    model_args_.scoring_func() = "sqrtsoftplus";
    model_args_.topk_method() = "noaux_tc";
    model_args_.swiglu_limit() = 10.0f;
    set_tp_ctx(/*world_size=*/1, /*ep_size=*/1);
  }

  void set_tp_ctx(int64_t world_size, int64_t ep_size) {
    global_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, world_size);
    tp_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, world_size);
    ep_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, ep_size);
    single_rank_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, /*world_size=*/1);
    dp_pg_.reset();

    parallel_args_ =
        ParallelArgs(/*rank=*/0, world_size, /*dp_size=*/1, global_pg_.get());
    parallel_args_.process_group_ = global_pg_.get();
    parallel_args_.tp_group_ = tp_pg_.get();
    parallel_args_.single_rank_group_ = single_rank_pg_.get();
    parallel_args_.cp_group_ = tp_pg_.get();
    parallel_args_.ep_size_ = ep_size;
    parallel_args_.moe_ep_group_ = ep_pg_.get();
    parallel_args_.moe_tp_group_ = tp_pg_.get();
  }

  void set_dp_ep_ctx(int64_t dp_size, bool with_dp_group) {
    const int64_t world_size = dp_size;
    global_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, world_size);
    tp_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, /*world_size=*/1);
    ep_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, /*world_size=*/2);
    single_rank_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, /*world_size=*/1);
    if (with_dp_group) {
      dp_pg_ = std::make_unique<test::MockProcessGroup>(
          options_.device(), /*rank=*/0, dp_size);
    } else {
      dp_pg_.reset();
    }

    parallel_args_ =
        ParallelArgs(/*rank=*/0, world_size, dp_size, global_pg_.get());
    parallel_args_.process_group_ = global_pg_.get();
    parallel_args_.tp_group_ = tp_pg_.get();
    parallel_args_.single_rank_group_ = single_rank_pg_.get();
    parallel_args_.cp_group_ = tp_pg_.get();
    parallel_args_.dp_local_process_group_ =
        with_dp_group ? dp_pg_.get() : nullptr;
    parallel_args_.ep_size_ = 2;
    parallel_args_.moe_ep_group_ = ep_pg_.get();
    parallel_args_.moe_tp_group_ = tp_pg_.get();
  }

  std::unordered_map<std::string, torch::Tensor> create_fp_weights() const {
    std::unordered_map<std::string, torch::Tensor> weight_dict;
    const int64_t num_experts = model_args_.n_routed_experts();
    const int64_t hidden_size = model_args_.hidden_size();
    const int64_t inter_size = model_args_.moe_intermediate_size();

    for (int64_t expert_id = 0; expert_id < num_experts; ++expert_id) {
      const std::string prefix = "experts." + std::to_string(expert_id) + ".";
      const std::string seed_prefix =
          "deepseek_v4_sparse_moe_block.expert_" + std::to_string(expert_id);
      weight_dict[prefix + "gate_proj.weight"] =
          test::seeded_tensor(seed_prefix + ".gate_proj",
                              {inter_size, hidden_size},
                              torch::kBFloat16,
                              options_.device());
      weight_dict[prefix + "up_proj.weight"] =
          test::seeded_tensor(seed_prefix + ".up_proj",
                              {inter_size, hidden_size},
                              torch::kBFloat16,
                              options_.device());
      weight_dict[prefix + "down_proj.weight"] =
          test::seeded_tensor(seed_prefix + ".down_proj",
                              {hidden_size, inter_size},
                              torch::kBFloat16,
                              options_.device());
    }

    weight_dict["gate.weight"] =
        test::seeded_tensor("deepseek_v4_sparse_moe_block.gate",
                            {num_experts, hidden_size},
                            torch::kBFloat16,
                            options_.device());
    const int64_t shared_size = inter_size * model_args_.n_shared_experts();
    weight_dict["shared_experts.gate_proj.weight"] =
        test::seeded_tensor("deepseek_v4_sparse_moe_block.shared.gate_proj",
                            {shared_size, hidden_size},
                            torch::kBFloat16,
                            options_.device());
    weight_dict["shared_experts.up_proj.weight"] =
        test::seeded_tensor("deepseek_v4_sparse_moe_block.shared.up_proj",
                            {shared_size, hidden_size},
                            torch::kBFloat16,
                            options_.device());
    weight_dict["shared_experts.down_proj.weight"] =
        test::seeded_tensor("deepseek_v4_sparse_moe_block.shared.down_proj",
                            {hidden_size, shared_size},
                            torch::kBFloat16,
                            options_.device());
    return weight_dict;
  }

  DeepseekV4SparseMoEBlock create_block(bool use_hash = false) const {
    return DeepseekV4SparseMoEBlock(
        model_args_, quant_args_, parallel_args_, options_, use_hash);
  }

  FusedMoE create_raw_moe(bool enable_result_reduction = true) const {
    const FusedMoEArgs moe_args{
        .is_gated = true,
        .enable_result_reduction = enable_result_reduction,
        .use_hash = false};
    return FusedMoE(
        model_args_, moe_args, quant_args_, parallel_args_, options_);
  }

  FusedMoEImpl::RouteInfo route_for(FusedMoE moe,
                                    torch::Tensor hidden_states) const {
    torch::Tensor rows =
        hidden_states.reshape({-1, hidden_states.size(-1)}).contiguous();
    return moe->prep_route(rows);
  }

  torch::Tensor make_hidden(const std::vector<int64_t>& shape) const {
    const int64_t numel = std::accumulate(
        shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
    return test::seeded_tensor("deepseek_v4_sparse_moe_block.hidden",
                               {numel},
                               torch::kFloat32,
                               options_.device())
        .reshape(shape)
        .to(options_.dtype());
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
  std::unique_ptr<test::MockProcessGroup> ep_pg_;
  std::unique_ptr<test::MockProcessGroup> single_rank_pg_;
};

TEST_F(DeepseekV4SparseMoEBlockTest, RoutedPgUsesEpGroupWhenEpEnabled) {
  set_tp_ctx(/*world_size=*/2, /*ep_size=*/2);
  DeepseekV4SparseMoEBlock block = create_block();

  EXPECT_EQ(DeepseekV4SparseMoEBlockTestPeer::routed_pg(*block), ep_pg_.get());
}

TEST_F(DeepseekV4SparseMoEBlockTest, RoutedPgFallsBackToTpGroup) {
  set_tp_ctx(/*world_size=*/2, /*ep_size=*/1);
  DeepseekV4SparseMoEBlock block = create_block();

  EXPECT_EQ(DeepseekV4SparseMoEBlockTestPeer::routed_pg(*block), tp_pg_.get());
}

TEST_F(DeepseekV4SparseMoEBlockTest, NeedGatherOnlyForDpEp) {
  set_dp_ep_ctx(/*dp_size=*/2, /*with_dp_group=*/true);
  DeepseekV4SparseMoEBlock block = create_block();
  EXPECT_TRUE(DeepseekV4SparseMoEBlockTestPeer::need_gather(*block));

  set_tp_ctx(/*world_size=*/2, /*ep_size=*/2);
  block = create_block();
  EXPECT_FALSE(DeepseekV4SparseMoEBlockTestPeer::need_gather(*block));
}

TEST_F(DeepseekV4SparseMoEBlockTest, ConvertsModelTokensToRowTokens) {
  set_dp_ep_ctx(/*dp_size=*/2, /*with_dp_group=*/true);
  DeepseekV4SparseMoEBlock block = create_block();
  ModelInputParams input_params;
  input_params.parallel.dp_global_token_nums = {3, 1};

  std::vector<int32_t> row_tokens =
      DeepseekV4SparseMoEBlockTestPeer::row_dp_tokens(
          *block, /*hidden_rows=*/6, input_params);

  EXPECT_EQ(row_tokens, std::vector<int32_t>({6, 2}));
}

TEST_F(DeepseekV4SparseMoEBlockTest, LoadStateDictAcceptsUnprefixedMoeKeys) {
  DeepseekV4SparseMoEBlock block = create_block();
  FusedMoE raw_moe = create_raw_moe();
  StateDict state_dict(create_fp_weights());
  block->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  torch::Tensor hidden_states = make_hidden({4, model_args_.hidden_size()});
  FusedMoEImpl::RouteInfo route = route_for(raw_moe, hidden_states);
  torch::Tensor expected = raw_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false, route);
  torch::Tensor actual = block->forward_selected(
      hidden_states, route.reduce_weight, route.expert_id, ModelInputParams());

  sync_dev();
  test::verify_tensor_close(actual, expected, 1e-3, 1e-4);
}

TEST_F(DeepseekV4SparseMoEBlockTest, SelectedRouteMergesSharedAndRouted) {
  DeepseekV4SparseMoEBlock block = create_block();
  FusedMoE raw_moe = create_raw_moe(/*enable_result_reduction=*/true);
  StateDict state_dict(create_fp_weights());
  block->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  torch::Tensor hidden_states = make_hidden({4, model_args_.hidden_size()});
  FusedMoEImpl::RouteInfo route = route_for(raw_moe, hidden_states);
  torch::Tensor actual = block->forward_selected(
      hidden_states, route.reduce_weight, route.expert_id, ModelInputParams());
  torch::Tensor expected = raw_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false, route);

  sync_dev();
  test::verify_tensor_close(actual, expected, 1e-3, 1e-4);
}

TEST_F(DeepseekV4SparseMoEBlockTest, SelectedRouteRestores3DShape) {
  DeepseekV4SparseMoEBlock block = create_block();
  FusedMoE raw_moe = create_raw_moe();
  StateDict state_dict(create_fp_weights());
  block->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  const int64_t token_num = 2;
  const int64_t hc_mult = 3;
  torch::Tensor hidden_states =
      make_hidden({token_num, hc_mult, model_args_.hidden_size()});
  FusedMoEImpl::RouteInfo route = route_for(raw_moe, hidden_states);
  torch::Tensor topk_weights =
      route.reduce_weight.reshape({token_num, hc_mult, -1});
  torch::Tensor topk_ids = route.expert_id.reshape({token_num, hc_mult, -1});

  torch::Tensor actual = block->forward_selected(
      hidden_states, topk_weights, topk_ids, ModelInputParams());

  sync_dev();
  EXPECT_EQ(actual.sizes(), hidden_states.sizes());
}

TEST_F(DeepseekV4SparseMoEBlockTest, SelectedRouteRejectsInvalidTopkShape) {
  DeepseekV4SparseMoEBlock block = create_block();
  torch::Tensor hidden_states = make_hidden({3, model_args_.hidden_size()});
  torch::Tensor topk_weights = torch::ones({2, 2}, options_);
  torch::Tensor topk_ids = torch::zeros({3, 2}, options_.dtype(torch::kInt64));

  EXPECT_ANY_THROW(block->forward_selected(
      hidden_states, topk_weights, topk_ids, ModelInputParams()));
}

TEST_F(DeepseekV4SparseMoEBlockTest, SelectedRouteRequiresDpTokensForGather) {
  set_dp_ep_ctx(/*dp_size=*/2, /*with_dp_group=*/true);
  DeepseekV4SparseMoEBlock block = create_block();
  torch::Tensor hidden_states = make_hidden({2, model_args_.hidden_size()});
  torch::Tensor topk_weights = torch::ones({2, 2}, options_);
  torch::Tensor topk_ids = torch::zeros({2, 2}, options_.dtype(torch::kInt64));

  EXPECT_DEATH(block->forward_selected(
                   hidden_states, topk_weights, topk_ids, ModelInputParams()),
               "dp_global_token_nums is empty");
}

}  // namespace layer
}  // namespace xllm
