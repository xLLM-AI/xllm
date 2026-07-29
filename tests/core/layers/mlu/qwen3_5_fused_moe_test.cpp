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

#include "layers/mlu/qwen3_5/qwen3_5_fused_moe.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/tests_utils.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {
namespace {

class Qwen3_5FusedMoETest : public ::testing::Test {
 protected:
  void SetUp() override {
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(Platform::type_torch(), 0)
                   .requires_grad(false);
    parallel_args_ = test::create_default_parallel_args(process_group_);
  }

  Qwen3_5FusedMoE create_moe() {
    constexpr int64_t kNumExperts = 4;
    constexpr int64_t kHiddenSize = 8;
    constexpr int64_t kIntermediateSize = 4;

    ModelArgs model_args = test::create_default_model_args();
    model_args.n_routed_experts() = kNumExperts;
    model_args.num_experts_per_tok() = 2;
    model_args.n_group() = 1;
    model_args.topk_group() = 1;
    model_args.routed_scaling_factor() = 1.0f;
    model_args.hidden_size() = kHiddenSize;
    model_args.moe_intermediate_size() = kIntermediateSize;
    model_args.n_shared_experts() = 0;
    model_args.norm_topk_prob() = true;
    model_args.scoring_func() = "softmax";

    QuantArgs quant_args = test::create_default_quant_args();
    const FusedMoEArgs moe_args{.is_gated = true,
                                .enable_result_reduction = true};
    return Qwen3_5FusedMoE(Qwen3_5FusedMoEImpl(
        model_args, moe_args, quant_args, parallel_args_, options_));
  }

  StateDict create_gate_up_dict() {
    std::unordered_map<std::string, torch::Tensor> weights;
    weights["experts.gate_up_proj.qweight"] =
        torch::ones({4, 8, 8}, options_.dtype(torch::kInt8));
    weights["experts.gate_up_proj.per_channel_scale"] =
        torch::ones({4, 8}, options_.dtype(torch::kFloat32));
    weights["experts.gate_up_proj.smooth"] =
        torch::ones({8}, options_.dtype(torch::kFloat32));
    return StateDict(std::move(weights));
  }

  StateDict create_down_dict() {
    std::unordered_map<std::string, torch::Tensor> weights;
    weights["experts.down_proj.qweight"] =
        torch::ones({4, 8, 4}, options_.dtype(torch::kInt8));
    weights["experts.down_proj.per_channel_scale"] =
        torch::ones({4, 8}, options_.dtype(torch::kFloat32));
    weights["experts.down_proj.smooth"] =
        torch::ones({4}, options_.dtype(torch::kFloat32));
    return StateDict(std::move(weights));
  }

  torch::TensorOptions options_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  std::unique_ptr<ProcessGroup> process_group_;
};

TEST_F(Qwen3_5FusedMoETest, LoadsFusedWeightsAcrossStateDicts) {
  Qwen3_5FusedMoE moe = create_moe();
  StateDict gate_up_dict = create_gate_up_dict();
  StateDict down_dict = create_down_dict();

  moe->load_state_dict(gate_up_dict);
  moe->load_state_dict(down_dict);

  moe->verify_loaded_weights();
}

TEST_F(Qwen3_5FusedMoETest, RejectsPartialFusedWeightGroup) {
  Qwen3_5FusedMoE moe = create_moe();
  std::unordered_map<std::string, torch::Tensor> weights;
  weights["experts.gate_up_proj.qweight"] =
      torch::ones({4, 8, 8}, options_.dtype(torch::kInt8));
  StateDict state_dict(std::move(weights));

  EXPECT_DEATH(moe->load_state_dict(state_dict),
               "failed to load gate_up smoothquant weights");
}

}  // namespace
}  // namespace layer
}  // namespace xllm
