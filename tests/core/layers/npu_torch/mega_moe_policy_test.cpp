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

#include <optional>
#include <string>

#include "layers/npu_torch/mega_moe_policy.h"

namespace xllm {
namespace layer {
namespace {

MegaMoeCapability supported_capability() {
  MegaMoeCapability capability;
  capability.soc_name = "Ascend910_93";
  capability.cann_version = "9.1.0-beta.3";
  capability.vendor_version = "9.1.0-beta.3";
  capability.dtype = MegaMoeDType::BFLOAT16;
  capability.activation = "swiglu";
  capability.hidden_size = 2048;
  capability.intermediate_size = 512;
  capability.num_experts = 256;
  capability.topk = 8;
  capability.ep_size = 16;
  capability.tp_size = 1;
  capability.dp_size = 16;
  capability.workspace_symbol_ready = true;
  capability.execute_symbol_ready = true;
  capability.comm_context_ready = true;
  return capability;
}

void expect_rejected(const MegaMoeCapability& capability,
                     MegaMoeRejectionReason reason) {
  const MegaMoeDecision decision =
      decide_mega_moe(MegaMoeMode::ON, capability);
  EXPECT_FALSE(decision.use_mega_moe);
  EXPECT_TRUE(decision.fatal);
  EXPECT_EQ(decision.reason, reason);
}

TEST(MegaMoePolicyTest, ParsesExplicitModes) {
  EXPECT_EQ(parse_mega_moe_mode("off"), MegaMoeMode::OFF);
  EXPECT_EQ(parse_mega_moe_mode("auto"), MegaMoeMode::AUTO);
  EXPECT_EQ(parse_mega_moe_mode("on"), MegaMoeMode::ON);
  EXPECT_EQ(parse_mega_moe_mode("invalid"), std::nullopt);
}

TEST(MegaMoePolicyTest, OffKeepsLegacyExecution) {
  const MegaMoeDecision decision =
      decide_mega_moe(MegaMoeMode::OFF, supported_capability());
  EXPECT_FALSE(decision.use_mega_moe);
  EXPECT_FALSE(decision.fatal);
  EXPECT_EQ(decision.reason, MegaMoeRejectionReason::DISABLED);
}

TEST(MegaMoePolicyTest, OnAndAutoSelectExactWhitelist) {
  for (MegaMoeMode mode : {MegaMoeMode::ON, MegaMoeMode::AUTO}) {
    const MegaMoeDecision decision =
        decide_mega_moe(mode, supported_capability());
    EXPECT_TRUE(decision.use_mega_moe);
    EXPECT_FALSE(decision.fatal);
    EXPECT_EQ(decision.reason, MegaMoeRejectionReason::NONE);
  }
}

TEST(MegaMoePolicyTest, OnSupportsEp16MoeTp1WithAttentionDp16) {
  MegaMoeCapability capability = supported_capability();
  capability.ep_size = 16;
  capability.tp_size = 1;
  capability.dp_size = 16;

  const MegaMoeDecision decision =
      decide_mega_moe(MegaMoeMode::ON, capability);
  EXPECT_TRUE(decision.use_mega_moe);
  EXPECT_FALSE(decision.fatal);
  EXPECT_EQ(decision.reason, MegaMoeRejectionReason::NONE);
}

TEST(MegaMoePolicyTest, OnSupportsEp8MoeTp1WithAttentionDp8) {
  MegaMoeCapability capability = supported_capability();
  capability.ep_size = 8;
  capability.tp_size = 1;
  capability.dp_size = 8;

  const MegaMoeDecision decision =
      decide_mega_moe(MegaMoeMode::ON, capability);

  EXPECT_TRUE(decision.use_mega_moe);
  EXPECT_FALSE(decision.fatal);
  EXPECT_EQ(decision.reason, MegaMoeRejectionReason::NONE);
}

TEST(MegaMoePolicyTest, AttentionDpDoesNotAffectOperatorEligibility) {
  for (int64_t dp_size : {1, 2, 8, 16, 32}) {
    SCOPED_TRACE(dp_size);
    MegaMoeCapability capability = supported_capability();
    capability.dp_size = dp_size;

    const MegaMoeDecision decision =
        decide_mega_moe(MegaMoeMode::ON, capability);
    EXPECT_TRUE(decision.use_mega_moe);
    EXPECT_FALSE(decision.fatal);
    EXPECT_EQ(decision.reason, MegaMoeRejectionReason::NONE);
  }
}

TEST(MegaMoePolicyTest, AutoFallsBackOnlyBeforeCollectivesStart) {
  MegaMoeCapability capability = supported_capability();
  capability.ep_size = 3;
  const MegaMoeDecision before_collectives =
      decide_mega_moe(MegaMoeMode::AUTO, capability);
  EXPECT_FALSE(before_collectives.use_mega_moe);
  EXPECT_FALSE(before_collectives.fatal);
  EXPECT_EQ(before_collectives.reason, MegaMoeRejectionReason::UNSUPPORTED_EP);

  capability.collectives_started = true;
  const MegaMoeDecision after_collectives =
      decide_mega_moe(MegaMoeMode::AUTO, capability);
  EXPECT_FALSE(after_collectives.use_mega_moe);
  EXPECT_TRUE(after_collectives.fatal);
  EXPECT_EQ(after_collectives.reason,
            MegaMoeRejectionReason::COLLECTIVES_ALREADY_STARTED);
}

TEST(MegaMoePolicyTest, WeightBudgetIsFatalOnAndFallbackAuto) {
  MegaMoeCapability capability = supported_capability();
  capability.weight_budget_ready = false;

  const auto forced = decide_mega_moe(MegaMoeMode::ON, capability);
  EXPECT_FALSE(forced.use_mega_moe);
  EXPECT_TRUE(forced.fatal);
  EXPECT_EQ(forced.reason,
            MegaMoeRejectionReason::WEIGHT_BUDGET_EXCEEDED);

  const auto automatic = decide_mega_moe(MegaMoeMode::AUTO, capability);
  EXPECT_FALSE(automatic.use_mega_moe);
  EXPECT_FALSE(automatic.fatal);
  EXPECT_EQ(automatic.reason,
            MegaMoeRejectionReason::WEIGHT_BUDGET_EXCEEDED);
}

TEST(MegaMoePolicyTest, FusedMc2ConflictFailsFast) {
  MegaMoeCapability capability = supported_capability();
  capability.fused_mc2_enabled = true;
  expect_rejected(capability, MegaMoeRejectionReason::FUSED_MC2_CONFLICT);

  const MegaMoeDecision auto_decision =
      decide_mega_moe(MegaMoeMode::AUTO, capability);
  EXPECT_FALSE(auto_decision.use_mega_moe);
  EXPECT_TRUE(auto_decision.fatal);
  EXPECT_EQ(auto_decision.reason,
            MegaMoeRejectionReason::FUSED_MC2_CONFLICT);
}

TEST(MegaMoePolicyTest, RejectsUnsupportedPlatformAndVersions) {
  MegaMoeCapability capability = supported_capability();
  capability.soc_name = "Ascend910B";
  expect_rejected(capability, MegaMoeRejectionReason::UNSUPPORTED_SOC);
  capability = supported_capability();
  capability.cann_version = "9.0.0";
  expect_rejected(capability,
                  MegaMoeRejectionReason::UNSUPPORTED_CANN_VERSION);
  capability = supported_capability();
  capability.vendor_version = "9.0.0";
  expect_rejected(capability,
                  MegaMoeRejectionReason::UNSUPPORTED_VENDOR_VERSION);
}

TEST(MegaMoePolicyTest, RejectsMissingSymbolsAndContext) {
  MegaMoeCapability capability = supported_capability();
  capability.workspace_symbol_ready = false;
  expect_rejected(capability, MegaMoeRejectionReason::OP_SYMBOLS_UNAVAILABLE);
  capability = supported_capability();
  capability.execute_symbol_ready = false;
  expect_rejected(capability, MegaMoeRejectionReason::OP_SYMBOLS_UNAVAILABLE);
  capability = supported_capability();
  capability.comm_context_ready = false;
  expect_rejected(capability,
                  MegaMoeRejectionReason::COMM_CONTEXT_UNAVAILABLE);
}

TEST(MegaMoePolicyTest, RejectsUnsupportedDtypeActivationAndShape) {
  MegaMoeCapability capability = supported_capability();
  capability.dtype = MegaMoeDType::OTHER;
  expect_rejected(capability, MegaMoeRejectionReason::UNSUPPORTED_DTYPE);
  capability = supported_capability();
  capability.activation = "gelu";
  expect_rejected(capability,
                  MegaMoeRejectionReason::UNSUPPORTED_ACTIVATION);
  capability = supported_capability();
  capability.hidden_size = 4096;
  expect_rejected(capability, MegaMoeRejectionReason::UNSUPPORTED_SHAPE);
  capability = supported_capability();
  capability.intermediate_size = 1024;
  expect_rejected(capability, MegaMoeRejectionReason::UNSUPPORTED_SHAPE);
  capability = supported_capability();
  capability.num_experts = 128;
  expect_rejected(capability, MegaMoeRejectionReason::UNSUPPORTED_SHAPE);
  capability = supported_capability();
  capability.topk = 4;
  expect_rejected(capability, MegaMoeRejectionReason::UNSUPPORTED_SHAPE);
}

TEST(MegaMoePolicyTest, RejectsUnsupportedParallelAndRuntimeModes) {
  MegaMoeCapability capability = supported_capability();
  capability.ep_size = 3;
  expect_rejected(capability, MegaMoeRejectionReason::UNSUPPORTED_EP);
  capability = supported_capability();
  capability.tp_size = 2;
  expect_rejected(capability, MegaMoeRejectionReason::UNSUPPORTED_TP);
  capability = supported_capability();
  capability.graph_enabled = true;
  expect_rejected(capability, MegaMoeRejectionReason::GRAPH_ENABLED);
  capability = supported_capability();
  capability.quantization_enabled = true;
  expect_rejected(capability,
                  MegaMoeRejectionReason::QUANTIZATION_ENABLED);
}

TEST(MegaMoePolicyTest, MegaMoePreservesGlobalRoutingAndAddsSharedOnce) {
  const MegaMoeDecision decision =
      decide_mega_moe(MegaMoeMode::ON, supported_capability());
  const MegaMoeExecutionContract contract =
      make_mega_moe_execution_contract(
          decision, /*ep_size=*/16, /*has_shared_expert=*/true);
  EXPECT_TRUE(contract.preserve_global_topk);
  EXPECT_FALSE(contract.apply_local_expert_mask);
  EXPECT_FALSE(contract.reduce_routed_output_across_ep);
  EXPECT_EQ(contract.shared_output_additions, 1);
}

TEST(MegaMoePolicyTest, LegacyMasksLocalRoutingAndReducesAcrossEp) {
  const MegaMoeDecision decision =
      decide_mega_moe(MegaMoeMode::OFF, supported_capability());
  const MegaMoeExecutionContract contract =
      make_mega_moe_execution_contract(
          decision, /*ep_size=*/16, /*has_shared_expert=*/true);
  EXPECT_FALSE(contract.preserve_global_topk);
  EXPECT_TRUE(contract.apply_local_expert_mask);
  EXPECT_TRUE(contract.reduce_routed_output_across_ep);
  EXPECT_EQ(contract.shared_output_additions, 1);
}

}  // namespace
}  // namespace layer
}  // namespace xllm
