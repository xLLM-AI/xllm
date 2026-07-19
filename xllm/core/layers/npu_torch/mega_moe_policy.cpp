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

#include "layers/npu_torch/mega_moe_policy.h"

namespace xllm {
namespace layer {
namespace {

constexpr char kSupportedSocPrefix[] = "Ascend910_93";
constexpr char kSupportedVersion[] = "9.1.0-beta.3";

MegaMoeRejectionReason find_rejection_reason(
    const MegaMoeCapability& capability) {
  if (capability.fused_mc2_enabled) {
    return MegaMoeRejectionReason::FUSED_MC2_CONFLICT;
  }
  if (capability.soc_name.rfind(kSupportedSocPrefix, 0) != 0) {
    return MegaMoeRejectionReason::UNSUPPORTED_SOC;
  }
  if (capability.cann_version != kSupportedVersion) {
    return MegaMoeRejectionReason::UNSUPPORTED_CANN_VERSION;
  }
  if (capability.vendor_version != kSupportedVersion) {
    return MegaMoeRejectionReason::UNSUPPORTED_VENDOR_VERSION;
  }
  if (!capability.workspace_symbol_ready ||
      !capability.execute_symbol_ready) {
    return MegaMoeRejectionReason::OP_SYMBOLS_UNAVAILABLE;
  }
  if (!capability.comm_context_ready) {
    return MegaMoeRejectionReason::COMM_CONTEXT_UNAVAILABLE;
  }
  if (capability.dtype != MegaMoeDType::BFLOAT16) {
    return MegaMoeRejectionReason::UNSUPPORTED_DTYPE;
  }
  if (capability.activation != "swiglu") {
    return MegaMoeRejectionReason::UNSUPPORTED_ACTIVATION;
  }
  if (capability.hidden_size != 2048 ||
      capability.intermediate_size != 512 ||
      capability.num_experts != 256 || capability.topk != 8) {
    return MegaMoeRejectionReason::UNSUPPORTED_SHAPE;
  }
  if (capability.ep_size != 16) {
    return MegaMoeRejectionReason::UNSUPPORTED_EP;
  }
  if (capability.tp_size != 1) {
    return MegaMoeRejectionReason::UNSUPPORTED_TP;
  }
  if (capability.dp_size != 1) {
    return MegaMoeRejectionReason::UNSUPPORTED_DP;
  }
  if (capability.graph_enabled) {
    return MegaMoeRejectionReason::GRAPH_ENABLED;
  }
  if (capability.quantization_enabled) {
    return MegaMoeRejectionReason::QUANTIZATION_ENABLED;
  }
  if (!capability.weight_budget_ready) {
    return MegaMoeRejectionReason::WEIGHT_BUDGET_EXCEEDED;
  }
  return MegaMoeRejectionReason::NONE;
}

}  // namespace

std::optional<MegaMoeMode> parse_mega_moe_mode(const std::string& value) {
  if (value == "off") {
    return MegaMoeMode::OFF;
  }
  if (value == "auto") {
    return MegaMoeMode::AUTO;
  }
  if (value == "on") {
    return MegaMoeMode::ON;
  }
  return std::nullopt;
}

MegaMoeDecision decide_mega_moe(
    MegaMoeMode mode,
    const MegaMoeCapability& capability) {
  if (mode == MegaMoeMode::OFF) {
    return MegaMoeDecision{
        .use_mega_moe = false,
        .fatal = false,
        .reason = MegaMoeRejectionReason::DISABLED};
  }

  const MegaMoeRejectionReason rejection_reason =
      find_rejection_reason(capability);
  if (rejection_reason == MegaMoeRejectionReason::NONE) {
    return MegaMoeDecision{
        .use_mega_moe = true,
        .fatal = false,
        .reason = MegaMoeRejectionReason::NONE};
  }

  // MegaMoe and fused_mc2 both own the routed-expert communication sequence.
  // Falling back silently would make precedence configuration-dependent.
  if (rejection_reason == MegaMoeRejectionReason::FUSED_MC2_CONFLICT) {
    return MegaMoeDecision{.use_mega_moe = false,
                           .fatal = true,
                           .reason = rejection_reason};
  }

  if (mode == MegaMoeMode::AUTO && !capability.collectives_started) {
    return MegaMoeDecision{
        .use_mega_moe = false,
        .fatal = false,
        .reason = rejection_reason};
  }

  const MegaMoeRejectionReason fatal_reason =
      mode == MegaMoeMode::AUTO
          ? MegaMoeRejectionReason::COLLECTIVES_ALREADY_STARTED
          : rejection_reason;
  return MegaMoeDecision{
      .use_mega_moe = false,
      .fatal = true,
      .reason = fatal_reason};
}

MegaMoeExecutionContract make_mega_moe_execution_contract(
    const MegaMoeDecision& decision,
    int64_t ep_size,
    bool has_shared_expert) {
  const bool distributed_ep = ep_size > 1;
  if (decision.use_mega_moe) {
    return MegaMoeExecutionContract{
        .preserve_global_topk = true,
        .apply_local_expert_mask = false,
        .reduce_routed_output_across_ep = false,
        .shared_output_additions = has_shared_expert ? 1 : 0};
  }
  return MegaMoeExecutionContract{
      .preserve_global_topk = false,
      .apply_local_expert_mask = distributed_ep,
      .reduce_routed_output_across_ep = distributed_ep,
      .shared_output_additions = has_shared_expert ? 1 : 0};
}

}  // namespace layer
}  // namespace xllm
