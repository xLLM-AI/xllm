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

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace xllm {
namespace layer {

enum class MegaMoeMode : int8_t {
  OFF = 0,
  AUTO = 1,
  ON = 2,
};

enum class MegaMoeDType : int8_t {
  BFLOAT16 = 0,
  OTHER = 1,
};

enum class MegaMoeRejectionReason : int8_t {
  NONE = 0,
  DISABLED = 1,
  FUSED_MC2_CONFLICT = 2,
  COLLECTIVES_ALREADY_STARTED = 3,
  UNSUPPORTED_SOC = 4,
  UNSUPPORTED_CANN_VERSION = 5,
  UNSUPPORTED_VENDOR_VERSION = 6,
  OP_SYMBOLS_UNAVAILABLE = 7,
  COMM_CONTEXT_UNAVAILABLE = 8,
  UNSUPPORTED_DTYPE = 9,
  UNSUPPORTED_ACTIVATION = 10,
  UNSUPPORTED_SHAPE = 11,
  UNSUPPORTED_EP = 12,
  UNSUPPORTED_TP = 13,
  UNSUPPORTED_DP = 14,
  GRAPH_ENABLED = 15,
  QUANTIZATION_ENABLED = 16,
  WEIGHT_BUDGET_EXCEEDED = 17,
};

struct MegaMoeCapability {
  std::string soc_name;
  std::string cann_version;
  std::string vendor_version;
  MegaMoeDType dtype = MegaMoeDType::OTHER;
  std::string activation;
  int64_t hidden_size = 0;
  int64_t intermediate_size = 0;
  int64_t num_experts = 0;
  int64_t topk = 0;
  int64_t ep_size = 0;
  int64_t tp_size = 0;
  int64_t dp_size = 0;
  bool graph_enabled = false;
  bool quantization_enabled = false;
  bool workspace_symbol_ready = false;
  bool execute_symbol_ready = false;
  bool comm_context_ready = false;
  bool fused_mc2_enabled = false;
  bool collectives_started = false;
  bool weight_budget_ready = true;
};

struct MegaMoeDecision {
  bool use_mega_moe = false;
  bool fatal = false;
  MegaMoeRejectionReason reason = MegaMoeRejectionReason::NONE;
};

struct MegaMoeExecutionContract {
  bool preserve_global_topk = false;
  bool apply_local_expert_mask = true;
  bool reduce_routed_output_across_ep = true;
  int32_t shared_output_additions = 0;
};

std::optional<MegaMoeMode> parse_mega_moe_mode(const std::string& value);

MegaMoeDecision decide_mega_moe(MegaMoeMode mode,
                                const MegaMoeCapability& capability);

MegaMoeExecutionContract make_mega_moe_execution_contract(
    const MegaMoeDecision& decision,
    int64_t ep_size,
    bool has_shared_expert);

}  // namespace layer
}  // namespace xllm
