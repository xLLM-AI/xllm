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

#include "core/framework/config/kernel_config.h"

#include <glog/logging.h>

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"
#include "core/framework/config/eplb_config.h"

#if defined(USE_NPU)
DEFINE_bool(enable_customize_mla_kernel, false, "enable customize mla kernel");

DEFINE_string(npu_kernel_backend,
              "AUTO",
              "NPU kernel backend. Supported options: AUTO, ATB, TORCH.");

DEFINE_bool(enable_intralayer_addnorm,
            false,
            "enable fused intralayer addnorm ops.");

DEFINE_int32(enable_fused_mc2,
             -1,
             "Fused MC2 mode for NPU. -1 uses auto default, 0 disables fused "
             "MC2, positive values enable dense matmul-allreduce, 1 uses "
             "DispatchFFNCombine for MoE, 2 uses DispatchGmmCombineDecode for "
             "MoE.");
DEFINE_bool(enable_interlayer_addnorm,
            false,
            "enable fused interlayer addnorm ops.");

DEFINE_bool(enable_split_rmsnorm_rope,
            false,
            "enable fused split rmsnorm rope ops.");

DEFINE_bool(enable_aclnn_matmul,
            false,
            "enable ACLNN matmul backend for supported NPU ATB layers.");

DEFINE_bool(enable_aclnn_swiglu,
            false,
            "enable ACLNN SwiGLU backend for supported NPU ATB layers.");

DEFINE_bool(enable_flashcomm1,
            false,
            "Enable Flash Communication 1 sequence-parallel optimization.");

DEFINE_int32(flashcomm1_min_prefill_tokens,
             8192,
             "Minimum prefill token count to activate FC1.");

DEFINE_bool(enable_mmrs_fusion,
            false,
            "Enable Matmul+ReduceScatter fusion kernel for FC1.");

DEFINE_string(mmrs_comm_mode,
              "aiv",
              "Communication mode for torch_npu MMRS: ai_cpu, aiv, or none.");
#endif

namespace xllm {
namespace {

#if defined(USE_NPU)
int32_t resolve_fused_mc2_mode(int32_t mode) {
  CHECK_GE(mode, -1) << "--enable_fused_mc2 must be -1, 0, 1, or 2.";
  CHECK_LE(mode, 2) << "--enable_fused_mc2 must be -1, 0, 1, or 2.";

  if (mode != -1) {
    return mode;
  }
  if (EPLBConfig::get_instance().expert_parallel_degree() == 2) {
    return 1;
  }
  return 0;
}
#endif

}  // namespace

void KernelConfig::from_flags() {
#if defined(USE_NPU)
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_customize_mla_kernel);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(npu_kernel_backend);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_intralayer_addnorm);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_fused_mc2);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_interlayer_addnorm);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_split_rmsnorm_rope);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_aclnn_matmul);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_aclnn_swiglu);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_flashcomm1);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(flashcomm1_min_prefill_tokens);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_mmrs_fusion);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(mmrs_comm_mode);
#endif
}

void KernelConfig::from_json(const JsonReader& json) {
#if defined(USE_NPU)
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_customize_mla_kernel);
  XLLM_CONFIG_ASSIGN_FROM_JSON(npu_kernel_backend);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_intralayer_addnorm);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_fused_mc2);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_interlayer_addnorm);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_split_rmsnorm_rope);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_aclnn_matmul);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_aclnn_swiglu);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_flashcomm1);
  XLLM_CONFIG_ASSIGN_FROM_JSON(flashcomm1_min_prefill_tokens);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_mmrs_fusion);
  XLLM_CONFIG_ASSIGN_FROM_JSON(mmrs_comm_mode);
#endif
}

void KernelConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
#if defined(USE_NPU)
  const KernelConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_customize_mla_kernel);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, npu_kernel_backend);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_intralayer_addnorm);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_fused_mc2);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_interlayer_addnorm);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_split_rmsnorm_rope);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_aclnn_matmul);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_aclnn_swiglu);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_flashcomm1);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, flashcomm1_min_prefill_tokens);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_mmrs_fusion);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, mmrs_comm_mode);
#endif
}

KernelConfig& KernelConfig::get_instance() {
  static KernelConfig config;
  return config;
}

void KernelConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
#if defined(USE_NPU)
  enable_fused_mc2(resolve_fused_mc2_mode(enable_fused_mc2()));
#endif
}

}  // namespace xllm
