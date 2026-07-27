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

#include "core/framework/config/parallel_config.h"

#include <glog/logging.h>

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

DEFINE_int32(dp_size, 1, "Data parallel size for MLA attention.");

DEFINE_int32(ep_size, 1, "Expert parallel size for MoE model.");

DEFINE_int32(cp_size, 1, "Context parallel size for DSA attention.");

DEFINE_int32(kv_split_size,
             1,
             "KV-cache split width. 0 falls back to cp_size (legacy); 1 means "
             "no KV split (each CP rank stores full KV, skips prefix "
             "AllGather); other K (K divides cp_size) means KV is sharded "
             "across K ranks while token-CP still uses cp_size.");

DEFINE_int64(tp_size, 1, "Tensor parallelism size, only used for DiT model.");

DEFINE_int64(sp_size, 1, "Sequence parallelism size, only used for DiT model.");

DEFINE_int64(cfg_size,
             1,
             "Classifier-free guidiance parallelism size, only used for DiT "
             "model.");

DEFINE_int64(vae_size, 1, "Vae patch parallelism size");

DEFINE_string(
    communication_backend,
    "hccl",
    "NPU communication backend.(e.g. lccl, hccl). When enable dp, use hccl.");

DEFINE_bool(enable_mm_encoder_dp,
            false,
            "Enable encoder data parallelism for multi-modal models.");

DEFINE_bool(
    enable_multi_stream_parallel,
    false,
    "Whether to enable computation communication parallel by two streams "
    "and two micro batches in prefill stage.");

DEFINE_int32(micro_batch_num,
             1,
             "Default use two micro batches for multi-stream parallel.");

DEFINE_bool(
    enable_dp_balance,
    false,
    "Whether to enable dp load balance, if true, sequences within a single "
    "dp batch will be shuffled.");

DEFINE_bool(
    enable_runtime_cp_dp_switch,
    false,
    "Enable runtime CP<->DP switching (dual-graph mode). When on the "
    "worker brings up TWO CollectiveCommunicators at startup -- one "
    "with cp_size=N dp_size=1 and one with cp_size=1 dp_size=N -- and "
    "holds both ATB graph pairs resident. See "
    "Desktop/cp_docs_work/probes_results_20260629/DESIGN_DOC_v0.2.md "
    "for the budget. Costs ~600 MB / NPU plus extra ATB graph workspace.");

DEFINE_bool(enable_flip_verbose_log,
            false,
            "Enable per-request/per-step FLIPDIAG diagnostic logs for CP<->DP "
            "flip debugging (dispatch, recv, per-step batch shape, per-worker "
            "forward shard, dp_global_token_nums broadcast). Off by default -- "
            "flip lifecycle events (switch_mode/rebuild/relink/backdoor) are "
            "still logged unconditionally. Turn on when investigating flip "
            "regressions; expect 20-100x log volume during DP burst traffic. "
            "Runtime-toggleable via brpc /flags endpoint.");
// Registering a no-op validator marks the flag as reloadable so brpc's
// /flags?setvalue=true HTTP endpoint accepts runtime toggles. Without it
// brpc rejects with "A reloadable gflag must have validator" and forces
// a service restart just to flip verbose logging.
static bool ValidateFlipVerboseLog(const char*, bool) { return true; }
DEFINE_validator(enable_flip_verbose_log, &ValidateFlipVerboseLog);

DEFINE_int32(dual_mode_port_stride,
             256,
             "Port stride between the CP and DP communicators when "
             "enable_runtime_cp_dp_switch is on. 256 was validated by the v3 "
             "probe; smaller values risk colliding with TIME_WAIT residual "
             "TCPStore ports between iterations.");

namespace xllm {

void ParallelConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dp_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(ep_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(cp_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(kv_split_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(tp_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(sp_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(cfg_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(vae_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(communication_backend);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_mm_encoder_dp);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_multi_stream_parallel);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(micro_batch_num);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_dp_balance);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_runtime_cp_dp_switch);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dual_mode_port_stride);
}

void ParallelConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(dp_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(ep_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(cp_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(tp_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(sp_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(cfg_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(vae_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(communication_backend);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_mm_encoder_dp);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_multi_stream_parallel);
  XLLM_CONFIG_ASSIGN_FROM_JSON(micro_batch_num);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_dp_balance);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_runtime_cp_dp_switch);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dual_mode_port_stride);
}

void ParallelConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const ParallelConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(config_json, default_config, dp_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(config_json, default_config, ep_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(config_json, default_config, cp_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(config_json, default_config, tp_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(config_json, default_config, sp_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, cfg_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, vae_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, communication_backend);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_mm_encoder_dp);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_multi_stream_parallel);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, micro_batch_num);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_dp_balance);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_runtime_cp_dp_switch);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dual_mode_port_stride);
}

ParallelConfig& ParallelConfig::get_instance() {
  static ParallelConfig config;
  return config;
}

void ParallelConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
