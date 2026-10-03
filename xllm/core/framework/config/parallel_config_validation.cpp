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

#include "core/framework/config/parallel_config_validation.h"

#include <boost/algorithm/string.hpp>
#include <string_view>
#include <unordered_set>

#include "core/framework/config/execution_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/platform/platform.h"
#include "core/util/utils.h"
#include "framework/kv_cache/layerwise_split_layout.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

bool is_python_cp_compatible_graph_backend(std::string_view graph_backend) {
  std::string normalized_backend(graph_backend);
  boost::algorithm::to_lower(normalized_backend);
  return normalized_backend.empty() || normalized_backend == "off" ||
         normalized_backend == "none" || normalized_backend == "0" ||
         normalized_backend == "aclgraph";
}

}  // namespace

void validate_layerwise_split_size_startup_config(const Options& options,
                                                  const std::string& model_type,
                                                  int32_t global_world_size) {
  const ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const int32_t layerwise_split_size = parallel_config.layerwise_split_size();
  validate_layerwise_split_size_config(layerwise_split_size);
  if (layerwise_split_size <= 1) {
    return;
  }

  CHECK_GE(options.dp_size(), 1) << "dp_size must be >= 1.";
  CHECK_GT(global_world_size, 0) << "world_size must be > 0.";
  const int32_t dp_cp_size = options.dp_size() * options.cp_size();
  CHECK_EQ(global_world_size % dp_cp_size, 0)
      << "world_size (" << global_world_size
      << ") must be divisible by dp_size * cp_size (" << dp_cp_size << ").";
  const int32_t attn_tp_size = global_world_size / dp_cp_size;
  validate_layerwise_split_enablement(
      layerwise_split_size, attn_tp_size, model_type);

  CHECK_LE(options.host_blocks_factor(), 1.0)
      << "layerwise_split_size > 1 does not support hierarchy host cache.";
  CHECK_EQ(parallel_config.cp_size(), 1)
      << "layerwise_split_size > 1 does not support context parallelism.";
  CHECK_EQ(parallel_config.kv_split_size_effective(), 1)
      << "layerwise_split_size > 1 does not support KV split.";
}

std::optional<std::string> validate_context_parallel_config(
    const Options& options,
    EngineType engine_type,
    const std::string& model_type,
    int32_t global_world_size) {
  if (options.cp_size() < 1) {
    return "cp_size must be greater than or equal to 1";
  }

  if (options.cp_size() == 1) {
    const int32_t kv_split =
        ParallelConfig::get_instance().kv_split_size_effective();
    if (Platform::is_npu() &&
        ModelConfig::is_python_model_impl(
            ModelConfig::get_instance().model_impl()) &&
        kv_split > 1 && options.dp_size() > 1) {
      // The Python DCP initializer currently forms KV groups over the whole
      // world, not inside each DP request domain. Reject before the PCP early
      // return and before workers can enter mismatched collectives.
      return "Python DCP requires dp_size == 1 until DP-local KV groups are "
             "implemented (world_size=" +
             std::to_string(global_world_size) +
             ", dp_size=" + std::to_string(options.dp_size()) + ", tp_size=" +
             std::to_string(global_world_size / options.dp_size()) +
             ", cp_size=1, kv_split_size=" + std::to_string(kv_split) + ").";
    }
    return std::nullopt;
  }

  if (Platform::is_mlu()) {
    if (engine_type != EngineType::LLM && engine_type != EngineType::SSM) {
      return "MLU CP supports only LLM text generation";
    }
    if (options.task_type() != "generate") {
      return "MLU CP supports only the generate task";
    }
    if (!is_mlu_model_cp_capable(model_type)) {
      return "MLU CP does not support model_type=" + model_type;
    }

    if (global_world_size % (options.dp_size() * options.cp_size()) != 0) {
      return "MLU CP requires world_size divisible by dp_size * cp_size "
             "(orthogonal PCP x TP layout)";
    }

    if (options.dp_size() != 1) {
      return "MLU CP requires dp_size == 1";
    }

    if (ParallelConfig::get_instance().kv_split_size() != 1) {
      return "MLU CP requires kv_split_size == 1";
    }

    if (options.ep_size() != global_world_size) {
      return "MLU CP requires ep_size == global world size";
    }
    return std::nullopt;
  }

  if (Platform::is_npu()) {
    if (engine_type != EngineType::LLM && engine_type != EngineType::SSM) {
      return "Model-side CP supports only LLM text generation";
    }
    if (options.task_type() != "generate") {
      return "Model-side CP supports only the generate task";
    }
    const bool is_dsv4_model = util::is_deepseek_v4_model_type(model_type);
    if (engine_type == EngineType::SSM &&
        SpeculativeConfig::requires_aux_hidden_capture(
            options.speculative_algorithm()) &&
        !is_dsv4_model) {
      return "Current model-side CP does not support aux-hidden-capture "
             "speculative algorithms (Eagle3/DFlash/DSpark); run speculative "
             "decoding on a cp_size=1 Decode instance.";
    }
    // Native model-side CP is compatible with graph mode because the two are
    // phase-disjoint: CP only engages on batch_forward_type.no_decode(), while
    // ACL graph captures/replays pure decode. The Python CP path applies the
    // same phase split only to decode-only ACLGraph; other graph backends are
    // rejected below.
    //
    // Native spec-verify chunked prefill still follows the admission guard in
    // AclGraphExecutorImpl::run(). Python handles expanded MTP verification as
    // replicated decode without CP row sharding, so it requires complete KV
    // caches as checked below.
    if (options.instance_role() != InstanceRole::DEFAULT &&
        options.instance_role() != InstanceRole::PREFILL) {
      return "Model-side CP supports only DEFAULT or PREFILL roles";
    }

    // Python model executor runs a standalone torch CP path (all-gather KV,
    // eager prefill only) that does not go through the ATB fused-attention op,
    // so it bypasses the ATB-backend requirement and the ATB CP capability
    // allowlist below. The safety constraints above (LLM/generate,
    // DEFAULT/PREFILL) still apply. Orthogonal TP x CP is supported (both may
    // be > 1, sharing world = cp * tp); the collective communicator builds
    // the narrowed TP group and the strided CP group as separate torch
    // subgroups off the shared world rendezvous endpoint. DP > 1 stays
    // unsupported: the Python executor does not implement the dp * cp * tp
    // rank layout.
    if (ModelConfig::is_python_model_impl(
            ModelConfig::get_instance().model_impl())) {
      // Only models whose Python forward actually shards the sequence (via
      // cp_shard_rows / cp_merge_rows) may enable CP. Other Python models keep
      // a full-sequence forward, so a cp_context would be built but never
      // consumed: qwen3_5 would reach _prefill_cp with unsharded rows (garbled
      // or out-of-bounds output) and unsupported MLA models would silently
      // recompute the whole sequence on every rank. Mirror the implementations
      // that explicitly consume cp_context rather than admitting every Python
      // model type.
      static const std::unordered_set<std::string> kPythonCpCapableModels = {
          "qwen3",
          "glm_moe_dsa",
      };
      if (kPythonCpCapableModels.find(model_type) ==
          kPythonCpCapableModels.end()) {
        return "Python model-side CP does not support model_type=" +
               model_type + "; supported models are qwen3 and glm_moe_dsa.";
      }
      // On NPU, the Python executor resolves enable_graph=true with an
      // off-like backend to ACLGraph. ACLGraph handles Decode only, so Prefill
      // still runs through EagerRunner and receives cp_context.
      if (!is_python_cp_compatible_graph_backend(
              ExecutionConfig::get_instance().python_graph_backend())) {
        return "Python model-side CP requires Prefill to use EagerRunner; use "
               "--python_graph_backend=off or decode-only aclgraph";
      }
      if (options.dp_size() != 1) {
        return "Python CP requires dp_size == 1";
      }
      if (global_world_size % (options.dp_size() * options.cp_size()) != 0) {
        return "Python CP requires world_size divisible by dp_size * cp_size";
      }
      const int32_t kv_split =
          ParallelConfig::get_instance().kv_split_size_effective();
      if (kv_split < 1 || options.cp_size() % kv_split != 0) {
        return "Python CP requires kv_split_size effective value to be a "
               "positive divisor of cp_size";
      }
      if (model_type == "glm_moe_dsa" && engine_type == EngineType::SSM &&
          SpeculativeConfig::is_mtp_algorithm(
              options.speculative_algorithm()) &&
          kv_split != 1) {
        // Prefill gathers the CP shards into each rank's full cache. MTP
        // verification and draft decode then retain the global token rows.
        return "Python GLM CP with MTP requires replicated KV caches; use "
               "kv_split_size=1";
      }
      if (model_type == "glm_moe_dsa" && kv_split > 1 &&
          (!options.enable_disagg_pd() ||
           options.instance_role() != InstanceRole::PREFILL)) {
        return "Python GLM CP with kv_split_size > 1 requires disaggregated "
               "PD with the PREFILL role; set enable_disagg_pd=true and "
               "instance_role=PREFILL";
      }
      return std::nullopt;
    }

    // Require registered NPU model-side CP capability. The backend is not
    // constrained: ATB models drive CP through NpuCpPlan, while TORCH models
    // (deepseek_v4) own their CP split inside the model. Both rely on the
    // orthogonal dp * cp * attn_tp == world layout validated below.
    std::string effective_backend;
    std::string resolved_name;
    std::string resolve_error;
    // Runtime platform branches are type-checked in every platform build.
    const std::string requested_backend = options.npu_kernel_backend();
    if (!resolve_model_registration(model_type,
                                    requested_backend,
                                    &effective_backend,
                                    &resolved_name,
                                    &resolve_error)) {
      return "Model-side CP rejected model_type=" + model_type + ": " +
             resolve_error;
    }
    if (!is_npu_model_cp_capable(resolved_name)) {
      return "NPU model-side CP does not support model_type=" + model_type +
             " (resolved=" + resolved_name +
             "); only deepseek_v32, deepseek_v32_mtp, deepseek_v4, "
             "deepseek_v4_mtp, glm_moe_dsa, glm_moe_dsa_mtp are registered as "
             "CP-capable.";
    }
    if (global_world_size % (options.dp_size() * options.cp_size()) != 0) {
      return "NPU CP requires world_size divisible by dp_size * cp_size "
             "(orthogonal CP x TP layout)";
    }
    const int32_t attn_tp_size =
        global_world_size / (options.dp_size() * options.cp_size());
    if (attn_tp_size < 1) {
      return "NPU CP requires attn_tp_size >= 1";
    }
    const int32_t kv_split =
        ParallelConfig::get_instance().kv_split_size_effective();
    if (kv_split < 1 || options.cp_size() % kv_split != 0) {
      return "NPU CP requires kv_split_size effective value to be a positive "
             "divisor of cp_size";
    }
    return std::nullopt;
  }

  return "cp_size > 1 is only supported on platforms with model-side CP "
         "(MLU/NPU); disable CP (cp_size=1) or use MLU/NPU.";
}

}  // namespace xllm
