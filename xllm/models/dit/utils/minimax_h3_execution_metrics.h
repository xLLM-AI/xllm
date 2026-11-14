/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/framework/model/dit_model_params.h"
#include "core/framework/request/minimax_h3_conditioning.h"
#include "models/dit/utils/minimax_h3_rainfusion_policy.h"

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "acl/acl.h"
#endif

namespace xllm {

struct H3QualifiedLaunchSnapshot {
  int64_t rf_blocks = 0;
  int64_t selector_launches = 0;
  int64_t attention_launches = 0;
  int64_t qkv_a2a_launches = 0;
  int64_t qkv_a2a_waits = 0;
  int64_t inverse_a2a_calls = 0;
};

struct H3QualifiedStepObservation {
  int64_t step_index = 0;
  bool cache_hit = false;
  int64_t executed_blocks = 0;
  int64_t rf_blocks = 0;
  int64_t dense_blocks = 0;
  int64_t selector_launches = 0;
  int64_t attention_launches = 0;
  int64_t qkv_a2a_launches = 0;
  int64_t qkv_a2a_waits = 0;
  int64_t inverse_a2a_calls = 0;
};

// Request-local observations, available before either CacheDiT or RainFusion.
// No field below creates device state or communication groups.
struct H3QualifiedExecutionMetrics {
  std::string request_id;
  std::string profile = std::string(kMiniMaxH3ExecutionProfile);
  int64_t infer_steps = 49;
  int64_t front_blocks = 1;
  int64_t back_blocks = 0;
  int64_t executed_blocks = 0;
  int64_t rf_executed_blocks = 0;
  int64_t dense_executed_blocks = 0;
  int64_t selector_launches = 0;
  int64_t attention_launches = 0;
  int64_t qkv_a2a_launches = 0;
  int64_t qkv_a2a_waits = 0;
  int64_t inverse_a2a_calls = 0;
  std::vector<int64_t> hit_steps;
  std::vector<H3QualifiedStepObservation> observed_steps;
  bool qkv_observed = false;
  H3RainfusionPolicy rf_policy = kH3SparseNone;
  bool rf_ready = false;
  std::string fallback_reason;
  bool completion_synchronized = false;
};

inline std::string h3_execution_request_id(const DiTModelParams& params) {
  const auto source = params.tensor_sources.get_named_tensor("prompt_embed");
  if (params.batch_size != 1 || !source.has_value()) {
    throw std::invalid_argument(
        "H3 observation identity requires a single prompt_embed");
  }
  const auto* id = get_tensor_parameter<std::string>(
      source->get().parameters, kMiniMaxH3InternalRequestId);
  if (id == nullptr || id->empty()) {
    throw std::invalid_argument(
        "H3 observation is missing its server request ID");
  }
  return *id;
}

// `actual_blocks` must be the delta of the actual block-forward counter.
// Before RF exists, default-zero snapshots measure dense/cache/overlap paths.
inline void record_h3_qualified_execution_step(
    H3QualifiedExecutionMetrics& metrics,
    int32_t global_rank,
    int64_t step,
    bool cache_hit,
    int64_t actual_blocks,
    const H3QualifiedLaunchSnapshot& before = {},
    const H3QualifiedLaunchSnapshot& after = {}) {
  if (metrics.request_id.empty() ||
      step != static_cast<int64_t>(metrics.observed_steps.size()) || step < 0 ||
      step >= metrics.infer_steps || actual_blocks < 1 || actual_blocks > 50 ||
      metrics.completion_synchronized) {
    throw std::logic_error(
        "H3 observation has an invalid identity, step or actual block count");
  }
  if (metrics.front_blocks < 1 || metrics.front_blocks >= 50 ||
      metrics.back_blocks < 0 ||
      metrics.back_blocks >= 50 - metrics.front_blocks ||
      actual_blocks !=
          (cache_hit ? metrics.front_blocks + metrics.back_blocks : 50)) {
    throw std::logic_error(
        "H3 actual block count disagrees with its cache decision");
  }
  H3QualifiedStepObservation observation;
  observation.step_index = step;
  observation.cache_hit = cache_hit;
  observation.executed_blocks = actual_blocks;
  observation.rf_blocks = after.rf_blocks - before.rf_blocks;
  observation.selector_launches =
      after.selector_launches - before.selector_launches;
  observation.attention_launches =
      after.attention_launches - before.attention_launches;
  observation.qkv_a2a_launches =
      after.qkv_a2a_launches - before.qkv_a2a_launches;
  observation.qkv_a2a_waits = after.qkv_a2a_waits - before.qkv_a2a_waits;
  observation.inverse_a2a_calls =
      after.inverse_a2a_calls - before.inverse_a2a_calls;
  const int64_t expected_qkv = metrics.qkv_observed ? 3 * actual_blocks : 0;
  if (observation.qkv_a2a_launches != expected_qkv ||
      observation.qkv_a2a_waits != expected_qkv ||
      observation.inverse_a2a_calls !=
          (metrics.qkv_observed ? actual_blocks : 0)) {
    throw std::logic_error(
        "H3 QKV submission/wait observations disagree with actual blocks");
  }
  observation.dense_blocks = actual_blocks - observation.rf_blocks;
  int64_t expected_rf = 0;
  if (metrics.rf_ready && metrics.rf_policy.includes_step(step)) {
    const auto& policy = metrics.rf_policy;
    expected_rf = cache_hit
                      ? policy.layer_count(0, metrics.front_blocks) +
                            policy.layer_count(50 - metrics.back_blocks, 50)
                      : policy.layer_count(0, 50);
  }
  if (observation.rf_blocks != expected_rf ||
      observation.selector_launches != observation.rf_blocks ||
      observation.attention_launches != observation.rf_blocks) {
    throw std::logic_error(
        "H3 actual RainFusion submissions disagree with the executed block "
        "path");
  }
  metrics.executed_blocks += actual_blocks;
  metrics.rf_executed_blocks += observation.rf_blocks;
  metrics.dense_executed_blocks += observation.dense_blocks;
  metrics.selector_launches += observation.selector_launches;
  metrics.attention_launches += observation.attention_launches;
  metrics.qkv_a2a_launches += observation.qkv_a2a_launches;
  metrics.qkv_a2a_waits += observation.qkv_a2a_waits;
  metrics.inverse_a2a_calls += observation.inverse_a2a_calls;
  if (cache_hit) {
    metrics.hit_steps.push_back(step);
  }
  metrics.observed_steps.push_back(observation);
  const nlohmann::json json = {
      {"kind", "h3_execution_step"},
      {"request_id", metrics.request_id},
      {"global_rank", global_rank},
      {"step_index", step},
      {"cache_hit", cache_hit},
      {"executed_blocks", actual_blocks},
      {"rf_blocks", observation.rf_blocks},
      {"dense_blocks", observation.dense_blocks},
      {"selector_launches", observation.selector_launches},
      {"attention_launches", observation.attention_launches},
      {"qkv_a2a_launches", observation.qkv_a2a_launches},
      {"qkv_a2a_waits", observation.qkv_a2a_waits},
      {"inverse_a2a_calls", observation.inverse_a2a_calls}};
  LOG(INFO) << "H3_EXECUTION_JSON " << json.dump();
}

// Call after decode inside an all-rank completion phase. Emit success only
// after that phase has confirmed every rank completed this synchronization.
inline void finish_h3_qualified_request(H3QualifiedExecutionMetrics& metrics,
                                        const torch::Device& device) {
  if (metrics.request_id.empty() || metrics.infer_steps != 49 ||
      metrics.observed_steps.size() != 49 || metrics.completion_synchronized) {
    throw std::logic_error(
        "H3 completion requires a request ID and 49 observed steps");
  }
#if defined(USE_NPU)
  if (device.type() != c10::DeviceType::PrivateUse1) {
    throw std::invalid_argument("H3 completion requires an NPU device");
  }
  const aclError status = aclrtSynchronizeStream(
      c10_npu::getCurrentNPUStream(device.index()).stream());
  if (status != ACL_SUCCESS) {
    throw std::runtime_error("H3 final stream synchronize failed: " +
                             std::to_string(status));
  }
  metrics.completion_synchronized = true;
#else
  throw std::runtime_error("H3 completion requires the NPU build");
#endif
}

inline void emit_h3_qualified_execution_summary(
    const H3QualifiedExecutionMetrics& metrics,
    int32_t global_rank) {
  if (metrics.request_id.empty() || !metrics.completion_synchronized ||
      metrics.observed_steps.size() != 49) {
    throw std::logic_error(
        "H3 success summary requires completed device synchronization");
  }
  const nlohmann::json json = {
      {"kind", "h3_execution_summary"},
      {"status", "success"},
      {"step_base", 0},
      {"request_id", metrics.request_id},
      {"global_rank", global_rank},
      {"profile", metrics.profile},
      {"steps", metrics.infer_steps},
      {"front_blocks", metrics.front_blocks},
      {"back_blocks", metrics.back_blocks},
      {"cache_hit_steps", metrics.hit_steps},
      {"rf_policy", metrics.rf_policy.name},
      {"rf_window", {metrics.rf_policy.step_begin, metrics.rf_policy.step_end}},
      {"rf_layer_window",
       {metrics.rf_policy.layer_begin, metrics.rf_policy.layer_end}},
      {"rf_blocks", metrics.rf_executed_blocks},
      {"dense_blocks", metrics.dense_executed_blocks},
      {"executed_blocks", metrics.executed_blocks},
      {"selector_launches", metrics.selector_launches},
      {"attention_launches", metrics.attention_launches},
      {"qkv_observed", metrics.qkv_observed},
      {"qkv_a2a_launches", metrics.qkv_a2a_launches},
      {"qkv_a2a_waits", metrics.qkv_a2a_waits},
      {"inverse_a2a_calls", metrics.inverse_a2a_calls},
      {"rf_ready", metrics.rf_ready},
      {"fallback_reason", metrics.fallback_reason}};
  LOG(INFO) << "H3_EXECUTION_JSON " << json.dump();
}

}  // namespace xllm
