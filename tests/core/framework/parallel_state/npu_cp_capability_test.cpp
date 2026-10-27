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

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>

#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/parallel_config_validation.h"
#include "core/framework/config/scheduler_config.h"
#include "core/util/scope_guard.h"

namespace xllm {
namespace {

TEST(NpuCpCapabilityTest, PythonDcpRejectsCrossDpGroupsBeforePcpEarlyReturn) {
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  ScopeGuard config_guard([&] {
    model_config.model_impl(original_model_impl);
    parallel_config.kv_split_size(original_kv_split_size);
  });
  model_config.model_impl("python");
  parallel_config.kv_split_size(2);
  Options options;
  options.cp_size(1).dp_size(2).instance_role(InstanceRole::DECODE);

  const std::optional<std::string> error = validate_context_parallel_config(
      options, EngineType::LLM, "glm_moe_dsa", /*global_world_size=*/8);
  ASSERT_TRUE(error.has_value());
  EXPECT_NE(error->find("Python DCP requires dp_size == 1"), std::string::npos);

  model_config.model_impl("py");
  EXPECT_TRUE(validate_context_parallel_config(options,
                                               EngineType::LLM,
                                               "glm_moe_dsa",
                                               /*global_world_size=*/8)
                  .has_value());
  options.dp_size(1);
  EXPECT_FALSE(validate_context_parallel_config(options,
                                                EngineType::LLM,
                                                "glm_moe_dsa",
                                                /*global_world_size=*/8)
                   .has_value());
  options.dp_size(2);
  parallel_config.kv_split_size(1);
  EXPECT_FALSE(validate_context_parallel_config(options,
                                                EngineType::LLM,
                                                "glm_moe_dsa",
                                                /*global_world_size=*/8)
                   .has_value());
  parallel_config.kv_split_size(2);
  model_config.model_impl("native");
  EXPECT_TRUE(validate_context_parallel_config(options,
                                               EngineType::LLM,
                                               "glm_moe_dsa",
                                               /*global_world_size=*/8)
                  .has_value());
}

TEST(NpuCpCapabilityTest, PythonCpAllowsAclGraphAndRejectsCompileBackends) {
  ExecutionConfig& execution_config = ExecutionConfig::get_instance();
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const std::string original_python_graph_backend =
      execution_config.python_graph_backend();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  ScopeGuard config_guard([&] {
    parallel_config.kv_split_size(original_kv_split_size);
    model_config.model_impl(original_model_impl);
    execution_config.python_graph_backend(original_python_graph_backend);
  });
  execution_config.python_graph_backend("off");
  model_config.model_impl("python");
  parallel_config.kv_split_size(1);

  Options options;
  options.task_type("generate")
      .cp_size(4)
      .dp_size(1)
      .ep_size(16)
      .instance_role(InstanceRole::PREFILL)
      .enable_graph(true);
  EXPECT_FALSE(validate_context_parallel_config(options,
                                                EngineType::LLM,
                                                "glm_moe_dsa",
                                                /*global_world_size=*/16)
                   .has_value());

  options.enable_graph(false);
  execution_config.python_graph_backend("aclgraph");
  EXPECT_FALSE(validate_context_parallel_config(options,
                                                EngineType::LLM,
                                                "glm_moe_dsa",
                                                /*global_world_size=*/16)
                   .has_value());

  execution_config.python_graph_backend("inductor");
  EXPECT_NE(validate_context_parallel_config(options,
                                             EngineType::LLM,
                                             "glm_moe_dsa",
                                             /*global_world_size=*/16)
                .value_or("")
                .find("EagerRunner"),
            std::string::npos);

  execution_config.python_graph_backend("off");
  EXPECT_FALSE(validate_context_parallel_config(options,
                                                EngineType::LLM,
                                                "glm_moe_dsa",
                                                /*global_world_size=*/16)
                   .has_value());
}

TEST(NpuCpCapabilityTest, PythonCpPreservesQwenAndRestrictsGlm) {
  ExecutionConfig& execution_config = ExecutionConfig::get_instance();
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const std::string original_python_graph_backend =
      execution_config.python_graph_backend();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  ScopeGuard config_guard([&] {
    parallel_config.kv_split_size(original_kv_split_size);
    model_config.model_impl(original_model_impl);
    execution_config.python_graph_backend(original_python_graph_backend);
  });
  execution_config.python_graph_backend("off");
  model_config.model_impl("python");
  parallel_config.kv_split_size(1);

  Options options;
  options.task_type("generate")
      .cp_size(4)
      .dp_size(1)
      .ep_size(16)
      .instance_role(InstanceRole::PREFILL)
      .enable_graph(false)
      .speculative_algorithm("MTP")
      .num_speculative_tokens(2);
  EXPECT_FALSE(validate_context_parallel_config(options,
                                                EngineType::LLM,
                                                "qwen3",
                                                /*global_world_size=*/16)
                   .has_value());
  EXPECT_FALSE(validate_context_parallel_config(options,
                                                EngineType::LLM,
                                                "glm_moe_dsa",
                                                /*global_world_size=*/16)
                   .has_value());

  options.speculative_algorithm("DSpark").draft_model_path("/tmp/draft");
  EXPECT_EQ(validate_context_parallel_config(options,
                                             EngineType::LLM,
                                             "glm_moe_dsa",
                                             /*global_world_size=*/16)
                .value_or("")
                .find("aux-hidden-capture"),
            std::string::npos);

  options.cp_size(1).instance_role(InstanceRole::DECODE).enable_disagg_pd(true);
  EXPECT_FALSE(validate_context_parallel_config(options,
                                                EngineType::LLM,
                                                "glm_moe_dsa",
                                                /*global_world_size=*/16)
                   .has_value());
  options.cp_size(4)
      .instance_role(InstanceRole::PREFILL)
      .enable_disagg_pd(false)
      .speculative_algorithm("MTP");

  // EP topology validation belongs to model construction, not this Python CP
  // capability gate.
  options.ep_size(2);
  EXPECT_FALSE(validate_context_parallel_config(options,
                                                EngineType::LLM,
                                                "glm_moe_dsa",
                                                /*global_world_size=*/16)
                   .has_value());

  parallel_config.kv_split_size(2);
  EXPECT_FALSE(validate_context_parallel_config(options,
                                                EngineType::LLM,
                                                "qwen3",
                                                /*global_world_size=*/16)
                   .has_value());
  EXPECT_FALSE(validate_context_parallel_config(options,
                                                EngineType::LLM,
                                                "glm_moe_dsa",
                                                /*global_world_size=*/16)
                   .has_value());

  parallel_config.kv_split_size(1);
  EXPECT_NE(validate_context_parallel_config(options,
                                             EngineType::LLM,
                                             "glm_moe_dsa_mtp",
                                             /*global_world_size=*/16)
                .value_or("")
                .find("model_type=glm_moe_dsa_mtp"),
            std::string::npos);

  parallel_config.kv_split_size(3);
  EXPECT_NE(validate_context_parallel_config(options,
                                             EngineType::LLM,
                                             "glm_moe_dsa",
                                             /*global_world_size=*/16)
                .value_or("")
                .find("positive divisor of cp_size"),
            std::string::npos);
}

TEST(NpuCpCapabilityTest, PythonGlm5NextCapabilityGate) {
  ExecutionConfig& execution_config = ExecutionConfig::get_instance();
  KVCacheConfig& kv_cache_config = KVCacheConfig::get_instance();
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  SchedulerConfig& scheduler_config = SchedulerConfig::get_instance();
  const std::string original_python_graph_backend =
      execution_config.python_graph_backend();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  const bool original_chunked_prefill =
      scheduler_config.enable_chunked_prefill();
  const bool original_mix_batch = scheduler_config.enable_mix_batch();
  const bool original_prefix_cache = kv_cache_config.enable_prefix_cache();
  const bool original_schedule_overlap =
      scheduler_config.enable_schedule_overlap();
  ScopeGuard config_guard([&] {
    scheduler_config.enable_schedule_overlap(original_schedule_overlap);
    kv_cache_config.enable_prefix_cache(original_prefix_cache);
    scheduler_config.enable_mix_batch(original_mix_batch);
    scheduler_config.enable_chunked_prefill(original_chunked_prefill);
    parallel_config.kv_split_size(original_kv_split_size);
    model_config.model_impl(original_model_impl);
    execution_config.python_graph_backend(original_python_graph_backend);
  });
  model_config.model_impl("python");
  execution_config.python_graph_backend("off");
  parallel_config.kv_split_size(1);
  scheduler_config.enable_chunked_prefill(false);
  scheduler_config.enable_mix_batch(false);
  kv_cache_config.enable_prefix_cache(false);
  scheduler_config.enable_schedule_overlap(false);

  Options options;
  options.task_type("generate")
      .cp_size(2)
      .dp_size(1)
      .ep_size(1)
      .instance_role(InstanceRole::PREFILL)
      .enable_graph(false);
  EXPECT_FALSE(validate_context_parallel_config(options,
                                                EngineType::LLM,
                                                "glm5_next",
                                                /*global_world_size=*/8)
                   .has_value());

  options.instance_role(InstanceRole::DEFAULT);
  EXPECT_FALSE(validate_context_parallel_config(options,
                                                EngineType::LLM,
                                                "glm5_next",
                                                /*global_world_size=*/8)
                   .has_value());
  options.instance_role(InstanceRole::PREFILL);

  execution_config.python_graph_backend("aclgraph");
  EXPECT_FALSE(validate_context_parallel_config(options,
                                                EngineType::LLM,
                                                "glm5_next",
                                                /*global_world_size=*/8)
                   .has_value());

  options.speculative_algorithm("MTP")
      .num_speculative_tokens(2)
      .draft_model_path("/tmp/draft");
  EXPECT_EQ(
      validate_context_parallel_config(options,
                                       EngineType::LLM,
                                       "glm5_next",
                                       /*global_world_size=*/8),
      std::optional<std::string>(
          "Python GLM-5 Next CP does not support target-side speculative "
          "verification; run speculation on a cp_size=1 Decode instance"));

  options.num_speculative_tokens(0).draft_model_path(std::nullopt);

  execution_config.python_graph_backend("inductor");
  EXPECT_NE(validate_context_parallel_config(options,
                                             EngineType::LLM,
                                             "glm5_next",
                                             /*global_world_size=*/8)
                .value_or("")
                .find("EagerRunner"),
            std::string::npos);

  execution_config.python_graph_backend("off");
  for (int32_t dp_size : {1, 2}) {
    for (int32_t ep_size : {1, 2, 8}) {
      options.dp_size(dp_size).ep_size(ep_size);
      EXPECT_FALSE(validate_context_parallel_config(options,
                                                    EngineType::LLM,
                                                    "glm5_next",
                                                    /*global_world_size=*/8)
                       .has_value());
    }
  }
  options.dp_size(1);
  for (int32_t ep_size : {0, 3}) {
    options.ep_size(ep_size);
    EXPECT_NE(validate_context_parallel_config(options,
                                               EngineType::LLM,
                                               "glm5_next",
                                               /*global_world_size=*/8)
                  .value_or("")
                  .find("ep_size to be a positive divisor"),
              std::string::npos);
  }
  options.ep_size(1);

  options.expert_parallel_degree(2).ep_size(8);
  EXPECT_FALSE(validate_context_parallel_config(options,
                                                EngineType::LLM,
                                                "glm5_next",
                                                /*global_world_size=*/8)
                   .has_value());
  options.ep_size(4);
  EXPECT_NE(validate_context_parallel_config(options,
                                             EngineType::LLM,
                                             "glm5_next",
                                             /*global_world_size=*/8)
                .value_or("")
                .find("ep_size equal to world_size"),
            std::string::npos);
  options.expert_parallel_degree(1).ep_size(1);

  parallel_config.kv_split_size(2);
  EXPECT_NE(validate_context_parallel_config(options,
                                             EngineType::LLM,
                                             "glm5_next",
                                             /*global_world_size=*/8)
                .value_or("")
                .find("kv_split_size == 1"),
            std::string::npos);
  parallel_config.kv_split_size(1);

  scheduler_config.enable_chunked_prefill(true);
  EXPECT_NE(validate_context_parallel_config(options,
                                             EngineType::LLM,
                                             "glm5_next",
                                             /*global_world_size=*/8)
                .value_or("")
                .find("enable_chunked_prefill=false"),
            std::string::npos);
  scheduler_config.enable_chunked_prefill(false);

  scheduler_config.enable_mix_batch(true);
  EXPECT_FALSE(validate_context_parallel_config(options,
                                                EngineType::LLM,
                                                "glm5_next",
                                                /*global_world_size=*/8)
                   .has_value());
  EXPECT_TRUE(scheduler_config.enable_mix_batch());
  scheduler_config.enable_mix_batch(false);

  kv_cache_config.enable_prefix_cache(true);
  EXPECT_NE(validate_context_parallel_config(options,
                                             EngineType::LLM,
                                             "glm5_next",
                                             /*global_world_size=*/8)
                .value_or("")
                .find("enable_prefix_cache=false"),
            std::string::npos);
  kv_cache_config.enable_prefix_cache(false);

  scheduler_config.enable_schedule_overlap(true);
  EXPECT_NE(validate_context_parallel_config(options,
                                             EngineType::LLM,
                                             "glm5_next",
                                             /*global_world_size=*/8)
                .value_or("")
                .find("enable_schedule_overlap=false"),
            std::string::npos);
  scheduler_config.enable_schedule_overlap(false);

  options.enable_disagg_pd(true);
  EXPECT_NE(validate_context_parallel_config(options,
                                             EngineType::LLM,
                                             "glm5_next",
                                             /*global_world_size=*/8)
                .value_or("")
                .find("disaggregated PD"),
            std::string::npos);
  options.instance_role(InstanceRole::DEFAULT);
  EXPECT_NE(validate_context_parallel_config(options,
                                             EngineType::LLM,
                                             "glm5_next",
                                             /*global_world_size=*/8)
                .value_or("")
                .find("disaggregated PD"),
            std::string::npos);
}

TEST(NpuCpCapabilityTest, PythonGlmMtpSupportsCpWithDecodeAclGraph) {
  ExecutionConfig& execution_config = ExecutionConfig::get_instance();
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const std::string original_python_graph_backend =
      execution_config.python_graph_backend();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  const int32_t original_cp_size = parallel_config.cp_size();
  ScopeGuard config_guard([&] {
    parallel_config.cp_size(original_cp_size);
    parallel_config.kv_split_size(original_kv_split_size);
    model_config.model_impl(original_model_impl);
    execution_config.python_graph_backend(original_python_graph_backend);
  });
  model_config.model_impl("python");
  execution_config.python_graph_backend("aclgraph");
  parallel_config.cp_size(2).kv_split_size(2);

  Options options;
  options.task_type("generate")
      .cp_size(2)
      .dp_size(1)
      .ep_size(1)
      .instance_role(InstanceRole::DEFAULT)
      .enable_graph(true)
      .speculative_algorithm("MTP")
      .num_speculative_tokens(2);
  EXPECT_FALSE(validate_context_parallel_config(options,
                                                EngineType::LLM,
                                                "glm_moe_dsa",
                                                /*global_world_size=*/16)
                   .has_value());

  // The automatic KV split follows CP and uses the same DCP owners during
  // prefill, target verification and draft decoding.
  options.instance_role(InstanceRole::PREFILL).enable_disagg_pd(true);
  parallel_config.kv_split_size(0);
  EXPECT_FALSE(validate_context_parallel_config(options,
                                                EngineType::LLM,
                                                "glm_moe_dsa",
                                                /*global_world_size=*/16)
                   .has_value());
}

}  // namespace
}  // namespace xllm
