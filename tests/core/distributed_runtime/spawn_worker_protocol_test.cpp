/* Copyright 2026 The xLLM Authors.

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

#include "core/distributed_runtime/spawn_worker_server/spawn_worker_protocol.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

namespace xllm::spawn_worker_protocol {
namespace {

std::vector<char*> mutable_argv(std::vector<std::string>* arguments) {
  std::vector<char*> argv;
  argv.reserve(arguments->size());
  for (std::string& argument : *arguments) {
    argv.emplace_back(argument.data());
  }
  return argv;
}

TEST(SpawnWorkerProtocolTest, UsesExplicitDtypeFromCurrentArguments) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  arguments[kIndexerCacheDtypeArgumentIndex] = "int8";
  std::vector<char*> argv = mutable_argv(&arguments);

  const std::optional<std::string> indexer_cache_dtype =
      parse_indexer_cache_dtype(kArgumentCount, argv.data());

  ASSERT_TRUE(indexer_cache_dtype.has_value());
  EXPECT_EQ(indexer_cache_dtype.value(), "int8");
}

TEST(SpawnWorkerProtocolTest, UsesDefaultDtypeFromLegacyArguments) {
  std::vector<std::string> arguments(kMinimumArgumentCount, "unused");
  std::vector<char*> argv = mutable_argv(&arguments);

  const std::optional<std::string> indexer_cache_dtype =
      parse_indexer_cache_dtype(kMinimumArgumentCount, argv.data());

  ASSERT_TRUE(indexer_cache_dtype.has_value());
  EXPECT_EQ(indexer_cache_dtype.value(), kDefaultIndexerCacheDtype);
}

TEST(SpawnWorkerProtocolTest, RejectsArgumentsBelowLegacyMinimum) {
  std::vector<std::string> arguments(kMinimumArgumentCount - 1, "unused");
  std::vector<char*> argv = mutable_argv(&arguments);

  EXPECT_FALSE(parse_indexer_cache_dtype(kMinimumArgumentCount - 1, argv.data())
                   .has_value());
}

TEST(SpawnWorkerProtocolTest, IgnoresUnknownTrailingArguments) {
  std::vector<std::string> arguments(kArgumentCount + 1, "unused");
  arguments[kIndexerCacheDtypeArgumentIndex] = "int8";
  std::vector<char*> argv = mutable_argv(&arguments);

  const std::optional<std::string> indexer_cache_dtype =
      parse_indexer_cache_dtype(kArgumentCount + 1, argv.data());

  ASSERT_TRUE(indexer_cache_dtype.has_value());
  EXPECT_EQ(indexer_cache_dtype.value(), "int8");
}

TEST(SpawnWorkerProtocolTest, PreservesExplicitEmptyDtype) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  arguments[kIndexerCacheDtypeArgumentIndex] = "";
  std::vector<char*> argv = mutable_argv(&arguments);

  const std::optional<std::string> indexer_cache_dtype =
      parse_indexer_cache_dtype(kArgumentCount, argv.data());

  ASSERT_TRUE(indexer_cache_dtype.has_value());
  EXPECT_TRUE(indexer_cache_dtype->empty());
}

TEST(SpawnWorkerProtocolTest, PreservesExplicitModelImplementation) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  arguments[kModelImplArgumentIndex] = "python";
  std::vector<char*> argv = mutable_argv(&arguments);

  EXPECT_EQ(parse_model_impl(kArgumentCount, argv.data()), "python");
}

TEST(SpawnWorkerProtocolTest, PreservesExplicitEmptyModelImplementation) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  arguments[kModelImplArgumentIndex] = "";
  std::vector<char*> argv = mutable_argv(&arguments);

  EXPECT_TRUE(parse_model_impl(kArgumentCount, argv.data()).empty());
}

TEST(SpawnWorkerProtocolTest, DefaultsModelImplementationForLegacyArguments) {
  for (int32_t argc = kMinimumArgumentCount; argc <= kModelImplArgumentIndex;
       ++argc) {
    std::vector<std::string> arguments(argc, "unused");
    std::vector<char*> argv = mutable_argv(&arguments);

    EXPECT_TRUE(parse_model_impl(argc, argv.data()).empty());
  }
}

TEST(SpawnWorkerProtocolTest,
     PreservesModelImplementationWithTrailingArguments) {
  std::vector<std::string> arguments(kArgumentCount + 1, "unused");
  arguments[kModelImplArgumentIndex] = "python";
  std::vector<char*> argv = mutable_argv(&arguments);

  EXPECT_EQ(parse_model_impl(kArgumentCount + 1, argv.data()), "python");
}

TEST(SpawnWorkerProtocolTest, PreservesIndependentExtensionImplementation) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  arguments[kModelImplArgumentIndex] = "python";
  arguments[kExtensionImplArgumentIndex] = "custom";
  std::vector<char*> argv = mutable_argv(&arguments);

  EXPECT_EQ(parse_model_impl(kArgumentCount, argv.data()), "python");
  EXPECT_EQ(parse_extension_impl(kArgumentCount, argv.data()), "custom");
}

TEST(SpawnWorkerProtocolTest, PreservesExplicitEmptyExtensionImplementation) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  arguments[kExtensionImplArgumentIndex] = "";
  std::vector<char*> argv = mutable_argv(&arguments);

  EXPECT_TRUE(parse_extension_impl(kArgumentCount, argv.data()).empty());
}

TEST(SpawnWorkerProtocolTest,
     DefaultsExtensionImplementationForLegacyArguments) {
  for (int32_t argc = kMinimumArgumentCount;
       argc <= kExtensionImplArgumentIndex;
       ++argc) {
    std::vector<std::string> arguments(argc, "unused");
    std::vector<char*> argv = mutable_argv(&arguments);

    EXPECT_TRUE(parse_extension_impl(argc, argv.data()).empty());
  }
}

TEST(SpawnWorkerProtocolTest, DefaultsExtensionImplementationForNullArguments) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  std::vector<char*> argv = mutable_argv(&arguments);
  argv[kExtensionImplArgumentIndex] = nullptr;

  EXPECT_TRUE(parse_extension_impl(kArgumentCount, nullptr).empty());
  EXPECT_TRUE(parse_extension_impl(kArgumentCount, argv.data()).empty());
}

TEST(SpawnWorkerProtocolTest,
     PreservesExtensionImplementationWithTrailingArguments) {
  std::vector<std::string> arguments(kArgumentCount + 1, "unused");
  arguments[kExtensionImplArgumentIndex] = "custom";
  std::vector<char*> argv = mutable_argv(&arguments);

  EXPECT_EQ(parse_extension_impl(kArgumentCount + 1, argv.data()), "custom");
}

TEST(SpawnWorkerProtocolTest, PreservesResolvedPrefillSettings) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  arguments[kPrefillChunkArgumentIndex] = "512";
  arguments[kEnableChunkedPrefillArgumentIndex] = "0";
  arguments[kEnableDpFairTokenBudgetArgumentIndex] = "1";
  std::vector<char*> argv = mutable_argv(&arguments);
  const std::optional<PrefillConfig> config =
      parse_prefill_config(kArgumentCount, argv.data());
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->max_chunk_tokens, 512);
  EXPECT_FALSE(config->enable_chunked_prefill);
  EXPECT_TRUE(config->enable_dp_fair_token_budget);
}

TEST(SpawnWorkerProtocolTest, PreservesLegacyPrefillDefaults) {
  std::vector<std::string> arguments(kEplbArgumentCount, "unused");
  std::vector<char*> argv = mutable_argv(&arguments);
  const std::optional<PrefillConfig> config =
      parse_prefill_config(kEplbArgumentCount, argv.data());
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->max_chunk_tokens, -1);
  EXPECT_TRUE(config->enable_chunked_prefill);
  EXPECT_TRUE(config->enable_dp_fair_token_budget);
}

TEST(SpawnWorkerProtocolTest, RejectsPartialOrInvalidPrefillSettings) {
  std::vector<std::string> arguments(kArgumentCount, "1");
  std::vector<char*> argv = mutable_argv(&arguments);
  for (int32_t argc = kPrefillChunkArgumentIndex + 1; argc < kArgumentCount;
       ++argc) {
    EXPECT_FALSE(parse_prefill_config(argc, argv.data()).has_value());
  }
  for (const char* invalid : {"0", "-1", "2147483648", "512tail"}) {
    arguments[kPrefillChunkArgumentIndex] = invalid;
    argv = mutable_argv(&arguments);
    EXPECT_FALSE(parse_prefill_config(kArgumentCount, argv.data()).has_value());
  }
  arguments[kPrefillChunkArgumentIndex] = "512";
  arguments[kEnableChunkedPrefillArgumentIndex] = "2";
  argv = mutable_argv(&arguments);
  EXPECT_FALSE(parse_prefill_config(kArgumentCount, argv.data()).has_value());
}

TEST(SpawnWorkerProtocolTest, RejectsMissingPrefillArguments) {
  EXPECT_FALSE(parse_prefill_config(kArgumentCount, nullptr).has_value());
  for (int32_t index = kPrefillChunkArgumentIndex; index < kArgumentCount;
       ++index) {
    std::vector<std::string> arguments(kArgumentCount, "1");
    std::vector<char*> argv = mutable_argv(&arguments);
    argv[index] = nullptr;
    EXPECT_FALSE(parse_prefill_config(kArgumentCount, argv.data()).has_value());
  }
}

TEST(SpawnWorkerProtocolTest, PreservesExplicitEplbWorkerConfig) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  arguments[kEnableEplbArgumentIndex] = "1";
  arguments[kRedundantExpertsNumArgumentIndex] = "4";
  arguments[kUseDecodeOnlyLoadArgumentIndex] = "1";
  std::vector<char*> argv = mutable_argv(&arguments);

  const std::optional<EplbWorkerConfig> config =
      parse_eplb_worker_config(kArgumentCount, argv.data());

  ASSERT_TRUE(config.has_value());
  EXPECT_TRUE(config->enabled);
  EXPECT_EQ(config->redundant_experts_num, 4);
  EXPECT_TRUE(config->use_decode_only_load);
}

TEST(SpawnWorkerProtocolTest, PreservesLoadPolicyWhenEplbIsDisabled) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  arguments[kEnableEplbArgumentIndex] = "0";
  arguments[kRedundantExpertsNumArgumentIndex] = "0";
  arguments[kUseDecodeOnlyLoadArgumentIndex] = "1";
  std::vector<char*> argv = mutable_argv(&arguments);

  const std::optional<EplbWorkerConfig> config =
      parse_eplb_worker_config(kArgumentCount, argv.data());

  ASSERT_TRUE(config.has_value());
  EXPECT_FALSE(config->enabled);
  EXPECT_EQ(config->redundant_experts_num, 0);
  EXPECT_TRUE(config->use_decode_only_load);
}

TEST(SpawnWorkerProtocolTest, DefaultsEplbWorkerConfigForLegacyArguments) {
  for (int32_t argc = kMinimumArgumentCount; argc <= kEnableEplbArgumentIndex;
       ++argc) {
    SCOPED_TRACE(argc);
    std::vector<std::string> arguments(argc, "unused");
    std::vector<char*> argv = mutable_argv(&arguments);

    const std::optional<EplbWorkerConfig> config =
        parse_eplb_worker_config(argc, argv.data());

    ASSERT_TRUE(config.has_value());
    EXPECT_FALSE(config->enabled);
    EXPECT_EQ(config->redundant_experts_num, 1);
    EXPECT_FALSE(config->use_decode_only_load);
  }
}

TEST(SpawnWorkerProtocolTest, RejectsPartialEplbWorkerConfig) {
  for (int32_t argc = kEnableEplbArgumentIndex + 1; argc < kEplbArgumentCount;
       ++argc) {
    SCOPED_TRACE(argc);
    std::vector<std::string> arguments(argc, "1");
    std::vector<char*> argv = mutable_argv(&arguments);

    EXPECT_FALSE(parse_eplb_worker_config(argc, argv.data()).has_value());
  }
}

TEST(SpawnWorkerProtocolTest, RejectsInvalidEplbBooleanValues) {
  for (const int32_t index :
       {kEnableEplbArgumentIndex, kUseDecodeOnlyLoadArgumentIndex}) {
    SCOPED_TRACE(index);
    for (const char* invalid : {"", "2", "-1", "true", "1tail", " 1"}) {
      SCOPED_TRACE(invalid);
      std::vector<std::string> arguments(kArgumentCount, "1");
      arguments[index] = invalid;
      std::vector<char*> argv = mutable_argv(&arguments);

      EXPECT_FALSE(
          parse_eplb_worker_config(kArgumentCount, argv.data()).has_value());
    }
  }
}

TEST(SpawnWorkerProtocolTest, RejectsInvalidRedundantExpertCounts) {
  for (const char* invalid :
       {"", "-1", "2147483648", "-2147483649", "1tail", " 1", "+1"}) {
    SCOPED_TRACE(invalid);
    std::vector<std::string> arguments(kArgumentCount, "1");
    arguments[kRedundantExpertsNumArgumentIndex] = invalid;
    std::vector<char*> argv = mutable_argv(&arguments);

    EXPECT_FALSE(
        parse_eplb_worker_config(kArgumentCount, argv.data()).has_value());
  }
}

TEST(SpawnWorkerProtocolTest, AcceptsMaximumRedundantExpertCount) {
  std::vector<std::string> arguments(kArgumentCount, "1");
  arguments[kRedundantExpertsNumArgumentIndex] = "2147483647";
  std::vector<char*> argv = mutable_argv(&arguments);

  const std::optional<EplbWorkerConfig> config =
      parse_eplb_worker_config(kArgumentCount, argv.data());

  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->redundant_experts_num, 2147483647);
}

TEST(SpawnWorkerProtocolTest, RejectsMissingEplbWorkerArguments) {
  EXPECT_FALSE(parse_eplb_worker_config(kArgumentCount, nullptr).has_value());
  std::vector<std::string> arguments(kMinimumArgumentCount - 1, "1");
  std::vector<char*> argv = mutable_argv(&arguments);
  EXPECT_FALSE(parse_eplb_worker_config(kMinimumArgumentCount - 1, argv.data())
                   .has_value());

  for (int32_t index = kEnableEplbArgumentIndex; index < kEplbArgumentCount;
       ++index) {
    SCOPED_TRACE(index);
    std::vector<std::string> current_arguments(kArgumentCount, "1");
    std::vector<char*> current_argv = mutable_argv(&current_arguments);
    current_argv[index] = nullptr;

    EXPECT_FALSE(parse_eplb_worker_config(kArgumentCount, current_argv.data())
                     .has_value());
  }
}

TEST(SpawnWorkerProtocolTest, PreservesEplbConfigWithUnknownTrailingArguments) {
  std::vector<std::string> arguments(kArgumentCount + 1, "unused");
  arguments[kEnableEplbArgumentIndex] = "1";
  arguments[kRedundantExpertsNumArgumentIndex] = "8";
  arguments[kUseDecodeOnlyLoadArgumentIndex] = "0";
  std::vector<char*> argv = mutable_argv(&arguments);

  const std::optional<EplbWorkerConfig> config =
      parse_eplb_worker_config(kArgumentCount + 1, argv.data());

  ASSERT_TRUE(config.has_value());
  EXPECT_TRUE(config->enabled);
  EXPECT_EQ(config->redundant_experts_num, 8);
  EXPECT_FALSE(config->use_decode_only_load);
}

TEST(SpawnWorkerProtocolTest, PreservesExplicitGraphDoubleBufferMode) {
  for (const bool enabled : {false, true}) {
    SCOPED_TRACE(enabled);
    std::vector<std::string> arguments(
        kEnableGraphDoubleBufferArgumentIndex + 2, "unused");
    arguments[kEnableGraphDoubleBufferArgumentIndex] = enabled ? "1" : "0";
    std::vector<char*> argv = mutable_argv(&arguments);
    const std::optional<bool> parsed = parse_graph_double_buffer(
        static_cast<int32_t>(argv.size()), argv.data());
    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed.value(), enabled);
  }
}

TEST(SpawnWorkerProtocolTest, DefaultsGraphDoubleBufferForLegacyArguments) {
  for (int32_t argc = kMinimumArgumentCount;
       argc <= kEnableGraphDoubleBufferArgumentIndex;
       ++argc) {
    SCOPED_TRACE(argc);
    std::vector<std::string> arguments(argc, "unused");
    std::vector<char*> argv = mutable_argv(&arguments);
    const std::optional<bool> parsed =
        parse_graph_double_buffer(argc, argv.data());
    ASSERT_TRUE(parsed.has_value());
    EXPECT_TRUE(parsed.value());
  }
}

TEST(SpawnWorkerProtocolTest, RejectsInvalidGraphDoubleBufferMode) {
  for (const char* invalid : {"", "2", "-1", "true", "1tail", " 1"}) {
    SCOPED_TRACE(invalid);
    std::vector<std::string> arguments(
        kEnableGraphDoubleBufferArgumentIndex + 1, "unused");
    arguments[kEnableGraphDoubleBufferArgumentIndex] = invalid;
    std::vector<char*> argv = mutable_argv(&arguments);
    EXPECT_FALSE(parse_graph_double_buffer(static_cast<int32_t>(argv.size()),
                                           argv.data())
                     .has_value());
  }
  EXPECT_FALSE(
      parse_graph_double_buffer(kMinimumArgumentCount, nullptr).has_value());
  std::vector<std::string> arguments(kEnableGraphDoubleBufferArgumentIndex + 1,
                                     "unused");
  std::vector<char*> argv = mutable_argv(&arguments);
  argv[kEnableGraphDoubleBufferArgumentIndex] = nullptr;
  EXPECT_FALSE(
      parse_graph_double_buffer(static_cast<int32_t>(argv.size()), argv.data())
          .has_value());
  EXPECT_FALSE(parse_graph_double_buffer(kMinimumArgumentCount - 1, argv.data())
                   .has_value());
}

TEST(SpawnWorkerProtocolTest, PreservesGraphDecodeLimitAndTrailingArguments) {
  for (const int32_t limit : {-1, 0, 1, 20, 2147483647}) {
    SCOPED_TRACE(limit);
    for (const int32_t trailing : {0, 1}) {
      std::vector<std::string> arguments(
          kAclGraphDecodeBatchSizeLimitArgumentIndex + 1 + trailing, "unused");
      arguments[kAclGraphDecodeBatchSizeLimitArgumentIndex] =
          std::to_string(limit);
      std::vector<char*> argv = mutable_argv(&arguments);
      const auto parsed = parse_graph_decode_batch_size_limit(
          static_cast<int32_t>(argv.size()), argv.data());
      ASSERT_TRUE(parsed.has_value());
      EXPECT_EQ(parsed.value(), limit);
    }
  }
}

TEST(SpawnWorkerProtocolTest, DefaultsGraphDecodeLimitForLegacyArguments) {
  for (int32_t argc = kMinimumArgumentCount;
       argc <= kAclGraphDecodeBatchSizeLimitArgumentIndex;
       ++argc) {
    std::vector<std::string> arguments(argc, "unused");
    std::vector<char*> argv = mutable_argv(&arguments);
    const auto parsed = parse_graph_decode_batch_size_limit(argc, argv.data());
    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed.value(), 16);
  }
}

TEST(SpawnWorkerProtocolTest, RejectsMalformedGraphDecodeLimit) {
  for (const char* text :
       {"", " 1", "1x", "1.0", "2147483648", "-2147483649"}) {
    std::vector<std::string> arguments(
        kAclGraphDecodeBatchSizeLimitArgumentIndex + 1, "unused");
    arguments[kAclGraphDecodeBatchSizeLimitArgumentIndex] = text;
    std::vector<char*> argv = mutable_argv(&arguments);
    EXPECT_FALSE(parse_graph_decode_batch_size_limit(
                     static_cast<int32_t>(argv.size()), argv.data())
                     .has_value());
  }
  EXPECT_FALSE(
      parse_graph_decode_batch_size_limit(kMinimumArgumentCount, nullptr)
          .has_value());
  std::vector<std::string> arguments(
      kAclGraphDecodeBatchSizeLimitArgumentIndex + 1, "unused");
  std::vector<char*> argv = mutable_argv(&arguments);
  argv[kAclGraphDecodeBatchSizeLimitArgumentIndex] = nullptr;
  EXPECT_FALSE(parse_graph_decode_batch_size_limit(
                   static_cast<int32_t>(argv.size()), argv.data())
                   .has_value());
  EXPECT_FALSE(parse_graph_decode_batch_size_limit(kMinimumArgumentCount - 1,
                                                   argv.data())
                   .has_value());
}

TEST(SpawnWorkerProtocolTest, TopologyDefaultsForLegacyLaunchers) {
  for (int32_t argc = kMinimumArgumentCount;
       argc <= kLayerwiseSplitSizeArgumentIndex;
       ++argc) {
    std::vector<std::string> arguments(argc, "unused");
    std::vector<char*> argv = mutable_argv(&arguments);
    const auto parsed = parse_topology_config(argc, argv.data());
    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed->layerwise_split_size, 1);
    EXPECT_FALSE(parsed->enable_mm_encoder_dp);
  }
}

TEST(SpawnWorkerProtocolTest, PreservesTopologyPairWithTrailingArguments) {
  for (const int32_t split : {1, 2, 8}) {
    for (const bool encoder_dp : {false, true}) {
      for (const int32_t trailing : {0, 1}) {
        std::vector<std::string> arguments(
            kEnableMmEncoderDpArgumentIndex + 1 + trailing, "unused");
        arguments[kLayerwiseSplitSizeArgumentIndex] = std::to_string(split);
        arguments[kEnableMmEncoderDpArgumentIndex] = encoder_dp ? "1" : "0";
        std::vector<char*> argv = mutable_argv(&arguments);
        const auto parsed = parse_topology_config(
            static_cast<int32_t>(argv.size()), argv.data());
        ASSERT_TRUE(parsed.has_value());
        EXPECT_EQ(parsed->layerwise_split_size, split);
        EXPECT_EQ(parsed->enable_mm_encoder_dp, encoder_dp);
      }
    }
  }
}

TEST(SpawnWorkerProtocolTest, RejectsMalformedOrPartialTopologyPair) {
  std::vector<std::string> arguments(kEnableMmEncoderDpArgumentIndex + 1,
                                     "unused");
  arguments[kLayerwiseSplitSizeArgumentIndex] = "2";
  arguments[kEnableMmEncoderDpArgumentIndex] = "1";
  std::vector<char*> argv = mutable_argv(&arguments);
  const int32_t argc = static_cast<int32_t>(argv.size());
  EXPECT_FALSE(parse_topology_config(argc, nullptr).has_value());
  EXPECT_FALSE(parse_topology_config(kMinimumArgumentCount - 1, argv.data())
                   .has_value());
  EXPECT_FALSE(
      parse_topology_config(kEnableMmEncoderDpArgumentIndex, argv.data())
          .has_value());
  for (const char* invalid : {"", "0", "-1", " 2", "2x", "2147483648"}) {
    arguments[kLayerwiseSplitSizeArgumentIndex] = invalid;
    argv = mutable_argv(&arguments);
    EXPECT_FALSE(parse_topology_config(argc, argv.data()).has_value());
  }
  arguments[kLayerwiseSplitSizeArgumentIndex] = "2";
  for (const char* invalid : {"", "true", "2", "-1"}) {
    arguments[kEnableMmEncoderDpArgumentIndex] = invalid;
    argv = mutable_argv(&arguments);
    EXPECT_FALSE(parse_topology_config(argc, argv.data()).has_value());
  }
  arguments[kEnableMmEncoderDpArgumentIndex] = "1";
  argv = mutable_argv(&arguments);
  argv[kLayerwiseSplitSizeArgumentIndex] = nullptr;
  EXPECT_FALSE(parse_topology_config(argc, argv.data()).has_value());
  argv = mutable_argv(&arguments);
  argv[kEnableMmEncoderDpArgumentIndex] = nullptr;
  EXPECT_FALSE(parse_topology_config(argc, argv.data()).has_value());
}

TEST(SpawnWorkerProtocolTest, DefaultsGraphVmmPoolForLegacyLaunchers) {
  for (int32_t argc = kMinimumArgumentCount;
       argc <= kEnableGraphVmmPoolArgumentIndex;
       ++argc) {
    std::vector<std::string> arguments(argc, "unused");
    std::vector<char*> argv = mutable_argv(&arguments);
    const auto parsed = parse_graph_vmm_pool(argc, argv.data());
    ASSERT_TRUE(parsed.has_value());
    EXPECT_TRUE(parsed.value());
  }
}

TEST(SpawnWorkerProtocolTest, PreservesGraphVmmPoolWithTrailingArguments) {
  for (const bool enabled : {false, true}) {
    for (const int32_t trailing : {0, 1}) {
      std::vector<std::string> arguments(
          kEnableGraphVmmPoolArgumentIndex + 1 + trailing, "unused");
      arguments[kEnableGraphVmmPoolArgumentIndex] = enabled ? "1" : "0";
      std::vector<char*> argv = mutable_argv(&arguments);
      const auto parsed =
          parse_graph_vmm_pool(static_cast<int32_t>(argv.size()), argv.data());
      ASSERT_TRUE(parsed.has_value());
      EXPECT_EQ(parsed.value(), enabled);
    }
  }
}

TEST(SpawnWorkerProtocolTest, RejectsMalformedGraphVmmPool) {
  std::vector<std::string> arguments(kEnableGraphVmmPoolArgumentIndex + 1,
                                     "unused");
  for (const char* invalid : {"", "true", "false", "2", "-1", " 1", "1x"}) {
    arguments[kEnableGraphVmmPoolArgumentIndex] = invalid;
    std::vector<char*> argv = mutable_argv(&arguments);
    EXPECT_FALSE(
        parse_graph_vmm_pool(static_cast<int32_t>(argv.size()), argv.data())
            .has_value());
  }
  std::vector<char*> argv = mutable_argv(&arguments);
  const int32_t argc = static_cast<int32_t>(argv.size());
  EXPECT_FALSE(parse_graph_vmm_pool(argc, nullptr).has_value());
  EXPECT_FALSE(
      parse_graph_vmm_pool(kMinimumArgumentCount - 1, argv.data()).has_value());
  argv[kEnableGraphVmmPoolArgumentIndex] = nullptr;
  EXPECT_FALSE(parse_graph_vmm_pool(argc, argv.data()).has_value());
}

}  // namespace
}  // namespace xllm::spawn_worker_protocol
