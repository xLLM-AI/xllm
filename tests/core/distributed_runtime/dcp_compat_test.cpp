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

#include "core/distributed_runtime/dcp_compat.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>

namespace xllm {
namespace {

Options dcp_options_with_supported_feature_flags() {
  Options options;
  options.decode_context_parallel_size(2)
      .enable_chunked_prefill(false)
      .enable_prefix_cache(false)
      .enable_schedule_overlap(false)
      .enable_disagg_pd(false)
      .instance_role(InstanceRole::DEFAULT)
      .num_speculative_tokens(0);
  return options;
}

void expect_error_contains(const std::optional<std::string>& error,
                           const std::string& expected) {
  ASSERT_TRUE(error.has_value());
  EXPECT_NE(error->find(expected), std::string::npos) << error.value();
}

TEST(DcpCompatTest, DcpOneDoesNotRejectDefaultOptions) {
  Options options;
  options.decode_context_parallel_size(1);

  EXPECT_FALSE(
      validate_dcp_first_version_options(options, EngineType::LLM).has_value());
}

TEST(DcpCompatTest, AllowsSupportedFirstVersionFeatureFlags) {
  const Options options = dcp_options_with_supported_feature_flags();

  EXPECT_FALSE(
      validate_dcp_first_version_options(options, EngineType::LLM).has_value());
}

TEST(DcpCompatTest, RejectsDefaultChunkedPrefillFirst) {
  Options options;
  options.decode_context_parallel_size(2);

  expect_error_contains(
      validate_dcp_first_version_options(options, EngineType::LLM),
      "enable_chunked_prefill=false");
}

TEST(DcpCompatTest, RejectsPrefixCache) {
  Options options = dcp_options_with_supported_feature_flags();
  options.enable_prefix_cache(true);

  expect_error_contains(
      validate_dcp_first_version_options(options, EngineType::LLM),
      "enable_prefix_cache=false");
}

TEST(DcpCompatTest, RejectsScheduleOverlap) {
  Options options = dcp_options_with_supported_feature_flags();
  options.enable_schedule_overlap(true);

  expect_error_contains(
      validate_dcp_first_version_options(options, EngineType::LLM),
      "enable_schedule_overlap=false");
}

TEST(DcpCompatTest, RejectsDisaggregatedPrefillDecodeFlag) {
  Options options = dcp_options_with_supported_feature_flags();
  options.enable_disagg_pd(true);

  expect_error_contains(
      validate_dcp_first_version_options(options, EngineType::LLM),
      "enable_disagg_pd=false");
}

TEST(DcpCompatTest, RejectsDisaggregatedPrefillDecodeRole) {
  Options options = dcp_options_with_supported_feature_flags();
  options.instance_role(InstanceRole::DECODE);

  expect_error_contains(
      validate_dcp_first_version_options(options, EngineType::LLM),
      "instance_role=DEFAULT");
}

TEST(DcpCompatTest, RejectsSpeculativeEngineType) {
  const Options options = dcp_options_with_supported_feature_flags();

  expect_error_contains(
      validate_dcp_first_version_options(options, EngineType::SSM),
      "speculative decoding");
}

TEST(DcpCompatTest, RejectsDraftModelPath) {
  Options options = dcp_options_with_supported_feature_flags();
  options.draft_model_path("/tmp/draft-model");

  expect_error_contains(
      validate_dcp_first_version_options(options, EngineType::LLM),
      "draft_model");
}

TEST(DcpCompatTest, RejectsSpeculativeTokens) {
  Options options = dcp_options_with_supported_feature_flags();
  options.num_speculative_tokens(1);

  expect_error_contains(
      validate_dcp_first_version_options(options, EngineType::LLM),
      "num_speculative_tokens=0");
}

TEST(DcpCompatTest, AllowsDenseQwen35ModelType) {
  EXPECT_FALSE(
      validate_dcp_first_version_model_type("qwen3_5_text").has_value());
}

TEST(DcpCompatTest, RejectsUnvalidatedQwen35MoeModelType) {
  expect_error_contains(
      validate_dcp_first_version_model_type("qwen3_5_moe_text"), "MoE");
}

}  // namespace
}  // namespace xllm
