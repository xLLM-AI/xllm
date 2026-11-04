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

#include "core/framework/batch/dit_forward_input_factory.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/framework/batch/dit_batch.h"
#include "core/framework/config/dit_config.h"
#include "core/framework/request/minimax_h3_conditioning.h"
#include "models/dit/utils/minimax_h3_execution_metrics.h"

namespace xllm {
namespace {

std::shared_ptr<DiTRequest> make_request(const std::string& prompt,
                                         float tensor_value) {
  DiTRequestState state;
  state.input_params().prompt = prompt;
  state.input_params().media_sources.add(
      "image", "image", torch::full({3, 2, 2}, tensor_value));
  state.input_params().tensor_sources.add("prompt_embed",
                                          torch::full({2, 4}, tensor_value));
  return std::make_shared<DiTRequest>(prompt, "", "", state);
}

TEST(DiTForwardInputFactoryTest, PreservesRequestOrderAndNamedInputs) {
  auto first = make_request("first", /*tensor_value=*/1.0f);
  auto second = make_request("second", /*tensor_value=*/2.0f);
  DiTBatch batch;
  batch.add(first);
  batch.add(second);
  DiTForwardInputFactory factory;
  auto input = factory.create_input(batch);

  EXPECT_EQ(input.model_params.batch_size, 2);
  EXPECT_EQ(input.model_params.prompts,
            (std::vector<std::string>{"first", "second"}));
  EXPECT_TRUE(input.model_params.generation_params ==
              first->state().generation_params());
  ASSERT_EQ(input.model_params.media_sources.size(), 1u);
  EXPECT_EQ(input.model_params.media_sources.at(0).name, "image");
  EXPECT_TRUE(
      torch::equal(input.model_params.media_sources.at(0).tensor[0],
                   first->state().input_params().media_sources.at(0).tensor));
  EXPECT_TRUE(
      torch::equal(input.model_params.media_sources.at(0).tensor[1],
                   second->state().input_params().media_sources.at(0).tensor));
  const auto prompt_embed =
      input.model_params.tensor_sources.get("prompt_embed");
  ASSERT_TRUE(prompt_embed.has_value());
  EXPECT_EQ(prompt_embed->sizes().vec(), (std::vector<int64_t>{2, 2, 4}));
  EXPECT_TRUE(torch::equal(
      (*prompt_embed)[0],
      *first->state().input_params().tensor_sources.get("prompt_embed")));
  EXPECT_TRUE(torch::equal(
      (*prompt_embed)[1],
      *second->state().input_params().tensor_sources.get("prompt_embed")));
}

TEST(DiTForwardInputFactoryTest, ReusesSingleRequestTensorStorage) {
  auto request = make_request("single", /*tensor_value=*/3.0f);
  DiTBatch batch;
  batch.add(request);
  DiTForwardInputFactory factory;
  auto input = factory.create_input(batch);

  const torch::Tensor& image = input.model_params.media_sources.at(0).tensor;
  EXPECT_EQ(image.sizes(), torch::IntArrayRef({1, 3, 2, 2}));
  EXPECT_EQ(
      image.data_ptr(),
      request->state().input_params().media_sources.at(0).tensor.data_ptr());
  const auto prompt_embed =
      input.model_params.tensor_sources.get("prompt_embed");
  ASSERT_TRUE(prompt_embed.has_value());
  EXPECT_EQ(prompt_embed->data_ptr(),
            request->state()
                .input_params()
                .tensor_sources.get("prompt_embed")
                ->data_ptr());
}

TEST(DiTForwardInputFactoryDeathTest, RejectsMixedGenerationParameters) {
  auto first = make_request("first", /*tensor_value=*/1.0f);
  auto second = make_request("second", /*tensor_value=*/2.0f);
  second->state().generation_params().width += 1;
  DiTBatch batch;
  batch.add(first);
  batch.add(second);
  DiTForwardInputFactory factory;
  EXPECT_DEATH(factory.create_input(batch), "generation params");
}

class H3RequestIdentityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    profile_ = DiTConfig::get_instance().dit_h3_execution_profile();
    timing_ = DiTConfig::get_instance().dit_h3_phase_timing();
    DiTConfig::get_instance().dit_h3_phase_timing(false);
    DiTConfig::get_instance().dit_h3_execution_profile(
        std::string(kMiniMaxH3ExecutionProfile));
  }
  void TearDown() override {
    DiTConfig::get_instance().dit_h3_execution_profile(profile_);
    DiTConfig::get_instance().dit_h3_phase_timing(timing_);
  }

  std::string profile_;
  bool timing_ = false;
};

TEST_F(H3RequestIdentityTest, ReplacesClientIdentityWithoutMutatingRequest) {
  auto request = make_request("server-request-1", 3.0F);
  auto& source = request->state().input_params().tensor_sources.entries()[0];
  source.parameters[std::string(kMiniMaxH3InternalRequestId)] =
      std::string("spoofed-client-id");
  DiTBatch batch;
  batch.add(request);
  const auto input = DiTForwardInputFactory().create_input(batch);
  const auto named =
      input.model_params.tensor_sources.get_named_tensor("prompt_embed");
  ASSERT_TRUE(named.has_value());
  const auto* transported = get_tensor_parameter<std::string>(
      named->get().parameters, kMiniMaxH3InternalRequestId);
  ASSERT_NE(transported, nullptr);
  EXPECT_EQ(*transported, "server-request-1");
  EXPECT_EQ(*get_tensor_parameter<std::string>(source.parameters,
                                               kMiniMaxH3InternalRequestId),
            "spoofed-client-id");
  EXPECT_EQ(named->get().tensor.data_ptr(), source.tensor.data_ptr());
  EXPECT_EQ(input.model_params.tensor_sources.size(), 1);
  const auto moved = input.model_params.to(torch::kCPU, torch::kFloat32);
  EXPECT_EQ(*get_tensor_parameter<std::string>(
                moved.tensor_sources.entries()[0].parameters,
                kMiniMaxH3InternalRequestId),
            "server-request-1");
}

TEST_F(H3RequestIdentityTest, IdenticalConditioningKeepsDistinctServerIds) {
  std::vector<std::string> observed;
  for (const std::string& id : {"server-request-1", "server-request-2"}) {
    DiTBatch batch;
    batch.add(make_request(id, 1.0F));
    const auto input = DiTForwardInputFactory().create_input(batch);
    observed.push_back(*get_tensor_parameter<std::string>(
        input.model_params.tensor_sources.entries()[0].parameters,
        kMiniMaxH3InternalRequestId));
  }
  EXPECT_EQ(observed,
            (std::vector<std::string>{"server-request-1", "server-request-2"}));
}

TEST_F(H3RequestIdentityTest, NativeDoesNotInjectIdentity) {
  DiTConfig::get_instance().dit_h3_execution_profile("native");
  DiTBatch batch;
  batch.add(make_request("native-request", 1.0F));
  const auto input = DiTForwardInputFactory().create_input(batch);
  EXPECT_EQ(get_tensor_parameter<std::string>(
                input.model_params.tensor_sources.entries()[0].parameters,
                kMiniMaxH3InternalRequestId),
            nullptr);
}

TEST_F(H3RequestIdentityTest, RejectsMissingAndAmbiguousServerIdentity) {
  DiTBatch empty_id;
  empty_id.add(make_request("", 1.0F));
  EXPECT_THROW(DiTForwardInputFactory().create_input(empty_id),
               std::invalid_argument);
  DiTBatch multiple;
  multiple.add(make_request("first", 1.0F));
  multiple.add(make_request("second", 1.0F));
  EXPECT_THROW(DiTForwardInputFactory().create_input(multiple),
               std::invalid_argument);
}

TEST_F(H3RequestIdentityTest, NativeTimingOptInUsesServerIdentity) {
  DiTConfig::get_instance().dit_h3_execution_profile("native");
  DiTConfig::get_instance().dit_h3_phase_timing(true);
  DiTBatch batch;
  batch.add(make_request("native-measurement-request", 1.0F));
  const auto input = DiTForwardInputFactory().create_input(batch);
  EXPECT_EQ(h3_execution_request_id(input.model_params),
            "native-measurement-request");
}

TEST(H3ExecutionMetricsTest,
     DenseAndCachedRequestsHaveIndependentActualCounts) {
  for (int32_t request_index = 0; request_index < 2; ++request_index) {
    H3QualifiedExecutionMetrics metrics;
    metrics.request_id = "fixture-" + std::to_string(request_index);
    for (int64_t step = 0; step < 49; ++step) {
      const bool hit = request_index == 1 &&
                       (step == 30 || step == 32 || step == 34 || step == 36);
      record_h3_qualified_execution_step(metrics, 0, step, hit, hit ? 1 : 50);
    }
    EXPECT_EQ(metrics.executed_blocks, request_index == 0 ? 2450 : 2254);
    EXPECT_EQ(metrics.dense_executed_blocks, metrics.executed_blocks);
    EXPECT_EQ(metrics.rf_executed_blocks, 0);
    EXPECT_EQ(metrics.selector_launches, 0);
    EXPECT_EQ(metrics.attention_launches, 0);
    EXPECT_EQ(metrics.hit_steps.size(), request_index == 0 ? 0 : 4);
    EXPECT_FALSE(metrics.completion_synchronized);
    EXPECT_THROW(emit_h3_qualified_execution_summary(metrics, 0),
                 std::logic_error);
  }
}

TEST(H3ExecutionMetricsTest,
     RejectsMissingIdentityClockAndCounterDisagreement) {
  H3QualifiedExecutionMetrics metrics;
  EXPECT_THROW(record_h3_qualified_execution_step(metrics, 0, 0, false, 50),
               std::logic_error);
  metrics.request_id = "fixture";
  EXPECT_THROW(record_h3_qualified_execution_step(metrics, 0, 1, false, 50),
               std::logic_error);
  EXPECT_THROW(record_h3_qualified_execution_step(metrics, 0, 0, false, 0),
               std::logic_error);
  H3QualifiedLaunchSnapshot after{.rf_blocks = 1};
  EXPECT_THROW(
      record_h3_qualified_execution_step(metrics, 0, 0, false, 50, {}, after),
      std::logic_error);
  EXPECT_EQ(metrics.executed_blocks, 0);
  EXPECT_TRUE(metrics.observed_steps.empty());
  EXPECT_THROW(finish_h3_qualified_request(metrics, torch::Device(torch::kCPU)),
               std::logic_error);
}

// Counter arithmetic only; injected observations do not prove device launches.
TEST(H3ExecutionMetricsTest,
     RainFusionWindowUsesActualDeltasAndHalfOpenBounds) {
  H3QualifiedExecutionMetrics metrics;
  metrics.request_id = "fixture-rf-counter-arithmetic";
  metrics.rf_ready = true;
  H3QualifiedLaunchSnapshot totals;
  for (int64_t step = 0; step < 49; ++step) {
    const H3QualifiedLaunchSnapshot before = totals;
    const bool hit = step == 30 || step == 32 || step == 34 || step == 36;
    const int64_t actual = hit ? 1 : 50;
    if (step >= 8 && step < 36) {
      totals.rf_blocks += actual;
      totals.selector_launches += actual;
      totals.attention_launches += actual;
    }
    record_h3_qualified_execution_step(
        metrics, 0, step, hit, actual, before, totals);
  }
  EXPECT_EQ(metrics.rf_executed_blocks, 1253);
  EXPECT_EQ(metrics.dense_executed_blocks, 1001);
  EXPECT_EQ(metrics.executed_blocks, 2254);
  EXPECT_EQ(metrics.selector_launches, 1253);
  EXPECT_EQ(metrics.attention_launches, 1253);
}

TEST(H3ExecutionMetricsTest, RequestIdHasNoHashOrCounterFallback) {
  DiTModelParams params;
  params.batch_size = 1;
  EXPECT_THROW(h3_execution_request_id(params), std::invalid_argument);
  params.tensor_sources.add("prompt_embed", torch::ones({1, 1, 1}));
  auto& parameters = params.tensor_sources.entries()[0].parameters;
  EXPECT_THROW(h3_execution_request_id(params), std::invalid_argument);
  parameters[std::string(kMiniMaxH3InternalRequestId)] = int64_t{7};
  EXPECT_THROW(h3_execution_request_id(params), std::invalid_argument);
  parameters[std::string(kMiniMaxH3InternalRequestId)] = std::string();
  EXPECT_THROW(h3_execution_request_id(params), std::invalid_argument);
  parameters[std::string(kMiniMaxH3InternalRequestId)] =
      std::string("fixture-id");
  EXPECT_EQ(h3_execution_request_id(params), "fixture-id");
}

TEST(H3ExecutionMetricsTest,
     CommunicationCountsRequireAllSubmittedWorksToComplete) {
  H3QualifiedExecutionMetrics metrics;
  metrics.request_id = "fixture-qkv";
  metrics.qkv_observed = true;
  H3QualifiedLaunchSnapshot after{
      .qkv_a2a_launches = 150, .qkv_a2a_waits = 149, .inverse_a2a_calls = 50};
  EXPECT_THROW(
      record_h3_qualified_execution_step(metrics, 0, 0, false, 50, {}, after),
      std::logic_error);
  EXPECT_TRUE(metrics.observed_steps.empty());
  after.qkv_a2a_waits = 150;
  record_h3_qualified_execution_step(metrics, 0, 0, false, 50, {}, after);
  EXPECT_EQ(metrics.qkv_a2a_launches, 150);
  EXPECT_EQ(metrics.qkv_a2a_waits, 150);
  EXPECT_EQ(metrics.inverse_a2a_calls, 50);
}

}  // namespace
}  // namespace xllm
