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

#include "core/runtime/schedule_overlap_state.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace xllm {
namespace {

using namespace std::chrono_literals;

ForwardOutput token_output(int32_t token_id) {
  ForwardOutput output;
  output.sample_output.next_tokens = torch::tensor({token_id}, torch::kInt64);
  return output;
}

TEST(ScheduleOverlapStateTest, ConsumedOutputRemainsAvailableForContinuation) {
  ScheduleOverlapState state(/*publish_to_host=*/true,
                             /*retain_tokenless_output=*/false);
  ASSERT_TRUE(state.record(token_output(7), {"request"}, {"sequence"}));

  ForwardOutput consumed = state.consume();
  ASSERT_TRUE(consumed.sample_output.next_tokens.defined());
  EXPECT_EQ(consumed.sample_output.next_tokens.item<int64_t>(), 7);
  EXPECT_TRUE(state.has_tokens());
  EXPECT_TRUE(state.prior_output().sample_output.next_tokens.defined());
  EXPECT_EQ(state.request_ids(), (std::vector<std::string>{"request"}));
  EXPECT_EQ(state.sample_sequence_ids(),
            (std::vector<std::string>{"sequence"}));

  ASSERT_TRUE(
      state.record(token_output(9), {"next_request"}, {"next_sequence"}));
  consumed = state.consume();
  EXPECT_EQ(consumed.sample_output.next_tokens.item<int64_t>(), 9);
  EXPECT_EQ(state.prior_output().sample_output.next_tokens.item<int64_t>(), 9);
  EXPECT_EQ(state.request_ids(), (std::vector<std::string>{"next_request"}));
  EXPECT_EQ(state.sample_sequence_ids(),
            (std::vector<std::string>{"next_sequence"}));
}

TEST(ScheduleOverlapStateTest, ChecksContinuationTokensAndRequestOwnership) {
  ScheduleOverlapState state(/*publish_to_host=*/true,
                             /*retain_tokenless_output=*/false);
  EXPECT_FALSE(state.can_reuse_tokens({"request"}));

  ASSERT_TRUE(state.record(token_output(7), {"request"}, {"sequence"}));
  EXPECT_TRUE(state.can_reuse_tokens({"request", "next"}));
  EXPECT_FALSE(state.can_reuse_tokens({"other"}));
}

TEST(ScheduleOverlapStateTest,
     CacheOnlyRecordsRefreshContinuationWithoutHostConsumption) {
  ScheduleOverlapState state(/*publish_to_host=*/false,
                             /*retain_tokenless_output=*/false);
  ASSERT_TRUE(
      state.record(token_output(7), {"prior_request"}, {"prior_sequence"}));

  auto next_record = std::async(std::launch::async, [&state]() {
    return state.record(token_output(9), {"next_request"}, {"next_sequence"});
  });
  const std::future_status next_status = next_record.wait_for(1s);
  if (next_status != std::future_status::ready) {
    state.shutdown();
  }
  ASSERT_EQ(next_status, std::future_status::ready);
  ASSERT_TRUE(next_record.get());
  EXPECT_EQ(state.prior_output().sample_output.next_tokens.item<int64_t>(), 9);
  EXPECT_EQ(state.request_ids(), (std::vector<std::string>{"next_request"}));
  EXPECT_EQ(state.sample_sequence_ids(),
            (std::vector<std::string>{"next_sequence"}));
}

TEST(ScheduleOverlapStateTest, EmptyPublicationClearsPriorContinuation) {
  for (const bool retain_tokenless_output : {false, true}) {
    ScheduleOverlapState state(/*publish_to_host=*/true,
                               retain_tokenless_output);
    ASSERT_TRUE(
        state.record(token_output(7), {"prior_request"}, {"prior_sequence"}));
    (void)state.consume();
    ASSERT_TRUE(state.has_tokens());
    ASSERT_TRUE(state.record(std::nullopt, {"request"}, {"sequence"}));

    const ForwardOutput consumed = state.consume();
    EXPECT_FALSE(consumed.sample_output.next_tokens.defined());
    EXPECT_FALSE(state.has_tokens());
    EXPECT_FALSE(state.prior_output().sample_output.next_tokens.defined());
    EXPECT_TRUE(state.request_ids().empty());
    EXPECT_TRUE(state.sample_sequence_ids().empty());
  }
}

TEST(ScheduleOverlapStateTest, TokenlessMetadataCanBePublishedAndConsumed) {
  ScheduleOverlapState state(/*publish_to_host=*/true,
                             /*retain_tokenless_output=*/false);
  ForwardOutput output;
  output.json_object_errors.push_back({"sequence", "row mismatch"});
  ASSERT_TRUE(state.record(std::move(output), {"request"}, {"sequence"}));

  const ForwardOutput consumed = state.consume();
  EXPECT_FALSE(consumed.sample_output.next_tokens.defined());
  ASSERT_EQ(consumed.json_object_errors.size(), 1U);
  EXPECT_EQ(state.sample_sequence_ids(),
            (std::vector<std::string>{"sequence"}));
}

TEST(ScheduleOverlapStateTest, FiltersOrdinaryTokenlessOutputFromPublication) {
  ScheduleOverlapState state(/*publish_to_host=*/true,
                             /*retain_tokenless_output=*/false);
  ForwardOutput output;
  output.expert_load_data = torch::tensor({3, 5}, torch::kInt64);
  output.prepared_token = 11;
  ASSERT_TRUE(state.record(std::move(output), {"request"}, {"sequence"}));

  const ForwardOutput consumed = state.consume();
  EXPECT_FALSE(consumed.expert_load_data.defined());
  EXPECT_EQ(consumed.prepared_token, -1);
  EXPECT_FALSE(state.has_tokens());
  EXPECT_TRUE(state.request_ids().empty());
  EXPECT_TRUE(state.sample_sequence_ids().empty());
}

TEST(ScheduleOverlapStateTest, RetainsTokenlessOutputWhenRequested) {
  ScheduleOverlapState state(/*publish_to_host=*/true,
                             /*retain_tokenless_output=*/true);
  ForwardOutput output;
  output.expert_load_data = torch::tensor({3, 5}, torch::kInt64);
  output.prepared_token = 11;
  ASSERT_TRUE(state.record(std::move(output), {"request"}, {"sequence"}));

  const ForwardOutput consumed = state.consume();
  EXPECT_FALSE(consumed.sample_output.next_tokens.defined());
  ASSERT_TRUE(consumed.expert_load_data.defined());
  EXPECT_TRUE(torch::equal(consumed.expert_load_data,
                           torch::tensor({3, 5}, torch::kInt64)));
  EXPECT_EQ(consumed.prepared_token, 11);
  EXPECT_EQ(state.request_ids(), (std::vector<std::string>{"request"}));
  EXPECT_EQ(state.sample_sequence_ids(),
            (std::vector<std::string>{"sequence"}));
}

TEST(ScheduleOverlapStateTest, RecordAppliesBackpressureUntilConsumed) {
  ScheduleOverlapState state(/*publish_to_host=*/true,
                             /*retain_tokenless_output=*/false);
  ASSERT_TRUE(state.record(token_output(1), {}, {}));

  auto second_publish = std::async(std::launch::async, [&state]() {
    return state.record(token_output(2), {}, {});
  });
  EXPECT_EQ(second_publish.wait_for(50ms), std::future_status::timeout);
  EXPECT_EQ(state.consume().sample_output.next_tokens.item<int64_t>(), 1);
  ASSERT_EQ(second_publish.wait_for(1s), std::future_status::ready);
  EXPECT_TRUE(second_publish.get());
  EXPECT_EQ(state.consume().sample_output.next_tokens.item<int64_t>(), 2);
}

TEST(ScheduleOverlapStateTest, ShutdownWakesWaitingConsumerAndProducer) {
  ScheduleOverlapState state(/*publish_to_host=*/true,
                             /*retain_tokenless_output=*/false);
  auto consumer =
      std::async(std::launch::async, [&state]() { return state.consume(); });
  EXPECT_EQ(consumer.wait_for(50ms), std::future_status::timeout);
  state.shutdown();
  ASSERT_EQ(consumer.wait_for(1s), std::future_status::ready);
  EXPECT_FALSE(consumer.get().sample_output.next_tokens.defined());

  EXPECT_FALSE(state.record(token_output(3), {}, {}));
}

TEST(ScheduleOverlapStateTest, ShutdownUnblocksPublicationBackpressure) {
  ScheduleOverlapState state(/*publish_to_host=*/true,
                             /*retain_tokenless_output=*/false);
  ASSERT_TRUE(state.record(token_output(1), {}, {}));
  auto second_publish = std::async(std::launch::async, [&state]() {
    return state.record(token_output(2), {}, {});
  });
  EXPECT_EQ(second_publish.wait_for(50ms), std::future_status::timeout);
  state.shutdown();
  ASSERT_EQ(second_publish.wait_for(1s), std::future_status::ready);
  EXPECT_FALSE(second_publish.get());
}

TEST(ScheduleOverlapStateTest,
     ShutdownRejectsDeviceUpdatesWithoutReplacingPriorOutput) {
  std::weak_ptr<int32_t> weak_owner;
  {
    ScheduleOverlapState state(/*publish_to_host=*/false,
                               /*retain_tokenless_output=*/false);
    auto owner = std::make_shared<int32_t>(42);
    weak_owner = owner;
    ForwardOutput output = token_output(7);
    output.retained_inputs.emplace_back(owner);
    ASSERT_TRUE(
        state.record(std::move(output), {"prior_request"}, {"prior_sequence"}));
    owner.reset();
    state.shutdown();

    EXPECT_FALSE(
        state.record(token_output(9), {"new_request"}, {"new_sequence"}));

    EXPECT_TRUE(state.has_tokens());
    EXPECT_EQ(state.prior_output().sample_output.next_tokens.item<int64_t>(),
              7);
    EXPECT_EQ(state.request_ids(), (std::vector<std::string>{"prior_request"}));
    EXPECT_EQ(state.sample_sequence_ids(),
              (std::vector<std::string>{"prior_sequence"}));
    EXPECT_FALSE(weak_owner.expired());
  }
  EXPECT_TRUE(weak_owner.expired());
}

TEST(ScheduleOverlapStateTest, RetainedInputsOutliveConsumedCopy) {
  std::weak_ptr<int32_t> weak_owner;
  {
    ScheduleOverlapState state(/*publish_to_host=*/true,
                               /*retain_tokenless_output=*/false);
    auto owner = std::make_shared<int32_t>(42);
    weak_owner = owner;
    ForwardOutput output = token_output(9);
    output.retained_inputs.emplace_back(owner);
    ASSERT_TRUE(state.record(std::move(output), {}, {}));
    owner.reset();
    ForwardOutput consumed = state.consume();
    consumed.retained_inputs.clear();
    EXPECT_FALSE(weak_owner.expired());
  }
  EXPECT_TRUE(weak_owner.expired());
}

}  // namespace
}  // namespace xllm
