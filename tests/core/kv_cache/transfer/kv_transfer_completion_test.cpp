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

#include "core/kv_cache/transfer/kv_transfer_completion.h"

#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace xllm {
namespace {

using namespace std::chrono_literals;

TEST(KVTransferCompletionTest, WaitsForEveryTransfer) {
  folly::Promise<std::vector<KVTransferTaskResult>> first_promise;
  folly::Promise<std::vector<KVTransferTaskResult>> second_promise;
  KVTransferCompletion completion;
  completion.add(first_promise.getSemiFuture(), {"request-first"});
  completion.add(second_promise.getSemiFuture(), {"request-second"});

  std::promise<void> waiter_started;
  std::future<void> started = waiter_started.get_future();
  std::future<std::unordered_set<std::string>> result =
      std::async(std::launch::async, [&]() {
        waiter_started.set_value();
        return completion.wait();
      });

  started.wait();
  first_promise.setValue(std::vector<KVTransferTaskResult>{});
  EXPECT_EQ(result.wait_for(50ms), std::future_status::timeout);
  second_promise.setValue(std::vector<KVTransferTaskResult>{});
  EXPECT_TRUE(result.get().empty());
}

TEST(KVTransferCompletionTest, ReportsTransferFailure) {
  folly::Promise<std::vector<KVTransferTaskResult>> success_promise;
  folly::Promise<std::vector<KVTransferTaskResult>> failure_promise;
  KVTransferCompletion completion;
  completion.add(success_promise.getSemiFuture(), {"request-success"});
  completion.add(failure_promise.getSemiFuture(), {"request-failure"});
  success_promise.setValue(std::vector<KVTransferTaskResult>{});
  failure_promise.setValue(std::vector<KVTransferTaskResult>{
      KVTransferTaskResult{{"request-failure"}, KVTransferErrorCode::FAILED}});

  EXPECT_EQ(completion.wait(),
            (std::unordered_set<std::string>{"request-failure"}));
}

TEST(KVTransferCompletionTest, RejectsPendingTransferAfterTimeout) {
  EXPECT_DEATH(
      {
        folly::Promise<std::vector<KVTransferTaskResult>> promise;
        KVTransferCompletion completion(1ms);
        completion.add(promise.getSemiFuture(), {"request-pending"});
        completion.wait();
      },
      "KV cache push did not finish after timeout");
}

TEST(KVTransferCompletionTest, RejectsPendingTransferAtDestruction) {
  EXPECT_DEATH(
      {
        folly::Promise<std::vector<KVTransferTaskResult>> promise;
        KVTransferCompletion completion;
        completion.add(promise.getSemiFuture(), {"request-pending"});
      },
      "pending KV transfers");
}

TEST(KVTransferTrackerTest, WaitsForEveryTrackedTransfer) {
  KVTransferTracker tracker;
  std::shared_ptr<KVTransferTracker::Completion> first = tracker.track();
  std::shared_ptr<KVTransferTracker::Completion> second = tracker.track();

  std::promise<void> waiter_started;
  std::future<void> started = waiter_started.get_future();
  std::future<void> result = std::async(std::launch::async, [&]() {
    waiter_started.set_value();
    tracker.wait();
  });

  started.wait();
  first.reset();
  EXPECT_EQ(result.wait_for(50ms), std::future_status::timeout);
  second.reset();
  EXPECT_EQ(result.wait_for(1s), std::future_status::ready);
}

TEST(KVTransferTrackerTest, ReportsPendingUntilEveryTransferFinishes) {
  KVTransferTracker tracker;
  EXPECT_FALSE(tracker.has_pending());

  std::shared_ptr<KVTransferTracker::Completion> first = tracker.track();
  std::shared_ptr<KVTransferTracker::Completion> second = tracker.track();
  EXPECT_TRUE(tracker.has_pending());

  first.reset();
  EXPECT_TRUE(tracker.has_pending());

  second.reset();
  EXPECT_FALSE(tracker.has_pending());
}

TEST(KVTransferTrackerTest, DestructionWaitsForTrackedTransfer) {
  auto tracker = std::make_unique<KVTransferTracker>();
  std::shared_ptr<KVTransferTracker::Completion> completion = tracker->track();

  std::promise<void> destruction_started;
  std::future<void> started = destruction_started.get_future();
  std::future<void> result = std::async(
      std::launch::async,
      [tracker = std::move(tracker), &destruction_started]() mutable {
        destruction_started.set_value();
        tracker.reset();
      });

  started.wait();
  EXPECT_EQ(result.wait_for(50ms), std::future_status::timeout);
  completion.reset();
  EXPECT_EQ(result.wait_for(1s), std::future_status::ready);
}

}  // namespace
}  // namespace xllm
