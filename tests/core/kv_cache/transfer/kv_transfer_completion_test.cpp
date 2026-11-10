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
#include <stdexcept>
#include <string>
#include <vector>

#include "core/platform/device.h"

namespace xllm {
namespace {

using namespace std::chrono_literals;

TEST(KVTransferCompletionTest, WaitsForTransferAndReportsFailures) {
  folly::Promise<std::vector<std::string>> promise;
  KVTransferCompletion completion(promise.getSemiFuture(),
                                  {"request-success", "request-failure"});
  std::promise<void> waiter_started;
  std::future<void> started = waiter_started.get_future();
  auto result = std::async(std::launch::async, [&]() {
    waiter_started.set_value();
    return completion.wait({}, Device(torch::kCPU));
  });

  started.wait();
  EXPECT_EQ(result.wait_for(50ms), std::future_status::timeout);
  promise.setValue(std::vector<std::string>{"request-failure"});
  EXPECT_EQ(result.get(), (std::vector<std::string>{"request-failure"}));
}

TEST(KVTransferCompletionTest, MarksAllRequestsFailedIfTransferThrows) {
  folly::Promise<std::vector<std::string>> promise;
  KVTransferCompletion completion(promise.getSemiFuture(), {"first", "second"});
  promise.setException(std::runtime_error("push failed"));

  EXPECT_EQ(completion.wait({}, Device(torch::kCPU)),
            (std::vector<std::string>{"first", "second"}));
}

TEST(KVTransferCompletionTest, RejectsPendingTransferAtDestruction) {
  EXPECT_DEATH(
      {
        folly::Promise<std::vector<std::string>> promise;
        KVTransferCompletion completion(promise.getSemiFuture(),
                                        {"request-pending"});
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
