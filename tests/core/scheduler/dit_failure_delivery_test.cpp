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

#include <gtest/gtest.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/request/dit_request.h"
#include "core/scheduler/dit_scheduler.h"

namespace xllm {
namespace {

TEST(DiTFailureDeliveryTest,
     FailureCompletesEachRequestOnceAndNextSuccessIsClean) {
  std::mutex mutex;
  std::unordered_map<std::string, std::vector<DiTRequestOutput>> delivered;
  const DiTOutputFunc callback = [&](const DiTRequestOutput& output) {
    std::lock_guard<std::mutex> lock(mutex);
    delivered[output.request_id].emplace_back(output);
    return true;
  };
  DiTAsyncResponseProcessor processor(/*disable_log_stats=*/true);
  std::vector<std::shared_ptr<DiTRequest>> requests;
  requests.reserve(2);
  std::vector<std::weak_ptr<DiTRequest>> lifetimes;
  lifetimes.reserve(2);
  for (const std::string& id : {"failed-0", "failed-1"}) {
    DiTRequestState state({}, {}, callback, DiTRequestKind::kText);
    auto request = std::make_shared<DiTRequest>(
        id, "x-rid", "time", std::move(state), "service-" + id);
    request->handle_forward_text_output("must not be delivered as success");
    lifetimes.emplace_back(request);
    requests.emplace_back(std::move(request));
  }
  const Status status(StatusCode::INVALID_ARGUMENT, "rank=7: bad condition");
  fail_dit_requests(requests, processor, status);
  EXPECT_TRUE(requests.empty());
  processor.wait_completion();
  {
    std::lock_guard<std::mutex> lock(mutex);
    ASSERT_EQ(delivered.size(), 2u);
    for (const std::string& id : {"failed-0", "failed-1"}) {
      ASSERT_EQ(delivered.at(id).size(), 1u);
      const DiTRequestOutput& output = delivered.at(id).front();
      ASSERT_TRUE(output.status.has_value());
      EXPECT_EQ(output.status->code(), StatusCode::INVALID_ARGUMENT);
      EXPECT_EQ(output.status->message(), "rank=7: bad condition");
      EXPECT_EQ(output.service_request_id, "service-" + id);
      EXPECT_TRUE(output.finished);
      EXPECT_FALSE(output.cancelled);
      EXPECT_TRUE(output.outputs.empty());
    }
  }
  for (const auto& lifetime : lifetimes) {
    EXPECT_TRUE(lifetime.expired());
  }
  DiTRequestState next_state({}, {}, callback, DiTRequestKind::kText);
  auto next = std::make_shared<DiTRequest>(
      "next", "x-rid", "time", std::move(next_state));
  next->handle_forward_text_output("next success");
  processor.process_completed_request(std::move(next));
  processor.wait_completion();
  std::lock_guard<std::mutex> lock(mutex);
  ASSERT_EQ(delivered.at("next").size(), 1u);
  const DiTRequestOutput& output = delivered.at("next").front();
  EXPECT_TRUE(!output.status.has_value() || output.status->ok());
  EXPECT_TRUE(output.finished);
  ASSERT_EQ(output.outputs.size(), 1u);
  EXPECT_EQ(output.outputs.front().text, "next success");
  EXPECT_EQ(delivered.at("failed-0").size(), 1u);
  EXPECT_EQ(delivered.at("failed-1").size(), 1u);
}

}  // namespace
}  // namespace xllm
