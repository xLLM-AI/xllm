/* Copyright 2026 The xLLM Authors.

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

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "framework/request/request.h"
#include "framework/request/request_state.h"
#include "scheduler/request_priority_queue.h"
#include "scheduler/scheduler_policy.h"

namespace xllm {
namespace {

std::shared_ptr<Request> make_request(const std::string& request_id,
                                      size_t prompt_tokens) {
  std::vector<int32_t> prompt_token_ids(prompt_tokens, 1);
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  stopping_checker.set_max_context_len(4096);
  stopping_checker.set_ignore_eos(true);

  RequestState state("prompt",
                     prompt_token_ids,
                     sampling_param,
                     scheduler_param,
                     stopping_checker,
                     prompt_token_ids.size() + 8,
                     /*n=*/1,
                     /*best_of=*/1,
                     /*logprobs=*/false,
                     /*stream=*/false,
                     /*echo=*/false,
                     /*skip_special_tokens=*/false,
                     /*enable_schedule_overlap=*/false,
                     /*output_func=*/nullptr,
                     /*outputs_func=*/nullptr);
  return std::make_shared<Request>(
      request_id, "x-request-id", "x-request-time", state);
}

std::vector<std::string> collect_ids(const RequestPriorityQueue& queue) {
  std::vector<std::string> request_ids;
  for (auto it = queue.begin(); it != queue.end(); ++it) {
    request_ids.emplace_back((*it)->request_id());
  }
  return request_ids;
}

TEST(ShortRequestFirstPolicyTest, ClassifiesFreshRequestsByThreshold) {
  EXPECT_EQ(classify_short_request_first(*make_request("short", 64),
                                         /*threshold=*/256),
            ShortRequestFirstRequestClass::SHORT);
  EXPECT_EQ(classify_short_request_first(*make_request("long", 512),
                                         /*threshold=*/256),
            ShortRequestFirstRequestClass::LONG);
}

TEST(ShortRequestFirstPolicyTest, ClassifiesPreemptedAsImmediate) {
  std::shared_ptr<Request> request =
      make_request("preempted", /*prompt_tokens=*/512);
  request->set_preempted();
  EXPECT_EQ(classify_short_request_first(*request, /*threshold=*/256),
            ShortRequestFirstRequestClass::IMMEDIATE);
}

TEST(ShortRequestFirstPolicyTest, SortsImmediateBeforeShortBeforeLong) {
  DequeQueue queue;
  std::shared_ptr<Request> immediate =
      make_request("immediate", /*prompt_tokens=*/512);
  immediate->set_preempted();
  queue.push(immediate, /*if_back=*/true);
  queue.push(make_request("long", /*prompt_tokens=*/512),
             /*if_back=*/true);
  queue.push(make_request("short", /*prompt_tokens=*/64),
             /*if_back=*/true);

  sort_short_request_first_queue(queue,
                                 /*threshold=*/256,
                                 /*long_max_wait_ms=*/0.0);

  EXPECT_EQ(collect_ids(queue),
            std::vector<std::string>({"immediate", "short", "long"}));
}

TEST(ShortRequestFirstPolicyTest, SortsShortBeforeLongByDefault) {
  DequeQueue queue;
  queue.push(make_request("long", /*prompt_tokens=*/512),
             /*if_back=*/true);
  queue.push(make_request("short", /*prompt_tokens=*/64),
             /*if_back=*/true);

  sort_short_request_first_queue(queue,
                                 /*threshold=*/256,
                                 /*long_max_wait_ms=*/0.0);

  EXPECT_EQ(collect_ids(queue), std::vector<std::string>({"short", "long"}));
}

TEST(ShortRequestFirstPolicyTest, DoesNotPromoteWhenAgingOff) {
  DequeQueue queue;
  std::shared_ptr<Request> long_request =
      make_request("long", /*prompt_tokens=*/512);
  queue.push(long_request, /*if_back=*/true);
  queue.push(make_request("short", /*prompt_tokens=*/64),
             /*if_back=*/true);

  sort_short_request_first_queue(queue,
                                 /*threshold=*/256,
                                 /*long_max_wait_ms=*/0.0,
                                 long_request->created_time() + absl::Hours(1));

  EXPECT_EQ(collect_ids(queue), std::vector<std::string>({"short", "long"}));
}

TEST(ShortRequestFirstPolicyTest, PromotesAgedLongHeadAtWaitBoundary) {
  DequeQueue queue;
  std::shared_ptr<Request> long_request =
      make_request("long", /*prompt_tokens=*/512);
  queue.push(long_request, /*if_back=*/true);
  queue.push(make_request("short", /*prompt_tokens=*/64),
             /*if_back=*/true);

  sort_short_request_first_queue(
      queue,
      /*threshold=*/256,
      /*long_max_wait_ms=*/100.0,
      long_request->created_time() + absl::Milliseconds(99));

  EXPECT_EQ(collect_ids(queue), std::vector<std::string>({"short", "long"}));

  sort_short_request_first_queue(
      queue,
      /*threshold=*/256,
      /*long_max_wait_ms=*/100.0,
      long_request->created_time() + absl::Milliseconds(100));

  EXPECT_EQ(collect_ids(queue), std::vector<std::string>({"long", "short"}));
}

TEST(ShortRequestFirstPolicyTest, PromotesOnlyTheOldestLongHead) {
  DequeQueue queue;
  std::shared_ptr<Request> long_0 =
      make_request("long-0", /*prompt_tokens=*/512);
  queue.push(long_0, /*if_back=*/true);
  queue.push(make_request("long-1", /*prompt_tokens=*/768),
             /*if_back=*/true);
  queue.push(make_request("short", /*prompt_tokens=*/64),
             /*if_back=*/true);

  sort_short_request_first_queue(
      queue,
      /*threshold=*/256,
      /*long_max_wait_ms=*/100.0,
      long_0->created_time() + absl::Milliseconds(100));

  EXPECT_EQ(collect_ids(queue),
            std::vector<std::string>({"long-0", "short", "long-1"}));
}

}  // namespace
}  // namespace xllm
