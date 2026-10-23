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

#include "core/distributed_runtime/pipeline_dispatcher.h"

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace xllm {

TEST(PipelineDispatcherTest, StagesOverlapWithoutReorderingMicrobatches) {
  PipelineDispatcher dispatcher(/*stages=*/2);
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<std::vector<int32_t>> seen(2);
  bool first_stage_advanced = false;
  bool second_stage_started = false;
  dispatcher.run(/*microbatches=*/3, [&](int32_t stage, int32_t microbatch) {
    std::unique_lock<std::mutex> lock(mutex);
    seen[stage].emplace_back(microbatch);
    if (stage == 0 && microbatch == 1) {
      first_stage_advanced = true;
      cv.notify_all();
      EXPECT_TRUE(cv.wait_for(
          lock, std::chrono::seconds(5), [&] { return second_stage_started; }));
    }
    if (stage == 1 && microbatch == 0) {
      second_stage_started = true;
      cv.notify_all();
      EXPECT_TRUE(cv.wait_for(
          lock, std::chrono::seconds(5), [&] { return first_stage_advanced; }));
    }
  });
  EXPECT_EQ(seen[0], (std::vector<int32_t>{0, 1, 2}));
  EXPECT_EQ(seen[1], seen[0]);
  // A second invocation reuses the same stage threads after retirement.
  dispatcher.run(/*microbatches=*/1, [&](int32_t stage, int32_t microbatch) {
    EXPECT_EQ(microbatch, 0);
    seen[stage].emplace_back(3);
  });
  EXPECT_EQ(seen[0], (std::vector<int32_t>{0, 1, 2, 3}));
  EXPECT_EQ(seen[1], seen[0]);
}

}  // namespace xllm
