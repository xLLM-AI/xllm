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

#include "core/kv_cache/transfer/host_store_writer.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <future>
#include <stdexcept>
#include <vector>

namespace xllm {
namespace {

using namespace std::chrono_literals;
using Put = std::packaged_task<uint32_t()>;

TEST(HostStoreWriterTest, HoldsCompletionAndDrainsAcceptedWritesInOrder) {
  std::promise<void> started;
  std::promise<void> release;
  auto gate = release.get_future();
  std::vector<int32_t> completed;
  completed.reserve(2);
  std::future<uint32_t> first;
  std::future<uint32_t> second;
  {
    HostStoreWriter writer(/*max_blocks=*/3);
    first = writer.submit(/*blocks=*/2, Put([&] {
                            started.set_value();
                            gate.wait();
                            completed.push_back(1);
                            return 2;
                          }));
    started.get_future().wait();
    second = writer.submit(/*blocks=*/1, Put([&] {
                             completed.push_back(2);
                             return 1;
                           }));
    EXPECT_EQ(first.wait_for(0s), std::future_status::timeout);
    EXPECT_EQ(second.wait_for(0s), std::future_status::timeout);
    release.set_value();
  }
  EXPECT_EQ(first.wait_for(0s), std::future_status::ready);
  EXPECT_EQ(second.wait_for(0s), std::future_status::ready);
  EXPECT_EQ(first.get(), 2U);
  EXPECT_EQ(second.get(), 1U);
  EXPECT_EQ(completed, (std::vector<int32_t>{1, 2}));
}

TEST(HostStoreWriterTest, CountsBothRunningAndQueuedBlocksAgainstCapacity) {
  HostStoreWriter writer(/*max_blocks=*/4);
  std::promise<void> release;
  auto gate = release.get_future();
  auto running = writer.submit(/*blocks=*/2, Put([&] {
                                 gate.wait();
                                 return 2U;
                               }));
  auto queued = writer.submit(/*blocks=*/1, Put([] { return 1U; }));
  std::promise<void> submitting;
  auto next = std::async(std::launch::async, [&] {
    submitting.set_value();
    return writer.submit(/*blocks=*/2, Put([] { return 2U; }));
  });
  submitting.get_future().wait();
  EXPECT_EQ(next.wait_for(20ms), std::future_status::timeout);
  release.set_value();
  EXPECT_EQ(running.get(), 2U);
  EXPECT_EQ(queued.get(), 1U);
  EXPECT_EQ(next.get().get(), 2U);
}

TEST(HostStoreWriterTest, OversizedWriteHoldsCreditsAndReleasesThemOnFailure) {
  HostStoreWriter writer(/*max_blocks=*/2);
  std::promise<void> release;
  auto gate = release.get_future();
  auto oversized =
      writer.submit(/*blocks=*/3, Put([&]() -> uint32_t {
                      gate.wait();
                      throw std::runtime_error("backend PUT failed");
                    }));
  std::promise<void> submitting;
  auto next = std::async(std::launch::async, [&] {
    submitting.set_value();
    return writer.submit(/*blocks=*/1, Put([] { return 1U; }));
  });
  submitting.get_future().wait();
  EXPECT_EQ(next.wait_for(20ms), std::future_status::timeout);
  release.set_value();
  EXPECT_THROW(oversized.get(), std::runtime_error);
  EXPECT_EQ(next.get().get(), 1U);
}

}  // namespace
}  // namespace xllm
