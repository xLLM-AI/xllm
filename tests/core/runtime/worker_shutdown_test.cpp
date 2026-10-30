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

#include <acl/acl.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <future>

#include "core/kv_cache/transfer/kv_cache_transfer.h"
#include "core/runtime/worker_impl.h"

namespace xllm {
namespace {

using namespace std::chrono_literals;

class ShutdownTestWorker final : public WorkerImpl {
 public:
  explicit ShutdownTestWorker(bool with_output = true)
      : WorkerImpl(ParallelArgs(/*rank=*/0,
                                /*world_size=*/1,
                                /*dp_size=*/1,
                                /*process_group=*/nullptr),
                   torch::Device(torch::kPrivateUse1, /*index=*/0),
                   options()),
        with_output_(with_output) {}

  bool init_model(ModelContext& /*context*/) override { return false; }

  void prepare_work_before_execute(const LlmForwardInput& input,
                                   LlmForwardInput& prepared) override {
    prepared = input.clone();
  }

  std::future<void> second_step_entered() {
    return second_step_entered_.get_future();
  }

  void block_compute(std::function<void()>& callback) {
    auto guard = compute_stream_->set_stream_guard();
    CHECK_EQ(
        aclrtLaunchHostFunc(
            compute_stream_->get_stream()->stream(),
            [](void* data) { (*static_cast<std::function<void()>*>(data))(); },
            &callback),
        ACL_SUCCESS);
  }

  // Keep a failing wakeup assertion from leaving the test's future blocked.
  void release_result_waiter() { begin_shutdown(); }

 protected:
  std::optional<ForwardOutput> step_for_schedule_overlap(
      const LlmForwardInput& /*input*/) override {
    if (++step_count_ == 2) {
      second_step_entered_.set_value();
    }
    if (!with_output_) {
      return std::nullopt;
    }
    ForwardOutput output;
    output.sample_output.next_tokens = torch::tensor({7}, torch::kInt64);
    return output;
  }

 private:
  static runtime::Options options() {
    runtime::Options options;
    options.enable_schedule_overlap(true);
    return options;
  }

  bool with_output_;
  int32_t step_count_ = 0;
  std::promise<void> second_step_entered_;
};

LlmForwardInput empty_input() {
  LlmForwardInput input;
  input.token_ids = torch::empty({0}, torch::kInt32);
  return input;
}

TEST(WorkerShutdownTest, WakesResultWaitersWithoutPublishingAnOutput) {
  ShutdownTestWorker worker;
  std::promise<void> waiter_started;
  auto started = waiter_started.get_future();
  auto result = std::async(std::launch::async, [&]() {
    waiter_started.set_value();
    return worker.get_last_step_result();
  });
  started.wait();
  EXPECT_EQ(result.wait_for(50ms), std::future_status::timeout);

  worker.begin_shutdown();
  const std::future_status status = result.wait_for(2s);
  EXPECT_EQ(status, std::future_status::ready);
  if (status != std::future_status::ready) {
    worker.release_result_waiter();
  }
  EXPECT_FALSE(result.get().sample_output.next_tokens.defined());
  worker.drain_pending_execution();
}

TEST(WorkerShutdownTest, DoesNotReturnAnAlreadyConsumedOutputDuringShutdown) {
  ShutdownTestWorker worker;
  auto step = worker.step_async(empty_input());
  EXPECT_TRUE(std::move(step).get().has_value());
  EXPECT_TRUE(
      worker.get_last_step_result().sample_output.next_tokens.defined());

  worker.begin_shutdown();
  EXPECT_FALSE(
      worker.get_last_step_result().sample_output.next_tokens.defined());
  worker.drain_pending_execution();
}

class OverlapWorkerShutdownTest : public ::testing::TestWithParam<bool> {};

TEST_P(OverlapWorkerShutdownTest, DrainsStepsBlockedByUnconsumedOutput) {
  ShutdownTestWorker worker(GetParam());
  auto entered = worker.second_step_entered();
  auto first = worker.step_async(empty_input());
  EXPECT_EQ(std::move(first).get().has_value(), GetParam());
  auto second = worker.step_async(empty_input());
  EXPECT_EQ(entered.wait_for(2s), std::future_status::ready);
  EXPECT_FALSE(second.isReady());
  auto drained = std::async(std::launch::async,
                            [&]() { worker.drain_pending_execution(); });
  const std::future_status status = drained.wait_for(2s);
  EXPECT_EQ(status, std::future_status::ready);
  if (status != std::future_status::ready) {
    // Retire the recorded result if shutdown fails to wake the publisher.
    worker.get_last_step_result();
  }
  drained.get();
  EXPECT_TRUE(second.isReady());
  EXPECT_EQ(std::move(second).get().has_value(), GetParam());
  worker.drain_pending_execution();
}

INSTANTIATE_TEST_SUITE_P(OutputPresence,
                         OverlapWorkerShutdownTest,
                         ::testing::Bool());

TEST(WorkerShutdownTest, WaitsForDeviceWorkBeforeReturningFromDrain) {
  std::promise<void> callback_entered;
  auto entered = callback_entered.get_future();
  std::promise<void> release_callback;
  auto release = release_callback.get_future().share();
  std::atomic<bool> finished{false};
  std::function<void()> callback = [&]() {
    callback_entered.set_value();
    release.wait_for(10s);
    finished.store(true);
  };
  ShutdownTestWorker worker;
  worker.block_compute(callback);
  EXPECT_EQ(entered.wait_for(2s), std::future_status::ready);

  std::promise<void> drain_started;
  auto started = drain_started.get_future();
  auto drained = std::async(std::launch::async, [&]() {
    drain_started.set_value();
    worker.drain_pending_execution();
  });
  started.wait();
  EXPECT_EQ(drained.wait_for(50ms), std::future_status::timeout);
  release_callback.set_value();
  EXPECT_EQ(drained.wait_for(2s), std::future_status::ready);
  drained.get();
  EXPECT_TRUE(finished.load());
}

class BlockingKVCacheTransfer final : public KVCacheTransfer {
 public:
  BlockingKVCacheTransfer() : release_future_(release_.get_future().share()) {}

  void get_cache_info(uint64_t& /*cluster_id*/,
                      std::string& /*addr*/) override {}
  bool link_clusters(const std::vector<uint64_t>& /*cluster_ids*/,
                     const std::vector<std::string>& /*remote_addrs*/,
                     const std::vector<uint16_t>& /*ports*/) override {
    return true;
  }
  bool unlink_cluster(const uint64_t& /*cluster_id*/,
                      const std::string& /*remote_addr*/,
                      const uint16_t /*port*/,
                      bool /*force_flag*/) override {
    return true;
  }
  bool pull_kv_blocks(
      const uint64_t /*src_cluster_id*/,
      const std::string& /*src_addr*/,
      const std::vector<KVTransferMapping>& /*mappings*/) override {
    entered_.set_value();
    release_future_.wait_for(10s);
    return true;
  }
  bool push_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& /*merged_kv_infos*/,
      std::shared_ptr<KVPushSynchronizerImpl>& /*layer_synchronizer*/,
      bool /*is_draft_model*/,
      int32_t /*kv_split_rank*/,
      int32_t /*kv_split_size*/) override {
    return true;
  }

  std::future<void> entered() { return entered_.get_future(); }
  void release() { release_.set_value(); }

 private:
  std::promise<void> entered_;
  std::promise<void> release_;
  std::shared_future<void> release_future_;
};

TEST(KVCacheTransferDrainTest, WaitsForQueuedLocalTransfers) {
  BlockingKVCacheTransfer transfer;
  auto entered = transfer.entered();
  auto pulled = transfer.pull_kv_blocks_async(
      /*src_cluster_id=*/1,
      /*src_addr=*/"local-test",
      {KVTransferMapping{0, {1}, {2}, 0}});
  EXPECT_EQ(entered.wait_for(2s), std::future_status::ready);

  std::promise<void> drain_started;
  auto started = drain_started.get_future();
  auto drained = std::async(std::launch::async, [&]() {
    drain_started.set_value();
    transfer.drain();
  });
  started.wait();
  EXPECT_EQ(drained.wait_for(50ms), std::future_status::timeout);
  transfer.release();
  EXPECT_EQ(drained.wait_for(2s), std::future_status::ready);
  drained.get();
  EXPECT_TRUE(pulled.isReady());
  EXPECT_TRUE(std::move(pulled).get());
  transfer.drain();
}

}  // namespace
}  // namespace xllm
