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

#include "scheduler/scheduler_metrics.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <folly/ExceptionWrapper.h>
#include <folly/futures/Future.h>
#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "core/distributed_runtime/distributed_worker_manager.h"
#include "core/kv_cache/layout/static_kv_cache_layout.h"
#include "framework/block/block_manager_pool.h"
#include "framework/request/request.h"
#include "framework/request/request_state.h"

namespace xllm {

namespace {

class ActivationMemoryWorkerClient final : public WorkerClient {
 public:
  explicit ActivationMemoryWorkerClient(int64_t activation_memory,
                                        bool fail = false)
      : activation_memory_(activation_memory), fail_(fail) {}

  folly::SemiFuture<int64_t> get_active_activation_memory_async() override {
    if (fail_) {
      folly::Promise<int64_t> promise;
      auto future = promise.getSemiFuture();
      promise.setException(folly::make_exception_wrapper<std::runtime_error>(
          "worker activation memory query failed"));
      return future;
    }
    return folly::makeSemiFuture(activation_memory_);
  }

 private:
  int64_t activation_memory_;
  bool fail_;
};

}  // namespace

class SchedulerMetricsTestPeer final {
 public:
  static std::shared_ptr<DistributedWorkerManager> make_manager(
      const std::vector<int64_t>& activation_memories) {
    std::vector<std::shared_ptr<WorkerClient>> worker_clients;
    worker_clients.reserve(activation_memories.size());
    for (int64_t activation_memory : activation_memories) {
      worker_clients.emplace_back(
          std::make_shared<ActivationMemoryWorkerClient>(activation_memory));
    }
    return std::shared_ptr<DistributedWorkerManager>(
        new DistributedWorkerManager(std::move(worker_clients)));
  }

  static std::shared_ptr<DistributedWorkerManager> make_failing_manager() {
    std::vector<std::shared_ptr<WorkerClient>> worker_clients;
    worker_clients.emplace_back(
        std::make_shared<ActivationMemoryWorkerClient>(0, true));
    return std::shared_ptr<DistributedWorkerManager>(
        new DistributedWorkerManager(std::move(worker_clients)));
  }

  static std::vector<int64_t> active_activation_in_bytes(
      const SchedulerMetrics& metrics) {
    return metrics.get_active_activation_in_bytes();
  }

  static std::vector<int64_t> occupied_slots(
      const SchedulerMetrics& metrics,
      const std::vector<Sequence*>& sequences) {
    return metrics.get_num_occupied_slots(sequences);
  }
};

namespace {

std::unique_ptr<BlockManagerPool> make_block_manager_pool(int32_t dp_size = 1) {
  BlockManagerPool::Options options;
  options.num_blocks(8).block_size(2).enable_prefix_cache(false);
  return std::make_unique<BlockManagerPool>(options, dp_size);
}

std::shared_ptr<Request> make_request() {
  const std::vector<int32_t> prompt_token_ids{1, 2, 3, 4};
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  stopping_checker.set_max_context_len(64);
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
                     /*enable_schedule_overlap=*/true,
                     /*output_func=*/nullptr,
                     /*outputs_func=*/nullptr);
  return std::make_shared<Request>(
      "req", "x-request-id", "x-request-time", std::move(state));
}

}  // namespace

TEST(SchedulerMetricsTest, CollectsAndDrainsLatencySamples) {
  auto block_manager = make_block_manager_pool();
  SchedulerMetrics metrics(SchedulerMetricsTestPeer::make_manager({0}),
                           block_manager.get(),
                           /*dp_size=*/1,
                           /*num_speculative_tokens=*/0,
                           /*collect_recent_latency=*/true);
  std::shared_ptr<Request> request = make_request();
  Sequence* sequence = request->sequences().front().get();
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  sequence->append_token(Token(-1));
  const absl::Time start = absl::Now() - absl::Seconds(5);
  sequence->tbt_microseconds(start);
  std::vector<Sequence*> sequences = {sequence};
  std::vector<int64_t> ttft;
  std::vector<int64_t> tbt;

  sequence->update_last_step_token(Token(10), /*token_offset=*/0);
  metrics.update_token_latency_metrics(sequences);
  metrics.get_latency_metrics(ttft, tbt);
  ASSERT_EQ(ttft.size(), 1u);
  EXPECT_GE(ttft.front(), 1000);
  EXPECT_TRUE(tbt.empty());

  metrics.get_latency_metrics(ttft, tbt);
  EXPECT_TRUE(ttft.empty());
  EXPECT_TRUE(tbt.empty());

  sequence->append_token(Token(-1));
  sequence->update_last_step_token(Token(11), /*token_offset=*/0);
  metrics.update_token_latency_metrics(sequences);
  metrics.get_latency_metrics(ttft, tbt);
  EXPECT_TRUE(ttft.empty());
  ASSERT_EQ(tbt.size(), 1u);

  metrics.get_latency_metrics(ttft, tbt);
  EXPECT_TRUE(ttft.empty());
  EXPECT_TRUE(tbt.empty());
}

TEST(SchedulerMetricsTest, SkipsRecentSamplesWhenCollectionDisabled) {
  auto block_manager = make_block_manager_pool();
  SchedulerMetrics metrics(SchedulerMetricsTestPeer::make_manager({0}),
                           block_manager.get(),
                           /*dp_size=*/1,
                           /*num_speculative_tokens=*/0,
                           /*collect_recent_latency=*/false);
  std::shared_ptr<Request> request = make_request();
  Sequence* sequence = request->sequences().front().get();
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  sequence->append_token(Token(-1));
  sequence->tbt_microseconds(absl::Now() - absl::Seconds(5));
  sequence->update_last_step_token(Token(10), /*token_offset=*/0);
  std::vector<Sequence*> sequences = {sequence};

  metrics.update_token_latency_metrics(sequences);

  std::vector<int64_t> ttft;
  std::vector<int64_t> tbt;
  metrics.get_latency_metrics(ttft, tbt);
  EXPECT_TRUE(ttft.empty());
  EXPECT_TRUE(tbt.empty());
  EXPECT_GT(sequence->time_to_first_token_latency_seconds(), 0.0);
}

TEST(SchedulerMetricsTest, SamplesFirstWorkerOfEachDataParallelRank) {
  auto block_manager = make_block_manager_pool(/*dp_size=*/3);
  SchedulerMetrics metrics(
      SchedulerMetricsTestPeer::make_manager(
          {1025, 8192, 16384, 2049, 32768, 65536, 3073, 131072, 262144}),
      block_manager.get(),
      /*dp_size=*/3,
      /*num_speculative_tokens=*/0,
      /*collect_recent_latency=*/false);

  EXPECT_EQ(SchedulerMetricsTestPeer::active_activation_in_bytes(metrics),
            (std::vector<int64_t>{1025, 2049, 3073}));
}

TEST(SchedulerMetricsTest, SharedArenaCountsFullTailsAndKeepsLinearState) {
  BlockManagerPool::Options options;
  options.num_blocks(18)
      .block_size(2)
      .enable_prefix_cache(false)
      .enable_linear_state(true)
      .linear_state_num_slots(18)
      .enable_shared_kv_cache(true)
      .shared_linear_state_slots(2);
  BlockManagerPool block_manager(options);
  SchedulerMetrics metrics(nullptr, &block_manager, 1, 0, false);
  std::shared_ptr<Request> request = make_request();
  Sequence* sequence = request->sequences().front().get();
  ASSERT_TRUE(block_manager.allocate(sequence, /*num_tokens=*/4));
  ASSERT_EQ(block_manager.num_used_blocks()[0], 4u);

  sequence->kv_state().set_kv_cache_tokens_num(3);
  EXPECT_EQ(SchedulerMetricsTestPeer::occupied_slots(metrics, {sequence}),
            (std::vector<int64_t>{7}));
  sequence->kv_state().set_kv_cache_tokens_num(4);
  EXPECT_EQ(SchedulerMetricsTestPeer::occupied_slots(metrics, {sequence}),
            (std::vector<int64_t>{8}));
  block_manager.deallocate(sequence);
}

TEST(SchedulerMetricsTest, SharedSlabCountsPayloadBytesAndPartialFullTails) {
  StaticKVCacheLayoutSpec spec;
  spec.num_slots = 64;
  spec.num_slabs = 9;
  spec.full_row_bytes = 16;
  spec.conv_row_bytes = 8;
  spec.ssm_row_bytes = 32;
  spec.checkpoint_stride = 2;
  spec.full_layer_ids = {0, 3};
  spec.linear_layer_ids = {1, 2};
  BlockManagerPool::Options options;
  options.num_blocks(64)
      .block_size(2)
      .enable_prefix_cache(false)
      .enable_linear_state(true)
      .linear_state_num_slots(64)
      .enable_shared_kv_cache(true)
      .shared_layout(
          std::make_shared<const StaticKVCacheLayout>(std::move(spec)));
  BlockManagerPool block_manager(options);
  SchedulerMetrics metrics(nullptr, &block_manager, 1, 0, false);
  std::shared_ptr<Request> request = make_request();
  Sequence* sequence = request->sequences().front().get();
  ASSERT_TRUE(block_manager.allocate(sequence, /*num_tokens=*/4));
  ASSERT_EQ(block_manager.num_used_blocks()[0], 272u);
  ASSERT_EQ(block_manager.admission_capacity(0), 512u);
  ASSERT_EQ(block_manager.num_blocks(), 64u);

  sequence->kv_state().set_kv_cache_tokens_num(3);
  EXPECT_EQ(SchedulerMetricsTestPeer::occupied_slots(metrics, {sequence}),
            (std::vector<int64_t>{480}));
  sequence->kv_state().set_kv_cache_tokens_num(4);
  EXPECT_EQ(SchedulerMetricsTestPeer::occupied_slots(metrics, {sequence}),
            (std::vector<int64_t>{544}));
  bvar::LatencyRecorder* recorder =
      MULTI_HISTOGRAM_active_kv_cache_size_in_kilobytes.get_stats({"0"});
  ASSERT_NE(recorder, nullptr);
  const int64_t samples_before = recorder->count();
  std::vector<Sequence*> sequences = {sequence};
  metrics.update(sequences);
  EXPECT_EQ(recorder->count(), samples_before + 1);
  block_manager.deallocate(sequence);
}

TEST(SchedulerMetricsTest, PropagatesActivationMemoryQueryFailure) {
  auto block_manager = make_block_manager_pool();
  SchedulerMetrics metrics(SchedulerMetricsTestPeer::make_failing_manager(),
                           block_manager.get(),
                           /*dp_size=*/1,
                           /*num_speculative_tokens=*/0,
                           /*collect_recent_latency=*/false);
  std::shared_ptr<Request> request = make_request();
  std::vector<Sequence*> sequences = {request->sequences().front().get()};

  EXPECT_THROW(metrics.update(sequences), std::runtime_error);
}

TEST(SchedulerMetricsTest, RejectsIncompleteDataParallelSamples) {
  auto block_manager = make_block_manager_pool(/*dp_size=*/2);
  SchedulerMetrics metrics(SchedulerMetricsTestPeer::make_manager({1024}),
                           block_manager.get(),
                           /*dp_size=*/2,
                           /*num_speculative_tokens=*/0,
                           /*collect_recent_latency=*/false);

  EXPECT_DEATH(SchedulerMetricsTestPeer::active_activation_in_bytes(metrics),
               "samples must cover every DP rank");
}

TEST(SchedulerMetricsTest, RejectsUnequalWorkerCountsPerDataParallelRank) {
  auto block_manager = make_block_manager_pool(/*dp_size=*/2);
  SchedulerMetrics metrics(
      SchedulerMetricsTestPeer::make_manager({1024, 2048, 4096}),
      block_manager.get(),
      /*dp_size=*/2,
      /*num_speculative_tokens=*/0,
      /*collect_recent_latency=*/false);

  EXPECT_DEATH(SchedulerMetricsTestPeer::active_activation_in_bytes(metrics),
               "samples must have equal worker counts per DP rank");
}

TEST(SchedulerMetricsTest, AmortizedTokenLatencyRoundsHalfUp) {
  // Amortized per-token latency is round(latency / n) via (latency + n/2) / n.
  EXPECT_EQ(SchedulerMetrics::amortized_token_latency(100, 4), 25);
  EXPECT_EQ(SchedulerMetrics::amortized_token_latency(101, 4), 25);
  EXPECT_EQ(SchedulerMetrics::amortized_token_latency(102, 4), 26);
  EXPECT_EQ(SchedulerMetrics::amortized_token_latency(50, 5), 10);
  EXPECT_EQ(SchedulerMetrics::amortized_token_latency(53, 5), 11);
  // With a single committed token amortized latency equals the raw latency.
  EXPECT_EQ(SchedulerMetrics::amortized_token_latency(37, 1), 37);
}

}  // namespace xllm
