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

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "core/kv_cache/manager/block_manager_pool.h"
#include "core/scheduler/continuous_scheduler.h"
#include "core/scheduler/fake_scheduling_engine_test_utils.h"

namespace xllm {
namespace {

class FiniteGenerationEngine final : public FakeSchedulingEngine {
 public:
  explicit FiniteGenerationEngine(bool overlap) : overlap_(overlap) {
    BlockManagerPool::Options options;
    options.num_blocks(64)
        .block_size(4)
        .max_seqs_per_batch(1)
        .enable_prefix_cache(false);
    block_manager_ = std::make_unique<BlockManagerPool>(options);
  }

  ForwardOutput step(BatchGroup& batches) override {
    ++submitted_;
    if (overlap_ && submitted_ > completed_ + 2) {
      throw std::logic_error("Pending generation results were not collected");
    }
    // Emulate forward state updates without allocating accelerator-pinned
    // inputs.
    for (Batch& batch : batches) {
      for (Sequence* sequence : batch.get_sequences()) {
        sequence->kv_state().set_kv_cache_tokens_num(sequence->tokens().size());
        sequence->append_token(Token(overlap_ ? -1 : 7));
      }
    }
    return {};
  }

  std::vector<RawForwardOutput> collect_step_outputs(
      const BatchGroup& batches) override {
    ++completed_;
    for (const Batch& batch : batches) {
      for (Sequence* sequence : batch.get_sequences()) {
        if (sequence->cancelled() || sequence->finished()) {
          continue;
        }
        sequence->update_last_step_token(Token(/*id=*/7));
      }
    }
    return std::vector<RawForwardOutput>(batches.size());
  }

  size_t submitted() const { return submitted_; }
  size_t completed() const { return completed_; }

 private:
  const bool overlap_;
  size_t submitted_ = 0;
  size_t completed_ = 0;
};

SchedulerOptions generation_options(bool overlap) {
  SchedulerOptions options;
  options.max_tokens_per_batch(64)
      .max_seqs_per_batch(1)
      .max_tokens_per_chunk_for_prefill(64)
      .dp_size(1)
      .enable_schedule_overlap(overlap)
      .disable_log_stats(true);
  return options;
}

std::shared_ptr<Request> make_request(int32_t max_tokens, bool overlap) {
  StoppingChecker stopping;
  stopping.set_max_generated_tokens(max_tokens);
  stopping.set_max_context_len(64);
  stopping.set_ignore_eos(true);
  RequestState state(/*prompt=*/"x",
                     /*prompt_tokens=*/{1, 2},
                     RequestSamplingParam{},
                     SchedulerParam{},
                     std::move(stopping),
                     /*seq_capacity=*/64,
                     /*n=*/1,
                     /*best_of=*/1,
                     /*logprobs=*/false,
                     /*stream=*/false,
                     /*echo=*/false,
                     /*skip_special_tokens=*/false,
                     overlap,
                     /*output_func=*/nullptr,
                     /*outputs_func=*/nullptr);
  return std::make_shared<Request>(/*request_id=*/"1",
                                   /*x_request_id=*/"1",
                                   /*x_request_time=*/"1",
                                   std::move(state),
                                   /*service_request_id=*/"1");
}

class OfflineGenerationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    bool& prefix_cache = KVCacheConfig::get_instance().enable_prefix_cache();
    prefix_cache_enabled_ = std::exchange(prefix_cache, false);
  }

  void TearDown() override {
    KVCacheConfig::get_instance().enable_prefix_cache() = prefix_cache_enabled_;
  }

 private:
  bool prefix_cache_enabled_ = false;
};

TEST_F(OfflineGenerationTest, OverlapCommitsFinalTokensAndCanRunAgain) {
  SchedulerOptions options = generation_options(/*overlap=*/true);
  FiniteGenerationEngine engine(/*overlap=*/true);
  TestableContinuousScheduler scheduler(&engine, options);
  const auto initial_free_blocks =
      engine.kv_cache_manager()->num_free_blocks();

  scheduler.generate_with_schedule_overlap();
  EXPECT_EQ(engine.submitted(), 0u);
  EXPECT_EQ(engine.completed(), 0u);
  for (const int32_t max_tokens : {1, 3}) {
    SCOPED_TRACE(max_tokens);
    auto request = make_request(max_tokens, /*overlap=*/true);
    std::vector<RequestOutput> outputs;
    outputs.reserve(1);
    request->state().output_func = [&](const RequestOutput& output) {
      outputs.emplace_back(output);
      return true;
    };
    ASSERT_TRUE(scheduler.add_request(request));

    scheduler.generate_with_schedule_overlap();

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_TRUE(outputs.front().finished);
    ASSERT_EQ(outputs.front().outputs.size(), 1u);
    EXPECT_EQ(outputs.front().outputs.front().token_ids,
              std::vector<int32_t>(max_tokens, 7));
    EXPECT_EQ(engine.submitted(), engine.completed());
    EXPECT_EQ(engine.kv_cache_manager()->num_free_blocks(),
              initial_free_blocks);
  }
}

TEST_F(OfflineGenerationTest, OverlapDrainsPreviouslySubmittedWork) {
  SchedulerOptions options = generation_options(/*overlap=*/true);
  FiniteGenerationEngine engine(/*overlap=*/true);
  TestableContinuousScheduler scheduler(&engine, options);
  auto request = make_request(/*max_tokens=*/1, /*overlap=*/true);
  request->set_cancel();
  const std::weak_ptr<Request> retained_request = request;
  scheduler.seed_pending_batch(request);
  request.reset();

  scheduler.generate_with_schedule_overlap();

  EXPECT_EQ(engine.completed(), 1u);
  EXPECT_TRUE(retained_request.expired());
  scheduler.generate_with_schedule_overlap();
  EXPECT_EQ(engine.completed(), 1u);
}

TEST_F(OfflineGenerationTest, OverlapWaitsForAdmissionAndResponse) {
  SchedulerOptions options = generation_options(/*overlap=*/true);
  FiniteGenerationEngine engine(/*overlap=*/true);
  TestableContinuousScheduler scheduler(&engine, options);
  auto request = make_request(/*max_tokens=*/1, /*overlap=*/true);
  std::promise<void> callback_started;
  std::future<void> callback = callback_started.get_future();
  std::promise<void> release_callback;
  std::shared_future<void> released = release_callback.get_future().share();
  request->state().output_func = [&](const RequestOutput&) {
    callback_started.set_value();
    released.wait();
    return true;
  };
  scheduler.incr_pending_requests(/*count=*/1);
  auto generated = std::async(std::launch::async, [&]() {
    scheduler.generate_with_schedule_overlap();
  });
  EXPECT_TRUE(scheduler.wait_until_armed());
  EXPECT_NE(generated.wait_for(std::chrono::seconds(0)),
            std::future_status::ready);
  EXPECT_TRUE(scheduler.add_request(request));
  scheduler.decr_pending_requests();
  EXPECT_EQ(callback.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  EXPECT_NE(generated.wait_for(std::chrono::seconds(0)),
            std::future_status::ready);
  release_callback.set_value();
  generated.get();
  EXPECT_EQ(engine.submitted(), engine.completed());
}

TEST_F(OfflineGenerationTest, LegacyGenerationKeepsSynchronousOutput) {
  SchedulerOptions options = generation_options(/*overlap=*/false);
  FiniteGenerationEngine engine(/*overlap=*/false);
  TestableContinuousScheduler scheduler(&engine, options);
  auto request = make_request(/*max_tokens=*/3, /*overlap=*/false);
  std::vector<RequestOutput> outputs;
  outputs.reserve(1);
  request->state().output_func = [&](const RequestOutput& output) {
    outputs.emplace_back(output);
    return true;
  };
  ASSERT_TRUE(scheduler.add_request(request));

  scheduler.generate();

  ASSERT_EQ(outputs.size(), 1u);
  ASSERT_EQ(outputs.front().outputs.size(), 1u);
  EXPECT_EQ(outputs.front().outputs.front().token_ids,
            std::vector<int32_t>(3, 7));
  EXPECT_EQ(engine.submitted(), 3u);
  EXPECT_EQ(engine.completed(), 0u);
}

}  // namespace
}  // namespace xllm
