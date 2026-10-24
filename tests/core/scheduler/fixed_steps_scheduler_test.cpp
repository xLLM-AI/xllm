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

#include "core/scheduler/fixed_steps_scheduler.h"

#include <absl/time/time.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/distributed_runtime/rec_engine.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/request/rec_type.h"
#include "core/runtime/options.h"

namespace xllm {

namespace {

static_assert(std::is_base_of_v<Scheduler, FixedStepsScheduler>);
static_assert(std::is_constructible_v<FixedStepsScheduler,
                                      RecEngine*,
                                      const SchedulerOptions&>);
static_assert(!std::is_constructible_v<FixedStepsScheduler,
                                       Engine*,
                                       const SchedulerOptions&>);

class ControllablePrefetchBlockManagerPool final : public BlockManagerPool {
 public:
  ControllablePrefetchBlockManagerPool(const Options& options,
                                       bool enable_storage_prefetch,
                                       int32_t dp_size)
      : BlockManagerPool(options, dp_size),
        enable_storage_prefetch_(enable_storage_prefetch) {}

  bool has_storage_prefetch() const override {
    return enable_storage_prefetch_;
  }

  bool allocate(Sequence* sequence, size_t num_tokens) override {
    ++allocation_attempts_;
    if (fail_next_allocations_ > 0) {
      --fail_next_allocations_;
      return false;
    }
    return BlockManagerPool::allocate(sequence, num_tokens);
  }

  void fail_next_allocations(size_t count) { fail_next_allocations_ = count; }

  size_t allocation_attempts() const { return allocation_attempts_; }

  void prefetch_from_storage(std::shared_ptr<Request> request,
                             PrefetchDoneCallback done) override {
    ++prefetch_calls_;
    pending_.emplace_back(std::move(request), std::move(done));
    if (prefetch_started_callback_) {
      prefetch_started_callback_();
    }
  }

  void complete_prefetches() {
    auto pending = std::move(pending_);
    pending_.clear();
    for (auto& [request, done] : pending) {
      done(std::move(request));
    }
    if (prefetch_completion_callback_) {
      prefetch_completion_callback_();
    }
  }

  void drain_prefetch_completions() override {
    const size_t calls =
        drain_calls_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (drain_hook_) {
      drain_hook_(calls);
    }
  }

  size_t drain_calls() const {
    return drain_calls_.load(std::memory_order_relaxed);
  }

  void set_drain_hook(std::function<void(size_t)> hook) {
    drain_hook_ = std::move(hook);
  }

  size_t prefetch_calls() const { return prefetch_calls_; }

  void set_prefetch_completion_callback(
      PrefetchCompletionCallback callback) override {
    prefetch_completion_callback_ = std::move(callback);
  }

  void set_prefetch_started_callback(std::function<void()> callback) {
    prefetch_started_callback_ = std::move(callback);
  }

 private:
  const bool enable_storage_prefetch_;
  size_t fail_next_allocations_ = 0;
  size_t allocation_attempts_ = 0;
  size_t prefetch_calls_ = 0;
  std::vector<std::pair<std::shared_ptr<Request>, PrefetchDoneCallback>>
      pending_;
  PrefetchCompletionCallback prefetch_completion_callback_;
  std::function<void()> prefetch_started_callback_;
  std::atomic<size_t> drain_calls_{0};
  std::function<void(size_t)> drain_hook_;
};

class FakeTokenizer : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true) const override {
    (void)text;
    (void)ids;
    (void)add_special_tokens;
    return false;
  }
  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const override {
    (void)ids;
    (void)skip_special_tokens;
    return "";
  }
  std::optional<int32_t> token_to_id(
      const std::string_view& token) const override {
    (void)token;
    return std::nullopt;
  }
  std::string id_to_token(int32_t id) const override {
    (void)id;
    return "";
  }
  size_t vocab_size() const override { return 0; }
  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>();
  }
};

class FakeEngine final : public RecEngine {
 public:
  FakeEngine(int32_t num_blocks,
             int32_t block_size,
             bool enable_storage_prefetch = false,
             int32_t dp_size = 1)
      : RecEngine(make_options()) {
    BlockManagerPool::Options opt;
    opt.num_blocks_ = num_blocks;
    opt.block_size_ = block_size;
    opt.enable_prefix_cache_ = false;
    fake_tokenizer_ = std::make_unique<FakeTokenizer>();
    fake_block_manager_ =
        std::make_unique<ControllablePrefetchBlockManagerPool>(
            opt, enable_storage_prefetch, dp_size);
  }
  ForwardOutput step(RecBatchGroup& batch) override {
    if (step_hook_) {
      step_hook_();
    }
    last_step_had_request_.store(std::any_of(batch.begin(),
                                             batch.end(),
                                             [](const RecBatch& rec_batch) {
                                               return !rec_batch.empty();
                                             }),
                                 std::memory_order_relaxed);
    step_calls_.fetch_add(1, std::memory_order_relaxed);
    return ForwardOutput();
  }
  const Tokenizer* tokenizer() const override { return fake_tokenizer_.get(); }
  BlockManagerPool* block_manager_pool() const override {
    return fake_block_manager_.get();
  }
  const ModelArgs& model_args() const override {
    static ModelArgs args;
    return args;
  }
  const TokenizerArgs& tokenizer_args() const override {
    static TokenizerArgs args;
    return args;
  }
  std::vector<int64_t> get_active_activation_memory() const override {
    return {};
  }
  bool init() override { return true; }

  size_t step_calls() const {
    return step_calls_.load(std::memory_order_relaxed);
  }
  bool last_step_had_request() const {
    return last_step_had_request_.load(std::memory_order_relaxed);
  }
  void complete_prefetches() { fake_block_manager_->complete_prefetches(); }
  void fail_next_allocations(size_t count) {
    fake_block_manager_->fail_next_allocations(count);
  }
  size_t allocation_attempts() const {
    return fake_block_manager_->allocation_attempts();
  }
  size_t prefetch_calls() const {
    return fake_block_manager_->prefetch_calls();
  }
  void set_prefetch_started_callback(std::function<void()> callback) {
    fake_block_manager_->set_prefetch_started_callback(std::move(callback));
  }
  void set_step_hook(std::function<void()> step_hook) {
    step_hook_ = std::move(step_hook);
  }
  size_t drain_calls() const { return fake_block_manager_->drain_calls(); }
  void set_drain_hook(std::function<void(size_t)> hook) {
    fake_block_manager_->set_drain_hook(std::move(hook));
  }

 private:
  static runtime::Options make_options() {
    runtime::Options options;
    options.devices({torch::Device(torch::kCUDA, 0)});
    return options;
  }

  std::unique_ptr<Tokenizer> fake_tokenizer_;
  std::unique_ptr<ControllablePrefetchBlockManagerPool> fake_block_manager_;
  std::atomic<size_t> step_calls_{0};
  std::atomic<bool> last_step_had_request_{false};
  std::function<void()> step_hook_;
};

template <typename T>
class ScopedConfigValue final {
 public:
  ScopedConfigValue(T& value, T new_value) : value_(value), old_(value) {
    value_ = new_value;
  }

  ~ScopedConfigValue() { value_ = old_; }

 private:
  T& value_;
  T old_;
};

SchedulerOptions CreateOptions(int32_t max_tokens_per_batch = 10000,
                               int32_t max_seqs_per_batch = 256,
                               int32_t dp_size = 1,
                               bool enable_schedule_overlap = false,
                               int32_t rec_worker_max_concurrency = 1) {
  SchedulerOptions opt;
  opt.max_tokens_per_batch_ = max_tokens_per_batch;
  opt.max_seqs_per_batch_ = max_seqs_per_batch;
  opt.dp_size_ = dp_size;
  opt.enable_schedule_overlap_ = enable_schedule_overlap;
  opt.rec_worker_max_concurrency_ = rec_worker_max_concurrency;
  opt.max_tokens_per_chunk_for_prefill_ = 1024;
  opt.num_speculative_tokens_ = 0;
  return opt;
}

std::vector<std::shared_ptr<Request>> GenRequests(
    const std::vector<int32_t>& prompt_lens,
    const std::vector<int32_t>& max_tokens,
    RecType rec_type,
    int32_t max_context_len = 30000,
    size_t best_of = 1) {
  std::vector<std::shared_ptr<Request>> requests;
  EXPECT_EQ(prompt_lens.size(), max_tokens.size());
  for (size_t i = 0; i < prompt_lens.size(); ++i) {
    std::vector<int32_t> prompt_token_ids(prompt_lens[i], 0);
    RequestSamplingParam sampling_param;
    SchedulerParam scheduler_param;
    scheduler_param.offline = false;
    scheduler_param.priority = RequestPriority::NORMAL;
    StoppingChecker stopping_checker;
    stopping_checker.set_max_generated_tokens(max_tokens[i]);
    stopping_checker.set_max_context_len(max_context_len);
    stopping_checker.set_ignore_eos(true);
    RequestState req_state("x",
                           prompt_token_ids,
                           sampling_param,
                           scheduler_param,
                           stopping_checker,
                           static_cast<size_t>(prompt_lens[i]) + 30000,
                           1,
                           best_of,
                           false,
                           false,
                           false,
                           false,
                           false,
                           nullptr,
                           nullptr);
    req_state.rec_type = rec_type;
    auto request =
        std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
    requests.emplace_back(request);
  }
  return requests;
}

class TestableFixedStepsScheduler final : public FixedStepsScheduler {
 public:
  using FixedStepsScheduler::FixedStepsScheduler;

  RecBatchGroup prepare_batch_test() { return prepare_rec_batch(); }

  std::vector<std::shared_ptr<Request>> get_running_requests() {
    return running_requests_;
  }
};

}  // namespace

TEST(FixedStepsSchedulerTest, AddRequestSuccess) {
  auto engine = std::make_unique<FakeEngine>(32, 32);
  auto opt = CreateOptions();
  FixedStepsScheduler scheduler(engine.get(), opt);
  auto requests = GenRequests({64}, {10}, RecType::kOneRec);
  std::shared_ptr<Request> req = requests[0];
  EXPECT_TRUE(scheduler.add_request(req));
}

TEST(FixedStepsSchedulerTest, QueueCapacityRejectsBeforeScheduling) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeEngine>(64, 32);
  auto options = CreateOptions();
  options.request_queue_size(1);
  TestableFixedStepsScheduler scheduler(engine.get(), options);
  auto requests = GenRequests({32, 32}, {10, 10}, RecType::kOneRec);

  ASSERT_TRUE(scheduler.add_request(requests[0]));
  EXPECT_FALSE(scheduler.add_request(requests[1]));
  scheduler.prepare_batch_test();
  EXPECT_TRUE(scheduler.add_request(requests[1]));
}

TEST(FixedStepsSchedulerTest, PendingRequestCountTracksUpdates) {
  auto engine = std::make_unique<FakeEngine>(32, 32);
  FixedStepsScheduler scheduler(engine.get(), CreateOptions());

  EXPECT_EQ(scheduler.num_pending_requests(), 0u);
  scheduler.incr_pending_requests(2);
  EXPECT_EQ(scheduler.num_pending_requests(), 2u);
  scheduler.decr_pending_requests();
  EXPECT_EQ(scheduler.num_pending_requests(), 1u);
  scheduler.decr_pending_requests();
  EXPECT_EQ(scheduler.num_pending_requests(), 0u);
}

TEST(FixedStepsSchedulerTest, PrefetchReservesAdmissionUntilCompletion) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeEngine>(
      /*num_blocks=*/64, /*block_size=*/32, /*enable_storage_prefetch=*/true);
  auto options = CreateOptions();
  options.request_queue_size(1);
  TestableFixedStepsScheduler scheduler(engine.get(), options);
  auto requests = GenRequests({32, 32}, {10, 10}, RecType::kOneRec);

  ASSERT_TRUE(scheduler.add_request(requests[0]));
  EXPECT_TRUE(scheduler.has_pending_prefetch());
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 1u);
  EXPECT_EQ(engine->prefetch_calls(), 0u);
  EXPECT_FALSE(scheduler.add_request(requests[1]));

  RecBatchGroup batches = scheduler.prepare_batch_test();
  ASSERT_EQ(batches.size(), 1u);
  EXPECT_TRUE(batches.front().empty());
  EXPECT_EQ(engine->prefetch_calls(), 1u);
  EXPECT_FALSE(scheduler.add_request(requests[1]));

  engine->complete_prefetches();
  EXPECT_TRUE(scheduler.has_pending_prefetch());
  batches = scheduler.prepare_batch_test();
  ASSERT_EQ(batches.size(), 1u);
  EXPECT_EQ(batches.front().size(), 1u);
  EXPECT_FALSE(scheduler.has_pending_prefetch());
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
  EXPECT_TRUE(scheduler.add_request(requests[1]));
}

TEST(FixedStepsSchedulerTest, PrefetchCompletionWakesScheduler) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeEngine>(
      /*num_blocks=*/64, /*block_size=*/32, /*enable_storage_prefetch=*/true);
  auto scheduler =
      std::make_unique<FixedStepsScheduler>(engine.get(), CreateOptions());
  auto requests = GenRequests({32}, {10}, RecType::kOneRec);
  ASSERT_TRUE(scheduler->add_request(requests[0]));

  std::promise<void> prefetch_started;
  auto started = prefetch_started.get_future();
  engine->set_prefetch_started_callback(
      [&prefetch_started] { prefetch_started.set_value(); });
  std::promise<void> step_finished;
  auto finished = step_finished.get_future();
  std::thread scheduler_thread([&scheduler, &step_finished] {
    scheduler->step(absl::Seconds(1));
    step_finished.set_value();
  });

  ASSERT_EQ(started.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  engine->complete_prefetches();
  EXPECT_EQ(finished.wait_for(std::chrono::milliseconds(200)),
            std::future_status::ready);
  scheduler_thread.join();
  EXPECT_EQ(engine->step_calls(), 1u);
}

TEST(FixedStepsSchedulerTest, CancelledPrefetchReleasesReservedAdmission) {
  auto engine = std::make_unique<FakeEngine>(
      /*num_blocks=*/64, /*block_size=*/32, /*enable_storage_prefetch=*/true);
  auto options = CreateOptions();
  options.request_queue_size(1);
  TestableFixedStepsScheduler scheduler(engine.get(), options);
  auto requests = GenRequests({32, 32}, {10, 10}, RecType::kOneRec);

  ASSERT_TRUE(scheduler.add_request(requests[0]));
  requests[0]->set_cancel();
  RecBatchGroup batches = scheduler.prepare_batch_test();
  ASSERT_EQ(batches.size(), 1u);
  EXPECT_TRUE(batches.front().empty());
  EXPECT_EQ(engine->prefetch_calls(), 0u);
  EXPECT_FALSE(scheduler.has_pending_prefetch());
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
  EXPECT_TRUE(scheduler.add_request(requests[1]));
}

TEST(FixedStepsSchedulerTest, DestructionCancelsUnissuedPrefetch) {
  auto engine = std::make_unique<FakeEngine>(
      /*num_blocks=*/64, /*block_size=*/32, /*enable_storage_prefetch=*/true);
  auto requests = GenRequests({32}, {10}, RecType::kOneRec);
  {
    FixedStepsScheduler scheduler(engine.get(), CreateOptions());
    ASSERT_TRUE(scheduler.add_request(requests[0]));
    EXPECT_TRUE(scheduler.has_pending_prefetch());
  }

  EXPECT_TRUE(requests[0]->cancelled());
  EXPECT_EQ(engine->prefetch_calls(), 0u);
}

TEST(FixedStepsSchedulerTest, PrepareBatchEmptyWhenNoRequests) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  auto engine = std::make_unique<FakeEngine>(32, 32);
  auto opt = CreateOptions();
  TestableFixedStepsScheduler scheduler(engine.get(), opt);
  RecBatchGroup batches = scheduler.prepare_batch_test();
  EXPECT_FALSE(batches.empty());
  EXPECT_TRUE(batches[0].empty());
}

TEST(FixedStepsSchedulerTest, PrepareBatchOneRecSchedulesRequest) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeEngine>(64, 32);
  auto opt = CreateOptions(10000, 256);
  TestableFixedStepsScheduler scheduler(engine.get(), opt);
  auto requests = GenRequests({64, 64}, {10, 10}, RecType::kOneRec);
  for (auto& req : requests) {
    scheduler.add_request(req);
  }
  RecBatchGroup batches = scheduler.prepare_batch_test();
  EXPECT_FALSE(batches.empty());
  bool has_non_empty = false;
  for (const auto& b : batches) {
    if (!b.empty()) {
      has_non_empty = true;
      break;
    }
  }
  EXPECT_TRUE(has_non_empty);
  EXPECT_EQ(scheduler.get_running_requests().size(), 2u);
}

TEST(FixedStepsSchedulerTest, PrepareBatchRespectsTokenBudget) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeEngine>(64, 32);
  auto opt = CreateOptions(50, 1);
  TestableFixedStepsScheduler scheduler(engine.get(), opt);
  auto requests = GenRequests({40, 40}, {10, 10}, RecType::kOneRec);
  for (auto& req : requests) {
    scheduler.add_request(req);
  }
  scheduler.prepare_batch_test();
  EXPECT_LE(scheduler.get_running_requests().size(), 1u);
}

TEST(FixedStepsSchedulerTest,
     PreservesQueuedRequestsWhileMemoryThresholdBlocksPrefill) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      0.0);
  auto engine = std::make_unique<FakeEngine>(64, 32);
  TestableFixedStepsScheduler scheduler(engine.get(), CreateOptions());
  auto requests = GenRequests({32, 32}, {10, 10}, RecType::kOneRec);
  for (auto& request : requests) {
    ASSERT_TRUE(scheduler.add_request(request));
  }

  RecBatchGroup batches = scheduler.prepare_batch_test();
  ASSERT_EQ(batches.size(), 1u);
  EXPECT_TRUE(batches.front().empty());
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 2u);

  SchedulerConfig::get_instance().prefill_scheduling_memory_usage_threshold() =
      1.0;
  batches = scheduler.prepare_batch_test();
  ASSERT_EQ(batches.size(), 1u);
  EXPECT_EQ(batches.front().size(), 2u);
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
}

TEST(FixedStepsSchedulerTest, PreservesQueuedRequestsWhenKvAllocationFails) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeEngine>(
      /*num_blocks=*/4, /*block_size=*/32);
  engine->fail_next_allocations(1);
  TestableFixedStepsScheduler scheduler(engine.get(), CreateOptions());
  auto requests = GenRequests({32, 32}, {10, 10}, RecType::kLlmRec);
  for (auto& request : requests) {
    ASSERT_TRUE(scheduler.add_request(request));
  }

  RecBatchGroup batches = scheduler.prepare_batch_test();
  ASSERT_EQ(batches.size(), 1u);
  EXPECT_TRUE(batches.front().empty());
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 2u);
}

TEST(FixedStepsSchedulerTest, StepExecutesRecBatch) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeEngine>(64, 32);
  auto opt = CreateOptions(10000, 256);
  FixedStepsScheduler scheduler(engine.get(), opt);
  auto requests = GenRequests({32}, {10}, RecType::kOneRec);
  ASSERT_TRUE(scheduler.add_request(requests[0]));
  scheduler.step(absl::Milliseconds(500));
  EXPECT_EQ(engine->step_calls(), 1u);
  EXPECT_TRUE(engine->last_step_had_request());
}

TEST(FixedStepsSchedulerTest, IdleStepWaitsWithoutRepeatedScansUntilAdmission) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeEngine>(64, 32);
  std::promise<void> first_scan;
  auto scanned = first_scan.get_future();
  std::promise<void> repeated_scan;
  auto rescanned = repeated_scan.get_future();
  engine->set_drain_hook([&first_scan, &repeated_scan](size_t calls) {
    // Each prepare_rec_batch() drains completions twice.
    if (calls == 2) {
      first_scan.set_value();
    } else if (calls == 4) {
      repeated_scan.set_value();
    }
  });
  FixedStepsScheduler scheduler(engine.get(), CreateOptions());
  auto finished = std::async(
      std::launch::async, [&scheduler] { scheduler.step(absl::Seconds(2)); });

  EXPECT_EQ(scanned.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  EXPECT_EQ(rescanned.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  EXPECT_EQ(engine->drain_calls(), 2u);

  auto requests = GenRequests({32}, {10}, RecType::kOneRec);
  EXPECT_TRUE(scheduler.add_request(requests[0]));
  EXPECT_EQ(finished.wait_for(std::chrono::milliseconds(200)),
            std::future_status::ready);
  finished.get();
  engine->set_drain_hook({});
  EXPECT_EQ(engine->step_calls(), 1u);
  EXPECT_TRUE(engine->last_step_had_request());
}

TEST(FixedStepsSchedulerTest,
     QueuedRequestProgressesAfterOversizedHeadWithoutNewNotification) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeEngine>(64, 32);
  std::promise<Status> failure;
  auto rejected = failure.get_future();
  auto options = CreateOptions(/*max_tokens_per_batch=*/64,
                               /*max_seqs_per_batch=*/1);
  options.priority_strategy("priority");
  FixedStepsScheduler scheduler(engine.get(), options);
  auto requests = GenRequests({128, 32}, {10, 10}, RecType::kLlmRec);
  requests[0]->state().scheduler_param.priority = RequestPriority::HIGH;
  requests[0]->state().output_func = [&failure](const RequestOutput& output) {
    if (output.status.has_value()) {
      failure.set_value(output.status.value());
    }
    return true;
  };
  ASSERT_TRUE(scheduler.add_request(requests[0]));
  ASSERT_TRUE(scheduler.add_request(requests[1]));

  // Both admission notifications precede the first scan. After rejecting
  // the head, only the scheduler-local queue can trigger the next scan.
  auto finished = std::async(
      std::launch::async, [&scheduler] { scheduler.step(absl::Seconds(2)); });
  const std::future_status rejection_status =
      rejected.wait_for(std::chrono::seconds(1));
  EXPECT_EQ(rejection_status, std::future_status::ready);
  if (rejection_status == std::future_status::ready) {
    EXPECT_EQ(rejected.get().code(), StatusCode::RESOURCE_EXHAUSTED);
  }
  EXPECT_EQ(finished.wait_for(std::chrono::milliseconds(200)),
            std::future_status::ready);
  finished.get();
  EXPECT_EQ(engine->step_calls(), 1u);
  EXPECT_TRUE(engine->last_step_had_request());
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
}

TEST(FixedStepsSchedulerTest, RetriesRequestAfterTemporaryKvAllocationFailure) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeEngine>(64, 32);
  engine->fail_next_allocations(1);
  FixedStepsScheduler scheduler(engine.get(), CreateOptions());
  auto requests = GenRequests({32}, {10}, RecType::kLlmRec);
  ASSERT_TRUE(scheduler.add_request(requests[0]));

  scheduler.step(absl::Seconds(1));

  EXPECT_EQ(engine->allocation_attempts(), 2u);
  EXPECT_EQ(engine->step_calls(), 1u);
  EXPECT_TRUE(engine->last_step_had_request());
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
}

TEST(FixedStepsSchedulerTest,
     QueuedRequestProgressesAfterPermanentlyOversizedKvHead) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeEngine>(3, 32);
  std::promise<Status> failure;
  auto rejected = failure.get_future();
  auto options = CreateOptions(/*max_tokens_per_batch=*/10000,
                               /*max_seqs_per_batch=*/1);
  options.priority_strategy("priority");
  FixedStepsScheduler scheduler(engine.get(), options);
  // The first request needs three KV blocks; the following request fits in
  // the two-block pool. Both are admitted before the scheduler starts.
  auto requests = GenRequests({64, 32}, {10, 10}, RecType::kLlmRec);
  requests[0]->state().scheduler_param.priority = RequestPriority::HIGH;
  requests[0]->state().output_func = [&failure](const RequestOutput& output) {
    if (output.status.has_value()) {
      failure.set_value(output.status.value());
    }
    return true;
  };
  ASSERT_TRUE(scheduler.add_request(requests[0]));
  ASSERT_TRUE(scheduler.add_request(requests[1]));

  auto finished = std::async(
      std::launch::async, [&scheduler] { scheduler.step(absl::Seconds(2)); });
  const std::future_status rejection_status =
      rejected.wait_for(std::chrono::seconds(1));
  ASSERT_EQ(rejection_status, std::future_status::ready);
  EXPECT_EQ(rejected.get().code(), StatusCode::RESOURCE_EXHAUSTED);
  EXPECT_EQ(finished.wait_for(std::chrono::milliseconds(200)),
            std::future_status::ready);
  finished.get();
  EXPECT_EQ(engine->allocation_attempts(), 1u);
  EXPECT_EQ(engine->step_calls(), 1u);
  EXPECT_TRUE(engine->last_step_had_request());
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
}

TEST(FixedStepsSchedulerTest,
     RejectsAggregateKvReservationBeyondTotalCapacity) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  for (int32_t dp_size : {1, 2}) {
    SCOPED_TRACE(dp_size);
    auto engine = std::make_unique<FakeEngine>(
        4, 32, /*enable_storage_prefetch=*/false, dp_size);
    std::promise<Status> failure;
    auto rejected = failure.get_future();
    FixedStepsScheduler scheduler(engine.get(),
                                  CreateOptions(10000, 256, dp_size));
    const size_t best_of = static_cast<size_t>(dp_size) + 1;
    auto requests = GenRequests(
        {32}, {10}, RecType::kLlmRec, /*max_context_len=*/30000, best_of);
    requests[0]->state().output_func = [&failure](const RequestOutput& output) {
      if (output.status.has_value()) {
        failure.set_value(output.status.value());
      }
      return true;
    };
    ASSERT_TRUE(scheduler.add_request(requests[0]));

    // Each sequence needs two blocks. Padding leaves three usable blocks
    // per rank, so each rank can admit only one complete reservation.
    scheduler.step(absl::Milliseconds(10));

    ASSERT_EQ(rejected.wait_for(std::chrono::seconds(1)),
              std::future_status::ready);
    EXPECT_EQ(rejected.get().code(), StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_EQ(requests[0]->sequences().size(), best_of);
    EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
    EXPECT_EQ(engine->allocation_attempts(), 0u);
    EXPECT_EQ(engine->step_calls(), 0u);
  }
}

TEST(FixedStepsSchedulerTest, SchedulesAggregateKvReservationAtTotalCapacity) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeEngine>(5, 32);
  TestableFixedStepsScheduler scheduler(engine.get(), CreateOptions());
  auto requests = GenRequests(
      {32}, {10}, RecType::kLlmRec, /*max_context_len=*/30000, /*best_of=*/2);
  ASSERT_TRUE(scheduler.add_request(requests[0]));

  const RecBatchGroup batches = scheduler.prepare_batch_test();

  ASSERT_EQ(batches.size(), 1u);
  EXPECT_EQ(batches.front().size(), 2u);
  EXPECT_EQ(engine->allocation_attempts(), 2u);
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
}

TEST(FixedStepsSchedulerTest,
     SchedulesAggregateKvReservationAcrossRanksWithUnequalHeadroom) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeEngine>(
      4, 32, /*enable_storage_prefetch=*/false, /*dp_size=*/2);
  auto occupied = GenRequests({1}, {1}, RecType::kLlmRec);
  occupied[0]->sequences()[0]->set_dp_rank(1);
  ASSERT_TRUE(engine->block_manager_pool()->allocate(
      occupied[0]->sequences()[0].get(), 1));
  TestableFixedStepsScheduler scheduler(
      engine.get(), CreateOptions(10000, 256, /*dp_size=*/2));
  auto requests = GenRequests(
      {32}, {10}, RecType::kLlmRec, /*max_context_len=*/30000, /*best_of=*/2);
  ASSERT_TRUE(scheduler.add_request(requests[0]));

  // Sequential allocation can place two blocks on rank 0, then two on rank 1.
  // A capacity precheck must not pin both fresh sequences to the roomiest rank.
  const RecBatchGroup batches = scheduler.prepare_batch_test();

  ASSERT_EQ(batches.size(), 2u);
  EXPECT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[1].size(), 1u);
  ASSERT_EQ(requests[0]->sequences().size(), 2u);
  EXPECT_EQ(requests[0]->sequences()[0]->dp_rank(), 0);
  EXPECT_EQ(requests[0]->sequences()[1]->dp_rank(), 1);
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
  engine->block_manager_pool()->deallocate(occupied[0].get());
}

TEST(FixedStepsSchedulerTest,
     RetriesAggregateKvReservationAfterOtherRankReleasesCapacity) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeEngine>(
      4, 32, /*enable_storage_prefetch=*/false, /*dp_size=*/2);
  auto occupied = GenRequests({64}, {10}, RecType::kLlmRec);
  occupied[0]->sequences()[0]->set_dp_rank(1);
  ASSERT_TRUE(engine->block_manager_pool()->allocate(
      occupied[0]->sequences()[0].get(), 96));
  TestableFixedStepsScheduler scheduler(
      engine.get(), CreateOptions(10000, 256, /*dp_size=*/2));
  auto requests = GenRequests(
      {32}, {10}, RecType::kLlmRec, /*max_context_len=*/30000, /*best_of=*/2);
  ASSERT_TRUE(scheduler.add_request(requests[0]));

  RecBatchGroup batches = scheduler.prepare_batch_test();

  ASSERT_EQ(batches.size(), 2u);
  EXPECT_TRUE(batches[0].empty());
  EXPECT_TRUE(batches[1].empty());
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 1u);
  ASSERT_EQ(requests[0]->sequences().size(), 2u);
  EXPECT_EQ(requests[0]->sequences()[0]->dp_rank(), -1);
  EXPECT_EQ(requests[0]->sequences()[1]->dp_rank(), -1);

  engine->block_manager_pool()->deallocate(occupied[0].get());
  batches = scheduler.prepare_batch_test();

  ASSERT_EQ(batches.size(), 2u);
  EXPECT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[1].size(), 1u);
  EXPECT_NE(requests[0]->sequences()[0]->dp_rank(),
            requests[0]->sequences()[1]->dp_rank());
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
}

TEST(FixedStepsSchedulerTest, DestructionWaitsForAsynchronousRecExecution) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeEngine>(64, 32);
  auto options = CreateOptions();
  options.rec_worker_max_concurrency(2);
  auto requests = GenRequests({32}, {10}, RecType::kOneRec);
  std::promise<void> step_started;
  auto started = step_started.get_future();
  std::promise<void> release_step;
  auto release = release_step.get_future();
  engine->set_step_hook([&step_started, &release] {
    step_started.set_value();
    release.wait_for(std::chrono::seconds(2));
  });
  auto scheduler = std::make_unique<FixedStepsScheduler>(engine.get(), options);
  ASSERT_TRUE(scheduler->add_request(requests[0]));
  scheduler->step(absl::Milliseconds(500));
  ASSERT_EQ(started.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);

  std::promise<void> destruction_started;
  auto destroying = destruction_started.get_future();
  std::promise<void> destruction_finished;
  auto destroyed = destruction_finished.get_future();
  std::thread destroy(
      [&scheduler, &destruction_started, &destruction_finished] {
        destruction_started.set_value();
        scheduler.reset();
        destruction_finished.set_value();
      });
  EXPECT_EQ(destroying.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  EXPECT_EQ(destroyed.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  release_step.set_value();
  destroy.join();

  EXPECT_EQ(engine->step_calls(), 1u);
  EXPECT_TRUE(engine->last_step_had_request());
}

}  // namespace xllm
