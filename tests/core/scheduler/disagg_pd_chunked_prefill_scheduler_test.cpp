/* Copyright 2025-2026 The xLLM Authors.

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

#include "scheduler/disagg_pd_chunked_prefill_scheduler.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/metrics.h"
#include "core/framework/config/kv_cache_config.h"
#include "distributed_runtime/engine.h"
#include "framework/block/block_manager_pool.h"
#include "framework/model/model_args.h"
#include "framework/request/request.h"
#include "framework/request/request_state.h"
#include "framework/request/sequence.h"
#include "framework/tokenizer/tokenizer.h"
#include "scheduler/continuous_scheduler.h"

namespace xllm {
namespace {

BlockManagerPool::Options make_block_options(int32_t num_blocks,
                                             int32_t block_size) {
  BlockManagerPool::Options options;
  options.num_blocks(num_blocks)
      .block_size(block_size)
      .enable_prefix_cache(true)
      .enable_disagg_pd(true)
      .max_seqs_per_batch(1024);
  return options;
}

class EmptyMetricsBlockManagerPool final : public BlockManagerPool {
 public:
  explicit EmptyMetricsBlockManagerPool(const Options& options)
      : BlockManagerPool(options, /*dp_size=*/1) {}

  std::vector<size_t> num_blocks_in_prefix_cache() const override { return {}; }

  std::vector<size_t> num_free_blocks() const override { return {}; }

  std::vector<size_t> num_used_blocks() const override { return {}; }

  double kv_cache_utilization() const override { return 0.75; }
};

class FakeTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& /*text*/,
              std::vector<int32_t>* /*ids*/,
              bool /*add_special_tokens*/) const override {
    NOT_IMPLEMENTED();
  }

  std::string decode(const Slice<int32_t>& /*ids*/,
                     bool /*skip_special_tokens*/) const override {
    NOT_IMPLEMENTED();
  }

  std::optional<int32_t> token_to_id(
      const std::string_view& /*token*/) const override {
    NOT_IMPLEMENTED();
  }

  std::string id_to_token(int32_t /*id*/) const override { NOT_IMPLEMENTED(); }

  size_t vocab_size() const override { NOT_IMPLEMENTED(); }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>();
  }
};

class FakeEngine final : public Engine {
 public:
  FakeEngine(int32_t num_blocks,
             int32_t block_size,
             bool empty_metrics = false) {
    BlockManagerPool::Options options =
        make_block_options(num_blocks, block_size);
    tokenizer_ = std::make_unique<FakeTokenizer>();
    if (empty_metrics) {
      block_manager_ = std::make_unique<EmptyMetricsBlockManagerPool>(options);
    } else {
      block_manager_ =
          std::make_unique<BlockManagerPool>(options, /*dp_size=*/1);
    }
  }

  ForwardOutput step(std::vector<Batch>& /*batch*/) override {
    NOT_IMPLEMENTED();
  }

  void update_last_step_result(std::vector<Batch>& /*batch*/) override {
    NOT_IMPLEMENTED();
  }

  const Tokenizer* tokenizer() const override { return tokenizer_.get(); }

  BlockManagerPool* block_manager_pool() const override {
    return block_manager_.get();
  }

  const ModelArgs& model_args() const override { return model_args_; }

  const TokenizerArgs& tokenizer_args() const override { NOT_IMPLEMENTED(); }

  std::vector<int64_t> get_active_activation_memory() const override {
    NOT_IMPLEMENTED();
  }

  bool init() override { return true; }

 private:
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<BlockManagerPool> block_manager_;
  ModelArgs model_args_;
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

DisaggPDChunkedPrefillScheduler::Options make_options(
    int32_t max_tokens_per_batch,
    int32_t max_chunk,
    int32_t max_seqs = 1) {
  DisaggPDChunkedPrefillScheduler::Options options;
  options.enable_pd_ooc(true)
      .enable_disagg_pd(true)
      .enable_chunked_prefill(true)
      .enable_schedule_overlap(false)
      .instance_role(InstanceRole::PREFILL)
      .max_tokens_per_batch(max_tokens_per_batch)
      .max_seqs_per_batch(max_seqs)
      .max_tokens_per_chunk_for_prefill(max_chunk)
      .dp_size(1);
  return options;
}

std::shared_ptr<Request> make_request(
    const std::vector<int32_t>& prompt_token_ids) {
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
                     /*stream=*/false,
                     /*echo=*/false,
                     /*logprobs=*/false,
                     /*skip_special_tokens=*/false,
                     /*include_usage=*/false,
                     /*mm_data=*/nullptr,
                     /*service_request_id=*/nullptr);

  return std::make_shared<Request>(
      "req", "x-request-id", "x-request-time", state, "service-req");
}

size_t first_cache_size(const BlockManagerPool& block_manager) {
  const std::vector<size_t> cache_sizes =
      block_manager.num_blocks_in_prefix_cache();
  CHECK(!cache_sizes.empty());
  return cache_sizes[0];
}

void cache_prompt(BlockManagerPool* block_manager,
                  const std::vector<int32_t>& token_ids) {
  CHECK(block_manager != nullptr);
  std::shared_ptr<Request> request = make_request(token_ids);
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(block_manager->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  block_manager->cache(sequence);
  block_manager->deallocate(sequence);
}

void release_prefix_cache(BlockManagerPool* block_manager) {
  CHECK(block_manager != nullptr);
  const size_t num_data_blocks = block_manager->num_blocks() - 1;
  std::vector<int32_t> token_ids;
  token_ids.reserve(num_data_blocks * block_manager->block_size());
  for (size_t i = 0; i < num_data_blocks * block_manager->block_size(); ++i) {
    token_ids.push_back(static_cast<int32_t>(1000 + i));
  }

  std::shared_ptr<Request> request = make_request(token_ids);
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(block_manager->allocate(sequence));
  block_manager->deallocate(sequence);
  EXPECT_EQ(first_cache_size(*block_manager), 0u);
}

std::shared_ptr<Request> make_request_with_best_of(
    const std::vector<int32_t>& prompt_token_ids,
    size_t n,
    size_t best_of) {
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(8);
  stopping_checker.set_max_context_len(30000);
  stopping_checker.set_ignore_eos(true);

  RequestState req_state("x",
                         prompt_token_ids,
                         sampling_param,
                         scheduler_param,
                         stopping_checker,
                         prompt_token_ids.size() + 30000,
                         n,
                         best_of,
                         false,
                         false,
                         false,
                         false,
                         false,
                         nullptr,
                         nullptr);
  return std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
}

}  // namespace

TEST(DisaggPDChunkedPrefillSchedulerTest, PicksCurrentChunkBudget) {
  const PDChunkBudget budget = pick_pd_chunk_budget(32, 96, 40, 64);
  EXPECT_EQ(budget.next_tokens, 40);
  EXPECT_EQ(budget.max_tokens, 72);
}

TEST(DisaggPDChunkedPrefillSchedulerTest, LastPromptChunkStopsAtPromptEnd) {
  const PDChunkBudget budget = pick_pd_chunk_budget(80, 96, 40, 64);
  EXPECT_EQ(budget.next_tokens, 16);
  EXPECT_EQ(budget.max_tokens, 96);
}

TEST(DisaggPDChunkedPrefillSchedulerTest, EmptyBudgetRejectsSchedule) {
  const PDChunkBudget budget = pick_pd_chunk_budget(32, 96, 40, 0);
  EXPECT_EQ(budget.next_tokens, 0);
  EXPECT_EQ(budget.max_tokens, 32);
}

// A fresh prefill sequence that holds no blocks yet reserves its whole prompt
// footprint: ceil(num_prompt_tokens / block_size).
TEST(DisaggPDChunkedPrefillSchedulerTest, RemainingBlocksReservesFullPrompt) {
  EXPECT_EQ(pd_prefill_remaining_blocks(/*num_prompt_tokens=*/10,
                                        /*held_blocks=*/0,
                                        /*block_size=*/2),
            5u);
}

// Blocks already held (which include prefix-cache-shared blocks) are excluded
// from the reservation, so a request reusing a shared prefix reserves less.
TEST(DisaggPDChunkedPrefillSchedulerTest, RemainingBlocksExcludesHeldPrefix) {
  // Full prompt needs ceil(10/2)=5 blocks; 3 already held (e.g. shared prefix)
  // -> only 2 fresh blocks still required.
  EXPECT_EQ(pd_prefill_remaining_blocks(/*num_prompt_tokens=*/10,
                                        /*held_blocks=*/3,
                                        /*block_size=*/2),
            2u);
}

// Once the prompt is fully covered by held blocks nothing more is reserved,
// and a zero block_size (degenerate config) never reserves.
TEST(DisaggPDChunkedPrefillSchedulerTest, RemainingBlocksSaturatesAtZero) {
  EXPECT_EQ(pd_prefill_remaining_blocks(/*num_prompt_tokens=*/10,
                                        /*held_blocks=*/5,
                                        /*block_size=*/2),
            0u);
  EXPECT_EQ(pd_prefill_remaining_blocks(/*num_prompt_tokens=*/10,
                                        /*held_blocks=*/8,
                                        /*block_size=*/2),
            0u);
  EXPECT_EQ(pd_prefill_remaining_blocks(/*num_prompt_tokens=*/10,
                                        /*held_blocks=*/0,
                                        /*block_size=*/0),
            0u);
}

// A fresh request whose complete footprint still fits total capacity on top of
// the reserved set is admissible.
TEST(DisaggPDChunkedPrefillSchedulerTest, FootprintFitsWithinCapacity) {
  EXPECT_TRUE(pd_prefill_footprint_fits(/*reserved_blocks=*/3000,
                                        /*request_full_blocks=*/4000,
                                        /*total_blocks=*/8000));
}

// The check is inclusive: reserved + full exactly equal to total capacity still
// fits (the started set precisely saturates the cache).
TEST(DisaggPDChunkedPrefillSchedulerTest, FootprintFitsAtExactCapacity) {
  EXPECT_TRUE(pd_prefill_footprint_fits(/*reserved_blocks=*/4000,
                                        /*request_full_blocks=*/4000,
                                        /*total_blocks=*/8000));
}

// A footprint that would push the reserved set past total capacity does not
// fit. Reserving each started request's COMPLETE footprint against TOTAL (not
// its shrinking remainder against free) is what serializes near-capacity
// prompts and stops starts outrunning completions.
TEST(DisaggPDChunkedPrefillSchedulerTest, FootprintDoesNotFitBeyondCapacity) {
  EXPECT_FALSE(pd_prefill_footprint_fits(/*reserved_blocks=*/5000,
                                         /*request_full_blocks=*/4000,
                                         /*total_blocks=*/8000));
}

// The sole request in the system is admitted even when its own footprint
// exceeds the whole cache -- the caller bypasses the footprint check for it, so
// it makes progress (or fails cleanly via exceeds_block_capacity) instead of
// being deferred forever. This pins the one load-bearing use of that bypass.
TEST(DisaggPDChunkedPrefillSchedulerTest, AdmitsSoleRequestExceedingCapacity) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), true);
  // 4 blocks -> 3 usable, block_size 2. A 20-token prompt needs ceil(20/2)=10
  // blocks, far beyond capacity, so the footprint check alone would defer it.
  FakeEngine engine(/*num_blocks=*/4, /*block_size=*/2);
  BlockManagerPool* block_manager = engine.block_manager_pool();
  DisaggPDChunkedPrefillScheduler scheduler(&engine,
                                            make_options(
                                                /*max_tokens_per_batch=*/2,
                                                /*max_chunk=*/2));
  std::shared_ptr<Request> request = make_request(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  ASSERT_TRUE(scheduler.ContinuousScheduler::add_request(request));

  std::vector<Batch> batches = scheduler.prepare_batch_test();

  // Admitted (making progress on its first chunk), not left waiting forever.
  EXPECT_EQ(scheduler.get_running_requests().size(), 1u);
  EXPECT_EQ(scheduler.get_waiting_requests().size(), 0u);

  for (const auto& running : scheduler.get_running_requests()) {
    block_manager->deallocate(running.get());
  }
}

// Regression test: in disagg PD mode, expansion of best_of_n sequences must
// be deferred to the DECODE instance (where prefix cache lets seq[1..N-1]
// share seq[0]'s prompt KV). Expanding on the PREFILL instance would waste
// N x prefill compute. expand_sequences(false) is still the API used in
// non-PD ChunkedPrefillScheduler, so this test pins the contract: a fresh
// request created with best_of=4 starts with exactly one sequence, and the
// PD-PREFILL prepare_batch is expected NOT to call expand_sequences(false).
TEST(DisaggPDChunkedPrefillSchedulerTest,
     BestOfNRequestStartsWithSingleSequence) {
  auto request = make_request_with_best_of(
      {1, 2, 3, 4, 5, 6, 7, 8}, /*n=*/2, /*best_of=*/4);
  EXPECT_EQ(request->sequences().size(), 1u);
}

// Regression test: expand_sequences(true) should be a no-op while the first
// sequence has not finished prefill yet (kv_cache_tokens_num <
// num_prompt_tokens). This is the precondition that makes the two-phase
// flow on the decode instance correct: seq[1..best_of-1] are only created
// after seq[0]'s prompt KV is available for prefix-cache reuse.
TEST(DisaggPDChunkedPrefillSchedulerTest,
     ExpandSharePrefixWaitsForFirstSequencePrefill) {
  auto request = make_request_with_best_of(
      {1, 2, 3, 4, 5, 6, 7, 8}, /*n=*/2, /*best_of=*/4);
  EXPECT_FALSE(request->expand_sequences(/*share_prefix=*/true));
  EXPECT_EQ(request->sequences().size(), 1u);

  Sequence* seq0 = request->sequences()[0].get();
  seq0->kv_state().set_kv_cache_tokens_num(seq0->num_prompt_tokens());
  EXPECT_TRUE(request->expand_sequences(/*share_prefix=*/true));
  EXPECT_EQ(request->sequences().size(), 4u);
}

TEST(DisaggPDChunkedPrefillSchedulerTest, UpdatesBlockMetrics) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), true);
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  BlockManagerPool* block_manager = engine.block_manager_pool();
  DisaggPDChunkedPrefillScheduler scheduler(&engine,
                                            make_options(
                                                /*max_tokens_per_batch=*/4,
                                                /*max_chunk=*/4));
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  ASSERT_TRUE(scheduler.ContinuousScheduler::add_request(request));

  GAUGE_SET(num_free_blocks, 0);
  GAUGE_SET(num_used_blocks, 0);
  std::vector<Batch> batches = scheduler.prepare_batch_test();

  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches[0].size(), 1u);
  double num_free_blocks = GAUGE_VALUE(num_free_blocks);
  double num_used_blocks = GAUGE_VALUE(num_used_blocks);
  EXPECT_EQ(num_free_blocks, 5);
  EXPECT_EQ(num_used_blocks, 2);

  block_manager->deallocate(request.get());
}

TEST(DisaggPDChunkedPrefillSchedulerTest, UpdatesSchedulerMetrics) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), true);
  FakeEngine engine(/*num_blocks=*/16, /*block_size=*/2);
  BlockManagerPool* block_manager = engine.block_manager_pool();
  DisaggPDChunkedPrefillScheduler scheduler(&engine,
                                            make_options(
                                                /*max_tokens_per_batch=*/8,
                                                /*max_chunk=*/4));
  std::shared_ptr<Request> first_request = make_request({1, 2, 3, 4});
  std::shared_ptr<Request> second_request = make_request({5, 6, 7, 8});
  ASSERT_TRUE(scheduler.ContinuousScheduler::add_request(first_request));
  ASSERT_TRUE(scheduler.ContinuousScheduler::add_request(second_request));
  scheduler.incr_pending_requests(/*count=*/2);

  GAUGE_SET(num_pending_requests, 0);
  GAUGE_SET(num_running_requests, 0);
  GAUGE_SET(num_waiting_requests, 0);
  GAUGE_SET(num_running_sequences, 0);
  std::vector<Batch> batches = scheduler.prepare_batch_test();

  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches[0].size(), 1u);
  double num_pending_requests = GAUGE_VALUE(num_pending_requests);
  double num_running_requests = GAUGE_VALUE(num_running_requests);
  double num_waiting_requests = GAUGE_VALUE(num_waiting_requests);
  double num_running_sequences = GAUGE_VALUE(num_running_sequences);
  EXPECT_EQ(num_pending_requests, 2);
  EXPECT_EQ(num_running_requests, 1);
  EXPECT_EQ(num_waiting_requests, 1);
  EXPECT_EQ(num_running_sequences, 1);

  const std::vector<std::shared_ptr<Request>> running_requests =
      scheduler.get_running_requests();
  ASSERT_EQ(running_requests.size(), 1u);
  block_manager->deallocate(running_requests[0].get());
}

TEST(DisaggPDChunkedPrefillSchedulerTest, SkipsEmptyBlockMetrics) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), true);
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2, /*empty_metrics=*/true);
  BlockManagerPool* block_manager = engine.block_manager_pool();
  DisaggPDChunkedPrefillScheduler scheduler(&engine,
                                            make_options(
                                                /*max_tokens_per_batch=*/4,
                                                /*max_chunk=*/4));
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  ASSERT_TRUE(scheduler.ContinuousScheduler::add_request(request));

  GAUGE_SET(kv_cache_utilization_perc, 0);
  GAUGE_SET(num_blocks_in_prefix_cache, 11);
  GAUGE_SET(num_free_blocks, 12);
  GAUGE_SET(num_used_blocks, 13);
  std::vector<Batch> batches = scheduler.prepare_batch_test();

  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches[0].size(), 1u);
  double kv_cache_utilization_perc = GAUGE_VALUE(kv_cache_utilization_perc);
  double num_blocks_in_prefix_cache = GAUGE_VALUE(num_blocks_in_prefix_cache);
  double num_free_blocks = GAUGE_VALUE(num_free_blocks);
  double num_used_blocks = GAUGE_VALUE(num_used_blocks);
  EXPECT_EQ(kv_cache_utilization_perc, 0.75);
  EXPECT_EQ(num_blocks_in_prefix_cache, 11);
  EXPECT_EQ(num_free_blocks, 12);
  EXPECT_EQ(num_used_blocks, 13);

  block_manager->deallocate(request.get());
}

TEST(DisaggPDChunkedPrefillSchedulerTest,
     AppliesPrefixCacheBeforeBudgetCalculation) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), true);
  FakeEngine engine(/*num_blocks=*/16, /*block_size=*/2);
  BlockManagerPool* block_manager = engine.block_manager_pool();
  cache_prompt(block_manager, {1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_EQ(first_cache_size(*block_manager), 4u);

  DisaggPDChunkedPrefillScheduler scheduler(&engine,
                                            make_options(
                                                /*max_tokens_per_batch=*/4,
                                                /*max_chunk=*/4));
  std::shared_ptr<Request> request =
      make_request({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(scheduler.ContinuousScheduler::add_request(request));

  std::vector<Batch> batches = scheduler.prepare_batch_test();

  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches[0].size(), 1u);
  ASSERT_EQ(scheduler.get_running_sequences_budgets().size(), 1u);
  EXPECT_EQ(scheduler.get_running_sequences_budgets()[0], 2u);
  EXPECT_EQ(batches[0].get_allowed_max_tokens(), std::vector<uint32_t>({2}));
  EXPECT_EQ(sequence->kv_state().kv_cache_tokens_num(), 8u);
  EXPECT_EQ(sequence->kv_state().shared_blocks_num(BlockType::KV), 4u);
  EXPECT_EQ(sequence->kv_state().num_blocks(BlockType::KV), 5u);
  EXPECT_EQ(request->num_prefix_cache_tokens(), 8u);

  block_manager->deallocate(request.get());
  release_prefix_cache(block_manager);
}

// Completion-invariant admission gate: when the KV cache can hold at most one
// request's full prompt footprint, a second concurrent prefill request must be
// deferred even though a single next-chunk block is still free for it. Without
// the gate both are admitted, both grow toward their full prompt, and neither
// can obtain its final chunk -> hold-and-wait deadlock (the PD prefill hang).
TEST(DisaggPDChunkedPrefillSchedulerTest, DefersSecondWhenCacheHoldsOnlyOne) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), true);
  // 9 blocks -> 8 data blocks free (block 0 is padding), block_size 2.
  FakeEngine engine(/*num_blocks=*/9, /*block_size=*/2);
  BlockManagerPool* block_manager = engine.block_manager_pool();
  DisaggPDChunkedPrefillScheduler scheduler(&engine,
                                            make_options(
                                                /*max_tokens_per_batch=*/4,
                                                /*max_chunk=*/2,
                                                /*max_seqs=*/2));
  // Each prompt needs ceil(10/2)=5 full-footprint blocks; two of them (10) do
  // not fit in the 8 free blocks, but one does.
  std::shared_ptr<Request> first =
      make_request({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  std::shared_ptr<Request> second =
      make_request({11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  ASSERT_TRUE(scheduler.ContinuousScheduler::add_request(first));
  ASSERT_TRUE(scheduler.ContinuousScheduler::add_request(second));

  std::vector<Batch> batches = scheduler.prepare_batch_test();

  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(scheduler.get_running_requests().size(), 1u);
  EXPECT_EQ(scheduler.get_waiting_requests().size(), 1u);

  for (const auto& running : scheduler.get_running_requests()) {
    block_manager->deallocate(running.get());
  }
}

// The gate must not over-throttle: when the KV cache can hold both requests'
// full footprints, both are admitted in the same step.
TEST(DisaggPDChunkedPrefillSchedulerTest, AdmitsBothWhenCacheHoldsBoth) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), true);
  // 17 blocks -> 16 data blocks free, block_size 2.
  FakeEngine engine(/*num_blocks=*/17, /*block_size=*/2);
  BlockManagerPool* block_manager = engine.block_manager_pool();
  DisaggPDChunkedPrefillScheduler scheduler(&engine,
                                            make_options(
                                                /*max_tokens_per_batch=*/8,
                                                /*max_chunk=*/4,
                                                /*max_seqs=*/2));
  // Each prompt needs ceil(4/2)=2 full-footprint blocks; both (4) fit in 16.
  std::shared_ptr<Request> first = make_request({1, 2, 3, 4});
  std::shared_ptr<Request> second = make_request({5, 6, 7, 8});
  ASSERT_TRUE(scheduler.ContinuousScheduler::add_request(first));
  ASSERT_TRUE(scheduler.ContinuousScheduler::add_request(second));

  std::vector<Batch> batches = scheduler.prepare_batch_test();

  ASSERT_EQ(batches.size(), 1u);
  EXPECT_EQ(scheduler.get_running_requests().size(), 2u);
  EXPECT_EQ(scheduler.get_waiting_requests().size(), 0u);

  for (const auto& running : scheduler.get_running_requests()) {
    block_manager->deallocate(running.get());
  }
}

TEST(DisaggPDChunkedPrefillSchedulerTest, FullPrefixHitStillSchedulesStep) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), true);
  FakeEngine engine(/*num_blocks=*/16, /*block_size=*/2);
  BlockManagerPool* block_manager = engine.block_manager_pool();
  cache_prompt(block_manager, {1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_EQ(first_cache_size(*block_manager), 4u);

  DisaggPDChunkedPrefillScheduler scheduler(&engine,
                                            make_options(
                                                /*max_tokens_per_batch=*/4,
                                                /*max_chunk=*/4));
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4, 5, 6, 7, 8});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(scheduler.ContinuousScheduler::add_request(request));

  std::vector<Batch> batches = scheduler.prepare_batch_test();

  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches[0].size(), 1u);
  ASSERT_EQ(scheduler.get_running_sequences_budgets().size(), 1u);
  EXPECT_EQ(scheduler.get_running_sequences_budgets()[0], 2u);
  EXPECT_EQ(batches[0].get_allowed_max_tokens(), std::vector<uint32_t>({2}));
  EXPECT_EQ(sequence->kv_state().kv_cache_tokens_num(), 6u);
  EXPECT_EQ(sequence->kv_state().shared_blocks_num(BlockType::KV), 3u);
  EXPECT_EQ(sequence->kv_state().num_blocks(BlockType::KV), 4u);

  block_manager->deallocate(request.get());
  release_prefix_cache(block_manager);
}

}  // namespace xllm
