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

#include "scheduler/scheduler_policy.h"

#include <absl/time/clock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "continuous_scheduler.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/kv_cache_store_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/scheduler/fake_scheduling_engine_test_utils.h"
#include "core/scheduler/request_output_dispatcher.h"
#include "framework/block/block_manager_pool.h"
#include "framework/model/model_args.h"
#include "kv_cache/layout/static_kv_cache_layout.h"
#include "util/utils.h"

namespace xllm {

namespace {

class FakeEngine final : public FakeSchedulingEngine {
 public:
  FakeEngine(int32_t num_blocks,
             int32_t block_size,
             bool enable_prefix_cache = false,
             int32_t dp_size = 1,
             bool enable_linear_attention = false,
             int32_t speculative_tokens = 0) {
    BlockManagerPool::Options opt;
    opt.num_blocks_ = num_blocks;
    opt.block_size_ = block_size;
    opt.max_seqs_per_batch_ = 1024;
    opt.enable_prefix_cache_ = enable_prefix_cache;
    opt.enable_linear_state_ = enable_linear_attention;
    opt.num_speculative_tokens_ = speculative_tokens;
    opt.num_embedding_blocks_ =
        speculative_tokens > 0 ? opt.max_seqs_per_batch_ : 0;
    if (enable_linear_attention) {
      opt.linear_state_num_slots_ = 64;
      model_args_.layer_types({"linear_attention"});
    }
    block_manager_ = std::make_unique<BlockManagerPool>(opt, dp_size);
  }

  explicit FakeEngine(std::unique_ptr<BlockManagerPool> block_manager)
      : FakeSchedulingEngine(std::move(block_manager)) {}
};

class StaleCapacityBlockManagerPool final : public BlockManagerPool {
 public:
  explicit StaleCapacityBlockManagerPool(size_t advisory_capacity)
      : BlockManagerPool(make_options(), /*dp_size=*/1),
        advisory_capacity_(advisory_capacity) {}

  size_t admission_capacity(int32_t /*dp_rank*/) const override {
    return advisory_capacity_;
  }

  bool allocate(Sequence* sequence, size_t num_tokens) override {
    allocate_targets_.emplace_back(num_tokens);
    return BlockManagerPool::allocate(sequence, num_tokens);
  }

  const std::vector<size_t>& allocate_targets() const {
    return allocate_targets_;
  }

 private:
  static BlockManagerPool::Options make_options() {
    BlockManagerPool::Options options;
    options.num_blocks_ = 9;
    options.block_size_ = 4;
    options.max_seqs_per_batch_ = 16;
    options.enable_prefix_cache_ = false;
    return options;
  }

  const size_t advisory_capacity_;
  std::vector<size_t> allocate_targets_;
};

class FailOnAllocationBlockManagerPool final : public BlockManagerPool {
 public:
  explicit FailOnAllocationBlockManagerPool(size_t fail_on_attempt,
                                            bool enable_prefix_cache = false)
      : BlockManagerPool(make_options(enable_prefix_cache), /*dp_size=*/1),
        fail_on_attempt_(fail_on_attempt) {}

  bool allocate(Sequence* sequence, size_t num_tokens) override {
    ++allocation_attempts_;
    if (allocation_attempts_ == fail_on_attempt_) {
      return false;
    }
    return BlockManagerPool::allocate(sequence, num_tokens);
  }

  size_t allocation_attempts() const { return allocation_attempts_; }
  void set_fail_on_attempt(size_t attempt) { fail_on_attempt_ = attempt; }

 private:
  static BlockManagerPool::Options make_options(bool enable_prefix_cache) {
    BlockManagerPool::Options options;
    options.num_blocks_ = 32;
    options.block_size_ = 4;
    options.max_seqs_per_batch_ = 16;
    options.enable_prefix_cache_ = enable_prefix_cache;
    return options;
  }

  size_t fail_on_attempt_;
  size_t allocation_attempts_ = 0;
};

class TrackingBlockManagerPool final : public BlockManagerPool {
 public:
  TrackingBlockManagerPool()
      : BlockManagerPool(make_options(), /*dp_size=*/1) {}

  void allocate_shared(Sequence* sequence) override {
    ++allocate_shared_calls_;
    sequence->kv_state().set_prefix_cache_matched();
  }

  int32_t allocate_shared_calls() const { return allocate_shared_calls_; }

 private:
  static BlockManagerPool::Options make_options() {
    BlockManagerPool::Options options;
    options.num_blocks_ = 16;
    options.block_size_ = 128;
    options.max_seqs_per_batch_ = 16;
    options.enable_prefix_cache_ = true;
    return options;
  }

  int32_t allocate_shared_calls_ = 0;
};

class RetryRestoreBlockManagerPool final : public BlockManagerPool {
 public:
  RetryRestoreBlockManagerPool()
      : BlockManagerPool(make_options(), /*dp_size=*/1) {}

  bool allocate(Sequence* sequence, size_t num_tokens) override {
    allocate_targets_.emplace_back(num_tokens);
    if (allocate_targets_.size() == 1) {
      sequence->host_kv_state().reset();
      sequence->clear_host_cache_match();
      return false;
    }
    return true;
  }

  void allocate_shared(Sequence* sequence) override {
    ++allocate_shared_calls_;
    sequence->host_kv_state().set_prefix_cache_matched();
    if (sequence->kv_state().kv_cache_tokens_num() >= 32768) {
      sequence->clear_host_cache_match();
      return;
    }
    sequence->set_host_cache_match(/*restore_tokens=*/32768,
                                   /*copy_units=*/2);
  }

  const std::vector<size_t>& allocate_targets() const {
    return allocate_targets_;
  }

  bool needs_shared_reprobe(Sequence* sequence) const override {
    return !sequence->host_kv_state().prefix_cache_matched();
  }

  bool supports_host_cache_restore() const override { return true; }

  int32_t allocate_shared_calls() const { return allocate_shared_calls_; }

 private:
  static BlockManagerPool::Options make_options() {
    BlockManagerPool::Options options;
    options.num_blocks_ = 512;
    options.block_size_ = 128;
    options.max_seqs_per_batch_ = 16;
    options.enable_prefix_cache_ = true;
    return options;
  }

  std::vector<size_t> allocate_targets_;
  int32_t allocate_shared_calls_ = 0;
};

class PendingReleaseBlockManagerPool final : public BlockManagerPool {
 public:
  PendingReleaseBlockManagerPool()
      : BlockManagerPool(make_options(), /*dp_size=*/1) {}

  bool allocate(Sequence* /*sequence*/, size_t /*num_tokens*/) override {
    ++allocate_calls_;
    return false;
  }

  void allocate_shared(Sequence* sequence) override {
    sequence->kv_state().set_prefix_cache_matched();
  }

  bool has_pending_async_block_release() const override { return true; }

  int32_t allocate_calls() const { return allocate_calls_; }

 private:
  static BlockManagerPool::Options make_options() {
    BlockManagerPool::Options options;
    options.num_blocks_ = 256;
    options.block_size_ = 128;
    options.max_seqs_per_batch_ = 16;
    options.enable_prefix_cache_ = true;
    return options;
  }

  int32_t allocate_calls_ = 0;
};

class RestoreWaitingBlockManagerPool final : public BlockManagerPool {
 public:
  RestoreWaitingBlockManagerPool()
      : BlockManagerPool(make_options(), /*dp_size=*/1) {}

  bool allocate(Sequence* /*sequence*/, size_t /*num_tokens*/) override {
    ++allocate_calls_;
    return !pending_async_release_;
  }

  void allocate_shared(Sequence* sequence) override {
    ++allocate_shared_calls_;
    sequence->kv_state().set_prefix_cache_matched();
  }

  bool has_pending_async_block_release() const override {
    return pending_async_release_;
  }

  void set_pending_async_release(bool pending) {
    pending_async_release_ = pending;
  }

  int32_t allocate_calls() const { return allocate_calls_; }

  int32_t allocate_shared_calls() const { return allocate_shared_calls_; }

 private:
  static BlockManagerPool::Options make_options() {
    BlockManagerPool::Options options;
    options.num_blocks_ = 256;
    options.block_size_ = 128;
    options.max_seqs_per_batch_ = 16;
    options.enable_prefix_cache_ = true;
    return options;
  }

  bool pending_async_release_ = true;
  int32_t allocate_calls_ = 0;
  int32_t allocate_shared_calls_ = 0;
};

class DecodeVictimBlockManagerPool final : public BlockManagerPool {
 public:
  DecodeVictimBlockManagerPool()
      : BlockManagerPool(make_options(), /*dp_size=*/1) {}

  bool allocate(Sequence* /*sequence*/, size_t /*num_tokens*/) override {
    return false;
  }

  bool supports_host_cache_restore() const override { return true; }

  void deallocate(Request* request) override {
    BlockManagerPool::deallocate(request);
    pending_async_release_ = true;
  }

  bool has_pending_async_block_release() const override {
    return pending_async_release_;
  }

  void complete_async_release() { pending_async_release_ = false; }

 private:
  static BlockManagerPool::Options make_options() {
    BlockManagerPool::Options options;
    options.num_blocks_ = 256;
    options.block_size_ = 128;
    options.max_seqs_per_batch_ = 16;
    options.enable_prefix_cache_ = true;
    return options;
  }

  bool pending_async_release_ = false;
};

class RestorePriorityBlockManagerPool final : public BlockManagerPool {
 public:
  RestorePriorityBlockManagerPool()
      : BlockManagerPool(make_options(), /*dp_size=*/1) {}

  void set_sequences(Sequence* blocked, Sequence* victim) {
    blocked_ = blocked;
    victim_ = victim;
  }

  bool allocate(Sequence* sequence, size_t /*num_tokens*/) override {
    if (sequence == blocked_) {
      if (free_blocks_ < kBlockedGrowthBlocks) {
        return false;
      }
      free_blocks_ -= kBlockedGrowthBlocks;
      return true;
    }
    if (sequence == victim_) {
      if (free_blocks_ < kVictimRestoreBlocks) {
        return false;
      }
      free_blocks_ -= kVictimRestoreBlocks;
      return true;
    }
    return false;
  }

  void allocate_shared(Sequence* sequence) override {
    if (sequence != victim_) {
      return;
    }
    CHECK_GT(sequence->num_tokens(), 0u);
    sequence->kv_state().set_prefix_cache_matched();
    sequence->kv_state().set_kv_cache_tokens_num(sequence->num_tokens() - 1);
  }

  bool supports_host_cache_restore() const override { return true; }

  void deallocate(Request* request) override {
    BlockManagerPool::deallocate(request);
    pending_async_release_ = true;
  }

  bool has_pending_async_block_release() const override {
    return pending_async_release_;
  }

  void complete_release() {
    pending_async_release_ = false;
    free_blocks_ = kVictimRestoreBlocks;
  }

 private:
  static BlockManagerPool::Options make_options() {
    BlockManagerPool::Options options;
    options.num_blocks_ = 10;
    options.block_size_ = 128;
    options.max_seqs_per_batch_ = 16;
    options.enable_prefix_cache_ = true;
    return options;
  }

  static constexpr size_t kBlockedGrowthBlocks = 1;
  static constexpr size_t kVictimRestoreBlocks = 3;
  Sequence* blocked_ = nullptr;
  Sequence* victim_ = nullptr;
  size_t free_blocks_ = 0;
  bool pending_async_release_ = false;
};

class TestUnifiedPolicy final : public UnifiedPolicy {
 public:
  using SchedulerPolicy::allocate_for_prefill;
  using UnifiedPolicy::UnifiedPolicy;

  void allocate_shared_blocks_for_test(Sequence* sequence,
                                       SchedulerState& state) {
    allocate_shared_blocks_for(sequence, state);
  }
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

SchedulerOptions create_scheduler_options(
    int32_t max_tokens_per_batch,
    int32_t max_seqs_per_batch,
    int32_t num_speculative_tokens,
    int32_t max_tokens_per_chunk_for_prefill,
    int32_t dp_size,
    const std::string& priority_strategy = "fcfs",
    bool enable_profile_kv_blocks = true,
    bool enable_latency_aware_schedule = false,
    int32_t max_global_ttft_ms = std::numeric_limits<int32_t>::max(),
    int32_t max_global_tpot_ms = std::numeric_limits<int32_t>::max()) {
  SchedulerOptions opt;
  opt.num_speculative_tokens_ = num_speculative_tokens;
  opt.max_tokens_per_chunk_for_prefill_ = max_tokens_per_chunk_for_prefill;
  opt.max_tokens_per_batch_ = max_tokens_per_batch;
  opt.max_seqs_per_batch_ = max_seqs_per_batch;
  opt.dp_size_ = dp_size;
  opt.priority_strategy_ = priority_strategy;
  opt.enable_profile_kv_blocks_ = enable_profile_kv_blocks;
  opt.enable_latency_aware_schedule_ = enable_latency_aware_schedule;
  opt.max_global_ttft_ms_ = max_global_ttft_ms;
  opt.max_global_tpot_ms_ = max_global_tpot_ms;
  return opt;
}

std::vector<std::shared_ptr<Request>> generate_request(
    const std::vector<int32_t>& prompt_lens,
    const std::vector<int32_t>& max_tokens,
    std::optional<std::vector<bool>> offlines,
    std::optional<std::vector<int32_t>> priorities,
    int32_t max_context_len) {
  std::vector<std::shared_ptr<Request>> requests;
  EXPECT_TRUE(prompt_lens.size() == max_tokens.size());

  size_t batch_size = prompt_lens.size();
  std::vector<bool> offline_vec;
  std::vector<int32_t> priority_vec;
  if (offlines.has_value()) {
    offline_vec = *offlines;
  } else {
    offline_vec = std::vector<bool>(batch_size, false);
  }

  if (priorities.has_value()) {
    priority_vec = priorities.value();
  } else {
    priority_vec = std::vector<int32_t>(
        batch_size, static_cast<int32_t>(RequestPriority::NORMAL));
  }

  for (size_t i = 0; i < batch_size; ++i) {
    std::vector<int32_t> prompt_token_ids;
    prompt_token_ids.resize(prompt_lens[i]);
    RequestSamplingParam sampling_param;
    SchedulerParam scheduler_param;
    scheduler_param.offline = offline_vec[i];
    scheduler_param.priority = static_cast<RequestPriority>(priority_vec[i]);

    StoppingChecker stopping_checker;
    stopping_checker.set_max_generated_tokens(max_tokens[i]);
    stopping_checker.set_max_context_len(max_context_len);
    stopping_checker.set_ignore_eos(true);

    RequestState req_state(
        "x",
        prompt_token_ids,
        sampling_param,
        scheduler_param,
        stopping_checker,
        prompt_lens[i] + 30000,
        1,
        1,
        false,
        false,
        false,
        false,
        false,
        [](const RequestOutput&) { return true; },
        nullptr);

    auto request =
        std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
    requests.emplace_back(request);
  }

  return requests;
}

// dont not consider speculative decoding.
void update_requests(std::vector<std::shared_ptr<Request>> requests) {
  for (auto req : requests) {
    for (auto& seq : req->sequences()) {
      if (seq->kv_state().kv_cache_tokens_num() == 0) {
        seq->kv_state().incr_kv_cache_tokens_num(seq->num_prompt_tokens());
      } else {
        seq->kv_state().incr_kv_cache_tokens_num(1);
      }
      Token token(1);
      seq->append_token(token);
    }
  }
}

}  // namespace

// TEST-1:
// Three independent prefill requests, according to the configs,
// verify how many tokens are processed in one scheduling.
TEST(SchedulerPolicyTest, AddNewRequestBase) {
  std::vector<int32_t> prompt_len{10, 1024, 2048, 2048};
  std::vector<int32_t> num_blocks{16, 128, 129, 128};
  std::vector<int32_t> block_size{16, 16, 16, 256};
  std::vector<int32_t> chunk_size{1024, 1024, 1024, 768};
  std::vector<int32_t> kv_split_size{1, 1, 1, 2};
  std::vector<int32_t> validate_allowed_max_tokens{10, 1024, 2048, 768};
  for (size_t idx = 0; idx < prompt_len.size(); ++idx) {
    SchedulerOptions opt =
        create_scheduler_options(10000, 256, 0, chunk_size[idx], 1);
    ScopedConfigValue<int32_t> kv_split_guard(
        ParallelConfig::get_instance().kv_split_size(), kv_split_size[idx]);
    auto engine =
        std::make_unique<FakeEngine>(num_blocks[idx], block_size[idx]);
    auto scheduler =
        std::make_unique<TestableContinuousScheduler>(engine.get(), opt);

    // create requests
    auto requests = generate_request(
        {prompt_len[idx]}, {10}, std::nullopt, std::nullopt, 10000);
    for (auto req : requests) {
      scheduler->add_request(req);
    }

    auto batch = scheduler->prepare_batch();
    EXPECT_EQ(batch.size(), 1u);
    const std::vector<uint32_t>& allowed_max_tokens =
        batch[0].get_allowed_max_tokens();
    for (size_t i = 0; i < batch[0].size(); ++i) {
      auto seq = batch[0][i];
      EXPECT_EQ(allowed_max_tokens[i], validate_allowed_max_tokens[idx]);
    }
  }
}

TEST(SchedulerPolicyTest, NonChunkedPrefillAssignsDpBeforeFairBudget) {
  ScopedConfigValue<bool> fair_budget_guard(
      SchedulerConfig::get_instance().enable_dp_fair_token_budget(), true);
  ScopedConfigValue<bool> chunked_prefill_guard(
      SchedulerConfig::get_instance().enable_chunked_prefill(), false);
  ScopedConfigValue<bool> mix_batch_guard(
      SchedulerConfig::get_instance().enable_mix_batch(), false);
  constexpr int32_t kDpSize = 4;
  SchedulerOptions options = create_scheduler_options(
      /*max_tokens_per_batch=*/64,
      /*max_seqs_per_batch=*/4,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/64,
      /*dp_size=*/kDpSize);
  options.enable_chunked_prefill() = false;
  options.enable_disagg_pd() = true;
  options.instance_role() = InstanceRole::PREFILL;
  auto engine = std::make_unique<FakeEngine>(
      /*num_blocks=*/32,
      /*block_size=*/16,
      /*enable_prefix_cache=*/false,
      /*dp_size=*/kDpSize);
  auto scheduler =
      std::make_unique<TestableContinuousScheduler>(engine.get(), options);
  auto requests = generate_request(/*prompt_lens=*/{16, 16},
                                   /*max_tokens=*/{1, 1},
                                   /*offlines=*/std::nullopt,
                                   /*priorities=*/std::nullopt,
                                   /*max_context_len=*/128);
  ASSERT_EQ(requests[0]->sequences()[0]->dp_rank(), -1);
  requests[1]->sequences()[0]->set_dp_rank(/*dp_rank=*/3);
  scheduler->add_request(requests[0]);
  scheduler->add_request(requests[1]);

  BatchGroup batches = scheduler->prepare_batch();

  EXPECT_GE(requests[0]->sequences()[0]->dp_rank(), 0);
  EXPECT_LT(requests[0]->sequences()[0]->dp_rank(), kDpSize);
  EXPECT_EQ(requests[1]->sequences()[0]->dp_rank(), 3);
  size_t scheduled_sequences = 0;
  for (size_t dp_rank = 0; dp_rank < batches.size(); ++dp_rank) {
    scheduled_sequences += batches[dp_rank].size();
  }
  EXPECT_EQ(scheduled_sequences, 2u);
}

TEST(SchedulerPolicyTest, UnifiedPrefixHitIncludesScheduledSuffixCapacity) {
  ScopedConfigValue<bool> mix_batch_guard(
      SchedulerConfig::get_instance().enable_mix_batch(), true);
  constexpr int32_t kBlockSize = 128;
  constexpr int32_t kPrefixTokens = 256;
  constexpr int32_t kPromptTokens = 2313;
  constexpr int32_t kChunkTokens = 256;

  SchedulerOptions opt = create_scheduler_options(
      /*max_tokens_per_batch=*/kChunkTokens,
      /*max_seqs_per_batch=*/16,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/kChunkTokens,
      /*dp_size=*/1,
      /*priority_strategy=*/"multi_slo_and_prio");
  auto engine = std::make_unique<FakeEngine>(
      /*num_blocks=*/64, kBlockSize, /*enable_prefix_cache=*/true);

  auto cached_requests =
      generate_request({kPrefixTokens}, {1}, std::nullopt, std::nullopt, 10000);
  Sequence* cached_sequence = cached_requests[0]->sequences()[0].get();
  ASSERT_TRUE(engine->block_manager_pool()->allocate(cached_sequence));
  cached_sequence->kv_state().set_kv_cache_tokens_num(kPrefixTokens);
  engine->block_manager_pool()->publish_prefix(cached_sequence);
  engine->block_manager_pool()->deallocate(cached_sequence);

  auto scheduler =
      std::make_unique<TestableContinuousScheduler>(engine.get(), opt);
  auto requests =
      generate_request({kPromptTokens}, {1}, std::nullopt, std::nullopt, 10000);
  scheduler->add_request(requests[0]);

  BatchGroup batches = scheduler->prepare_batch();
  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches[0].size(), 1u);
  Sequence* hit_sequence = requests[0]->sequences()[0].get();
  EXPECT_EQ(hit_sequence->kv_cache_tokens_num(), kPrefixTokens);
  EXPECT_EQ(batches[0].get_allowed_max_tokens()[0], kChunkTokens);
  EXPECT_GE(hit_sequence->kv_state().current_max_tokens_capacity(),
            kPrefixTokens + kChunkTokens);
}

TEST(SchedulerPolicyTest, LinearPrefillPlanningAlignsAfterPartialBudget) {
  ScopedConfigValue<int32_t> stride_guard(
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 256);
  auto options = create_scheduler_options(1024, 16, 0, 256, 1);
  BatchMode mode{
      .enable_mix_batch = true,
      .enable_chunked_prefill = true,
      .priority_strategy = "fcfs",
  };
  TestUnifiedPolicy policy(mode, options);
  FakeEngine engine(64, 128, true, 1, true);
  auto requests =
      generate_request({777}, {1}, std::nullopt, std::nullopt, 10000);
  Sequence* sequence = requests.front()->sequences().front().get();
  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  std::list<std::shared_ptr<Request>> unified_queue;
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  const ModelArgs& model_args = engine.model_args();
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = engine.block_manager_pool(),
      .profile_manager = nullptr,
      .output_dispatcher = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = true,
  };
  for (const auto& [token_budget, expected_tokens] :
       {std::pair<size_t, size_t>{128, 128},
        {256, 128},
        {256, 256},
        {1024, 256},
        {256, 9}}) {
    size_t actual_tokens = 0;
    ASSERT_TRUE(policy.allocate_for_prefill(
        sequence, token_budget, &actual_tokens, state));
    EXPECT_EQ(actual_tokens, expected_tokens);
    sequence->kv_state().set_kv_cache_tokens_num(
        sequence->kv_cache_tokens_num() + actual_tokens);
  }
  engine.block_manager_pool()->deallocate(sequence);
}

TEST(SchedulerPolicyTest, LinearSchedulingPreservesReadSourceAcrossPolicies) {
  ScopedConfigValue<bool> chunked_guard(
      SchedulerConfig::get_instance().enable_chunked_prefill(), true);
  ScopedConfigValue<int32_t> stride_guard(
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 256);
  for (const bool prefix_cache : {false, true}) {
    ScopedConfigValue<bool> cache_guard(
        KVCacheConfig::get_instance().enable_prefix_cache(), prefix_cache);
    for (const std::string strategy : {"fcfs", "multi_slo_and_prio"}) {
      for (const bool mix_batch : {false, true}) {
        ScopedConfigValue<bool> mix_guard(
            SchedulerConfig::get_instance().enable_mix_batch(), mix_batch);
        auto options = create_scheduler_options(256, 16, 0, 256, 1, strategy);
        FakeEngine engine(128,
                          128,
                          prefix_cache,
                          /*dp_size=*/1,
                          /*enable_linear_attention=*/true);
        TestableContinuousScheduler scheduler(&engine, options);
        auto requests =
            generate_request({777}, {1}, std::nullopt, std::nullopt, 10000);
        Sequence* sequence = requests.front()->sequences().front().get();
        scheduler.add_request(requests.front());
        size_t cached_tokens = 0;
        int32_t previous_output_id = -1;
        for (const size_t expected_tokens : {256, 256, 256, 9}) {
          auto batches = scheduler.prepare_batch();
          ASSERT_EQ(batches.size(), 1u);
          ASSERT_EQ(batches.front().size(), 1u);
          EXPECT_EQ(batches.front().get_allowed_max_tokens().front(),
                    expected_tokens);
          const Slice<Block> linear_blocks =
              sequence->kv_state().blocks(BlockType::LINEAR);
          const bool rotated = prefix_cache && cached_tokens > 0;
          ASSERT_EQ(linear_blocks.size(), rotated ? 2u : 1u);
          if (rotated) {
            EXPECT_EQ(linear_blocks.front().id(), previous_output_id);
            EXPECT_NE(linear_blocks.back().id(), previous_output_id);
          } else if (cached_tokens > 0) {
            EXPECT_EQ(linear_blocks.back().id(), previous_output_id);
          }
          previous_output_id = linear_blocks.back().id();
          cached_tokens += expected_tokens;
          sequence->kv_state().set_kv_cache_tokens_num(cached_tokens);
        }
      }
    }
  }
}

TEST(SchedulerPolicyTest, DecodeFirstSkipsRedistributionForLinearModels) {
  ScopedConfigValue<bool> chunked_guard(
      SchedulerConfig::get_instance().enable_chunked_prefill(), true);
  ScopedConfigValue<bool> mix_guard(
      SchedulerConfig::get_instance().enable_mix_batch(), true);
  ScopedConfigValue<int32_t> stride_guard(
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  for (const bool prefix_cache : {false, true}) {
    ScopedConfigValue<bool> cache_guard(
        KVCacheConfig::get_instance().enable_prefix_cache(), prefix_cache);
    for (const bool overlap : {false, true}) {
      auto options = create_scheduler_options(6, 8, 0, 4, 1);
      options.enable_schedule_overlap(overlap);
      FakeEngine engine(64,
                        2,
                        prefix_cache,
                        /*dp_size=*/1,
                        /*enable_linear_attention=*/true);
      TestableContinuousScheduler scheduler(&engine, options);
      auto requests =
          generate_request({13}, {16}, std::nullopt, std::nullopt, 128);
      Sequence* sequence = requests.front()->sequences().front().get();
      int32_t previous_write_id = -1;
      scheduler.add_request(requests.front());
      size_t cached_tokens = 0;
      for (const size_t expected_tokens : {4u, 4u, 4u, 1u}) {
        auto batches = scheduler.prepare_batch();
        ASSERT_EQ(batches.size(), 1u);
        ASSERT_EQ(batches.front().size(), 1u);
        EXPECT_EQ(batches.front().get_allowed_max_tokens().front(),
                  expected_tokens);
        const Slice<Block> blocks =
            sequence->kv_state().blocks(BlockType::LINEAR);
        const bool rotated = prefix_cache && cached_tokens > 0;
        ASSERT_EQ(blocks.size(), rotated ? 2u : 1u);
        if (rotated) {
          EXPECT_EQ(blocks.front().id(), previous_write_id);
          EXPECT_NE(blocks.back().id(), previous_write_id);
        } else if (cached_tokens > 0) {
          EXPECT_EQ(blocks.back().id(), previous_write_id);
        }
        previous_write_id = blocks.back().id();
        cached_tokens += expected_tokens;
        sequence->kv_state().set_kv_cache_tokens_num(cached_tokens);
      }
    }
  }
}

TEST(SchedulerPolicyTest, DecodeQueuePlansColdBestOfCandidatesAsPrefill) {
  ScopedConfigValue<bool> chunked_guard(
      SchedulerConfig::get_instance().enable_chunked_prefill(), true);
  ScopedConfigValue<int32_t> stride_guard(
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  for (const int32_t token_budget : {5, 16}) {
    for (const bool prefix_cache : {false, true}) {
      ScopedConfigValue<bool> cache_guard(
          KVCacheConfig::get_instance().enable_prefix_cache(), prefix_cache);
      for (const bool mix_batch : {false, true}) {
        ScopedConfigValue<bool> mix_guard(
            SchedulerConfig::get_instance().enable_mix_batch(), mix_batch);
        for (const int32_t speculative_tokens : {0, 2}) {
          SCOPED_TRACE(::testing::Message()
                       << "token_budget=" << token_budget << ", prefix_cache="
                       << prefix_cache << ", mix_batch=" << mix_batch
                       << ", speculative_tokens=" << speculative_tokens);
          FakeEngine engine(64,
                            2,
                            prefix_cache,
                            /*dp_size=*/1,
                            /*enable_linear_attention=*/false,
                            speculative_tokens);
          auto options = create_scheduler_options(
              token_budget, 8, speculative_tokens, 4, 1);
          options.enable_schedule_overlap(false);
          TestableContinuousScheduler scheduler(&engine, options);
          auto requests =
              generate_request({7}, {16}, std::nullopt, std::nullopt, 128);
          RequestState request_state = requests.front()->state().clone();
          request_state.best_of = 2;
          auto request = std::make_shared<Request>(
              "best-of", "best-of", "", std::move(request_state), "");
          Sequence* first = request->sequences().front().get();
          ASSERT_TRUE(engine.block_manager_pool()->allocate(first, 7));
          std::vector<Batch> prefill(1);
          prefill.front().add(first, 7);
          first->kv_state().set_kv_cache_tokens_num(7);
          first->append_token(1);
          scheduler.add_request(request);
          auto batches = scheduler.prepare_batch();
          ASSERT_EQ(request->sequences().size(), 2u);
          ASSERT_EQ(batches.size(), 1u);
          ASSERT_EQ(batches.front().size(), 2u);
          Sequence* candidate = request->sequences().back().get();
          EXPECT_TRUE(candidate->is_prefill_stage());
          const size_t expected_cached_tokens = prefix_cache ? 6 : 0;
          EXPECT_EQ(candidate->kv_cache_tokens_num(), expected_cached_tokens);
          const size_t remaining_tokens = 7 - expected_cached_tokens;
          const size_t expected_prefill_tokens = std::min(
              static_cast<size_t>(token_budget - 1 - speculative_tokens),
              mix_batch && speculative_tokens == 0
                  ? remaining_tokens
                  : std::min(remaining_tokens, size_t{4}));
          EXPECT_EQ(
              candidate->kv_state().current_max_tokens_capacity(),
              (expected_cached_tokens + expected_prefill_tokens + 1) / 2 * 2);
          EXPECT_EQ(batches.front().get_allowed_max_tokens().back(),
                    expected_prefill_tokens);
          EXPECT_EQ(batches.front().get_allowed_max_tokens().front(),
                    static_cast<size_t>(1 + speculative_tokens));
          engine.block_manager_pool()->deallocate(request.get());
        }
      }
    }
  }
}

TEST(SchedulerPolicyTest, LinearLatencySchedulingRespectsRemainingTokenBudget) {
  ScopedConfigValue<bool> chunked_guard(
      SchedulerConfig::get_instance().enable_chunked_prefill(), true);
  ScopedConfigValue<int32_t> stride_guard(
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 256);
  for (const auto& [token_budget, checkpoint_tokens] :
       std::vector<std::pair<int32_t, uint32_t>>{{288, 224}, {384, 128}}) {
    for (const bool prefix_cache : {false, true}) {
      ScopedConfigValue<bool> cache_guard(
          KVCacheConfig::get_instance().enable_prefix_cache(), prefix_cache);
      for (const bool mix_batch : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "token_budget=" << token_budget
                                          << ", prefix_cache=" << prefix_cache
                                          << ", mix_batch=" << mix_batch);
        ScopedConfigValue<bool> mix_guard(
            SchedulerConfig::get_instance().enable_mix_batch(), mix_batch);
        auto options =
            create_scheduler_options(token_budget,
                                     /*max_seqs_per_batch=*/16,
                                     /*num_speculative_tokens=*/0,
                                     /*max_tokens_per_chunk_for_prefill=*/256,
                                     /*dp_size=*/1,
                                     /*priority_strategy=*/"multi_slo_and_prio",
                                     /*enable_profile_kv_blocks=*/false,
                                     /*enable_latency_aware_schedule=*/true,
                                     /*max_global_ttft_ms=*/1000000,
                                     /*max_global_tpot_ms=*/1000000);
        FakeEngine engine(128,
                          128,
                          prefix_cache,
                          /*dp_size=*/1,
                          /*enable_linear_attention=*/true);
        TestableContinuousScheduler scheduler(&engine, options);
        scheduler.get_profile_manager()->train_prefill_time_predictor(
            std::vector<std::pair<int32_t, double>>{
                {64, 8}, {128, 24}, {256, 80}, {512, 288}});
        auto requests = generate_request(
            {777, 777}, {1, 1}, std::nullopt, std::nullopt, 10000);
        for (auto& request : requests) {
          scheduler.add_request(request);
        }

        auto batches = scheduler.prepare_batch();

        ASSERT_EQ(batches.size(), 1u);
        ASSERT_EQ(batches.front().size(), 2u);
        EXPECT_EQ(batches.front().get_allowed_max_tokens(),
                  (std::vector<uint32_t>{
                      256, static_cast<uint32_t>(token_budget - 256)}));
        EXPECT_EQ(scheduler.get_running_requests().size(), 2u);
        for (size_t sequence_index = 0; sequence_index < batches.front().size();
             ++sequence_index) {
          batches.front()
              .get_sequences()[sequence_index]
              ->kv_state()
              .set_kv_cache_tokens_num(
                  batches.front().get_allowed_max_tokens()[sequence_index]);
        }
        batches = scheduler.prepare_batch();
        ASSERT_EQ(batches.size(), 1u);
        ASSERT_EQ(batches.front().size(), 2u);
        const uint32_t expected_first_tokens =
            prefix_cache && !mix_batch ? checkpoint_tokens : 256u;
        EXPECT_EQ(
            batches.front().get_allowed_max_tokens(),
            (std::vector<uint32_t>{
                expected_first_tokens,
                static_cast<uint32_t>(token_budget - expected_first_tokens)}));
      }
    }
  }
}

TEST(SchedulerPolicyTest, KvlessCompositeReprobesAfterPartialAllocation) {
  SchedulerOptions options = create_scheduler_options(
      /*max_tokens_per_batch=*/16384,
      /*max_seqs_per_batch=*/16,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/16384,
      /*dp_size=*/1,
      /*priority_strategy=*/"multi_slo_and_prio");
  BatchMode mode{
      .enable_mix_batch = true,
      .enable_chunked_prefill = true,
      .priority_strategy = "multi_slo_and_prio",
  };
  TestUnifiedPolicy policy(mode, options);
  TrackingBlockManagerPool block_manager_pool;

  std::vector<std::shared_ptr<Request>> requests = generate_request(
      {32769}, {1}, std::nullopt, std::nullopt, /*max_context_len=*/40000);
  Sequence* sequence = requests.front()->sequences().front().get();
  sequence->kv_state().add_blocks(BlockType::C4, std::vector<Block>(1));
  sequence->kv_state().set_prefix_cache_matched();
  ASSERT_TRUE(sequence->kv_state().has_any_blocks());
  ASSERT_EQ(sequence->kv_state().num_blocks(BlockType::KV), 0u);

  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  std::list<std::shared_ptr<Request>> unified_queue;
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = nullptr,
      .output_dispatcher = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = true,
  };

  policy.allocate_shared_blocks_for_test(sequence, state);

  EXPECT_EQ(block_manager_pool.allocate_shared_calls(), 1);
}

TEST(SchedulerPolicyTest, UnifiedRetryRefreshesHostRestoreBeforeChunkSizing) {
  constexpr size_t kPromptTokens = 34025;
  constexpr size_t kRestoreTokens = 32768;
  SchedulerOptions options = create_scheduler_options(
      /*max_tokens_per_batch=*/16384,
      /*max_seqs_per_batch=*/16,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/16384,
      /*dp_size=*/1,
      /*priority_strategy=*/"multi_slo_and_prio");
  BatchMode mode{
      .enable_mix_batch = true,
      .enable_chunked_prefill = true,
      .priority_strategy = "multi_slo_and_prio",
  };
  UnifiedPolicy policy(mode, options);
  RetryRestoreBlockManagerPool block_manager_pool;
  auto profile_engine = std::make_unique<FakeEngine>(512, 128);
  ProfileManager::Options profile_options;
  profile_options.max_tokens_per_batch(16384).max_seqs_per_batch(16);
  ProfileManager profile_manager =
      make_profile_manager(*profile_engine, profile_options);

  std::vector<std::shared_ptr<Request>> requests =
      generate_request({static_cast<int32_t>(kPromptTokens),
                        static_cast<int32_t>(kPromptTokens)},
                       {1, 16},
                       std::nullopt,
                       std::vector<int32_t>{1, 3},
                       /*max_context_len=*/40000);
  Sequence* decode = requests[1]->sequences().front().get();
  decode->kv_state().set_kv_cache_tokens_num(kPromptTokens);
  decode->append_token(Token(1));
  // Make the prefill urgent so the unified policy visits it before decode.
  requests[0]->state().scheduler_param.ttft_slo_ms = 0;
  requests[1]->state().scheduler_param.tpot_slo_ms = 1000000;
  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  std::list<std::shared_ptr<Request>> unified_queue{requests[0], requests[1]};
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = &profile_manager,
      .output_dispatcher = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = true,
  };
  const ScheduleBudget initial_budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  ScheduleBudget budget = initial_budget;
  std::vector<std::shared_ptr<Request>> finished;

  policy.schedule(state, budget, finished);

  ASSERT_EQ(block_manager_pool.allocate_targets().size(), 2u);
  EXPECT_EQ(block_manager_pool.allocate_targets()[0], kPromptTokens);
  EXPECT_EQ(block_manager_pool.allocate_targets()[1], kPromptTokens + 1);
  ASSERT_EQ(running_sequences.size(), 1u);
  EXPECT_EQ(running_sequences.front(), decode);
  EXPECT_EQ(running_sequence_budgets, (std::vector<size_t>{1}));
  ASSERT_EQ(unified_queue.size(), 1u);
  EXPECT_EQ(unified_queue.front(), requests[0]);
  EXPECT_TRUE(budget.kv_allocation_deferred);
  EXPECT_EQ(budget.num_preempted_requests, 0u);
  EXPECT_FALSE(requests[0]->preempted());
  EXPECT_FALSE(requests[1]->preempted());

  // A new round refreshes the failed Host match and retries the prefill,
  // while the decode retained its progress throughout the failed round.
  budget = initial_budget;
  policy.schedule(state, budget, finished);

  EXPECT_EQ(
      block_manager_pool.allocate_targets(),
      (std::vector<size_t>{
          kPromptTokens, kPromptTokens + 1, kPromptTokens, kPromptTokens + 1}));
  ASSERT_EQ(running_sequences.size(), 2u);
  EXPECT_EQ(running_sequences.front(), requests[0]->sequences().front().get());
  EXPECT_EQ(running_sequences.front()->kv_cache_tokens_num(), kRestoreTokens);
  EXPECT_EQ(running_sequences.back(), decode);
  EXPECT_EQ(running_sequence_budgets,
            (std::vector<size_t>{kPromptTokens - kRestoreTokens, 1}));
  EXPECT_EQ(budget.num_preempted_requests, 0u);
  EXPECT_TRUE(finished.empty());
}

TEST(SchedulerPolicyTest, UnifiedReclaimsMutuallyBlockedChunksWhenIdle) {
  ScopedConfigValue<bool> prefix_cache_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  for (const bool pending_release : {false, true}) {
    SCOPED_TRACE(pending_release);
    SchedulerOptions options = create_scheduler_options(
        /*max_tokens_per_batch=*/32,
        /*max_seqs_per_batch=*/2,
        /*num_speculative_tokens=*/0,
        /*max_tokens_per_chunk_for_prefill=*/16,
        /*dp_size=*/1,
        /*priority_strategy=*/"multi_slo_and_prio");
    BatchMode mode{
        .enable_mix_batch = true,
        .enable_chunked_prefill = true,
        .priority_strategy = "multi_slo_and_prio",
    };
    UnifiedPolicy policy(mode, options);
    FakeEngine engine(/*num_blocks=*/9, /*block_size=*/16);
    BlockManagerPool* pool = engine.block_manager_pool();
    ProfileManager::Options profile_options;
    profile_options.max_tokens_per_batch(32).max_seqs_per_batch(2);
    ProfileManager profile_manager =
        make_profile_manager(engine, profile_options);
    auto requests = generate_request({96, 96},
                                     {16, 16},
                                     std::nullopt,
                                     std::vector<int32_t>{1, 3},
                                     /*max_context_len=*/256);
    for (const auto& request : requests) {
      Sequence* sequence = request->sequences().front().get();
      ASSERT_TRUE(pool->allocate(sequence, 64));
      sequence->kv_state().set_kv_cache_tokens_num(64);
    }
    ASSERT_EQ(pool->num_free_blocks().front(), 0u);
    DequeQueue prefill_queue;
    DequeQueue chunk_queue;
    DequeQueue decode_queue;
    std::list<std::shared_ptr<Request>> unified_queue{requests[0], requests[1]};
    std::deque<DecodeRestoreEntry> decode_restore_waiting;
    std::vector<std::shared_ptr<Request>> running_requests;
    std::vector<Sequence*> running_sequences;
    std::vector<size_t> running_sequence_budgets;
    bool last_step_prefill = false;
    const ModelArgs& model_args = engine.model_args();
    SchedulerState state{
        .prefill_queue = prefill_queue,
        .chunk_queue = chunk_queue,
        .decode_queue = decode_queue,
        .unified_queue = unified_queue,
        .decode_restore_waiting = decode_restore_waiting,
        .running_requests = running_requests,
        .running_sequences = running_sequences,
        .running_sequences_budgets = running_sequence_budgets,
        .kv_cache_manager = pool,
        .profile_manager = &profile_manager,
        .output_dispatcher = nullptr,
        .model_args = model_args,
        .last_step_prefill = last_step_prefill,
        .options = options,
        .min_speculative_tokens_required = 0,
        .enable_prefix_cache = false,
        .has_pending_vmm_batch = pending_release,
    };
    const ScheduleBudget initial_budget{
        .remaining_token_budget = 32,
        .remaining_seq_budget = 2,
        .latency_budget = std::numeric_limits<double>::max(),
        .estimate_latency = 0,
        .num_preempted_requests = 0,
    };
    ScheduleBudget budget = initial_budget;
    std::vector<std::shared_ptr<Request>> finished;

    policy.schedule(state, budget, finished);

    EXPECT_TRUE(running_sequences.empty());
    EXPECT_TRUE(finished.empty());
    ASSERT_EQ(unified_queue.size(), 2u);
    EXPECT_TRUE(budget.kv_allocation_deferred);
    EXPECT_FALSE(requests[0]->preempted());
    EXPECT_EQ(budget.num_preempted_requests, pending_release ? 0u : 1u);
    EXPECT_EQ(requests[1]->preempted(), !pending_release);
    EXPECT_EQ(pool->num_free_blocks().front(), pending_release ? 0u : 4u);
    if (!pending_release) {
      // Reclaiming the queued victim, without any external owner releasing
      // memory, lets the higher-priority chunk execute on the next round.
      budget = initial_budget;
      policy.schedule(state, budget, finished);
      ASSERT_FALSE(running_sequences.empty());
      EXPECT_EQ(running_sequences.front(),
                requests[0]->sequences().front().get());
      EXPECT_EQ(running_sequence_budgets.front(), 16u);
      EXPECT_EQ(budget.num_preempted_requests, 0u);
    }
    for (const auto& request : requests) {
      pool->deallocate(request.get());
    }
  }
}

TEST(SchedulerPolicyTest, DecodeRestoreRetrySizesChunkAfterHostMatch) {
  ScopedConfigValue<int32_t> match_frequency(
      SchedulerConfig::get_instance().chunked_match_frequency(), 2);
  ScopedConfigValue<bool> in_batch_cache(
      KVCacheConfig::get_instance().enable_in_batch_prefix_cache(), false);
  constexpr size_t kHbmTokens = 8192;
  constexpr size_t kRestoreTokens = 32768;
  constexpr size_t kChunkTokens = 4096;
  SchedulerOptions options = create_scheduler_options(
      /*max_tokens_per_batch=*/kChunkTokens,
      /*max_seqs_per_batch=*/16,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/kChunkTokens,
      /*dp_size=*/1);
  BatchMode mode{
      .enable_mix_batch = true,
      .enable_chunked_prefill = true,
      .priority_strategy = "fcfs",
  };
  DecodeFirstPolicy policy(mode, options);
  RetryRestoreBlockManagerPool block_manager_pool;
  std::vector<std::shared_ptr<Request>> requests = generate_request(
      {40001}, {1}, std::nullopt, std::nullopt, /*max_context_len=*/50000);
  Sequence* sequence = requests.front()->sequences().front().get();
  ASSERT_TRUE(
      block_manager_pool.BlockManagerPool::allocate(sequence, kHbmTokens));
  Batch hbm_batch(sequence);
  sequence->kv_state().set_kv_cache_tokens_num(kHbmTokens);
  sequence->kv_state().set_prefix_cache_matched();
  ASSERT_GT(sequence->kv_state().num_blocks(BlockType::KV), 0u);
  block_manager_pool.allocate_shared(sequence);

  // The retained HBM prefix is at chunk 2, outside the regular match interval
  // of 5 chunks. A released Host match must still be refreshed on retry.
  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  std::list<std::shared_ptr<Request>> unified_queue;
  std::deque<DecodeRestoreEntry> decode_restore_waiting{
      DecodeRestoreEntry{requests.front(), absl::Now()}};
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = nullptr,
      .output_dispatcher = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = true,
  };
  const ScheduleBudget initial_budget{
      .remaining_token_budget = kChunkTokens,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  ScheduleBudget budget = initial_budget;
  std::vector<std::shared_ptr<Request>> finished;

  policy.schedule(state, budget, finished);

  ASSERT_EQ(decode_restore_waiting.size(), 1u);
  EXPECT_TRUE(running_sequences.empty());
  EXPECT_EQ(sequence->kv_cache_tokens_num(), kHbmTokens);
  EXPECT_FALSE(sequence->host_kv_state().prefix_cache_matched());
  EXPECT_EQ(budget.remaining_token_budget, kChunkTokens);

  budget = initial_budget;
  policy.schedule(state, budget, finished);

  EXPECT_EQ(block_manager_pool.allocate_targets(),
            (std::vector<size_t>{kRestoreTokens + kChunkTokens,
                                 kRestoreTokens + kChunkTokens}));
  EXPECT_TRUE(decode_restore_waiting.empty());
  ASSERT_EQ(running_sequences.size(), 1u);
  EXPECT_EQ(running_sequences.front(), sequence);
  EXPECT_EQ(sequence->kv_cache_tokens_num(), kRestoreTokens);
  EXPECT_EQ(running_sequence_budgets, (std::vector<size_t>{kChunkTokens}));
  EXPECT_EQ(budget.remaining_token_budget, 0u);
  EXPECT_TRUE(finished.empty());
}

class AsyncBlockReleasePolicyTest
    : public ::testing::TestWithParam<std::tuple<int32_t, bool, int32_t>> {};

TEST_P(AsyncBlockReleasePolicyTest, WaitsWithoutPreemptionThenRetries) {
  const auto [policy_kind, prefill, request_count] = GetParam();
  const std::string strategy = policy_kind == 2 ? "multi_slo_and_prio" : "fcfs";
  auto options = create_scheduler_options(1024, 16, 0, 512, 1, strategy);
  options.enable_schedule_overlap(true).enable_chunked_prefill(true);
  BatchMode mode{
      .enable_mix_batch = policy_kind != 0,
      .enable_chunked_prefill = true,
      .priority_strategy = strategy,
  };
  auto policy = create_scheduler_policy(mode, options);
  RestoreWaitingBlockManagerPool block_manager_pool;
  FakeEngine profile_engine(256, 128);
  ProfileManager::Options profile_options;
  profile_options.max_tokens_per_batch(1024).max_seqs_per_batch(16);
  ProfileManager profile_manager =
      make_profile_manager(profile_engine, profile_options);
  auto requests = generate_request(std::vector<int32_t>(request_count, 128),
                                   std::vector<int32_t>(request_count, 16),
                                   std::nullopt,
                                   std::nullopt,
                                   1024);
  if (!prefill) {
    for (const auto& request : requests) {
      Sequence* sequence = request->sequences().front().get();
      sequence->kv_state().set_kv_cache_tokens_num(
          sequence->num_prompt_tokens());
      sequence->append_token(1);
    }
  }
  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  std::list<std::shared_ptr<Request>> unified_queue;
  for (const auto& request : requests) {
    if (policy_kind == 2) {
      unified_queue.emplace_back(request);
    } else {
      (prefill ? prefill_queue : decode_queue).push(request);
    }
  }
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  model_args.layer_types({"linear_attention"});
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = &profile_manager,
      .output_dispatcher = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = false,
  };
  const ScheduleBudget initial_budget{
      .remaining_token_budget = 1024,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  ScheduleBudget budget = initial_budget;
  std::vector<std::shared_ptr<Request>> finished;

  policy->schedule(state, budget, finished);

  EXPECT_TRUE(running_sequences.empty());
  EXPECT_TRUE(finished.empty());
  EXPECT_EQ(budget.num_preempted_requests, 0u);
  EXPECT_EQ(prefill_queue.size() + decode_queue.size() + unified_queue.size(),
            static_cast<size_t>(request_count));
  for (const auto& request : requests) {
    EXPECT_FALSE(request->preempted());
  }

  block_manager_pool.set_pending_async_release(false);
  budget = initial_budget;
  policy->schedule(state, budget, finished);

  EXPECT_EQ(running_sequences.size(), static_cast<size_t>(request_count));
  EXPECT_EQ(budget.num_preempted_requests, 0u);
  EXPECT_TRUE(finished.empty());
}

INSTANTIATE_TEST_SUITE_P(AllPolicies,
                         AsyncBlockReleasePolicyTest,
                         ::testing::Combine(::testing::Values(0, 1, 2),
                                            ::testing::Bool(),
                                            ::testing::Values(1, 2)));

class PendingVmmBatchPolicyTest
    : public ::testing::TestWithParam<std::tuple<int32_t, bool, int32_t>> {};

TEST_P(PendingVmmBatchPolicyTest, RetainsRequestsUntilBatchReleasesBlocks) {
  const auto [policy_kind, prefill, request_count] = GetParam();
  const std::string strategy = policy_kind == 2 ? "multi_slo_and_prio" : "fcfs";
  auto options = create_scheduler_options(1024, 16, 0, 512, 1, strategy);
  options.enable_schedule_overlap(true).enable_chunked_prefill(true);
  BatchMode mode{
      .enable_mix_batch = policy_kind != 0,
      .enable_chunked_prefill = true,
      .priority_strategy = strategy,
  };
  auto policy = create_scheduler_policy(mode, options);
  BlockManagerPool::Options block_options;
  block_options.num_blocks_ = 8;
  block_options.block_size_ = 128;
  block_options.max_seqs_per_batch_ = 16;
  block_options.enable_prefix_cache_ = false;
  BlockManagerPool block_manager_pool(block_options, /*dp_size=*/1);
  const size_t free_blocks = block_manager_pool.num_free_blocks().front();
  int32_t dp_rank = 0;
  std::vector<Block> pending_blocks =
      block_manager_pool.allocate(free_blocks * 128, dp_rank);
  ASSERT_EQ(pending_blocks.size(), free_blocks);
  ASSERT_EQ(block_manager_pool.num_free_blocks().front(), 0u);

  FakeEngine profile_engine(256, 128);
  ProfileManager::Options profile_options;
  profile_options.max_tokens_per_batch(1024).max_seqs_per_batch(16);
  ProfileManager profile_manager =
      make_profile_manager(profile_engine, profile_options);
  auto requests = generate_request(std::vector<int32_t>(request_count, 128),
                                   std::vector<int32_t>(request_count, 16),
                                   std::nullopt,
                                   std::nullopt,
                                   1024);
  if (!prefill) {
    for (const auto& request : requests) {
      Sequence* sequence = request->sequences().front().get();
      sequence->kv_state().set_kv_cache_tokens_num(
          sequence->num_prompt_tokens());
      sequence->append_token(1);
    }
  }
  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  std::list<std::shared_ptr<Request>> unified_queue;
  for (const auto& request : requests) {
    if (policy_kind == 2) {
      unified_queue.emplace_back(request);
    } else {
      (prefill ? prefill_queue : decode_queue).push(request);
    }
  }
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = &profile_manager,
      .output_dispatcher = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = false,
      .has_pending_vmm_batch = true,
  };
  const ScheduleBudget initial_budget{
      .remaining_token_budget = 1024,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  ScheduleBudget budget = initial_budget;
  std::vector<std::shared_ptr<Request>> finished;

  policy->schedule(state, budget, finished);

  EXPECT_TRUE(running_sequences.empty());
  EXPECT_TRUE(finished.empty());
  EXPECT_EQ(budget.num_preempted_requests, 0u);
  EXPECT_EQ(prefill_queue.size() + decode_queue.size() + unified_queue.size(),
            static_cast<size_t>(request_count));
  for (const auto& request : requests) {
    EXPECT_FALSE(request->preempted());
  }

  pending_blocks.clear();
  state.has_pending_vmm_batch = false;
  budget = initial_budget;
  policy->schedule(state, budget, finished);

  EXPECT_EQ(running_sequences.size(), static_cast<size_t>(request_count));
  EXPECT_EQ(budget.num_preempted_requests, 0u);
  EXPECT_TRUE(finished.empty());
}

INSTANTIATE_TEST_SUITE_P(AllPolicies,
                         PendingVmmBatchPolicyTest,
                         ::testing::Combine(::testing::Values(0, 1, 2),
                                            ::testing::Bool(),
                                            ::testing::Values(1, 2)));

TEST(SchedulerPolicyTest, DefersWhileAsyncBlockReleaseIsPending) {
  SchedulerOptions options = create_scheduler_options(
      /*max_tokens_per_batch=*/16384,
      /*max_seqs_per_batch=*/16,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/16384,
      /*dp_size=*/1,
      /*priority_strategy=*/"multi_slo_and_prio");
  BatchMode mode{
      .enable_mix_batch = true,
      .enable_chunked_prefill = true,
      .priority_strategy = "multi_slo_and_prio",
  };
  UnifiedPolicy policy(mode, options);
  PendingReleaseBlockManagerPool block_manager_pool;
  auto profile_engine = std::make_unique<FakeEngine>(256, 128);
  ProfileManager::Options profile_options;
  profile_options.max_tokens_per_batch(16384).max_seqs_per_batch(16);
  ProfileManager profile_manager =
      make_profile_manager(*profile_engine, profile_options);

  std::vector<std::shared_ptr<Request>> requests = generate_request(
      {32000}, {2}, std::nullopt, std::nullopt, /*max_context_len=*/40000);
  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  std::list<std::shared_ptr<Request>> unified_queue{requests.front()};
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = &profile_manager,
      .output_dispatcher = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = true,
  };
  ScheduleBudget budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  std::vector<std::shared_ptr<Request>> finished;

  policy.schedule(state, budget, finished);

  EXPECT_EQ(block_manager_pool.allocate_calls(), 1);
  EXPECT_EQ(unified_queue.size(), 1u);
  EXPECT_TRUE(running_sequences.empty());
  EXPECT_TRUE(finished.empty());
}

TEST(SchedulerPolicyTest,
     RestoreWaitingDefersUntilAsyncReleaseThenSchedulesPrefill) {
  constexpr int32_t kPromptTokens = 128;
  constexpr int32_t kHostRestoreTokens = 256;
  SchedulerOptions options = create_scheduler_options(
      /*max_tokens_per_batch=*/16384,
      /*max_seqs_per_batch=*/16,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/16384,
      /*dp_size=*/1);
  BatchMode mode{
      .enable_mix_batch = false,
      .enable_chunked_prefill = true,
      .priority_strategy = "fcfs",
  };
  PrefillFirstPolicy policy(mode, options);
  RestoreWaitingBlockManagerPool block_manager_pool;
  BlockManagerPool::Options prefix_cache_options;
  prefix_cache_options.num_blocks_ = 4;
  prefix_cache_options.block_size_ = 128;
  prefix_cache_options.max_seqs_per_batch_ = 16;
  BlockManagerPool prefix_cache_pool(prefix_cache_options, /*dp_size=*/1);

  std::vector<std::shared_ptr<Request>> requests =
      generate_request({kPromptTokens},
                       {kPromptTokens + 1},
                       std::nullopt,
                       std::nullopt,
                       /*max_context_len=*/40000);
  std::vector<std::shared_ptr<Request>> cached_requests =
      generate_request({kHostRestoreTokens},
                       {1},
                       std::nullopt,
                       std::nullopt,
                       /*max_context_len=*/40000);
  Sequence* sequence = requests.front()->sequences().front().get();
  Sequence* cached_sequence =
      cached_requests.front()->sequences().front().get();
  sequence->kv_state().set_kv_cache_tokens_num(kPromptTokens);
  for (int32_t token_id = 0; token_id <= kPromptTokens; ++token_id) {
    sequence->append_token(token_id);
  }
  sequence->kv_state().set_kv_cache_tokens_num(0);
  ASSERT_TRUE(prefix_cache_pool.allocate(cached_sequence));
  sequence->host_kv_state().mount_composite_shared(
      BlockType::KV, cached_sequence->kv_state().take_blocks(BlockType::KV));
  sequence->host_kv_state().set_kv_cache_tokens_num(kHostRestoreTokens);
  sequence->host_kv_state().set_prefix_cache_matched();
  sequence->set_host_cache_match(/*restore_tokens=*/kHostRestoreTokens,
                                 /*copy_units=*/2);
  EXPECT_EQ(requests.front()->num_prefix_cache_tokens(), 0u);
  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  std::list<std::shared_ptr<Request>> unified_queue;
  std::deque<DecodeRestoreEntry> decode_restore_waiting{
      {requests.front(), absl::Now()}};
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = nullptr,
      .output_dispatcher = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = true,
  };
  ScheduleBudget budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  std::vector<std::shared_ptr<Request>> finished;

  policy.schedule(state, budget, finished);

  EXPECT_EQ(block_manager_pool.allocate_calls(), 0);
  EXPECT_EQ(decode_restore_waiting.size(), 1u);
  EXPECT_TRUE(running_sequences.empty());

  block_manager_pool.set_pending_async_release(false);
  policy.schedule(state, budget, finished);

  EXPECT_EQ(block_manager_pool.allocate_shared_calls(), 1);
  EXPECT_EQ(block_manager_pool.allocate_calls(), 1);
  EXPECT_TRUE(decode_restore_waiting.empty());
  ASSERT_EQ(running_sequences.size(), 1u);
  EXPECT_EQ(running_sequences.front(),
            requests.front()->sequences().front().get());
  EXPECT_EQ(requests.front()->num_prefix_cache_tokens(), 0u);
  EXPECT_LE(requests.front()->num_prefix_cache_tokens(),
            sequence->num_prompt_tokens());
  EXPECT_TRUE(finished.empty());
}

TEST(SchedulerPolicyTest, PendingDecodeReleaseStopsFurtherPreemption) {
  ScopedConfigValue<double> host_blocks_factor(
      KVCacheStoreConfig::get_instance().host_blocks_factor(), 2.0);
  SchedulerOptions options = create_scheduler_options(
      /*max_tokens_per_batch=*/16384,
      /*max_seqs_per_batch=*/16,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/16384,
      /*dp_size=*/1);
  options.enable_disagg_pd() = true;
  options.instance_role() = InstanceRole::DECODE;
  options.enable_schedule_overlap() = false;
  BatchMode mode{
      .enable_mix_batch = false,
      .enable_chunked_prefill = true,
      .priority_strategy = "fcfs",
  };
  PrefillFirstPolicy policy(mode, options);
  DecodeVictimBlockManagerPool block_manager_pool;

  std::vector<std::shared_ptr<Request>> requests =
      generate_request({128, 128, 128},
                       {2, 2, 2},
                       std::nullopt,
                       std::nullopt,
                       /*max_context_len=*/40000);
  for (const std::shared_ptr<Request>& request : requests) {
    Sequence* sequence = request->sequences().front().get();
    sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
    sequence->append_token(Token(1));
  }
  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  decode_queue.push(requests[0], /*if_back=*/true);
  decode_queue.push(requests[1], /*if_back=*/true);
  decode_queue.push(requests[2], /*if_back=*/true);
  std::list<std::shared_ptr<Request>> unified_queue;
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = nullptr,
      .output_dispatcher = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = true,
  };
  ScheduleBudget budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  std::vector<std::shared_ptr<Request>> finished;

  policy.schedule(state, budget, finished);

  EXPECT_EQ(decode_restore_waiting.size(), 1u);
  ASSERT_FALSE(decode_restore_waiting.empty());
  EXPECT_EQ(decode_restore_waiting.front().request.get(), requests[2].get());
  EXPECT_TRUE(requests[2]->preempted());
  EXPECT_TRUE(requests[2]->sequences().front()->is_prefill_stage());
  EXPECT_FALSE(requests[1]->preempted());
  EXPECT_EQ(decode_queue.size(), 2u);
  ASSERT_FALSE(decode_queue.empty());
  EXPECT_EQ(decode_queue.top().get(), requests[0].get());
  EXPECT_EQ(decode_queue.back().get(), requests[1].get());
  EXPECT_EQ(budget.num_preempted_requests, 1u);
  EXPECT_TRUE(prefill_queue.empty());
  EXPECT_TRUE(running_sequences.empty());
  EXPECT_TRUE(finished.empty());

  ScheduleBudget pending_budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  policy.schedule(state, pending_budget, finished);

  EXPECT_EQ(decode_restore_waiting.size(), 1u);
  EXPECT_FALSE(requests[1]->preempted());
  EXPECT_EQ(decode_queue.size(), 2u);
  EXPECT_EQ(decode_queue.back().get(), requests[1].get());
  EXPECT_EQ(pending_budget.num_preempted_requests, 0u);

  block_manager_pool.complete_async_release();
  ScheduleBudget released_budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  policy.schedule(state, released_budget, finished);

  EXPECT_EQ(decode_restore_waiting.size(), 2u);
  EXPECT_TRUE(requests[1]->preempted());
  EXPECT_EQ(decode_queue.size(), 1u);
  EXPECT_EQ(released_budget.num_preempted_requests, 1u);
}

TEST(SchedulerPolicyTest, RetriesBlockedDecodeBeforeRestoringVictim) {
  ScopedConfigValue<double> host_blocks_factor(
      KVCacheStoreConfig::get_instance().host_blocks_factor(), 2.0);
  SchedulerOptions options = create_scheduler_options(
      /*max_tokens_per_batch=*/16384,
      /*max_seqs_per_batch=*/16,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/16384,
      /*dp_size=*/1);
  options.enable_disagg_pd() = true;
  options.instance_role() = InstanceRole::DECODE;
  options.enable_schedule_overlap() = false;
  BatchMode mode{
      .enable_mix_batch = false,
      .enable_chunked_prefill = true,
      .priority_strategy = "fcfs",
  };
  PrefillFirstPolicy policy(mode, options);
  RestorePriorityBlockManagerPool block_manager_pool;

  std::vector<std::shared_ptr<Request>> requests =
      generate_request({128, 128},
                       {2, 2},
                       std::nullopt,
                       std::nullopt,
                       /*max_context_len=*/40000);
  Sequence* blocked = requests[0]->sequences().front().get();
  Sequence* victim = requests[1]->sequences().front().get();
  for (const std::shared_ptr<Request>& request : requests) {
    Sequence* sequence = request->sequences().front().get();
    sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
    sequence->append_token(Token(1));
  }
  block_manager_pool.set_sequences(blocked, victim);

  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  decode_queue.push(requests[0], /*if_back=*/true);
  decode_queue.push(requests[1], /*if_back=*/true);
  std::list<std::shared_ptr<Request>> unified_queue;
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = nullptr,
      .output_dispatcher = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = true,
  };
  ScheduleBudget initial_budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  std::vector<std::shared_ptr<Request>> finished;

  policy.schedule(state, initial_budget, finished);

  ASSERT_EQ(decode_restore_waiting.size(), 1u);
  EXPECT_EQ(decode_restore_waiting.front().request.get(), requests[1].get());
  EXPECT_EQ(decode_queue.size(), 1u);
  EXPECT_EQ(decode_queue.top().get(), requests[0].get());
  EXPECT_TRUE(running_sequences.empty());

  block_manager_pool.complete_release();
  ScheduleBudget released_budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };

  policy.schedule(state, released_budget, finished);

  ASSERT_EQ(running_sequences.size(), 1u);
  EXPECT_EQ(running_sequences.front(), blocked);
  ASSERT_EQ(decode_restore_waiting.size(), 1u);
  EXPECT_EQ(decode_restore_waiting.front().request.get(), requests[1].get());
  EXPECT_TRUE(decode_queue.empty());
}

TEST(SchedulerPolicyTest,
     DecodeFirstRetriesBlockedDecodeBeforeRestoringVictim) {
  ScopedConfigValue<double> host_blocks_factor(
      KVCacheStoreConfig::get_instance().host_blocks_factor(), 2.0);
  SchedulerOptions options = create_scheduler_options(
      /*max_tokens_per_batch=*/16384,
      /*max_seqs_per_batch=*/16,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/16384,
      /*dp_size=*/1);
  options.enable_disagg_pd() = true;
  options.instance_role() = InstanceRole::DECODE;
  options.enable_schedule_overlap() = false;
  BatchMode mode{
      .enable_mix_batch = true,
      .enable_chunked_prefill = true,
      .priority_strategy = "fcfs",
  };
  DecodeFirstPolicy policy(mode, options);
  RestorePriorityBlockManagerPool block_manager_pool;

  std::vector<std::shared_ptr<Request>> requests =
      generate_request({128, 128},
                       {2, 2},
                       std::nullopt,
                       std::nullopt,
                       /*max_context_len=*/40000);
  Sequence* blocked = requests[0]->sequences().front().get();
  Sequence* victim = requests[1]->sequences().front().get();
  for (const std::shared_ptr<Request>& request : requests) {
    Sequence* sequence = request->sequences().front().get();
    sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
    sequence->append_token(Token(1));
  }
  block_manager_pool.set_sequences(blocked, victim);

  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  decode_queue.push(requests[0], /*if_back=*/true);
  decode_queue.push(requests[1], /*if_back=*/true);
  std::list<std::shared_ptr<Request>> unified_queue;
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = nullptr,
      .output_dispatcher = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = true,
  };
  ScheduleBudget initial_budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  std::vector<std::shared_ptr<Request>> finished;

  policy.schedule(state, initial_budget, finished);

  ASSERT_EQ(decode_restore_waiting.size(), 1u);
  EXPECT_EQ(decode_restore_waiting.front().request.get(), requests[1].get());
  EXPECT_EQ(decode_queue.size(), 1u);
  EXPECT_EQ(decode_queue.top().get(), requests[0].get());
  EXPECT_TRUE(running_sequences.empty());

  block_manager_pool.complete_release();
  ScheduleBudget released_budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };

  policy.schedule(state, released_budget, finished);

  ASSERT_EQ(running_sequences.size(), 1u);
  EXPECT_EQ(running_sequences.front(), blocked);
  ASSERT_EQ(decode_restore_waiting.size(), 1u);
  EXPECT_EQ(decode_restore_waiting.front().request.get(), requests[1].get());
  EXPECT_TRUE(decode_queue.empty());
}

// TEST-2:
// memory or budget not enough
TEST(SchedulerPolicyTest, ResourceNotEnough) {
  // case1: max tokens budget not enough
  {
    // max token budget: 1
    SchedulerOptions opt = create_scheduler_options(1, 256, 0, 1024, 1);
    auto engine = std::make_unique<FakeEngine>(16, 16);
    auto scheduler =
        std::make_unique<TestableContinuousScheduler>(engine.get(), opt);

    // request prompt len: 100
    auto requests =
        generate_request({1, 100}, {1, 10}, std::nullopt, std::nullopt, 10000);
    scheduler->add_request(requests[0]);
    scheduler->add_request(requests[1]);

    auto batch = scheduler->prepare_batch();
    EXPECT_EQ(batch.size(), 1u);
    EXPECT_EQ(batch[0].size(), 1u);
  }

  // case2: blocks memory not enough
  {
    SchedulerOptions opt = create_scheduler_options(1000, 256, 0, 1024, 1);
    // free block slot: 1
    auto engine = std::make_unique<FakeEngine>(2, 8);
    auto scheduler =
        std::make_unique<TestableContinuousScheduler>(engine.get(), opt);

    // request prompt len: 1000
    auto requests = generate_request(
        {1, 1000}, {1, 100}, std::nullopt, std::nullopt, 10000);
    scheduler->add_request(requests[0]);
    scheduler->add_request(requests[1]);

    auto batch = scheduler->prepare_batch();
    EXPECT_EQ(batch.size(), 1u);
    EXPECT_EQ(batch[0].size(), 1u);
  }
}

// TEST-3:
// schedule decoding requests + some prefill requests
TEST(SchedulerPolicyTest, NormalSchedule) {
  // set max free blocks: 1024, support 1024*32=32768 tokens
  // (needs to be large enough to accommodate full-footprint reservation of
  // multiple in-flight chunked prefills plus decode blocks with margin)
  int block_num = 1024;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  SchedulerOptions opt = create_scheduler_options(
      10000, 256, 0, max_tokens_per_chunk_for_prefill, 1);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler =
      std::make_unique<TestableContinuousScheduler>(engine.get(), opt);

  // 1. schedule some new prefill requests
  auto requests = generate_request(
      {100, 200, 300}, {10, 20, 30}, std::nullopt, std::nullopt, 30000);
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto total_blocks =
      util::max(engine->block_manager_pool()->num_free_blocks());
  auto batch = scheduler->prepare_batch();
  EXPECT_EQ(batch.size(), 1u);
  EXPECT_TRUE(batch[0].size() == 3);
  auto seq_use_blocks =
      (100 / block_size + 1) + (200 / block_size + 1) + (300 / block_size + 1);
  EXPECT_TRUE(util::max(engine->block_manager_pool()->num_free_blocks()) ==
              (total_blocks - seq_use_blocks));
  update_requests(requests);

  // 2. schedule decoding requests
  batch = scheduler->prepare_batch();
  EXPECT_EQ(batch.size(), 1u);
  EXPECT_TRUE(batch[0].size() == 3);
  update_requests(requests);

  // 3. add new prefill requests
  auto requests1 =
      generate_request({400, 500}, {40, 50}, std::nullopt, std::nullopt, 30000);
  for (auto req : requests1) {
    scheduler->add_request(req);
  }
  batch = scheduler->prepare_batch();
  EXPECT_EQ(batch.size(), 1u);
  EXPECT_TRUE(batch[0].size() == 5);
  update_requests(requests);
  update_requests(requests1);

  // 4. add long new prefill requests,
  // memory is enough, can handled tokens from
  // max_tokens_per_chunk_for_prefill(1024) -> remain_max_tokens_budget
  auto requests2 =
      generate_request({10000}, {10}, std::nullopt, std::nullopt, 30000);
  for (auto req : requests2) {
    scheduler->add_request(req);
  }
  batch = scheduler->prepare_batch();
  EXPECT_EQ(batch.size(), 1u);
  EXPECT_TRUE(batch[0].size() == 6);
  const std::vector<uint32_t>& allowed_max_tokens =
      batch[0].get_allowed_max_tokens();
  // memory is enough, can handled tokens from 1024 -> remain_max_tokens_budget
  EXPECT_TRUE(allowed_max_tokens[5] == 10000 - 5);
  update_requests(requests);
  update_requests(requests1);
  update_requests(requests2);

  // 5. add long new prefill requests,
  // memory is not enough, only handled `max_tokens_per_chunk_for_prefill`(1024)
  // tokens
  auto requests3 =
      generate_request({10000}, {10}, std::nullopt, std::nullopt, 30000);
  for (auto req : requests3) {
    scheduler->add_request(req);
  }
  batch = scheduler->prepare_batch();
  EXPECT_EQ(batch.size(), 1u);
  EXPECT_TRUE(batch[0].size() == 7);
  const std::vector<uint32_t>& allowed_max_tokens1 =
      batch[0].get_allowed_max_tokens();
  // only can handle max_tokens_per_chunk_for_prefill tokens (block allocation
  // limits actual allocation even though admission gate passes).
  EXPECT_TRUE(allowed_max_tokens1[6] >= 1024);
}

// TEST-4:
// test preempt
TEST(SchedulerPolicyTest, PreemptSchedule) {
  // set max free blocks: 9, support 9*32=288 tokens
  // actually only 8 free blocks , because default 1 block is for padding
  int block_num = 9;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  SchedulerOptions opt = create_scheduler_options(
      10000, 256, 0, max_tokens_per_chunk_for_prefill, 1);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler =
      std::make_unique<TestableContinuousScheduler>(engine.get(), opt);

  std::vector<std::shared_ptr<Request>> running_requests;

  // 1. schedule some new prefill requests
  // request-1 has higher priority than request-2
  auto requests =
      generate_request({127, 127}, {10, 10}, std::nullopt, std::nullopt, 30000);
  running_requests = requests;
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch();
  EXPECT_EQ(batch.size(), 1u);
  EXPECT_TRUE(batch[0].size() == 2);
  update_requests(running_requests);

  batch = scheduler->prepare_batch();
  EXPECT_EQ(batch.size(), 1u);
  EXPECT_TRUE(batch[0].size() == 2);
  update_requests(running_requests);

  BlockManagerPool* block_manager_pool = engine->block_manager_pool();
  int free_blocks_before_preempt =
      util::max(block_manager_pool->num_free_blocks());
  batch = scheduler->prepare_batch();
  EXPECT_EQ(batch.size(), 1u);
  EXPECT_TRUE(batch[0].size() == 1);
  int free_blocks_after_preempt =
      util::max(block_manager_pool->num_free_blocks());
  EXPECT_TRUE(free_blocks_after_preempt > free_blocks_before_preempt);
  EXPECT_TRUE(scheduler->get_waiting_requests_num() == 1);
  // append a new block
  block_manager_pool->allocate(batch[0][0]);
  // remove preempted request from running_requests
  running_requests.pop_back();
  update_requests(running_requests);

  // 2.The preempted request was saved in waiting_priority_queue_,
  // it just like a new prefill request.
  // Continue to run, the new prefill request will not preempt
  // request in running_queue.
  batch = scheduler->prepare_batch();
  EXPECT_EQ(batch.size(), 1u);
  EXPECT_TRUE(batch[0].size() == 1);
}

// TEST-5:
// test priority schedule
TEST(SchedulerPolicyTest, PrioritySchedule) {
  // set max free blocks: 12
  // actually only 11 free blocks , because default 1 block is for padding
  int block_num = 12;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  SchedulerOptions opt = create_scheduler_options(
      10000, 256, 0, max_tokens_per_chunk_for_prefill, 1, "priority");
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler =
      std::make_unique<TestableContinuousScheduler>(engine.get(), opt);

  std::vector<std::shared_ptr<Request>> running_requests;

  // 1: HIGH, 2: NORMAL, 3: LOW
  auto requests = generate_request({127, 127, 127},
                                   {10, 10, 10},
                                   std::vector<bool>{false, false, false},
                                   std::vector<int32_t>{3, 3, 2},
                                   30000);
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch();
  EXPECT_EQ(batch.size(), 1u);
  EXPECT_TRUE(batch[0].size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests().size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests()[0]->priority() ==
              RequestPriority::NORMAL /*NORMAL*/);
  EXPECT_TRUE(scheduler->get_running_requests()[1]->priority() ==
              RequestPriority::LOW /*LOW*/);

  running_requests = scheduler->get_running_requests();
  update_requests(running_requests);

  // new HIGH priority request arrives, its prefill starts
  auto new_requests = generate_request({32},
                                       {10},
                                       std::vector<bool>{false},
                                       std::vector<int32_t>{1},
                                       30000);  // use 1 blocks
  scheduler->add_request(new_requests[0]);
  batch = scheduler->prepare_batch();
  // check there are 3 running requests owing to decode-maximal
  EXPECT_EQ(batch.size(), 1u);
  EXPECT_TRUE(batch[0].size() == 3);
  EXPECT_TRUE(scheduler->get_running_requests().size() == 3);
  running_requests.push_back(new_requests[0]);
  update_requests(running_requests);
  // preemption happens, only HIGH and NORMAL decode requests
  batch = scheduler->prepare_batch();
  EXPECT_EQ(batch.size(), 1u);
  EXPECT_TRUE(batch[0].size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests().size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests()[0]->priority() ==
              RequestPriority::HIGH /*HIGH*/);
  EXPECT_TRUE(scheduler->get_running_requests()[1]->priority() ==
              RequestPriority::NORMAL /*NORMAL*/);
}

// TEST-6:
// test latency budget
TEST(SchedulerPolicyTest, LatencySchedule) {
  int block_num = 12;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 4;
  // set chunked max_tokens budgets 10000 per step
  SchedulerOptions opt =
      create_scheduler_options(10000,
                               256,
                               0,
                               max_tokens_per_chunk_for_prefill,
                               1,
                               "fcfs",
                               false,
                               true,
                               350,
                               150);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler =
      std::make_unique<TestableContinuousScheduler>(engine.get(), opt);

  // manually created profile data for y=0.5x^2+10x
  std::vector<std::pair<int32_t, double>> created_profile_data = {
      {2, 22}, {4, 48}, {6, 78}, {8, 112}};
  auto profile_manager = scheduler->get_profile_manager();
  // fit y=0.5x^2+10x
  profile_manager->train_prefill_time_predictor(created_profile_data);

  auto requests = generate_request(
      {10, 10, 10}, {10, 10, 10}, std::nullopt, std::nullopt, 30000);
  // check if time equation fits well
  EXPECT_TRUE(
      static_cast<int32_t>(std::round(profile_manager->predict_step_time(
          requests[0]->sequences()[0].get(), true, true))) == 150);
  EXPECT_TRUE(static_cast<int32_t>(std::round(
                  profile_manager->predict_step_time(2, 0, true, true))) == 22);

  std::vector<std::shared_ptr<Request>> running_requests;

  // 1. three requests enter prefill
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch();
  EXPECT_EQ(batch.size(), 1);
  EXPECT_EQ(batch[0].size(), 3);
  EXPECT_EQ(scheduler->get_running_requests().size(), 3);
  running_requests = scheduler->get_running_requests();
  update_requests(running_requests);

  // 2. three requests enter decode, one new request chunked prefill
  auto new_requests = generate_request(
      {10}, {10}, std::nullopt, std::nullopt, 30000);  // use 1 blocks
  scheduler->add_request(new_requests[0]);
  batch = scheduler->prepare_batch();
  auto running_sequences_budgets = scheduler->get_running_sequences_budgets();

  EXPECT_EQ(batch.size(), 1);
  // tpot = 150 > 3 * 10 , all requests enter decode. But not enough for extra
  // whole prefill, then do chunked
  EXPECT_EQ(batch[0].size(), 4);
  EXPECT_EQ(scheduler->get_running_requests().size(), 4);
  EXPECT_EQ(running_sequences_budgets.size(), 4);
  EXPECT_EQ(running_sequences_budgets[3], max_tokens_per_chunk_for_prefill);
}

// TEST: Full-footprint admission gate
// When chunked prefill is enabled, a new request whose full footprint exceeds
// the available blocks (total - used) should NOT be admitted.
TEST(SchedulerPolicyTest, FullFootprintAdmissionGate) {
  // total_blocks = 8, block_size = 4.
  // Request A: 8 tokens → footprint = 2 blocks. Small, easily fits.
  // Request B: 32 tokens → footprint = 8 blocks. Needs all blocks.
  // max_tokens_per_chunk = 4 → each chunk = 1 block.
  //
  // A admitted, allocates 1 chunk (1 block used).
  // B check: available = 8 - 1 = 7, B footprint = 8 → 7 < 8 → blocked.
  const int32_t block_num = 9;  // 8 + 1 reserved
  const int32_t block_size = 4;
  const int32_t max_tokens_per_chunk = 4;

  SchedulerOptions opt = create_scheduler_options(
      /*max_tokens_per_batch=*/100,
      /*max_seqs=*/256,
      /*spec_tokens=*/0,
      max_tokens_per_chunk,
      /*dp_size=*/1);
  opt.enable_chunked_prefill() = true;
  opt.enable_disagg_pd() = true;
  opt.instance_role() = InstanceRole::PREFILL;

  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler =
      std::make_unique<TestableContinuousScheduler>(engine.get(), opt);
  ASSERT_NE(scheduler.get(), nullptr);

  // A (8 tokens = 2 blocks footprint), B (32 tokens = 8 blocks footprint).
  auto requests =
      generate_request({8, 32}, {10, 10}, std::nullopt, std::nullopt, 10000);
  scheduler->add_request(requests[0]);
  scheduler->add_request(requests[1]);

  auto batch = scheduler->prepare_batch();
  ASSERT_EQ(batch.size(), 1);
  // Only A admitted. B blocked: after A uses 1 block, available=8-1=7 < 8.
  EXPECT_EQ(batch[0].size(), 1);
}

// TEST: Full-footprint gate admits both when capacity is sufficient.
TEST(SchedulerPolicyTest, FullFootprintAdmitsBothWhenFits) {
  // total_blocks = 9, block_size = 4.
  // Request A: 8 tokens → footprint = 2 blocks.
  // Request B: 8 tokens → footprint = 2 blocks.
  // After A: reserved=2, B check: 2+2=4 <= 9 → admitted.
  const int32_t block_num = 9;
  const int32_t block_size = 4;
  const int32_t max_tokens_per_chunk = 4;

  SchedulerOptions opt = create_scheduler_options(
      /*max_tokens_per_batch=*/100,
      /*max_seqs=*/256,
      /*spec_tokens=*/0,
      max_tokens_per_chunk,
      /*dp_size=*/1);
  opt.enable_chunked_prefill() = true;
  opt.enable_disagg_pd() = true;
  opt.instance_role() = InstanceRole::PREFILL;

  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler =
      std::make_unique<TestableContinuousScheduler>(engine.get(), opt);
  ASSERT_NE(scheduler.get(), nullptr);

  auto requests =
      generate_request({8, 8}, {10, 10}, std::nullopt, std::nullopt, 10000);
  scheduler->add_request(requests[0]);
  scheduler->add_request(requests[1]);

  auto batch = scheduler->prepare_batch();
  ASSERT_EQ(batch.size(), 1);
  EXPECT_EQ(batch[0].size(), 2);  // Both admitted.
}

TEST(SchedulerPolicyTest, SharedArenaAdmissionReservesFullAndLinearResources) {
  ScopedConfigValue<bool> prefix_cache_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<bool> mix_batch_guard(
      SchedulerConfig::get_instance().enable_mix_batch(), true);
  ScopedConfigValue<int32_t> chunk_size_guard(
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 2);
  for (const uint32_t linear_slots : {1u, 4u}) {
    SCOPED_TRACE(linear_slots);
    const uint32_t prompt_slots = 4 + linear_slots;
    const uint32_t num_slots = 2 * prompt_slots;
    BlockManagerPool::Options pool_options;
    pool_options.num_blocks(num_slots)
        .block_size(2)
        .enable_prefix_cache(false)
        .enable_linear_state(true)
        .linear_state_num_slots(num_slots)
        .enable_shared_kv_cache(true)
        .shared_linear_state_slots(linear_slots);
    FakeEngine engine(std::make_unique<BlockManagerPool>(pool_options));
    SchedulerOptions options = create_scheduler_options(4, 4, 0, 2, 1);
    options.enable_chunked_prefill(true)
        .enable_schedule_overlap(false)
        .disable_log_stats(true);
    TestableContinuousScheduler scheduler(&engine, options);
    auto requests =
        generate_request({8, 8}, {2, 2}, std::nullopt, std::nullopt, 64);
    ASSERT_TRUE(scheduler.add_request(requests[0]));
    ASSERT_TRUE(scheduler.add_request(requests[1]));

    BatchGroup batches = scheduler.prepare_batch();

    ASSERT_EQ(batches.size(), 1u);
    EXPECT_EQ(batches.front().size(), 1u);
    EXPECT_EQ(scheduler.get_waiting_requests_num(), 1u);
    EXPECT_EQ(engine.block_manager_pool()->admission_capacity(0),
              num_slots - 1);
    EXPECT_EQ(engine.block_manager_pool()->sequence_resource_units(8),
              prompt_slots);
  }
}

TEST(SchedulerPolicyTest, SharedSlabPreservesByteUnitsAboveFourGiB) {
  constexpr size_t kGiB = size_t{1} << 30;
  for (const size_t num_allocatable_slabs : {4u, 5u}) {
    SCOPED_TRACE(num_allocatable_slabs);
    StaticKVCacheLayoutSpec spec;
    spec.num_slots = 64;
    spec.full_row_bytes = kGiB / 8;
    spec.conv_row_bytes = kGiB / 8;
    spec.ssm_row_bytes = kGiB / 4;
    spec.checkpoint_stride = 2;
    spec.full_layer_ids = {0, 2, 4, 6};
    spec.linear_layer_ids = {1, 3, 5, 7};
    const StaticKVCacheLayout layout(spec);
    ASSERT_EQ(layout.bank_count(), 4u);
    ASSERT_EQ(layout.slice_bytes(), kGiB / 2);
    ASSERT_EQ(layout.slab_bytes(), 2 * kGiB);
    ASSERT_EQ(layout.num_reserved_slabs(), 2u);
    spec.num_slabs = layout.num_reserved_slabs() + num_allocatable_slabs;
    ASSERT_LE(spec.num_slabs, layout.max_slab_count());
    BlockManagerPool::Options options;
    options.num_blocks(64)
        .block_size(2)
        .enable_prefix_cache(false)
        .enable_linear_state(true)
        .linear_state_num_slots(64)
        .enable_shared_kv_cache(true)
        .shared_layout(
            std::make_shared<const StaticKVCacheLayout>(std::move(spec)));
    BlockManagerPool pool(options);
    // Only mappings are allocated; no multi-GiB backing tensor is needed.
    EXPECT_EQ(pool.sequence_resource_units(4), 4 * kGiB + kGiB / 2);
    EXPECT_EQ(pool.admission_capacity(0), num_allocatable_slabs * 2 * kGiB);
  }
}

TEST(SchedulerPolicyTest, StaleCapacityReservesFullPromptBeforeFirstChunk) {
  ScopedConfigValue<bool> prefix_cache_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  for (bool enable_mix_batch : {false, true}) {
    SCOPED_TRACE(enable_mix_batch);
    ScopedConfigValue<bool> mix_batch_guard(
        SchedulerConfig::get_instance().enable_mix_batch(), enable_mix_batch);
    for (size_t advisory_capacity : {0u, 1u}) {
      SCOPED_TRACE(advisory_capacity);
      SchedulerOptions options = create_scheduler_options(
          /*max_tokens_per_batch=*/4,
          /*max_seqs_per_batch=*/1,
          /*num_speculative_tokens=*/0,
          /*max_tokens_per_chunk_for_prefill=*/4,
          /*dp_size=*/1);
      options.enable_chunked_prefill(true)
          .enable_schedule_overlap(false)
          .disable_log_stats(true);
      auto block_manager =
          std::make_unique<StaleCapacityBlockManagerPool>(advisory_capacity);
      StaleCapacityBlockManagerPool* pool = block_manager.get();
      FakeEngine engine(std::move(block_manager));
      TestableContinuousScheduler scheduler(&engine, options);
      const std::vector<size_t> initial_free_blocks = pool->num_free_blocks();
      auto request = generate_request({12},
                                      {4},
                                      std::nullopt,
                                      std::nullopt,
                                      /*max_context_len=*/64)[0];
      ASSERT_TRUE(scheduler.add_request(request));

      // A completed request can free physical pages before the next advisory
      // snapshot. The first chunk must probe the real allocator and retain
      // the full three-block prompt reservation without advancing computation.
      BatchGroup batches = scheduler.prepare_batch();
      ASSERT_EQ(batches.size(), 1u);
      ASSERT_EQ(batches.front().size(), 1u);
      Sequence* sequence = request->sequences().front().get();
      EXPECT_EQ(batches.front()[0], sequence);
      ASSERT_EQ(batches.front().get_allowed_max_tokens().size(), 1u);
      EXPECT_EQ(batches.front().get_allowed_max_tokens()[0], 4u);
      EXPECT_EQ(sequence->kv_state().num_blocks(BlockType::KV), 3u);
      EXPECT_EQ(sequence->kv_state().kv_cache_tokens_num(), 0u);
      ASSERT_FALSE(pool->allocate_targets().empty());
      EXPECT_EQ(pool->allocate_targets().front(), 12u);

      pool->deallocate(request.get());
      EXPECT_EQ(pool->num_free_blocks(), initial_free_blocks);
    }
  }
}

TEST(SchedulerPolicyTest, StaleCapacityDefersFailedFullPromptReservation) {
  ScopedConfigValue<bool> prefix_cache_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  for (bool enable_mix_batch : {false, true}) {
    SCOPED_TRACE(enable_mix_batch);

    ScopedConfigValue<bool> mix_batch_guard(
        SchedulerConfig::get_instance().enable_mix_batch(), enable_mix_batch);
    SchedulerOptions options = create_scheduler_options(
        /*max_tokens_per_batch=*/4,
        /*max_seqs_per_batch=*/1,
        /*num_speculative_tokens=*/0,
        /*max_tokens_per_chunk_for_prefill=*/4,
        /*dp_size=*/1);
    options.enable_chunked_prefill(true)
        .enable_schedule_overlap(false)
        .disable_log_stats(true);
    auto block_manager = std::make_unique<StaleCapacityBlockManagerPool>(
        /*advisory_capacity=*/0);
    StaleCapacityBlockManagerPool* pool = block_manager.get();
    FakeEngine engine(std::move(block_manager));
    TestableContinuousScheduler scheduler(&engine, options);
    const std::vector<size_t> initial_free_blocks = pool->num_free_blocks();
    auto blocker = generate_request({24},
                                    {4},
                                    std::nullopt,
                                    std::nullopt,
                                    /*max_context_len=*/64)[0];
    ASSERT_TRUE(pool->allocate(blocker->sequences().front().get(),
                               /*num_tokens=*/24));
    const std::vector<size_t> free_blocks_before = pool->num_free_blocks();
    RequestState request_state =
        generate_request({12}, {4}, std::nullopt, std::nullopt, 64)[0]
            ->state()
            .clone();

    auto request = std::make_shared<Request>("reservation-retry",
                                             "reservation-retry",
                                             "",
                                             std::move(request_state),
                                             "");
    std::optional<Status> callback_status;
    int32_t callback_count = 0;
    request->state().output_func = [&](const RequestOutput& output) {
      callback_status = output.status;
      ++callback_count;
      return true;
    };
    ASSERT_TRUE(scheduler.add_request(request));

    // Another owner temporarily holds the pages. Failed admission must leave
    // the request queued and return without retrying within the same round.
    for (size_t attempt = 0; attempt < 2; ++attempt) {
      BatchGroup batches = scheduler.prepare_batch();
      scheduler.wait_for_responses();
      ASSERT_EQ(batches.size(), 1u);
      EXPECT_TRUE(batches.front().empty());
      EXPECT_EQ(callback_count, 0);
      EXPECT_FALSE(callback_status.has_value());
      EXPECT_EQ(pool->allocate_targets().size(), attempt + 2);
      EXPECT_EQ(pool->allocate_targets().back(), 12u);
      EXPECT_EQ(
          request->sequences().front()->kv_state().num_blocks(BlockType::KV),
          0u);
      EXPECT_EQ(pool->num_free_blocks(), free_blocks_before);
      EXPECT_EQ(scheduler.get_waiting_requests_num(), 1u);
    }

    pool->deallocate(blocker.get());
    BatchGroup batches = scheduler.prepare_batch();
    scheduler.wait_for_responses();
    ASSERT_EQ(batches.size(), 1u);
    ASSERT_EQ(batches.front().size(), 1u);
    EXPECT_EQ(batches.front()[0], request->sequences().front().get());
    EXPECT_EQ(batches.front().get_allowed_max_tokens(),
              (std::vector<uint32_t>{4}));
    EXPECT_EQ(
        request->sequences().front()->kv_state().num_blocks(BlockType::KV), 3u);
    EXPECT_EQ(callback_count, 0);
    EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);

    pool->deallocate(request.get());
    EXPECT_EQ(pool->num_free_blocks(), initial_free_blocks);
  }
}

TEST(SchedulerPolicyTest, PersistentPrefillAllocationFailureUnblocksFollower) {
  ScopedConfigValue<bool> prefix_cache_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  for (const bool enable_mix_batch : {false, true}) {
    SCOPED_TRACE(enable_mix_batch);
    ScopedConfigValue<bool> mix_batch_guard(
        SchedulerConfig::get_instance().enable_mix_batch(), enable_mix_batch);
    SchedulerOptions options = create_scheduler_options(
        /*max_tokens_per_batch=*/4,
        /*max_seqs_per_batch=*/1,
        /*num_speculative_tokens=*/0,
        /*max_tokens_per_chunk_for_prefill=*/4,
        /*dp_size=*/1);
    options.enable_chunked_prefill(true)
        .enable_schedule_overlap(false)
        .disable_log_stats(true);
    auto block_manager = std::make_unique<FailOnAllocationBlockManagerPool>(
        /*fail_on_attempt=*/0);
    FailOnAllocationBlockManagerPool* pool = block_manager.get();
    FakeEngine engine(std::move(block_manager));
    TestableContinuousScheduler scheduler(&engine, options);
    const std::vector<size_t> initial_free_blocks = pool->num_free_blocks();
    auto requests =
        generate_request({12, 4}, {4, 4}, std::nullopt, std::nullopt, 64);
    std::optional<Status> callback_status;
    int32_t callback_count = 0;
    requests.front()->state().output_func = [&](const RequestOutput& output) {
      callback_status = output.status;
      ++callback_count;
      return true;
    };
    ASSERT_TRUE(scheduler.add_request(requests.front()));
    ASSERT_TRUE(scheduler.add_request(requests.back()));

    // A request can fit its logical arena yet never acquire physical backing.
    // Repeated empty rounds must eventually release the FCFS head.
    constexpr size_t kRetryLimit = 4096;
    for (size_t attempt = 0; attempt < kRetryLimit; ++attempt) {
      pool->set_fail_on_attempt(pool->allocation_attempts() + 1);
      BatchGroup batches = scheduler.prepare_batch();
      ASSERT_EQ(batches.size(), 1u);
      ASSERT_TRUE(batches.front().empty());
    }
    scheduler.wait_for_responses();
    ASSERT_TRUE(callback_status.has_value());
    EXPECT_EQ(callback_status->code(), StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_EQ(callback_count, 1);
    EXPECT_EQ(pool->allocation_attempts(), kRetryLimit);
    EXPECT_EQ(scheduler.get_waiting_requests_num(), 1u);
    EXPECT_EQ(pool->num_free_blocks(), initial_free_blocks);

    pool->set_fail_on_attempt(/*attempt=*/0);
    BatchGroup batches = scheduler.prepare_batch();
    ASSERT_EQ(batches.size(), 1u);
    ASSERT_EQ(batches.front().size(), 1u);
    EXPECT_EQ(batches.front()[0], requests.back()->sequences().front().get());
    EXPECT_EQ(batches.front().get_allowed_max_tokens(),
              (std::vector<uint32_t>{4}));
    EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
    pool->deallocate(requests.back().get());
    EXPECT_EQ(pool->num_free_blocks(), initial_free_blocks);
  }
}

TEST(SchedulerPolicyTest, SuccessfulPrefillResetsAllocationFailureWindow) {
  ScopedConfigValue<bool> prefix_cache_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  for (const bool enable_mix_batch : {false, true}) {
    SCOPED_TRACE(enable_mix_batch);
    ScopedConfigValue<bool> mix_batch_guard(
        SchedulerConfig::get_instance().enable_mix_batch(), enable_mix_batch);
    SchedulerOptions options = create_scheduler_options(
        /*max_tokens_per_batch=*/4,
        /*max_seqs_per_batch=*/1,
        /*num_speculative_tokens=*/0,
        /*max_tokens_per_chunk_for_prefill=*/4,
        /*dp_size=*/1);
    options.enable_chunked_prefill(true)
        .enable_schedule_overlap(false)
        .disable_log_stats(true);
    auto block_manager = std::make_unique<FailOnAllocationBlockManagerPool>(
        /*fail_on_attempt=*/0);
    FailOnAllocationBlockManagerPool* pool = block_manager.get();
    FakeEngine engine(std::move(block_manager));
    TestableContinuousScheduler scheduler(&engine, options);
    const std::vector<size_t> initial_free_blocks = pool->num_free_blocks();
    auto request =
        generate_request({12}, {4}, std::nullopt, std::nullopt, 64)[0];
    Sequence* sequence = request->sequences().front().get();
    std::optional<Status> callback_status;
    request->state().output_func = [&](const RequestOutput& output) {
      callback_status = output.status;
      return true;
    };
    ASSERT_TRUE(scheduler.add_request(request));

    // Two long pressure episodes separated by useful work are independent.
    // The second episode must receive the full retry window as well.
    constexpr size_t kFailuresBeforeRecovery = 4095;
    for (const size_t computed_tokens : {0u, 4u}) {
      sequence->kv_state().set_kv_cache_tokens_num(computed_tokens);
      for (size_t attempt = 0; attempt < kFailuresBeforeRecovery; ++attempt) {
        pool->set_fail_on_attempt(pool->allocation_attempts() + 1);
        BatchGroup batches = scheduler.prepare_batch();
        ASSERT_EQ(batches.size(), 1u);
        ASSERT_TRUE(batches.front().empty());
      }
      scheduler.wait_for_responses();
      ASSERT_FALSE(callback_status.has_value());

      pool->set_fail_on_attempt(/*attempt=*/0);
      BatchGroup batches = scheduler.prepare_batch();
      ASSERT_EQ(batches.size(), 1u);
      ASSERT_EQ(batches.front().size(), 1u);
      EXPECT_EQ(batches.front()[0], sequence);
      EXPECT_EQ(batches.front().get_allowed_max_tokens(),
                (std::vector<uint32_t>{4}));
      EXPECT_EQ(sequence->kv_cache_tokens_num(), computed_tokens);
      EXPECT_EQ(sequence->kv_state().num_blocks(BlockType::KV),
                computed_tokens / 4 + 1);
    }
    EXPECT_EQ(pool->allocation_attempts(), 2 * (kFailuresBeforeRecovery + 1));
    pool->deallocate(request.get());
    EXPECT_EQ(pool->num_free_blocks(), initial_free_blocks);
  }
}

TEST(SchedulerPolicyTest, TransientDecodeAllocationFailurePreservesRequest) {
  ScopedConfigValue<bool> prefix_cache_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  for (bool enable_mix_batch : {false, true}) {
    SCOPED_TRACE(enable_mix_batch);
    ScopedConfigValue<bool> mix_batch_guard(
        SchedulerConfig::get_instance().enable_mix_batch(), enable_mix_batch);
    SchedulerOptions options = create_scheduler_options(
        /*max_tokens_per_batch=*/4,
        /*max_seqs_per_batch=*/1,
        /*num_speculative_tokens=*/0,
        /*max_tokens_per_chunk_for_prefill=*/4,
        /*dp_size=*/1);
    options.enable_chunked_prefill(true)
        .enable_schedule_overlap(false)
        .disable_log_stats(true);
    auto block_manager = std::make_unique<FailOnAllocationBlockManagerPool>(
        /*fail_on_attempt=*/0);
    FailOnAllocationBlockManagerPool* pool = block_manager.get();
    FakeEngine engine(std::move(block_manager));
    TestableContinuousScheduler scheduler(&engine, options);
    const std::vector<size_t> initial_free_blocks = pool->num_free_blocks();
    auto request =
        generate_request({4}, {4}, std::nullopt, std::nullopt, 64)[0];
    std::optional<Status> callback_status;
    request->state().output_func = [&](const RequestOutput& output) {
      callback_status = output.status;
      return true;
    };
    ASSERT_TRUE(scheduler.add_request(request));
    BatchGroup batches = scheduler.prepare_batch();
    ASSERT_EQ(batches.size(), 1u);
    ASSERT_EQ(batches.front().size(), 1u);
    Sequence* sequence = request->sequences().front().get();
    ASSERT_EQ(sequence->kv_state().num_blocks(BlockType::KV), 1u);
    sequence->kv_state().set_kv_cache_tokens_num(4);
    sequence->append_token(Token(1));
    ASSERT_FALSE(sequence->is_prefill_stage());
    const int64_t original_block_id =
        sequence->kv_state().blocks(BlockType::KV).front().id();
    const std::vector<size_t> free_blocks_before = pool->num_free_blocks();
    const size_t successful_allocations = pool->allocation_attempts();
    pool->set_fail_on_attempt(successful_allocations + 1);

    // The first decode token crosses the four-token block boundary. A
    // transient mapping failure must not discard the completed prompt.
    batches = scheduler.prepare_batch();
    scheduler.wait_for_responses();
    ASSERT_EQ(batches.size(), 1u);
    EXPECT_TRUE(batches.front().empty());
    EXPECT_FALSE(callback_status.has_value());
    EXPECT_FALSE(request->cancelled());
    EXPECT_FALSE(request->finished());
    EXPECT_FALSE(request->preempted());
    EXPECT_EQ(sequence->kv_cache_tokens_num(), 4u);
    ASSERT_EQ(sequence->kv_state().num_blocks(BlockType::KV), 1u);
    EXPECT_EQ(sequence->kv_state().blocks(BlockType::KV).front().id(),
              original_block_id);
    EXPECT_EQ(pool->num_free_blocks(), free_blocks_before);
    EXPECT_EQ(pool->allocation_attempts(), successful_allocations + 1);

    batches = scheduler.prepare_batch();
    scheduler.wait_for_responses();
    ASSERT_EQ(batches.size(), 1u);
    ASSERT_EQ(batches.front().size(), 1u);
    EXPECT_EQ(batches.front()[0], sequence);
    EXPECT_EQ(batches.front().get_allowed_max_tokens(),
              (std::vector<uint32_t>{1}));
    EXPECT_FALSE(callback_status.has_value());
    EXPECT_EQ(sequence->kv_cache_tokens_num(), 4u);
    ASSERT_EQ(sequence->kv_state().num_blocks(BlockType::KV), 2u);
    EXPECT_EQ(sequence->kv_state().blocks(BlockType::KV).front().id(),
              original_block_id);
    EXPECT_EQ(pool->allocation_attempts(), successful_allocations + 2);
    pool->deallocate(request.get());
    EXPECT_EQ(pool->num_free_blocks(), initial_free_blocks);
  }
}

TEST(SchedulerPolicyTest, DecodeAllocationFailureDefersNewPrefill) {
  ScopedConfigValue<bool> prefix_cache_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<bool> mix_batch_guard(
      SchedulerConfig::get_instance().enable_mix_batch(), true);
  SchedulerOptions options = create_scheduler_options(
      /*max_tokens_per_batch=*/16,
      /*max_seqs_per_batch=*/3,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/4,
      /*dp_size=*/1);
  options.enable_chunked_prefill(true)
      .enable_schedule_overlap(false)
      .disable_log_stats(true);
  auto block_manager = std::make_unique<FailOnAllocationBlockManagerPool>(
      /*fail_on_attempt=*/0);
  FailOnAllocationBlockManagerPool* pool = block_manager.get();
  FakeEngine engine(std::move(block_manager));
  TestableContinuousScheduler scheduler(&engine, options);
  const std::vector<size_t> initial_free_blocks = pool->num_free_blocks();
  auto requests =
      generate_request({4, 4}, {16, 16}, std::nullopt, std::nullopt, 64);
  for (auto& request : requests) {
    ASSERT_TRUE(scheduler.add_request(request));
  }
  BatchGroup batches = scheduler.prepare_batch();
  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches.front().size(), 2u);
  update_requests(requests);
  Sequence* progressing = batches.front()[0];
  Sequence* deferred = batches.front()[1];
  const int64_t deferred_block_id =
      deferred->kv_state().blocks(BlockType::KV).front().id();
  const size_t previous_attempts = pool->allocation_attempts();
  pool->set_fail_on_attempt(previous_attempts + 2);
  auto fresh = generate_request({1}, {4}, std::nullopt, std::nullopt, 64)[0];
  std::optional<Status> callback_status;
  fresh->state().output_func = [&](const RequestOutput& output) {
    callback_status = output.status;
    return true;
  };
  ASSERT_TRUE(scheduler.add_request(fresh));

  // The first decode grows successfully; the second mapping fails. Keep the
  // successful decode, but do not spend more memory on a new prefill.
  batches = scheduler.prepare_batch();
  scheduler.wait_for_responses();
  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches.front().size(), 1u);
  EXPECT_EQ(batches.front()[0], progressing);
  EXPECT_EQ(pool->allocation_attempts(), previous_attempts + 2);
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 1u);
  EXPECT_FALSE(callback_status.has_value());
  EXPECT_FALSE(deferred->finished());
  EXPECT_EQ(deferred->kv_cache_tokens_num(), 4u);
  ASSERT_EQ(deferred->kv_state().num_blocks(BlockType::KV), 1u);
  EXPECT_EQ(deferred->kv_state().blocks(BlockType::KV).front().id(),
            deferred_block_id);

  // Pressure is scoped to the failed round. A recovered allocator admits the
  // waiting prefill alongside both existing decodes on the next pass.
  progressing->kv_state().set_kv_cache_tokens_num(5);
  progressing->append_token(Token(1));
  batches = scheduler.prepare_batch();
  scheduler.wait_for_responses();
  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches.front().size(), 3u);
  EXPECT_EQ(batches.front()[2], fresh->sequences().front().get());
  EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
  EXPECT_FALSE(callback_status.has_value());
  for (const auto& request : requests) {
    pool->deallocate(request.get());
  }
  pool->deallocate(fresh.get());
  EXPECT_EQ(pool->num_free_blocks(), initial_free_blocks);
}

TEST(SchedulerPolicyTest, ChunkAllocationFailureDefersOtherPrefillQueue) {
  ScopedConfigValue<bool> prefix_cache_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  for (const bool enable_mix_batch : {false, true}) {
    SCOPED_TRACE(enable_mix_batch);
    ScopedConfigValue<bool> mix_batch_guard(
        SchedulerConfig::get_instance().enable_mix_batch(), enable_mix_batch);
    SchedulerOptions options = create_scheduler_options(
        /*max_tokens_per_batch=*/4,
        /*max_seqs_per_batch=*/2,
        /*num_speculative_tokens=*/0,
        /*max_tokens_per_chunk_for_prefill=*/4,
        /*dp_size=*/1);
    options.enable_chunked_prefill(true)
        .enable_schedule_overlap(false)
        .disable_log_stats(true);
    auto block_manager = std::make_unique<FailOnAllocationBlockManagerPool>(
        /*fail_on_attempt=*/0);
    FailOnAllocationBlockManagerPool* pool = block_manager.get();
    FakeEngine engine(std::move(block_manager));
    TestableContinuousScheduler scheduler(&engine, options);
    const std::vector<size_t> initial_free_blocks = pool->num_free_blocks();
    auto holder =
        generate_request({120}, {4}, std::nullopt, std::nullopt, 256)[0];
    ASSERT_TRUE(pool->allocate(holder->sequences().front().get(), 112));
    auto chunked =
        generate_request({12}, {4}, std::nullopt, std::nullopt, 64)[0];
    ASSERT_TRUE(scheduler.add_request(chunked));
    BatchGroup batches = scheduler.prepare_batch();
    ASSERT_EQ(batches.size(), 1u);
    ASSERT_EQ(batches.front().size(), 1u);
    Sequence* sequence = chunked->sequences().front().get();
    sequence->kv_state().set_kv_cache_tokens_num(4);
    const int64_t original_block_id =
        sequence->kv_state().blocks(BlockType::KV).front().id();
    ASSERT_TRUE(pool->allocate(holder->sequences().front().get(), 120));
    ASSERT_EQ(pool->num_free_blocks().front(), 0u);
    const size_t previous_attempts = pool->allocation_attempts();
    auto fresh = generate_request({1}, {4}, std::nullopt, std::nullopt, 64)[0];
    std::optional<Status> callback_status;
    fresh->state().output_func = [&](const RequestOutput& output) {
      callback_status = output.status;
      return true;
    };
    ASSERT_TRUE(scheduler.add_request(fresh));

    // A blocked continuation must not trigger another allocation through the
    // separate fresh-prefill queue, or discard its already computed prefix.
    batches = scheduler.prepare_batch();
    scheduler.wait_for_responses();
    ASSERT_EQ(batches.size(), 1u);
    EXPECT_TRUE(batches.front().empty());
    EXPECT_EQ(pool->allocation_attempts(), previous_attempts + 1);
    EXPECT_EQ(scheduler.get_waiting_requests_num(), 2u);
    EXPECT_FALSE(callback_status.has_value());
    EXPECT_EQ(sequence->kv_cache_tokens_num(), 4u);
    ASSERT_EQ(sequence->kv_state().num_blocks(BlockType::KV), 1u);
    EXPECT_EQ(sequence->kv_state().blocks(BlockType::KV).front().id(),
              original_block_id);

    pool->deallocate(holder.get());
    for (const size_t completed_tokens : {4u, 8u}) {
      sequence->kv_state().set_kv_cache_tokens_num(completed_tokens);
      batches = scheduler.prepare_batch();
      ASSERT_EQ(batches.size(), 1u);
      ASSERT_EQ(batches.front().size(), 1u);
      EXPECT_EQ(batches.front()[0], sequence);
      EXPECT_EQ(batches.front().get_allowed_max_tokens(),
                (std::vector<uint32_t>{4}));
    }
    sequence->kv_state().set_kv_cache_tokens_num(12);
    sequence->append_token(Token(1));
    batches = scheduler.prepare_batch();
    scheduler.wait_for_responses();
    ASSERT_EQ(batches.size(), 1u);
    bool fresh_scheduled = false;
    for (uint64_t row = 0; row < batches.front().size(); ++row) {
      fresh_scheduled = fresh_scheduled || batches.front()[row] ==
                                               fresh->sequences().front().get();
    }
    EXPECT_TRUE(fresh_scheduled);
    EXPECT_FALSE(callback_status.has_value());
    pool->deallocate(chunked.get());
    pool->deallocate(fresh.get());
    EXPECT_EQ(pool->num_free_blocks(), initial_free_blocks);
  }
}

TEST(SchedulerPolicyTest, StaticCapacityRejectsDecodeBeyondArena) {
  ScopedConfigValue<bool> prefix_cache_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  for (bool enable_mix_batch : {false, true}) {
    SCOPED_TRACE(enable_mix_batch);
    ScopedConfigValue<bool> mix_batch_guard(
        SchedulerConfig::get_instance().enable_mix_batch(), enable_mix_batch);
    SchedulerOptions options = create_scheduler_options(
        /*max_tokens_per_batch=*/8,
        /*max_seqs_per_batch=*/1,
        /*num_speculative_tokens=*/0,
        /*max_tokens_per_chunk_for_prefill=*/8,
        /*dp_size=*/1);
    options.enable_chunked_prefill(true)
        .enable_schedule_overlap(false)
        .disable_log_stats(true);
    FakeEngine engine(/*num_blocks=*/3, /*block_size=*/4);
    BlockManagerPool* pool = engine.block_manager_pool();
    TestableContinuousScheduler scheduler(&engine, options);
    const std::vector<size_t> initial_free_blocks = pool->num_free_blocks();
    auto request =
        generate_request({8}, {4}, std::nullopt, std::nullopt, 64)[0];
    std::optional<Status> callback_status;
    request->state().output_func = [&](const RequestOutput& output) {
      callback_status = output.status;
      return true;
    };
    ASSERT_TRUE(scheduler.add_request(request));
    BatchGroup batches = scheduler.prepare_batch();
    ASSERT_EQ(batches.size(), 1u);
    ASSERT_EQ(batches.front().size(), 1u);
    Sequence* sequence = request->sequences().front().get();
    ASSERT_EQ(sequence->kv_state().num_blocks(BlockType::KV), 2u);
    sequence->kv_state().set_kv_cache_tokens_num(8);
    sequence->append_token(Token(1));

    // Padding leaves two usable blocks. The next decode token needs a third
    // block, which cannot fit even after all other owners release memory.
    batches = scheduler.prepare_batch();
    scheduler.wait_for_responses();
    ASSERT_EQ(batches.size(), 1u);
    EXPECT_TRUE(batches.front().empty());
    ASSERT_TRUE(callback_status.has_value());
    EXPECT_EQ(callback_status->code(), StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_EQ(sequence->kv_state().num_blocks(BlockType::KV), 0u);
    EXPECT_EQ(pool->num_free_blocks(), initial_free_blocks);
  }
}

TEST(SchedulerPolicyTest, IdleChunkContinuationExceedsUtilizationThreshold) {
  ScopedConfigValue<bool> prefix_cache_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> utilization_guard(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      0.5);
  for (bool enable_mix_batch : {false, true}) {
    SCOPED_TRACE(enable_mix_batch);
    ScopedConfigValue<bool> mix_batch_guard(
        SchedulerConfig::get_instance().enable_mix_batch(), enable_mix_batch);
    SchedulerOptions options = create_scheduler_options(
        /*max_tokens_per_batch=*/20,
        /*max_seqs_per_batch=*/1,
        /*num_speculative_tokens=*/0,
        /*max_tokens_per_chunk_for_prefill=*/20,
        /*dp_size=*/1);
    options.enable_chunked_prefill(true)
        .enable_schedule_overlap(false)
        .disable_log_stats(true);
    FakeEngine engine(/*num_blocks=*/8, /*block_size=*/4);
    BlockManagerPool* pool = engine.block_manager_pool();
    TestableContinuousScheduler scheduler(&engine, options);
    const std::vector<size_t> initial_free_blocks = pool->num_free_blocks();
    auto request =
        generate_request({24}, {4}, std::nullopt, std::nullopt, 64)[0];
    std::optional<Status> callback_status;
    request->state().output_func = [&](const RequestOutput& output) {
      callback_status = output.status;
      return true;
    };
    ASSERT_TRUE(scheduler.add_request(request));
    BatchGroup batches = scheduler.prepare_batch();
    ASSERT_EQ(batches.size(), 1u);
    ASSERT_EQ(batches.front().size(), 1u);
    EXPECT_EQ(batches.front().get_allowed_max_tokens(),
              (std::vector<uint32_t>{20}));
    Sequence* sequence = request->sequences().front().get();
    ASSERT_EQ(sequence->kv_state().num_blocks(BlockType::KV), 5u);
    sequence->kv_state().set_kv_cache_tokens_num(20);
    ASSERT_TRUE(sequence->is_chunked_prefill_stage());
    ASSERT_GT(pool->kv_cache_utilization(), 0.5);
    const int64_t original_block_id =
        sequence->kv_state().blocks(BlockType::KV).front().id();

    // No decode or other executable request can lower utilization. The
    // continuation must be allowed to allocate its last prompt block.
    batches = scheduler.prepare_batch();
    scheduler.wait_for_responses();
    ASSERT_EQ(batches.size(), 1u);
    ASSERT_EQ(batches.front().size(), 1u);
    EXPECT_EQ(batches.front()[0], sequence);
    EXPECT_EQ(batches.front().get_allowed_max_tokens(),
              (std::vector<uint32_t>{4}));
    EXPECT_FALSE(callback_status.has_value());
    EXPECT_FALSE(request->preempted());
    EXPECT_EQ(sequence->kv_cache_tokens_num(), 20u);
    ASSERT_EQ(sequence->kv_state().num_blocks(BlockType::KV), 6u);
    EXPECT_EQ(sequence->kv_state().blocks(BlockType::KV).front().id(),
              original_block_id);
    pool->deallocate(request.get());
    EXPECT_EQ(pool->num_free_blocks(), initial_free_blocks);
  }
}

TEST(SchedulerPolicyTest, StaticCapacityRejectsPromptWithoutReservation) {
  ScopedConfigValue<bool> prefix_cache_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  for (bool enable_mix_batch : {false, true}) {
    SCOPED_TRACE(enable_mix_batch);
    ScopedConfigValue<bool> mix_batch_guard(
        SchedulerConfig::get_instance().enable_mix_batch(), enable_mix_batch);
    SchedulerOptions options = create_scheduler_options(
        /*max_tokens_per_batch=*/4,
        /*max_seqs_per_batch=*/1,
        /*num_speculative_tokens=*/0,
        /*max_tokens_per_chunk_for_prefill=*/4,
        /*dp_size=*/1);
    options.enable_chunked_prefill(true)
        .enable_schedule_overlap(false)
        .disable_log_stats(true);
    auto block_manager = std::make_unique<StaleCapacityBlockManagerPool>(
        /*advisory_capacity=*/0);
    StaleCapacityBlockManagerPool* pool = block_manager.get();
    FakeEngine engine(std::move(block_manager));
    TestableContinuousScheduler scheduler(&engine, options);
    const std::vector<size_t> initial_free_blocks = pool->num_free_blocks();
    auto request = generate_request({40},
                                    {4},
                                    std::nullopt,
                                    std::nullopt,
                                    /*max_context_len=*/64)[0];
    std::optional<Status> callback_status;
    request->state().output_func = [&](const RequestOutput& output) {
      callback_status = output.status;
      return true;
    };
    ASSERT_TRUE(scheduler.add_request(request));

    // Ten prompt blocks exceed the configured nine-block arena. This is a
    // permanent size limit, so no physical reservation should be attempted.
    BatchGroup batches = scheduler.prepare_batch();
    scheduler.wait_for_responses();
    ASSERT_EQ(batches.size(), 1u);
    EXPECT_TRUE(batches.front().empty());
    ASSERT_TRUE(callback_status.has_value());
    EXPECT_EQ(callback_status->code(), StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_TRUE(pool->allocate_targets().empty());
    EXPECT_EQ(pool->num_free_blocks(), initial_free_blocks);
    EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
  }
}

TEST(SchedulerPolicyTest, StaticCapacityRejectsBestOfWithoutReservation) {
  ScopedConfigValue<bool> prefix_cache_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  for (bool enable_mix_batch : {false, true}) {
    SCOPED_TRACE(enable_mix_batch);
    ScopedConfigValue<bool> mix_batch_guard(
        SchedulerConfig::get_instance().enable_mix_batch(), enable_mix_batch);
    SchedulerOptions options = create_scheduler_options(
        /*max_tokens_per_batch=*/16,
        /*max_seqs_per_batch=*/2,
        /*num_speculative_tokens=*/0,
        /*max_tokens_per_chunk_for_prefill=*/4,
        /*dp_size=*/1);
    options.enable_chunked_prefill(true)
        .enable_schedule_overlap(false)
        .disable_log_stats(true);
    auto block_manager = std::make_unique<StaleCapacityBlockManagerPool>(
        /*advisory_capacity=*/0);
    StaleCapacityBlockManagerPool* pool = block_manager.get();
    FakeEngine engine(std::move(block_manager));
    TestableContinuousScheduler scheduler(&engine, options);
    const std::vector<size_t> initial_free_blocks = pool->num_free_blocks();
    RequestState request_state =
        generate_request({20}, {4}, std::nullopt, std::nullopt, 64)[0]
            ->state()
            .clone();
    request_state.best_of = 2;
    auto request = std::make_shared<Request>(
        "best-of-static", "best-of-static", "", std::move(request_state), "");
    std::optional<Status> callback_status;
    request->state().output_func = [&](const RequestOutput& output) {
      callback_status = output.status;
      return true;
    };
    ASSERT_TRUE(scheduler.add_request(request));

    BatchGroup batches = scheduler.prepare_batch();
    scheduler.wait_for_responses();
    ASSERT_EQ(batches.size(), 1u);
    EXPECT_TRUE(batches.front().empty());
    ASSERT_TRUE(callback_status.has_value());
    EXPECT_EQ(callback_status->code(), StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_TRUE(pool->allocate_targets().empty());
    EXPECT_EQ(pool->num_free_blocks(), initial_free_blocks);
    EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
  }
}

TEST(SchedulerPolicyTest, StaticCapacityRejectsPromptWithActiveRequest) {
  ScopedConfigValue<bool> prefix_cache_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  for (const auto& [enable_mix_batch, priority_strategy] :
       {std::pair<bool, std::string>{false, "fcfs"},
        {true, "fcfs"},
        {true, "multi_slo_and_prio"}}) {
    SCOPED_TRACE(enable_mix_batch);
    SCOPED_TRACE(priority_strategy);
    ScopedConfigValue<bool> mix_batch_guard(
        SchedulerConfig::get_instance().enable_mix_batch(), enable_mix_batch);
    SchedulerOptions options = create_scheduler_options(
        /*max_tokens_per_batch=*/16,
        /*max_seqs_per_batch=*/2,
        /*num_speculative_tokens=*/0,
        /*max_tokens_per_chunk_for_prefill=*/4,
        /*dp_size=*/1,
        priority_strategy);
    options.enable_chunked_prefill(true)
        .enable_schedule_overlap(false)
        .disable_log_stats(true);
    auto block_manager = std::make_unique<StaleCapacityBlockManagerPool>(
        /*advisory_capacity=*/0);
    StaleCapacityBlockManagerPool* pool = block_manager.get();
    FakeEngine engine(std::move(block_manager));
    TestableContinuousScheduler scheduler(&engine, options);
    auto requests = generate_request({4, 40},
                                     {4, 4},
                                     std::nullopt,
                                     std::vector<int32_t>{3, 1},
                                     /*max_context_len=*/64);
    std::shared_ptr<Request>& active = requests[0];
    std::shared_ptr<Request>& oversized = requests[1];
    ASSERT_TRUE(scheduler.add_request(active));
    BatchGroup batches = scheduler.prepare_batch();
    ASSERT_EQ(batches.size(), 1u);
    ASSERT_EQ(batches.front().size(), 1u);
    Sequence* active_sequence = active->sequences().front().get();
    active_sequence->kv_state().set_kv_cache_tokens_num(4);
    active_sequence->append_token(Token(1));
    const int64_t active_block_id =
        active_sequence->kv_state().blocks(BlockType::KV).front().id();
    std::optional<Status> callback_status;
    oversized->state().output_func = [&](const RequestOutput& output) {
      callback_status = output.status;
      return true;
    };
    ASSERT_TRUE(scheduler.add_request(oversized));

    batches = scheduler.prepare_batch();
    scheduler.wait_for_responses();
    ASSERT_EQ(batches.size(), 1u);
    ASSERT_EQ(batches.front().size(), 1u);
    EXPECT_EQ(batches.front()[0], active_sequence);
    ASSERT_TRUE(callback_status.has_value());
    EXPECT_EQ(callback_status->code(), StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_FALSE(active->preempted());
    EXPECT_EQ(active_sequence->kv_cache_tokens_num(), 4u);
    EXPECT_EQ(active_sequence->kv_state().blocks(BlockType::KV).front().id(),
              active_block_id);
    EXPECT_EQ(std::count(pool->allocate_targets().begin(),
                         pool->allocate_targets().end(),
                         40u),
              0);
    EXPECT_EQ(scheduler.get_waiting_requests_num(), 0u);
  }
}

TEST(SchedulerPolicyTest, BestOfAllocationFailureKeepsSuccessfulSequence) {
  ScopedConfigValue<bool> prefix_cache_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<bool> mix_batch_guard(
      SchedulerConfig::get_instance().enable_mix_batch(), false);
  SchedulerOptions options = create_scheduler_options(
      /*max_tokens_per_batch=*/16,
      /*max_seqs_per_batch=*/2,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/4,
      /*dp_size=*/1);
  options.enable_chunked_prefill(true)
      .enable_schedule_overlap(false)
      .disable_log_stats(true);
  auto block_manager =
      std::make_unique<FailOnAllocationBlockManagerPool>(/*fail_on_attempt=*/2);
  FailOnAllocationBlockManagerPool* pool = block_manager.get();
  FakeEngine engine(std::move(block_manager));
  TestableContinuousScheduler scheduler(&engine, options);
  const std::vector<size_t> initial_free_blocks = pool->num_free_blocks();
  RequestState request_state =
      generate_request({12}, {4}, std::nullopt, std::nullopt, 64)[0]
          ->state()
          .clone();
  request_state.best_of = 2;
  auto request = std::make_shared<Request>(
      "best-of-fresh", "best-of-fresh", "", std::move(request_state), "");
  ASSERT_TRUE(scheduler.add_request(request));

  BatchGroup batches = scheduler.prepare_batch();
  scheduler.wait_for_responses();
  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches.front().size(), 1u);
  ASSERT_EQ(request->sequences().size(), 2u);
  Sequence* first = request->sequences().front().get();
  Sequence* second = request->sequences().back().get();
  EXPECT_EQ(batches.front()[0], first);
  EXPECT_EQ(first->kv_state().num_blocks(BlockType::KV), 1u);
  EXPECT_EQ(first->kv_cache_tokens_num(), 0u);
  EXPECT_EQ(second->kv_state().num_blocks(BlockType::KV), 0u);
  EXPECT_EQ(second->kv_cache_tokens_num(), 0u);
  EXPECT_EQ(pool->allocation_attempts(), 2u);
  EXPECT_EQ(pool->num_free_blocks().front(), initial_free_blocks.front() - 1);
  const int64_t first_block_id =
      first->kv_state().blocks(BlockType::KV).front().id();

  first->kv_state().set_kv_cache_tokens_num(4);
  batches = scheduler.prepare_batch();
  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches.front().size(), 2u);
  EXPECT_EQ(batches.front()[0], first);
  EXPECT_EQ(batches.front()[1], second);
  EXPECT_EQ(first->kv_state().blocks(BlockType::KV).front().id(),
            first_block_id);
  EXPECT_EQ(first->kv_cache_tokens_num(), 4u);
  EXPECT_EQ(first->kv_state().num_blocks(BlockType::KV), 2u);
  EXPECT_EQ(second->kv_cache_tokens_num(), 0u);
  EXPECT_EQ(second->kv_state().num_blocks(BlockType::KV), 1u);
  EXPECT_EQ(pool->allocation_attempts(), 4u);
  pool->deallocate(request.get());
  EXPECT_EQ(pool->num_free_blocks(), initial_free_blocks);
}

TEST(SchedulerPolicyTest, BestOfAllocationFailurePreservesExistingChunk) {
  ScopedConfigValue<bool> prefix_cache_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<bool> mix_batch_guard(
      SchedulerConfig::get_instance().enable_mix_batch(), false);
  SchedulerOptions options = create_scheduler_options(
      /*max_tokens_per_batch=*/16,
      /*max_seqs_per_batch=*/2,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/4,
      /*dp_size=*/1);
  options.enable_chunked_prefill(true)
      .enable_schedule_overlap(false)
      .disable_log_stats(true);
  auto block_manager =
      std::make_unique<FailOnAllocationBlockManagerPool>(/*fail_on_attempt=*/2);
  FailOnAllocationBlockManagerPool* pool = block_manager.get();
  FakeEngine engine(std::move(block_manager));
  TestableContinuousScheduler scheduler(&engine, options);
  const std::vector<size_t> initial_free_blocks = pool->num_free_blocks();
  RequestState request_state =
      generate_request({12}, {4}, std::nullopt, std::nullopt, 64)[0]
          ->state()
          .clone();
  request_state.best_of = 2;
  auto request = std::make_shared<Request>(
      "best-of-existing", "best-of-existing", "", std::move(request_state), "");
  Sequence* first = request->sequences().front().get();
  ASSERT_TRUE(pool->BlockManagerPool::allocate(first, /*num_tokens=*/4));
  first->kv_state().set_kv_cache_tokens_num(4);
  const int64_t first_block_id =
      first->kv_state().blocks(BlockType::KV).front().id();
  const size_t free_blocks_before = pool->num_free_blocks().front();
  ASSERT_EQ(first->kv_state().num_blocks(BlockType::KV), 1u);
  ASSERT_TRUE(scheduler.add_request(request));

  BatchGroup batches = scheduler.prepare_batch();
  scheduler.wait_for_responses();
  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches.front().size(), 1u);
  EXPECT_EQ(batches.front()[0], first);
  ASSERT_EQ(request->sequences().size(), 2u);
  EXPECT_EQ(first->kv_state().blocks(BlockType::KV).front().id(),
            first_block_id);
  EXPECT_EQ(first->kv_state().kv_cache_tokens_num(), 4u);
  EXPECT_EQ(first->kv_state().num_blocks(BlockType::KV), 2u);
  EXPECT_EQ(request->sequences().back()->kv_state().num_blocks(BlockType::KV),
            0u);
  EXPECT_EQ(pool->num_free_blocks().front(), free_blocks_before - 1);
  EXPECT_EQ(pool->allocation_attempts(), 2u);
  pool->deallocate(request.get());
  EXPECT_EQ(pool->num_free_blocks(), initial_free_blocks);
}

TEST(SchedulerPolicyTest, BestOfPartialAdmissionReleasesPrefixUsage) {
  ScopedConfigValue<bool> prefix_cache_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), true);
  ScopedConfigValue<bool> in_batch_prefix_cache_guard(
      KVCacheConfig::get_instance().enable_in_batch_prefix_cache(), false);
  ScopedConfigValue<bool> mix_batch_guard(
      SchedulerConfig::get_instance().enable_mix_batch(), false);
  SchedulerOptions options = create_scheduler_options(
      /*max_tokens_per_batch=*/16,
      /*max_seqs_per_batch=*/2,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/4,
      /*dp_size=*/1);
  options.enable_chunked_prefill(true)
      .enable_schedule_overlap(false)
      .disable_log_stats(true);
  auto block_manager = std::make_unique<FailOnAllocationBlockManagerPool>(
      /*fail_on_attempt=*/2, /*enable_prefix_cache=*/true);
  FailOnAllocationBlockManagerPool* pool = block_manager.get();
  FakeEngine engine(std::move(block_manager));

  auto cached_request =
      generate_request({4}, {1}, std::nullopt, std::nullopt, 64)[0];
  Sequence* cached_sequence = cached_request->sequences().front().get();
  ASSERT_TRUE(pool->BlockManagerPool::allocate(cached_sequence,
                                               /*num_tokens=*/4));
  cached_sequence->kv_state().set_kv_cache_tokens_num(4);
  pool->publish_prefix(cached_sequence);
  pool->deallocate(cached_request.get());

  TestableContinuousScheduler scheduler(&engine, options);
  RequestState request_state =
      generate_request({12}, {4}, std::nullopt, std::nullopt, 64)[0]
          ->state()
          .clone();
  request_state.best_of = 2;
  auto request = std::make_shared<Request>(
      "best-of-prefix", "best-of-prefix", "", std::move(request_state), "");
  ASSERT_TRUE(request->expand_sequences(/*share_prefix=*/false));
  ASSERT_EQ(request->sequences().size(), 2u);
  ASSERT_TRUE(scheduler.add_request(request));

  const std::vector<size_t> used_blocks_before = pool->num_used_blocks();
  BatchGroup batches = scheduler.prepare_batch();
  scheduler.wait_for_responses();
  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches.front().size(), 1u);
  Sequence* first = request->sequences().front().get();
  Sequence* second = request->sequences().back().get();
  EXPECT_EQ(batches.front()[0], first);
  EXPECT_EQ(first->kv_cache_tokens_num(), 4u);
  EXPECT_EQ(first->kv_state().num_blocks(BlockType::KV), 2u);
  EXPECT_EQ(second->kv_cache_tokens_num(), 4u);
  EXPECT_EQ(second->kv_state().num_blocks(BlockType::KV), 1u);
  EXPECT_EQ(first->kv_state().blocks(BlockType::KV).front().id(),
            second->kv_state().blocks(BlockType::KV).front().id());
  const int64_t first_private_block_id =
      first->kv_state().blocks(BlockType::KV).back().id();

  first->kv_state().set_kv_cache_tokens_num(8);
  batches = scheduler.prepare_batch();
  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches.front().size(), 2u);
  EXPECT_EQ(batches.front()[0], first);
  EXPECT_EQ(batches.front()[1], second);
  EXPECT_EQ(first->kv_state().blocks(BlockType::KV)[1].id(),
            first_private_block_id);
  EXPECT_EQ(first->kv_cache_tokens_num(), 8u);
  EXPECT_GE(second->kv_state().num_blocks(BlockType::KV), 2u);
  EXPECT_EQ(pool->allocation_attempts(), 4u);

  request->set_cancel();
  batches = scheduler.prepare_batch();
  scheduler.wait_for_responses();
  ASSERT_EQ(batches.size(), 1u);
  EXPECT_TRUE(batches.front().empty());
  EXPECT_EQ(first->kv_state().num_blocks(BlockType::KV), 0u);
  EXPECT_EQ(second->kv_state().num_blocks(BlockType::KV), 0u);
  EXPECT_EQ(pool->num_used_blocks(), used_blocks_before);
}

TEST(SchedulerPolicyTest,
     QueuedCancelledDecodeReleasesResourcesBeforeSchedulingLiveFollower) {
  for (const bool enable_mix_batch : {false, true}) {
    SCOPED_TRACE(enable_mix_batch);
    for (const bool enable_linear_attention : {false, true}) {
      SCOPED_TRACE(enable_linear_attention);
      SchedulerOptions options = create_scheduler_options(
          /*max_tokens_per_batch=*/1,
          /*max_seqs_per_batch=*/1,
          /*num_speculative_tokens=*/0,
          /*max_tokens_per_chunk_for_prefill=*/4,
          /*dp_size=*/1,
          /*priority_strategy=*/"priority");
      BatchMode mode{
          .enable_mix_batch = enable_mix_batch,
          .enable_chunked_prefill = true,
          .priority_strategy = "priority",
      };
      std::unique_ptr<SchedulerPolicy> policy;
      if (enable_mix_batch) {
        policy = std::make_unique<DecodeFirstPolicy>(mode, options);
      } else {
        policy = std::make_unique<PrefillFirstPolicy>(mode, options);
      }
      FakeEngine engine(/*num_blocks=*/16,
                        /*block_size=*/4,
                        /*enable_prefix_cache=*/false,
                        /*dp_size=*/1,
                        enable_linear_attention);
      BlockManagerPool* pool = engine.block_manager_pool();
      const std::vector<size_t> initial_free_blocks = pool->num_free_blocks();
      std::vector<std::shared_ptr<Request>> requests = generate_request(
          {7, 7},
          {8, 8},
          std::nullopt,
          std::vector<int32_t>{static_cast<int32_t>(RequestPriority::LOW),
                               static_cast<int32_t>(RequestPriority::HIGH)},
          /*max_context_len=*/64);
      const std::shared_ptr<Request>& cancelled = requests[0];
      const std::shared_ptr<Request>& live = requests[1];
      for (const std::shared_ptr<Request>& request : requests) {
        Sequence* sequence = request->sequences().front().get();
        ASSERT_TRUE(pool->try_allocate(sequence));
        sequence->append_token(Token(1));
        ASSERT_FALSE(sequence->is_prefill_stage());
        ASSERT_GT(sequence->kv_state().num_blocks(BlockType::KV), 0u);
        ASSERT_EQ(sequence->has_linear_state_slot(), enable_linear_attention);
        ASSERT_FALSE(request->finished());
      }
      Sequence* cancelled_sequence = cancelled->sequences().front().get();
      Sequence* live_sequence = live->sequences().front().get();
      const size_t cancelled_kv_blocks =
          cancelled_sequence->kv_state().num_blocks(BlockType::KV);
      const std::vector<size_t> allocated_free_blocks = pool->num_free_blocks();
      ASSERT_EQ(allocated_free_blocks.size(), 1u);

      DequeQueue prefill_queue;
      DequeQueue chunk_queue;
      SetQueue decode_queue(create_comparator("priority", /*reverse=*/true));
      decode_queue.push(live);
      decode_queue.push(cancelled);
      ASSERT_EQ(decode_queue.top(), live);
      cancelled->set_cancel();
      std::list<std::shared_ptr<Request>> unified_queue;
      std::deque<DecodeRestoreEntry> decode_restore_waiting;
      std::vector<std::shared_ptr<Request>> running_requests;
      std::vector<Sequence*> running_sequences;
      std::vector<size_t> running_sequence_budgets;
      running_requests.reserve(1);
      running_sequences.reserve(1);
      running_sequence_budgets.reserve(1);
      bool last_step_prefill = false;
      int32_t released_failed_requests = 0;
      SchedulerState state{
          .prefill_queue = prefill_queue,
          .chunk_queue = chunk_queue,
          .decode_queue = decode_queue,
          .unified_queue = unified_queue,
          .decode_restore_waiting = decode_restore_waiting,
          .running_requests = running_requests,
          .running_sequences = running_sequences,
          .running_sequences_budgets = running_sequence_budgets,
          .kv_cache_manager = pool,
          .profile_manager = nullptr,
          .output_dispatcher = nullptr,
          .model_args = engine.model_args(),
          .last_step_prefill = last_step_prefill,
          .options = options,
          .min_speculative_tokens_required = 0,
          .enable_prefix_cache = false,
          .release_failed_request =
              [&](const std::shared_ptr<Request>& request) {
                EXPECT_EQ(request, cancelled);
                ++released_failed_requests;
              },
      };
      ScheduleBudget budget{
          .remaining_token_budget = 1,
          .remaining_seq_budget = 1,
          .latency_budget = std::numeric_limits<double>::max(),
          .estimate_latency = 0,
          .num_preempted_requests = 0,
      };

      // The cancelled waiter is below the live higher-priority request. It
      // must be retired before admission consumes the only sequence slot.
      std::vector<std::shared_ptr<Request>> finished =
          policy->collect_finished(state);
      EXPECT_EQ(finished, (std::vector<std::shared_ptr<Request>>{cancelled}));
      ASSERT_EQ(decode_queue.size(), 1u);
      policy->schedule(state, budget, finished);

      EXPECT_EQ(finished, (std::vector<std::shared_ptr<Request>>{cancelled}));
      EXPECT_EQ(released_failed_requests, 1);
      EXPECT_EQ(running_requests,
                (std::vector<std::shared_ptr<Request>>{live}));
      EXPECT_EQ(running_sequences, (std::vector<Sequence*>{live_sequence}));
      EXPECT_EQ(running_sequence_budgets, (std::vector<size_t>{1}));
      EXPECT_TRUE(decode_queue.empty());
      EXPECT_TRUE(prefill_queue.empty());
      EXPECT_EQ(budget.remaining_token_budget, 0u);
      EXPECT_EQ(budget.remaining_seq_budget, 0u);
      EXPECT_EQ(budget.num_preempted_requests, 0u);
      EXPECT_EQ(cancelled_sequence->kv_state().num_blocks(BlockType::KV), 0u);
      EXPECT_EQ(cancelled_sequence->kv_state().num_blocks(BlockType::LINEAR),
                0u);
      EXPECT_FALSE(cancelled_sequence->has_linear_state_slot());
      EXPECT_EQ(pool->num_free_blocks().front(),
                allocated_free_blocks.front() + cancelled_kv_blocks);
      EXPECT_GT(live_sequence->kv_state().num_blocks(BlockType::KV), 0u);
      EXPECT_EQ(live_sequence->has_linear_state_slot(),
                enable_linear_attention);

      pool->deallocate(live.get());
      EXPECT_EQ(pool->num_free_blocks(), initial_free_blocks);
    }
  }
}

TEST(SchedulerPolicyTest, FcfsDecodeHandoffAndTerminalCleanup) {
  SchedulerOptions options = create_scheduler_options(
      /*max_tokens_per_batch=*/1024,
      /*max_seqs_per_batch=*/1,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/1024,
      /*dp_size=*/1);
  BatchMode mode{
      .enable_mix_batch = true,
      .enable_chunked_prefill = true,
      .priority_strategy = "fcfs",
  };
  DecodeFirstPolicy policy(mode, options);
  BlockManagerPool::Options pool_options;
  pool_options.num_blocks_ = 64;
  pool_options.block_size_ = 128;
  pool_options.max_seqs_per_batch_ = 1;
  pool_options.enable_prefix_cache_ = true;
  BlockManagerPool block_manager_pool(pool_options, /*dp_size=*/1);

  std::vector<std::shared_ptr<Request>> requests =
      generate_request({128, 128},
                       {8, 8},
                       std::nullopt,
                       std::nullopt,
                       /*max_context_len=*/40000);
  for (const std::shared_ptr<Request>& request : requests) {
    Sequence* sequence = request->sequences().front().get();
    sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
    sequence->append_token(Token(1));
  }
  const std::shared_ptr<Request>& older_waiter = requests[0];
  const std::shared_ptr<Request>& handoff = requests[1];
  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  decode_queue.push(older_waiter, /*if_back=*/true);
  std::list<std::shared_ptr<Request>> unified_queue;
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = nullptr,
      .output_dispatcher = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = true,
  };
  folly::MPMCQueue<std::shared_ptr<Request>> request_queue(4);

  // A handoff with KV already present must not overtake an older waiter.
  ASSERT_TRUE(request_queue.write(handoff));
  policy.drain_request_queue(state, request_queue);
  EXPECT_TRUE(policy.collect_finished(state).empty());
  ASSERT_EQ(decode_queue.size(), 1u);

  ScheduleBudget budget{
      .remaining_token_budget = 1024,
      .remaining_seq_budget = 1,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  std::vector<std::shared_ptr<Request>> finished;
  policy.schedule(state, budget, finished);

  ASSERT_EQ(running_sequences.size(), 1u);
  EXPECT_EQ(running_sequences.front(), older_waiter->sequences().front().get());
  ASSERT_EQ(decode_queue.size(), 1u);
  EXPECT_EQ(decode_queue.top(), handoff);
  EXPECT_TRUE(finished.empty());

  // A terminal arrival and a queued handoff cancelled while idle must be
  // collected even though no budget admits them.
  std::vector<std::shared_ptr<Request>> late_arrivals =
      generate_request({128},
                       {8},
                       std::nullopt,
                       std::nullopt,
                       /*max_context_len=*/40000);
  const std::shared_ptr<Request>& terminal_arrival = late_arrivals.front();
  terminal_arrival->set_cancel();
  handoff->set_cancel();
  ASSERT_TRUE(request_queue.write(terminal_arrival));
  policy.drain_request_queue(state, request_queue);
  EXPECT_TRUE(prefill_queue.empty());
  finished = policy.collect_finished(state);

  EXPECT_EQ(finished.size(), 2u);
  EXPECT_TRUE(decode_queue.empty());
  ASSERT_FALSE(running_requests.empty());
  EXPECT_EQ(running_requests.front(), older_waiter);
}

}  // namespace xllm
