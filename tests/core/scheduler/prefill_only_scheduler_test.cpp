/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "prefill_only_scheduler.h"

#include <gtest/gtest.h>

#include <limits>
#include <optional>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"
#include "distributed_runtime/engine.h"
#include "framework/block/block.h"
#include "util/utils.h"

namespace xllm {
namespace {

class FakeTokenizer : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true) const {
    NOT_IMPLEMENTED();
  }
  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const {
    NOT_IMPLEMENTED();
  }
  std::optional<int32_t> token_to_id(const std::string_view& token) const {
    NOT_IMPLEMENTED();
  }
  std::string id_to_token(int32_t id) const { NOT_IMPLEMENTED(); }
  size_t vocab_size() const { NOT_IMPLEMENTED(); }
  std::unique_ptr<Tokenizer> clone() const {
    return std::make_unique<FakeTokenizer>();
  }
};

class FakeEngine : public Engine {
 public:
  FakeEngine(int32_t num_blocks, int32_t block_size, bool enable_prefix_cache) {
    BlockManagerPool::Options opt;
    opt.num_blocks_ = num_blocks;
    opt.block_size_ = block_size;
    opt.enable_prefix_cache_ = enable_prefix_cache;
    fake_tokenizer_ = std::make_unique<FakeTokenizer>();
    fake_block_manager_ = std::make_unique<BlockManagerPool>(opt, 1);
  }
  ForwardOutput step(std::vector<Batch>& batch) { NOT_IMPLEMENTED(); }
  void update_last_step_result(std::vector<Batch>& batch) { NOT_IMPLEMENTED(); }
  const Tokenizer* tokenizer() const { return fake_tokenizer_.get(); }
  BlockManagerPool* block_manager_pool() const {
    return fake_block_manager_.get();
  }
  const ModelArgs& model_args() const { NOT_IMPLEMENTED(); }
  const TokenizerArgs& tokenizer_args() const { NOT_IMPLEMENTED(); }
  std::vector<int64_t> get_active_activation_memory() const {
    NOT_IMPLEMENTED();
  }
  bool init() override { return true; }

 private:
  std::unique_ptr<Tokenizer> fake_tokenizer_;
  std::unique_ptr<BlockManagerPool> fake_block_manager_;
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

ContinuousScheduler::Options create_scheduler_options(
    int32_t max_tokens_per_batch,
    int32_t max_seqs_per_batch,
    int32_t num_speculative_tokens,
    int32_t max_tokens_per_chunk_for_prefill,
    int32_t dp_size) {
  ContinuousScheduler::Options opt;
  opt.num_speculative_tokens_ = num_speculative_tokens;
  opt.max_tokens_per_chunk_for_prefill_ = max_tokens_per_chunk_for_prefill;
  opt.max_tokens_per_batch_ = max_tokens_per_batch;
  opt.max_seqs_per_batch_ = max_seqs_per_batch;
  opt.dp_size_ = dp_size;
  opt.priority_strategy_ = "fcfs";
  opt.enable_profile_kv_blocks_ = true;
  opt.enable_latency_aware_schedule_ = false;
  opt.max_global_ttft_ms_ = std::numeric_limits<int32_t>::max();
  opt.max_global_tpot_ms_ = std::numeric_limits<int32_t>::max();
  return opt;
}

std::shared_ptr<Request> generate_request_with_prompt_tokens(
    const std::vector<int32_t>& prompt_token_ids,
    int32_t max_tokens,
    int32_t max_context_len) {
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(max_tokens);
  stopping_checker.set_max_context_len(max_context_len);
  stopping_checker.set_ignore_eos(true);

  RequestState req_state("x",
                         prompt_token_ids,
                         sampling_param,
                         scheduler_param,
                         stopping_checker,
                         prompt_token_ids.size() + 30000,
                         1,
                         1,
                         false,
                         false,
                         false,
                         false,
                         false,
                         nullptr,
                         nullptr);

  return std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
}

// Seed the prefix cache with `prefix_token_ids` by allocating a sequence,
// advancing its kv cache position over the full prefix and deallocating it so
// its full blocks are published to the prefix cache.
void seed_prefix_cache(BlockManagerPool* block_manager_pool,
                       const std::vector<int32_t>& prefix_token_ids) {
  auto cached_request = generate_request_with_prompt_tokens(
      prefix_token_ids, /*max_tokens=*/1, /*max_context_len=*/30000);
  Sequence* cached_sequence = cached_request->sequences()[0].get();
  ASSERT_TRUE(block_manager_pool->allocate(cached_sequence,
                                           cached_sequence->num_tokens()));
  cached_sequence->kv_state().set_kv_cache_tokens_num(
      cached_sequence->num_tokens());
  block_manager_pool->deallocate(cached_sequence);
}

}  // namespace

// A newly scheduled prefill sequence that shares a long cached prefix must end
// up with enough kv capacity to cover its matched prefix plus the tokens it is
// asked to compute this step, even when the per-step token budget is clamped to
// fewer tokens than the prefix length. This is the invariant enforced by
// batch_input_builder.cpp (current_max_tokens_capacity >= kv + q_seq_len).
TEST(PrefillOnlySchedulerTest,
     PrefixHitUnderClampedBudgetKeepsCapacitySufficient) {
  constexpr int32_t kBlockSize = 8;
  constexpr int32_t kNumBlocks = 128;
  // 12 full blocks worth of shared prefix (96 tokens).
  constexpr int32_t kPrefixLen = 96;
  // Full prompt shares the whole prefix and appends a small unique tail.
  constexpr int32_t kPromptLen = 120;
  // Per-step token budget deliberately clamped below the prefix length.
  constexpr int32_t kMaxTokensPerBatch = 40;

  ScopedConfigValue<bool> enable_prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), true);
  // Keep the prefill memory-usage guard from blocking the single sequence.
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      2.0);

  auto engine = std::make_unique<FakeEngine>(
      kNumBlocks, kBlockSize, /*enable_prefix_cache=*/true);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();

  ContinuousScheduler::Options opt = create_scheduler_options(
      kMaxTokensPerBatch, 256, /*num_speculative_tokens=*/5, 1024, 1);
  auto scheduler = std::make_unique<PrefillOnlyScheduler>(engine.get(), opt);
  ASSERT_TRUE(scheduler != nullptr);

  std::vector<int32_t> prefix_token_ids;
  prefix_token_ids.reserve(kPrefixLen);
  for (int32_t i = 0; i < kPrefixLen; ++i) {
    prefix_token_ids.emplace_back(i + 1);
  }
  seed_prefix_cache(block_manager_pool, prefix_token_ids);

  std::vector<int32_t> prompt_token_ids = prefix_token_ids;
  prompt_token_ids.reserve(kPromptLen);
  for (int32_t i = kPrefixLen; i < kPromptLen; ++i) {
    prompt_token_ids.emplace_back(i + 1);
  }
  auto request = generate_request_with_prompt_tokens(
      prompt_token_ids, /*max_tokens=*/10, /*max_context_len=*/30000);
  scheduler->add_request(request);

  auto batch = scheduler->prepare_batch_test();
  ASSERT_EQ(batch.size(), 1);
  ASSERT_EQ(batch[0].size(), 1);

  auto running_requests = scheduler->get_running_requests();
  auto budgets = scheduler->get_running_sequences_budgets();
  ASSERT_EQ(running_requests.size(), 1);
  ASSERT_EQ(budgets.size(), 1);

  Sequence* sequence = running_requests[0]->sequences()[0].get();
  const size_t kv_cache_tokens = sequence->kv_state().kv_cache_tokens_num();
  // Precondition of the regression: the cached prefix was matched and it is
  // longer than the clamped per-step budget.
  ASSERT_EQ(kv_cache_tokens, static_cast<size_t>(kPrefixLen));
  ASSERT_LT(static_cast<size_t>(kMaxTokensPerBatch), kv_cache_tokens);

  const size_t q_seq_len =
      std::min(sequence->num_tokens() - kv_cache_tokens, budgets[0]);
  EXPECT_GE(sequence->kv_state().current_max_tokens_capacity(),
            kv_cache_tokens + q_seq_len);

  // Cached prefix blocks stay resident in the prefix-cache table, so leak the
  // engine to skip the block manager's "all blocks freed" teardown check.
  scheduler.reset();
  (void)engine.release();
}

}  // namespace xllm
