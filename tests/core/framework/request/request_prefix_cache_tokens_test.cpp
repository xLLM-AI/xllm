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

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/framework/config/scheduler_config.h"
#include "framework/block/block_manager_pool.h"
#include "framework/request/incremental_decoder.h"
#include "request.h"
#include "request_state.h"

namespace xllm {
namespace {

template <typename T>
class ScopedValue final {
 public:
  ScopedValue(T* target, T value) : target_(target), old_(*target) {
    *target_ = value;
  }
  ~ScopedValue() { *target_ = old_; }

  ScopedValue(const ScopedValue&) = delete;
  ScopedValue& operator=(const ScopedValue&) = delete;

 private:
  T* target_;
  T old_;
};

Sequence MakeSequence(size_t index, const std::vector<int32_t>& prompt_tokens) {
  RequestSamplingParam sampling_param;
  sampling_param.beam_width = 0;
  sampling_param.is_embeddings = false;

  StoppingChecker stopping_checker;

  SequenceParams params;
  params.seq_capacity = prompt_tokens.size() + 8;
  params.echo = false;
  params.skip_special_tokens = true;
  params.streaming = false;
  params.enable_schedule_overlap = false;
  params.rec_type = RecType::kNone;
  params.bos_token_id = 0;
  params.request_id = "cached_tokens_test";
  params.sampling_param = &sampling_param;
  params.stopping_checker = &stopping_checker;

  IncrementalDecoder decoder(
      /*prompt=*/"prompt",
      /*num_prompt_tokens=*/prompt_tokens.size(),
      /*echo=*/params.echo,
      /*skip_special_tokens=*/params.skip_special_tokens);

  return Sequence(index,
                  prompt_tokens,
                  /*input_embedding=*/torch::Tensor(),
                  /*mm_data=*/MMData(),
                  decoder,
                  params);
}

std::shared_ptr<Request> MakeRequest(
    const std::vector<int32_t>& prompt_tokens) {
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  RequestState state(
      "prompt",
      prompt_tokens,
      sampling_param,
      SchedulerParam{},
      stopping_checker,
      /*seq_capacity=*/prompt_tokens.size() + 8,
      /*n=*/1,
      /*best_of=*/1,
      /*logprobs=*/false,
      /*stream=*/false,
      /*echo=*/false,
      /*skip_special_tokens=*/true,
      /*enable_schedule_overlap=*/false,
      [](const RequestOutput&) { return true; },
      OutputsFunc{});

  return std::make_shared<Request>("cached-tokens-test", "", "", state);
}

void EvictPrefixCache(BlockManagerPool& pool,
                      const BlockManagerPool::Options& options) {
  int32_t dp_rank = 0;
  pool.allocate((options.num_blocks() - 1) * options.block_size(), dp_rank);
}

TEST(RequestCachedTokensTest, RecordsCachedTokensFromPrefixCache) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(16).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.max_seqs_per_batch(1024);
  BlockManagerPool pool(options, /*dp_size=*/1);

  {
    // Populate prefix cache via a standalone sequence.
    Sequence seq1 =
        MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9});
    ASSERT_TRUE(pool.allocate(&seq1));
    seq1.kv_state().set_kv_cache_tokens_num(seq1.num_prompt_tokens());
    pool.cache(&seq1);

    // Create a Request whose prompt overlaps with the cached prefix.
    auto request = MakeRequest({1, 2, 3, 4, 5, 6, 7, 8, 10});
    auto* seq = request->sequences()[0].get();
    ASSERT_TRUE(pool.allocate(seq, seq->num_prompt_tokens()));

    EXPECT_EQ(seq->num_prefix_cache_tokens(), 8u);

    request->record_num_prefix_cache_tokens();
    EXPECT_EQ(request->num_prefix_cache_tokens(), 8u);
  }

  EvictPrefixCache(pool, options);
}

TEST(RequestCachedTokensTest, CachedTokensSurvivesKVBlockRelease) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(16).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.max_seqs_per_batch(1024);
  BlockManagerPool pool(options, /*dp_size=*/1);

  std::shared_ptr<Request> request;
  {
    Sequence seq1 =
        MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9});
    ASSERT_TRUE(pool.allocate(&seq1));
    seq1.kv_state().set_kv_cache_tokens_num(seq1.num_prompt_tokens());
    pool.cache(&seq1);

    request = MakeRequest({1, 2, 3, 4, 5, 6, 7, 8, 10});
    auto* seq = request->sequences()[0].get();
    ASSERT_TRUE(pool.allocate(seq, seq->num_prompt_tokens()));

    request->record_num_prefix_cache_tokens();
    EXPECT_EQ(request->num_prefix_cache_tokens(), 8u);

    // Simulate KV block release (preemption or finish).
    pool.deallocate(seq);
    EXPECT_EQ(seq->num_prefix_cache_tokens(), 0u);

    // The recorded value on Request must survive the release.
    EXPECT_EQ(request->num_prefix_cache_tokens(), 8u);
  }

  EvictPrefixCache(pool, options);
}

TEST(RequestCachedTokensTest, ZeroWhenPrefixCacheDisabled) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(16).host_num_blocks(0).block_size(4).enable_prefix_cache(
      false);
  BlockManagerPool pool(options, /*dp_size=*/1);

  auto request = MakeRequest({1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto* seq = request->sequences()[0].get();
  ASSERT_TRUE(pool.allocate(seq, seq->num_prompt_tokens()));

  request->record_num_prefix_cache_tokens();
  EXPECT_EQ(request->num_prefix_cache_tokens(), 0u);
}

TEST(RequestCachedTokensTest, RecordIsIdempotentWithMaxSemantics) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(16).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.max_seqs_per_batch(1024);
  BlockManagerPool pool(options, /*dp_size=*/1);

  std::shared_ptr<Request> request;
  {
    Sequence seq1 =
        MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9});
    ASSERT_TRUE(pool.allocate(&seq1));
    seq1.kv_state().set_kv_cache_tokens_num(seq1.num_prompt_tokens());
    pool.cache(&seq1);

    request = MakeRequest({1, 2, 3, 4, 5, 6, 7, 8, 10});
    auto* seq = request->sequences()[0].get();
    ASSERT_TRUE(pool.allocate(seq, seq->num_prompt_tokens()));

    request->record_num_prefix_cache_tokens();
    EXPECT_EQ(request->num_prefix_cache_tokens(), 8u);

    // Release KV blocks, then record again — max semantics preserves the value.
    pool.deallocate(seq);
    request->record_num_prefix_cache_tokens();
    EXPECT_EQ(request->num_prefix_cache_tokens(), 8u);
  }

  EvictPrefixCache(pool, options);
}

}  // namespace
}  // namespace xllm
