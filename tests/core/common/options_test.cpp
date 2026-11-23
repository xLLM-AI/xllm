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

#include "core/common/options.h"

#include <gtest/gtest.h>

#include "core/common/prefill_policy.h"

namespace xllm {
namespace {

TEST(OptionsTest, SpeculativeTokenCountRemainsNonnegative) {
  Options options;
  EXPECT_EQ(options.num_speculative_tokens(), 0);
  options.num_speculative_tokens(/*value=*/4);
  EXPECT_EQ(options.num_speculative_tokens(), 4);
  options.num_speculative_tokens(/*value=*/-1);
  EXPECT_EQ(options.num_speculative_tokens(), 0);
}

TEST(OptionsTest, ResolvePrefillAutoAndExplicitChunks) {
  Options default_options;
  default_options.resolve_prefill_options();
  EXPECT_EQ(default_options.max_tokens_per_chunk_for_prefill(),
            default_options.max_tokens_per_batch());
  for (const int32_t requested : {-1, -8, 128, 1024}) {
    Options options;
    options.max_tokens_per_batch(512).max_tokens_per_chunk_for_prefill(
        requested);
    options.resolve_prefill_options();
    EXPECT_EQ(options.max_tokens_per_chunk_for_prefill(),
              requested < 0 ? 512 : requested);
    options.resolve_prefill_options();
    EXPECT_EQ(options.max_tokens_per_chunk_for_prefill(),
              requested < 0 ? 512 : requested);
  }
}

TEST(OptionsTest, ResolvePrefillRejectsZeroChunkAndInvalidBatch) {
  Options options;
  options.max_tokens_per_chunk_for_prefill(0);
  EXPECT_DEATH(options.resolve_prefill_options(),
               "max_tokens_per_chunk_for_prefill");
  options.max_tokens_per_chunk_for_prefill(-1).max_tokens_per_batch(0);
  EXPECT_DEATH(options.resolve_prefill_options(), "max_tokens_per_batch");
}

TEST(OptionsTest, ResolvePrefillPreservesDisabledAndMultimodalEmbeddingModes) {
  Options options;
  options.max_tokens_per_batch(512).enable_chunked_prefill(false);
  options.resolve_prefill_options();
  EXPECT_EQ(options.max_tokens_per_chunk_for_prefill(), 512);
  EXPECT_FALSE(options.enable_chunked_prefill());

  Options embedding;
  embedding.task_type("mm_embed").max_tokens_per_batch(256);
  embedding.resolve_prefill_options();
  EXPECT_FALSE(embedding.enable_chunked_prefill());
  EXPECT_FALSE(embedding.enable_prefix_cache());
  EXPECT_EQ(embedding.max_tokens_per_chunk_for_prefill(), 256);
}

TEST(PrefillPolicyTest, FairBudgetPreservesChunkFloorAndRoleSemantics) {
  const PrefillPolicy chunked(/*max_batch_tokens=*/10,
                              /*requested_chunk_tokens=*/2,
                              /*enable_chunked_prefill=*/true,
                              /*enable_fair_budget=*/true);
  EXPECT_EQ(chunked.per_dp_group(3, InstanceRole::PREFILL), 4);
  EXPECT_EQ(chunked.per_dp_group(1, InstanceRole::PREFILL), 10);
  EXPECT_EQ(chunked.per_dp_group(3, InstanceRole::DECODE), 10);
  EXPECT_EQ(chunked.per_dp_group(3, InstanceRole::MIX), 10);
  const PrefillPolicy unfair(/*max_batch_tokens=*/10,
                             /*requested_chunk_tokens=*/2,
                             /*enable_chunked_prefill=*/true,
                             /*enable_fair_budget=*/false);
  EXPECT_EQ(unfair.per_dp_group(3, InstanceRole::PREFILL), 10);

  const PrefillPolicy large_chunk(/*max_batch_tokens=*/10,
                                  /*requested_chunk_tokens=*/6,
                                  /*enable_chunked_prefill=*/true,
                                  /*enable_fair_budget=*/true);
  EXPECT_EQ(large_chunk.per_dp_group(3, InstanceRole::PREFILL), 6);
  const PrefillPolicy oversized_chunk(/*max_batch_tokens=*/10,
                                      /*requested_chunk_tokens=*/16,
                                      /*enable_chunked_prefill=*/true,
                                      /*enable_fair_budget=*/true);
  EXPECT_EQ(oversized_chunk.chunk_tokens(), 16);
  EXPECT_EQ(oversized_chunk.per_dp_group(3, InstanceRole::PREFILL), 10);
  const PrefillPolicy full_prefill(/*max_batch_tokens=*/10,
                                   /*requested_chunk_tokens=*/2,
                                   /*enable_chunked_prefill=*/false,
                                   /*enable_fair_budget=*/true);
  EXPECT_EQ(full_prefill.per_dp_group(3, InstanceRole::PREFILL), 10);
}

TEST(PrefillPolicyTest, AutomaticChunkAndBudgetStayConsistent) {
  for (const int64_t requested : {-1, -8}) {
    const PrefillPolicy policy(/*max_batch_tokens=*/10,
                               requested,
                               /*enable_chunked_prefill=*/true,
                               /*enable_fair_budget=*/true);
    EXPECT_EQ(policy.chunk_tokens(), 10);
    EXPECT_EQ(policy.per_dp_group(3, InstanceRole::PREFILL), 10);
  }
}

TEST(OptionsTest, ContextParallelDefaultsToOne) {
  Options options;

  EXPECT_EQ(options.cp_size(), 1);
}

TEST(OptionsTest, ContextParallelAcceptsExplicitValues) {
  Options options;
  options.cp_size(4);

  EXPECT_EQ(options.cp_size(), 4);
}

}  // namespace
}  // namespace xllm
