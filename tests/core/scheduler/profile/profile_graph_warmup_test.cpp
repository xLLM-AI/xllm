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
#include <string>
#include <vector>

#include "core/framework/model/mtp_utils.h"
#include "core/framework/sampling/sampling_params.h"
#include "framework/block/block_manager_pool.h"
#include "framework/request/incremental_decoder.h"
#include "framework/request/sequence.h"
#include "framework/request/stopping_checker.h"
#include "scheduler/profile/graph_warmup.h"

namespace xllm {
namespace {

Sequence make_sequence(size_t index, const std::vector<int32_t>& tokens) {
  RequestSamplingParam sampling_param;
  sampling_param.beam_width = 0;
  sampling_param.is_embeddings = false;

  StoppingChecker stopping_checker;

  SequenceParams params;
  params.seq_capacity = tokens.size() + 8;
  params.echo = false;
  params.skip_special_tokens = true;
  params.streaming = false;
  params.enable_schedule_overlap = false;
  params.rec_type = RecType::kNone;
  params.bos_token_id = 0;
  params.request_id = "profile_graph_warmup_test";
  params.sampling_param = &sampling_param;
  params.stopping_checker = &stopping_checker;

  IncrementalDecoder decoder(
      /*prompt=*/"prompt",
      /*num_prompt_tokens=*/tokens.size(),
      /*echo=*/params.echo,
      /*skip_special_tokens=*/params.skip_special_tokens);

  return Sequence(index,
                  tokens,
                  /*input_embedding=*/torch::Tensor(),
                  /*mm_data=*/MMData(),
                  decoder,
                  params);
}

TEST(GraphWarmupTest, BuildsCanonicalBuckets) {
  const std::vector<int32_t> buckets = graph_warmup_buckets(64);

  EXPECT_EQ(buckets, (std::vector<int32_t>{1, 2, 4, 8, 16, 32, 48, 64}));
}

TEST(GraphWarmupTest, IncludesNonCanonicalMaxBucket) {
  const std::vector<int32_t> buckets = graph_warmup_buckets(40);

  EXPECT_EQ(buckets, (std::vector<int32_t>{1, 2, 4, 8, 16, 32, 40}));
}

TEST(GraphWarmupTest, SkipsBucketsBelowDpSize) {
  const std::vector<int32_t> buckets =
      graph_decode_buckets(/*max_seqs_per_batch=*/16, /*dp_size=*/4);

  EXPECT_EQ(buckets, (std::vector<int32_t>{4, 8, 16}));
}

TEST(GraphWarmupTest, AllowsAllBucketsSkipped) {
  const std::vector<int32_t> buckets =
      graph_decode_buckets(/*max_seqs_per_batch=*/2, /*dp_size=*/4);

  EXPECT_TRUE(buckets.empty());
}

TEST(GraphWarmupTest, PrefillRoleUsesPrefillOnlyPlan) {
  EXPECT_EQ(graph_warmup_plan(InstanceRole::PREFILL),
            GraphWarmupPlan::PREFILL_ONLY);
}

TEST(GraphWarmupTest, NonPrefillRolesUseUnifiedPlan) {
  EXPECT_EQ(graph_warmup_plan(InstanceRole::DEFAULT), GraphWarmupPlan::UNIFIED);
  EXPECT_EQ(graph_warmup_plan(InstanceRole::MIX), GraphWarmupPlan::UNIFIED);
  EXPECT_EQ(graph_warmup_plan(InstanceRole::INVALID), GraphWarmupPlan::UNIFIED);
}

TEST(GraphWarmupTest, DecodeRoleUsesDecodeOnlyPlan) {
  EXPECT_EQ(graph_warmup_plan(InstanceRole::DECODE),
            GraphWarmupPlan::DECODE_ONLY);
}

TEST(GraphWarmupTest, FormatsWarmupProgress) {
  const std::string progress = graph_warmup_progress(
      /*completed=*/3, /*total=*/8, /*bucket=*/8, /*latency_ms=*/12.5);

  EXPECT_EQ(progress,
            "Graph warmup progress: [########------------] 3/8 37.5%, "
            "bucket=8, latency=12.50 ms");
}

TEST(GraphWarmupTest, FormatsCompletedWarmupProgress) {
  const std::string progress = graph_warmup_progress(
      /*completed=*/8, /*total=*/8, /*bucket=*/64, /*latency_ms=*/100.0);

  EXPECT_EQ(progress,
            "Graph warmup progress: [####################] 8/8 100.0%, "
            "bucket=64, latency=100.00 ms");
}

TEST(GraphWarmupTest, InjectsBootstrapEmbeddingWhenSpeculativeEnabled) {
  Sequence sequence = make_sequence(/*index=*/0, /*tokens=*/{1, 2, 3});

  prepare_warmup_decode_sequence(&sequence,
                                 /*embedding_width=*/128,
                                 /*num_speculative_tokens=*/3);

  const torch::Tensor embedding = sequence.get_mtp_bootstrap_embedding();
  ASSERT_TRUE(embedding.defined());
  EXPECT_EQ(embedding.dim(), 2);
  EXPECT_EQ(embedding.size(0), 1);
  EXPECT_EQ(embedding.size(1), 128);
}

TEST(GraphWarmupTest, DeepseekV4MtpUsesFlattenedHyperConnectionWidth) {
  ModelArgs model_args;
  model_args.model_type() = "deepseek_v4_mtp";
  model_args.hidden_size() = 128;
  model_args.hc_mult() = 4;

  EXPECT_EQ(mtp_hidden_state_width(model_args), 512);
}

TEST(GraphWarmupTest, OtherMtpModelsUseDenseHiddenWidth) {
  ModelArgs model_args;
  model_args.model_type() = "deepseek_v32_mtp";
  model_args.hidden_size() = 128;
  model_args.hc_mult() = 4;

  EXPECT_EQ(mtp_hidden_state_width(model_args), 128);
}

TEST(GraphWarmupTest, SkipsBootstrapEmbeddingWhenSpeculativeDisabled) {
  Sequence sequence = make_sequence(/*index=*/0, /*tokens=*/{1, 2, 3});

  prepare_warmup_decode_sequence(&sequence,
                                 /*embedding_width=*/128,
                                 /*num_speculative_tokens=*/0);

  EXPECT_FALSE(sequence.get_mtp_bootstrap_embedding().defined());
}

TEST(GraphWarmupTest, ProducesUniqueWarmupRequestIds) {
  const std::string first = next_warmup_request_id();
  const std::string second = next_warmup_request_id();

  EXPECT_FALSE(first.empty());
  EXPECT_FALSE(second.empty());
  EXPECT_NE(first, second);
}

TEST(GraphWarmupTest, PresetDpRankControlsBlockAllocation) {
  BlockManagerPool::Options options;
  options.num_blocks(/*num_blocks=*/8)
      .block_size(/*block_size=*/2)
      .enable_prefix_cache(/*enable_prefix_cache=*/false)
      .max_seqs_per_batch(/*max_seqs_per_batch=*/4);
  BlockManagerPool pool(options, /*dp_size=*/2);
  Sequence sequence = make_sequence(/*index=*/0, /*tokens=*/{1, 2, 3});
  sequence.set_dp_rank(/*dp_rank=*/1);
  const std::vector<size_t> used_before = pool.num_used_blocks();

  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/3));

  EXPECT_EQ(sequence.dp_rank(), 1);
  EXPECT_EQ(pool.num_used_blocks()[0], used_before[0]);
  EXPECT_GT(pool.num_used_blocks()[1], used_before[1]);
}

}  // namespace
}  // namespace xllm
