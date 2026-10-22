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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/distributed_runtime/engine.h"
#include "core/framework/block/block_manager_pool.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/request/incremental_decoder.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/mtp_utils.h"
#include "platform/platform.h"
#include "runtime/decode_graph_bucket.h"
#include "scheduler/profile/decode_graph_warmup_plan.h"
#include "scheduler/profile/graph_warmup.h"
#include "scheduler/profile/profile_manager.h"

namespace xllm {
namespace {

class ScopedDecodeGraphConfigSnapshot final {
 public:
  ScopedDecodeGraphConfigSnapshot()
      : enable_graph_(ExecutionConfig::get_instance().enable_graph()),
        enable_no_padding_(ExecutionConfig::get_instance()
                               .enable_graph_mode_decode_no_padding()),
        decode_batch_limit_(ExecutionConfig::get_instance()
                                .acl_graph_decode_batch_size_limit()) {}

  ~ScopedDecodeGraphConfigSnapshot() {
    ExecutionConfig::get_instance()
        .enable_graph(enable_graph_)
        .enable_graph_mode_decode_no_padding(enable_no_padding_)
        .acl_graph_decode_batch_size_limit(decode_batch_limit_);
  }

 private:
  bool enable_graph_;
  bool enable_no_padding_;
  int32_t decode_batch_limit_;
};

class ScopedVmmConfigSnapshot final {
 public:
  ScopedVmmConfigSnapshot()
      : enabled_(KVCacheConfig::get_instance().enable_virtual_memory()),
        elastic_(KVCacheConfig::get_instance().vmm_elastic()) {}

  ~ScopedVmmConfigSnapshot() {
    KVCacheConfig::get_instance().enable_virtual_memory(enabled_).vmm_elastic(
        elastic_);
  }

 private:
  bool enabled_;
  bool elastic_;
};

class ScopedSpeculativeSamplingConfigSnapshot final {
 public:
  ScopedSpeculativeSamplingConfigSnapshot()
      : adaptive_(SpeculativeConfig::get_instance()
                      .enable_adaptive_speculative_decode()),
        algorithm_(SpeculativeConfig::get_instance().speculative_algorithm()) {}

  ~ScopedSpeculativeSamplingConfigSnapshot() {
    SpeculativeConfig::get_instance()
        .enable_adaptive_speculative_decode(adaptive_)
        .speculative_algorithm(algorithm_);
  }

 private:
  bool adaptive_;
  std::string algorithm_;
};

TEST(SpeculativeValidateProfileGateTest, AdaptiveAndTaskPipelineRemainOptIn) {
  SpeculativeConfig speculative;
  const ProfileManager::Options options;

  EXPECT_FALSE(speculative.enable_adaptive_speculative_decode());
  EXPECT_FALSE(options.enable_task_pipeline());
  EXPECT_FALSE(should_profile_legacy_speculative_validate(
      speculative, options.enable_task_pipeline()));

  speculative.speculative_algorithm("DSpark").num_speculative_tokens(7);
  EXPECT_FALSE(should_profile_legacy_speculative_validate(
      speculative, /*enable_task_pipeline=*/true));
  EXPECT_FALSE(should_profile_legacy_speculative_validate(
      speculative, /*enable_task_pipeline=*/false));
}

TEST(SpeculativeValidateProfileGateTest,
     DSparkPipelineUsesOwnGraphBucketCosts) {
  SpeculativeConfig speculative;
  speculative.enable_adaptive_speculative_decode(true).num_speculative_tokens(
      7);
  ProfileManager::Options options;
  options.enable_profile_step_time(true);

  for (const std::string algorithm : {"DSpark", "dspark", "DSPARK", "dSpArK"}) {
    SCOPED_TRACE(algorithm);
    speculative.speculative_algorithm(algorithm);
    options.enable_task_pipeline(true);
    EXPECT_FALSE(should_profile_legacy_speculative_validate(
        speculative, options.enable_task_pipeline()));
    // The separate scheduler prefill/decode profiling flag stays enabled.
    EXPECT_TRUE(options.enable_profile_step_time());

    options.enable_task_pipeline(false);
    EXPECT_TRUE(should_profile_legacy_speculative_validate(
        speculative, options.enable_task_pipeline()));
  }
}

TEST(SpeculativeValidateProfileGateTest, OtherAlgorithmsKeepLegacyRouting) {
  SpeculativeConfig speculative;
  speculative.enable_adaptive_speculative_decode(true).num_speculative_tokens(
      3);
  for (const std::string algorithm : {"MTP", "mtp", "DFlash", "DFlash2"}) {
    speculative.speculative_algorithm(algorithm);
    for (bool enable_task_pipeline : {false, true}) {
      EXPECT_TRUE(should_profile_legacy_speculative_validate(
          speculative, enable_task_pipeline))
          << algorithm;
    }
  }
}

TEST(SpeculativeValidateProfileGateTest, UnsupportedOrNonPrunableSkipsSweep) {
  SpeculativeConfig speculative;
  speculative.enable_adaptive_speculative_decode(true);
  for (const std::string algorithm : {"MTP", "DFlash", "DFlash2", "DSpark"}) {
    speculative.speculative_algorithm(algorithm);
    for (const int32_t num_speculative_tokens : {0, 1}) {
      speculative.num_speculative_tokens(num_speculative_tokens);
      for (bool enable_task_pipeline : {false, true}) {
        EXPECT_FALSE(should_profile_legacy_speculative_validate(
            speculative, enable_task_pipeline))
            << algorithm;
      }
    }
  }
  speculative.num_speculative_tokens(7);
  for (const std::string algorithm : {"Eagle3", "Suffix", "unknown"}) {
    speculative.speculative_algorithm(algorithm);
    for (bool enable_task_pipeline : {false, true}) {
      EXPECT_FALSE(should_profile_legacy_speculative_validate(
          speculative, enable_task_pipeline))
          << algorithm;
    }
  }
}

TEST(DecodeGraphWarmupConfigTest, OrdinaryTargetUsesSingleTokenShape) {
  ExecutionConfig config;
  config.acl_graph_decode_batch_size_limit(7);
  const DecodeGraphWarmupConfig warmup_config =
      config.decode_graph_warmup_config(
          /*num_speculative_tokens=*/0,
          /*enable_no_padding=*/false);

  EXPECT_EQ(warmup_config.num_decoding_tokens, 1);
  EXPECT_EQ(warmup_config.num_speculative_tokens, 0);
  EXPECT_FALSE(warmup_config.enable_graph_mode_decode_no_padding);
  EXPECT_EQ(warmup_config.max_graph_batch_size, Platform::is_npu() ? 7 : 0);
}

TEST(DecodeGraphWarmupConfigTest, PreservesResolvedSpeculativeTargetWidth) {
  ExecutionConfig config;
  config.acl_graph_decode_batch_size_limit(9);
  const DecodeGraphWarmupConfig warmup_config =
      config.decode_graph_warmup_config(
          /*num_speculative_tokens=*/3,
          /*enable_no_padding=*/false);

  EXPECT_EQ(warmup_config.num_decoding_tokens, 4);
  EXPECT_EQ(warmup_config.num_speculative_tokens, 3);
  EXPECT_EQ(warmup_config.max_graph_batch_size, Platform::is_npu() ? 9 : 0);
}

TEST(DecodeGraphWarmupConfigTest, PaddingFollowsTargetOptions) {
  ExecutionConfig config;
  for (bool enable_no_padding : {false, true}) {
    config.enable_graph_mode_decode_no_padding(!enable_no_padding);
    EXPECT_EQ(config
                  .decode_graph_warmup_config(
                      /*num_speculative_tokens=*/3, enable_no_padding)
                  .enable_graph_mode_decode_no_padding,
              enable_no_padding);
  }
}

TEST(StepTimeProfilePlanTest, NeverExceedsConfiguredSequenceCapacity) {
  const std::vector<int32_t> batch_sizes =
      build_step_time_profile_batch_sizes(/*max_seqs_per_batch=*/16);
  ASSERT_FALSE(batch_sizes.empty());
  EXPECT_EQ(batch_sizes.back(), 16);
  EXPECT_TRUE(std::all_of(batch_sizes.begin(),
                          batch_sizes.end(),
                          [](int32_t x) { return x >= 1 && x <= 16; }));
}

TEST(StepTimeProfilePlanTest, BatchSizesCoverBoundaries) {
  // Minimal input yields the single smallest bucket.
  EXPECT_EQ(build_step_time_profile_batch_sizes(/*max_seqs_per_batch=*/1),
            (std::vector<int32_t>{1}));
  // Even tail buckets are padded up to the configured max.
  EXPECT_EQ(build_step_time_profile_batch_sizes(/*max_seqs_per_batch=*/2),
            (std::vector<int32_t>{1, 2}));
  // The legacy upper bound keeps odd buckets only.
  EXPECT_EQ(build_step_time_profile_batch_sizes(/*max_seqs_per_batch=*/23),
            (std::vector<int32_t>{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23}));
  // Inputs beyond the legacy bound are capped by it.
  EXPECT_EQ(build_step_time_profile_batch_sizes(/*max_seqs_per_batch=*/100),
            build_step_time_profile_batch_sizes(/*max_seqs_per_batch=*/23));
}

TEST(StepTimeProfilePlanTest, WarmupDecodeSeqLenFollowsFiaAndContextBounds) {
  // Without FIA the legacy 16-sequence bound applies, capped by the context.
  const bool without_fia = false;
  const bool with_fia = true;
  EXPECT_EQ(warmup_decode_seq_len(without_fia, /*max_context_len=*/8), 8);
  EXPECT_EQ(warmup_decode_seq_len(without_fia, /*max_context_len=*/16), 16);
  EXPECT_EQ(warmup_decode_seq_len(without_fia, /*max_context_len=*/4096), 16);
  // With FIA the bound rises to 2048, still capped by the context length.
  EXPECT_EQ(warmup_decode_seq_len(with_fia, /*max_context_len=*/16), 16);
  EXPECT_EQ(warmup_decode_seq_len(with_fia, /*max_context_len=*/2048), 2048);
  EXPECT_EQ(warmup_decode_seq_len(with_fia, /*max_context_len=*/4096), 2048);
}

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

DecodeGraphWarmupConfig make_decode_graph_warmup_config(
    int64_t num_decoding_tokens,
    int32_t num_speculative_tokens,
    bool enable_no_padding,
    int32_t max_graph_batch_size = 0) {
  DecodeGraphWarmupConfig warmup_config;
  warmup_config.num_decoding_tokens = num_decoding_tokens;
  warmup_config.num_speculative_tokens = num_speculative_tokens;
  warmup_config.enable_graph_mode_decode_no_padding = enable_no_padding;
  warmup_config.max_graph_batch_size = max_graph_batch_size;
  return warmup_config;
}

class StaleCapacityBlockManagerPool final : public BlockManagerPool {
 public:
  using BlockManagerPool::BlockManagerPool;

  std::vector<size_t> num_free_blocks() const override {
    // A capacity sample can still report exhaustion after live KV is freed.
    return std::vector<size_t>(num_used_blocks().size(), 0);
  }
};

class RecordingProfileEngine final : public Engine {
 public:
  explicit RecordingProfileEngine(bool overlap = false,
                                  int32_t dp_size = 1,
                                  bool stale_capacity_snapshot = false)
      : overlap_(overlap) {
    BlockManagerPool::Options options;
    options.num_blocks(/*num_blocks=*/64)
        .block_size(/*block_size=*/4)
        .enable_prefix_cache(/*enable_prefix_cache=*/false)
        .max_seqs_per_batch(/*max_seqs_per_batch=*/8);
    if (stale_capacity_snapshot) {
      block_manager_ =
          std::make_unique<StaleCapacityBlockManagerPool>(options, dp_size);
    } else {
      block_manager_ = std::make_unique<BlockManagerPool>(options, dp_size);
    }
    observed_batches_.reserve(options.max_seqs_per_batch());
    observed_kv_capacities_.reserve(options.max_seqs_per_batch());
    observed_bootstrap_embeddings_.reserve(options.max_seqs_per_batch());
    observed_greedy_samples_.reserve(options.max_seqs_per_batch());
    model_args_.vocab_size(128)
        .eos_token_id(2)
        .max_position_embeddings(16)
        .hidden_size(8);
  }

  ForwardOutput step(BatchGroup& batches) {
    EXPECT_EQ(pending_steps_, 0);
    int32_t sequence_count = 0;
    std::vector<int32_t> dp_ranks;
    for (Batch& batch : batches) {
      for (Sequence* sequence : batch.get_sequences()) {
        ++sequence_count;
        dp_ranks.emplace_back(sequence->dp_rank());
        all_requests_marked_ =
            all_requests_marked_ && sequence->is_graph_warmup();
        observed_kv_capacities_.emplace_back(
            sequence->kv_state().current_max_tokens_capacity());
        observed_bootstrap_embeddings_.emplace_back(
            sequence->get_mtp_bootstrap_embedding().defined());
        // Follow the real request-to-sampler conversion rather than assuming
        // the presence of a per-row sampling tensor implies greedy admission.
        SamplingParameters sampling;
        sampling.init({sequence->sampling_param()}, {0}, {}, {}, {});
        observed_greedy_samples_.emplace_back(sampling.sampling_mode ==
                                                  SamplingMode::GREEDY &&
                                              !sampling.do_sample.defined());
      }
    }
    observed_batches_.emplace_back(sequence_count);
    observed_dp_ranks_.emplace_back(std::move(dp_ranks));
    pending_steps_ += overlap_ ? 1 : 0;
    return ForwardOutput();
  }

  void update_last_step_result(BatchGroup& batches) {
    EXPECT_EQ(pending_steps_, 1);
    const std::vector<size_t> used_blocks = block_manager_->num_used_blocks();
    for (const Batch& batch : batches) {
      for (const Sequence* sequence : batch.get_sequences()) {
        EXPECT_GT(used_blocks[sequence->dp_rank()], 0);
      }
    }
    --pending_steps_;
  }

  BlockManagerPool* block_manager_pool() const override {
    return block_manager_.get();
  }

  const ModelArgs& model_args() const override { return model_args_; }

  void reset_profile_markers() { all_requests_marked_ = true; }

  bool all_requests_marked() const { return all_requests_marked_; }
  int32_t pending_steps() const { return pending_steps_; }
  const std::vector<int32_t>& observed_batches() const {
    return observed_batches_;
  }
  const std::vector<size_t>& observed_kv_capacities() const {
    return observed_kv_capacities_;
  }
  const std::vector<bool>& observed_bootstrap_embeddings() const {
    return observed_bootstrap_embeddings_;
  }
  const std::vector<std::vector<int32_t>>& observed_dp_ranks() const {
    return observed_dp_ranks_;
  }
  const std::vector<bool>& observed_greedy_samples() const {
    return observed_greedy_samples_;
  }

 private:
  std::unique_ptr<BlockManagerPool> block_manager_;
  ModelArgs model_args_;
  bool all_requests_marked_ = true;
  bool overlap_ = false;
  int32_t pending_steps_ = 0;
  std::vector<int32_t> observed_batches_;
  std::vector<size_t> observed_kv_capacities_;
  std::vector<bool> observed_bootstrap_embeddings_;
  std::vector<std::vector<int32_t>> observed_dp_ranks_;
  std::vector<bool> observed_greedy_samples_;
};

TEST(GraphWarmupTest, OnlyAdaptiveDsparkPipelineUsesGreedySyntheticRequests) {
  ScopedDecodeGraphConfigSnapshot execution_snapshot;
  ScopedSpeculativeSamplingConfigSnapshot sampling_snapshot;
  ExecutionConfig::get_instance().enable_graph(false);
  for (const bool adaptive : {false, true}) {
    for (const bool pipeline : {false, true}) {
      for (const std::string algorithm :
           {"DSpark", "dspark", "MTP", "DFlash", "DFlash2"}) {
        SCOPED_TRACE(algorithm);
        SCOPED_TRACE(adaptive);
        SCOPED_TRACE(pipeline);
        SpeculativeConfig::get_instance()
            .enable_adaptive_speculative_decode(adaptive)
            .speculative_algorithm(algorithm);
        RecordingProfileEngine engine;
        ProfileManager::Options options;
        options.max_tokens_per_batch(8)
            .max_seqs_per_batch(2)
            .dp_size(1)
            .enable_task_pipeline(pipeline)
            .instance_role(InstanceRole::DECODE);
        ProfileManager profile_manager(&engine, options);

        // Exercise both synthetic factories, including non-capture profiling
        // requests that still execute through the same worker admission.
        profile_manager.run_request(/*token_length=*/4,
                                    /*prefix_length=*/0,
                                    /*batch_size=*/2,
                                    /*extra_token_length=*/0,
                                    /*is_graph_warmup=*/false);
        profile_manager.profile_decode_step_time(/*token_length=*/5,
                                                 /*batch_size=*/2,
                                                 /*min_context_len=*/5,
                                                 /*max_context_len=*/5);

        const bool expected_greedy =
            adaptive && pipeline &&
            SpeculativeConfig::is_dspark_algorithm(algorithm);
        ASSERT_EQ(engine.observed_greedy_samples().size(), 8);
        for (const bool greedy : engine.observed_greedy_samples()) {
          EXPECT_EQ(greedy, expected_greedy);
        }
      }
    }
  }
  // Serving request defaults are independent of synthetic profile requests.
  EXPECT_FLOAT_EQ(RequestSamplingParam().temperature, 1.0F);
}

TEST(GraphWarmupTest, BuildsCanonicalBuckets) {
  const DecodeGraphWarmupPlan plan = get_compatibility_decode_graph_warmup_plan(
      /*max_global_batch_size=*/64, /*dp_size=*/1);

  EXPECT_EQ(plan.batch_sizes,
            (std::vector<int32_t>{1, 2, 4, 8, 16, 32, 48, 64}));
}

TEST(GraphWarmupTest, IncludesNonCanonicalMaxBucket) {
  const DecodeGraphWarmupPlan plan = get_compatibility_decode_graph_warmup_plan(
      /*max_global_batch_size=*/40, /*dp_size=*/1);

  EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{1, 2, 4, 8, 16, 32, 40}));
}

TEST(GraphWarmupTest, SkipsBucketsBelowDpSize) {
  const DecodeGraphWarmupPlan plan = get_compatibility_decode_graph_warmup_plan(
      /*max_global_batch_size=*/16, /*dp_size=*/4);

  EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{4, 8, 16}));
}

TEST(GraphWarmupTest, AllowsAllBucketsSkipped) {
  const DecodeGraphWarmupPlan plan = get_compatibility_decode_graph_warmup_plan(
      /*max_global_batch_size=*/2, /*dp_size=*/4);

  EXPECT_TRUE(plan.batch_sizes.empty());
}

TEST(DecodeGraphWarmupPlanTest, CoversEveryPaddedMtpTokenBucket) {
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    GTEST_SKIP() << "MTP decode graph warmup is not supported.";
  }

  const DecodeGraphWarmupConfig warmup_config = make_decode_graph_warmup_config(
      /*num_decoding_tokens=*/4,
      /*num_speculative_tokens=*/3,
      /*enable_no_padding=*/false);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/64, /*dp_size=*/1);

  EXPECT_EQ(
      plan.batch_sizes,
      (std::vector<int32_t>{
          1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64}));
}

TEST(DecodeGraphWarmupPlanTest, ResolvesIndependentDraftInvocationShapes) {
  ExecutionConfig execution;
  auto config = execution.decode_graph_warmup_config(3, false, "mTp");
  ASSERT_EQ(config.draft_shapes.size(), 1);
  EXPECT_EQ(config.draft_shapes.front().tokens_per_sequence, 1);
  EXPECT_FALSE(config.draft_shapes.front().bucket_by_sequence);

  execution.enable_task_pipeline(true);
  config = execution.decode_graph_warmup_config(3, false, "MTP");
  ASSERT_EQ(config.draft_shapes.size(), 2);
  EXPECT_EQ(config.draft_shapes[0].tokens_per_sequence, 2);
  EXPECT_EQ(config.draft_shapes[1].tokens_per_sequence, 1);
  config = execution.decode_graph_warmup_config(1, false, "MTP");
  ASSERT_EQ(config.draft_shapes.size(), 1);
  EXPECT_EQ(config.draft_shapes.front().tokens_per_sequence, 2);

  for (const int32_t width : {5, 7}) {
    config = execution.decode_graph_warmup_config(width, false, "dSpArK");
    ASSERT_EQ(config.draft_shapes.size(), 1);
    EXPECT_EQ(config.draft_shapes.front().tokens_per_sequence, width);
    EXPECT_TRUE(config.draft_shapes.front().bucket_by_sequence);
  }
  for (const std::string algorithm : {"DFlash", "DFlash2", "Suffix"}) {
    EXPECT_TRUE(execution.decode_graph_warmup_config(7, false, algorithm)
                    .draft_shapes.empty());
  }
  EXPECT_TRUE(execution.decode_graph_warmup_config(0, false, "MTP")
                  .draft_shapes.empty());
  EXPECT_TRUE(execution.decode_graph_warmup_config(3, true, "MTP")
                  .draft_shapes.empty());
  execution.enable_unified_mtp_graph(true);
  EXPECT_TRUE(execution.decode_graph_warmup_config(7, false, "DSpark")
                  .draft_shapes.empty());
}

TEST(DecodeGraphWarmupPlanTest,
     DsparkCoversDraftAndTargetWithoutRedundantBatch) {
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    GTEST_SKIP() << "Token-bucket graph warmup is not supported.";
  }
  ExecutionConfig execution;
  execution.enable_task_pipeline(true).acl_graph_decode_batch_size_limit(8);
  for (const int32_t width : {5, 7}) {
    const auto config =
        execution.decode_graph_warmup_config(width, false, "DSpark");
    const auto plan = build_decode_graph_warmup_plan(config, 8, 1);
    EXPECT_EQ(plan.batch_sizes,
              width == 5 ? (std::vector<int32_t>{1, 2, 4, 8})
                         : (std::vector<int32_t>{1, 2, 4, 6, 8}));
  }
}

TEST(DecodeGraphWarmupPlanTest, JointCoverageUsesLocalDpGeometryAndGraphLimit) {
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    GTEST_SKIP() << "Token-bucket graph warmup is not supported.";
  }
  for (const std::string algorithm : {"MTP", "DSpark"}) {
    for (const int32_t width : {1, 3, 5, 7}) {
      for (const int32_t dp_size : {1, 2, 4}) {
        for (const int32_t local_capacity : {3, 5, 8, 17}) {
          SCOPED_TRACE(algorithm);
          SCOPED_TRACE(width);
          SCOPED_TRACE(dp_size);
          SCOPED_TRACE(local_capacity);
          ExecutionConfig execution;
          execution.enable_task_pipeline(true)
              .acl_graph_decode_batch_size_limit(8);
          const auto config =
              execution.decode_graph_warmup_config(width, false, algorithm);
          const int32_t global_capacity = local_capacity * dp_size;
          const auto plan =
              build_decode_graph_warmup_plan(config, global_capacity, dp_size);
          const int32_t local_limit = std::min(local_capacity, 8);
          const auto graph_rows =
              [local_limit](int32_t count, int32_t rows, bool by_sequence) {
                return by_sequence
                           ? std::min(runtime::get_decode_graph_token_bucket(
                                          count, false),
                                      static_cast<int64_t>(local_limit)) *
                                 rows
                           : std::min(
                                 runtime::get_decode_graph_token_bucket(
                                     static_cast<int64_t>(count) * rows, false),
                                 static_cast<int64_t>(local_limit) * rows);
              };
          for (int32_t local = 1; local <= local_limit; ++local) {
            const int64_t target_bucket =
                runtime::get_decode_graph_token_bucket(
                    static_cast<int64_t>(local) * (width + 1), false);
            EXPECT_TRUE(std::any_of(
                plan.batch_sizes.begin(),
                plan.batch_sizes.end(),
                [=](int32_t global) {
                  return global <= local_limit * dp_size &&
                         runtime::get_decode_graph_token_bucket(
                             static_cast<int64_t>(global / dp_size) *
                                 (width + 1),
                             false) == target_bucket;
                }));
            for (const auto& shape : config.draft_shapes) {
              const int64_t expected = graph_rows(
                  local, shape.tokens_per_sequence, shape.bucket_by_sequence);
              EXPECT_TRUE(std::any_of(
                  plan.batch_sizes.begin(),
                  plan.batch_sizes.end(),
                  [&](int32_t global) {
                    return global <= local_limit * dp_size &&
                           graph_rows(global / dp_size,
                                      shape.tokens_per_sequence,
                                      shape.bucket_by_sequence) == expected;
                  }));
            }
          }
          EXPECT_EQ(plan.batch_sizes.back(), global_capacity);
          EXPECT_TRUE(
              std::is_sorted(plan.batch_sizes.begin(), plan.batch_sizes.end()));
        }
      }
    }
  }
}

TEST(DecodeGraphBucketTest, UsesSharedWidthForActiveDpShards) {
  EXPECT_EQ(runtime::get_decode_graph_token_bucket(
                /*num_tokens=*/4, /*enable_no_padding=*/false),
            4);
  EXPECT_EQ(runtime::get_decode_graph_token_bucket(
                /*num_tokens=*/20, /*enable_no_padding=*/false),
            32);
  EXPECT_EQ(
      runtime::get_decode_graph_dp_token_counts({4, 0, 0, 0, 0, 0, 0, 0}, 4),
      (std::vector<int32_t>{4, 1, 1, 1, 1, 1, 1, 1}));
  EXPECT_EQ(runtime::get_decode_graph_dp_token_counts(
                {16, 16, 20, 12, 16, 16, 12, 20}, 32),
            (std::vector<int32_t>{32, 32, 32, 32, 32, 32, 32, 32}));
  EXPECT_EQ(runtime::get_decode_graph_dp_layout_token_count(
                /*dp_size=*/8, /*graph_token_count=*/32),
            256);
  EXPECT_EQ(runtime::get_decode_graph_dp_layout_token_count(
                /*dp_size=*/8, /*graph_token_count=*/4),
            32);
}

TEST(DecodeGraphWarmupPlanTest, UsesLocalDpBatchesAndKeepsPartialBatch) {
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    GTEST_SKIP() << "MTP decode graph warmup is not supported.";
  }

  const DecodeGraphWarmupConfig warmup_config = make_decode_graph_warmup_config(
      /*num_decoding_tokens=*/4,
      /*num_speculative_tokens=*/0,
      /*enable_no_padding=*/false);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/66, /*dp_size=*/4);

  EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{4, 8, 16, 32, 48, 64, 66}));
}

TEST(DecodeGraphWarmupPlanTest, Dp32C64UsesLocalGraphBuckets) {
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    GTEST_SKIP() << "MTP decode graph warmup is not supported.";
  }

  const DecodeGraphWarmupConfig warmup_config = make_decode_graph_warmup_config(
      /*num_decoding_tokens=*/4,
      /*num_speculative_tokens=*/3,
      /*enable_no_padding=*/false);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/64, /*dp_size=*/32);

  // The scheduler batch is global, while graph shapes are keyed by the
  // largest local DP batch. C64 therefore needs the local-batch-2 bucket.
  EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{32, 64}));
}

TEST(DecodeGraphWarmupPlanTest,
     GraphLimitKeepsCppBatchSemanticsAndCoversMtpBuckets) {
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    GTEST_SKIP() << "MTP decode graph warmup is not supported.";
  }

  const DecodeGraphWarmupConfig warmup_config = make_decode_graph_warmup_config(
      /*num_decoding_tokens=*/4,
      /*num_speculative_tokens=*/3,
      /*enable_no_padding=*/false,
      /*max_graph_batch_size=*/16);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/64, /*dp_size=*/1);

  EXPECT_EQ(plan.batch_sizes,
            (std::vector<int32_t>{1, 2, 4, 8, 12, 16, 32, 48, 64}));
}

TEST(DecodeGraphWarmupPlanTest,
     Mtp1GraphLimitPreservesCompatibilityBatchSizes) {
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    GTEST_SKIP() << "MTP decode graph warmup is not supported.";
  }

  const DecodeGraphWarmupConfig warmup_config = make_decode_graph_warmup_config(
      /*num_decoding_tokens=*/2,
      /*num_speculative_tokens=*/1,
      /*enable_no_padding=*/false,
      /*max_graph_batch_size=*/16);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/64, /*dp_size=*/1);

  EXPECT_EQ(plan.batch_sizes,
            (std::vector<int32_t>{1, 2, 4, 8, 16, 32, 48, 64}));
}

// The graph batch limit caps the DP-local decode batch, not the
// scheduler-global one, so with dp_size > 1 the sweep must still reach every
// graph-eligible local batch's token bucket.
TEST(DecodeGraphWarmupPlanTest, GraphLimitIsLocalAcrossDpGroups) {
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    GTEST_SKIP() << "MTP decode graph warmup is not supported.";
  }

  const DecodeGraphWarmupConfig warmup_config = make_decode_graph_warmup_config(
      /*num_decoding_tokens=*/6,
      /*num_speculative_tokens=*/5,
      /*enable_no_padding=*/false,
      /*max_graph_batch_size=*/16);
  const int32_t dp_size = 4;
  const int32_t max_global_batch_size = 64;
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, max_global_batch_size, dp_size);

  EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{4, 8, 20, 32, 40, 52, 64}));

  std::vector<int64_t> warmed_buckets;
  for (const int32_t batch_size : plan.batch_sizes) {
    const int64_t num_tokens = static_cast<int64_t>(batch_size / dp_size) *
                               warmup_config.num_decoding_tokens;
    warmed_buckets.push_back(runtime::get_decode_graph_token_bucket(
        num_tokens, /*enable_no_padding=*/false));
  }
  for (int32_t local_batch_size = 1;
       local_batch_size <= std::min(max_global_batch_size / dp_size,
                                    warmup_config.max_graph_batch_size);
       ++local_batch_size) {
    const int64_t num_tokens = static_cast<int64_t>(local_batch_size) *
                               warmup_config.num_decoding_tokens;
    const int64_t bucket = runtime::get_decode_graph_token_bucket(
        num_tokens, /*enable_no_padding=*/false);
    EXPECT_NE(std::find(warmed_buckets.begin(), warmed_buckets.end(), bucket),
              warmed_buckets.end())
        << "local batch " << local_batch_size
        << " maps to unwarmed token bucket " << bucket;
  }
}

TEST(DecodeGraphWarmupPlanTest, NoPaddingKeepsCompatibilityBatches) {
  const DecodeGraphWarmupConfig warmup_config = make_decode_graph_warmup_config(
      /*num_decoding_tokens=*/4,
      /*num_speculative_tokens=*/3,
      /*enable_no_padding=*/true);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/64, /*dp_size=*/1);

  EXPECT_EQ(plan.batch_sizes,
            (std::vector<int32_t>{1, 2, 4, 8, 16, 32, 48, 64}));
}

TEST(DecodeGraphWarmupPlanTest,
     NoPaddingGraphLimitAddsIntermediateMtpTokenBuckets) {
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    GTEST_SKIP() << "MTP decode graph warmup is not supported.";
  }

  const DecodeGraphWarmupConfig warmup_config = make_decode_graph_warmup_config(
      /*num_decoding_tokens=*/4,
      /*num_speculative_tokens=*/3,
      /*enable_no_padding=*/true,
      /*max_graph_batch_size=*/16);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/16, /*dp_size=*/1);

  EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{1, 2, 4, 8, 12, 16}));
}

TEST(DecodeGraphWarmupPlanTest, NoPaddingMtpKeepsNonDivisibleDpTailBuckets) {
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    GTEST_SKIP() << "MTP decode graph warmup is not supported.";
  }

  const DecodeGraphWarmupConfig warmup_config = make_decode_graph_warmup_config(
      /*num_decoding_tokens=*/4,
      /*num_speculative_tokens=*/3,
      /*enable_no_padding=*/true,
      /*max_graph_batch_size=*/16);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/7, /*dp_size=*/2);

  EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{2, 4, 6, 7}));
}

TEST(DecodeGraphWarmupPlanTest, PreservesSuppliedWarmupConfig) {
  const DecodeGraphWarmupConfig warmup_config = make_decode_graph_warmup_config(
      /*num_decoding_tokens=*/4,
      /*num_speculative_tokens=*/3,
      /*enable_no_padding=*/false);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/16, /*dp_size=*/1);

  EXPECT_EQ(plan.warmup_config.num_decoding_tokens, 4);
  EXPECT_EQ(plan.warmup_config.num_speculative_tokens, 3);
  EXPECT_FALSE(plan.warmup_config.enable_graph_mode_decode_no_padding);
  EXPECT_EQ(plan.warmup_config.max_graph_batch_size, 0);
  if (Platform::supports_mtp_decode_graph_warmup()) {
    EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{1, 2, 4, 8, 12, 16}));
  } else {
    EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{1, 2, 4, 8, 16}));
  }
}

TEST(DecodeGraphWarmupPlanTest, RespectsReducedWarmupCapacity) {
  const DecodeGraphWarmupConfig warmup_config = make_decode_graph_warmup_config(
      /*num_decoding_tokens=*/1,
      /*num_speculative_tokens=*/0,
      /*enable_no_padding=*/false);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/15, /*dp_size=*/1);

  EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{1, 2, 4, 8, 15}));
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
      /*completed=*/3, /*total=*/8, /*token_bucket=*/8, /*latency_ms=*/12.5);

  EXPECT_EQ(progress,
            "Graph warmup progress: [########------------] 3/8 37.5%, "
            "token_bucket=8, latency=12.50 ms");
}

TEST(GraphWarmupTest, FormatsCompletedWarmupProgress) {
  const std::string progress = graph_warmup_progress(
      /*completed=*/8,
      /*total=*/8,
      /*token_bucket=*/64,
      /*latency_ms=*/100.0);

  EXPECT_EQ(progress,
            "Graph warmup progress: [####################] 8/8 100.0%, "
            "token_bucket=64, latency=100.00 ms");
}

TEST(GraphWarmupTest, InjectsBootstrapEmbeddingWhenSpeculativeEnabled) {
  Sequence sequence = make_sequence(/*index=*/0, /*tokens=*/{1, 2, 3});

  ModelArgs model_args;
  model_args.hidden_size(128);
  prepare_warmup_decode_sequence(&sequence,
                                 model_args,
                                 /*num_speculative_tokens=*/3);

  const torch::Tensor embedding = sequence.get_mtp_bootstrap_embedding();
  ASSERT_TRUE(embedding.defined());
  EXPECT_EQ(embedding.dim(), 2);
  EXPECT_EQ(embedding.size(0), 1);
  EXPECT_EQ(embedding.size(1), 128);
}

TEST(GraphWarmupTest, BlockDraftBootstrapIncludesEveryCapturedTargetLayer) {
  Sequence sequence = make_sequence(/*index=*/0, /*tokens=*/{1, 2, 3});
  ModelArgs model_args;
  model_args.hidden_size(6144).layers_to_capture({5, 19, 33, 47, 61, 75});

  prepare_warmup_decode_sequence(&sequence,
                                 model_args,
                                 /*num_speculative_tokens=*/7);

  const torch::Tensor embedding = sequence.get_mtp_bootstrap_embedding();
  ASSERT_TRUE(embedding.defined());
  EXPECT_TRUE(embedding.device().is_cpu());
  EXPECT_EQ(embedding.sizes(), (torch::IntArrayRef{1, 36864}));
  EXPECT_EQ(embedding.count_nonzero().item<int64_t>(), 0);
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

  ModelArgs model_args;
  model_args.hidden_size(128);
  prepare_warmup_decode_sequence(&sequence,
                                 model_args,
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

TEST(GraphWarmupTest, MarksOrdinaryProfileRequestsAsSyntheticLoad) {
  RecordingProfileEngine engine;
  ProfileManager::Options options;
  options.max_tokens_per_batch(4).max_seqs_per_batch(1).dp_size(1);
  ProfileManager profile_manager(&engine, options);
  engine.reset_profile_markers();

  profile_manager.run_request(/*token_length=*/4, /*prefix_length=*/0);

  EXPECT_TRUE(engine.all_requests_marked());
}

TEST(GraphWarmupTest, DecodeProfileUsesConfiguredSpeculativeGeometry) {
  ScopedDecodeGraphConfigSnapshot snapshot;
  ExecutionConfig::get_instance().enable_graph(false);
  RecordingProfileEngine engine;
  ProfileManager::Options options;
  options.max_tokens_per_batch(5)
      .max_seqs_per_batch(1)
      .dp_size(1)
      .instance_role(InstanceRole::DECODE)
      .decode_graph_warmup_config(
          ExecutionConfig::get_instance().decode_graph_warmup_config(
              /*num_speculative_tokens=*/3,
              /*enable_no_padding=*/false));
  ProfileManager profile_manager(&engine, options);

  profile_manager.profile_decode_step_time(/*token_length=*/5,
                                           /*batch_size=*/1,
                                           /*min_context_len=*/5,
                                           /*max_context_len=*/5);

  // Five context tokens plus four decode tokens need three four-token blocks.
  EXPECT_EQ(engine.observed_kv_capacities(), (std::vector<size_t>{12, 12, 12}));
  EXPECT_EQ(engine.observed_bootstrap_embeddings(),
            (std::vector<bool>{true, true, true}));
}

TEST(GraphWarmupTest, OrdinaryStepsWarmBucketsAndDrainBeforeReleasingKv) {
#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU)
  auto& config = ExecutionConfig::get_instance();
  const bool previous_graph = config.enable_graph();
  const bool previous_disabled = config.disable_graph_warmup();
  const int32_t previous_limit = config.acl_graph_decode_batch_size_limit();
  config.enable_graph(true);
  config.disable_graph_warmup(false);
  config.acl_graph_decode_batch_size_limit(4);
  for (bool overlap : {false, true}) {
    RecordingProfileEngine engine(overlap);
    const auto used_blocks = engine.block_manager_pool()->num_used_blocks();
    ProfileManager::Options options;
    options.max_tokens_per_batch(4)
        .max_seqs_per_batch(4)
        .dp_size(1)
        .enable_schedule_overlap(overlap)
        .enable_profile_step_time(false)
        .enable_profile_kv_blocks(false)
        .instance_role(InstanceRole::DECODE);
    ProfileManager profile_manager(&engine, options);
    EXPECT_EQ(engine.observed_batches(), (std::vector<int32_t>{4, 2, 1}));
    EXPECT_EQ(engine.pending_steps(), 0);
    EXPECT_EQ(engine.block_manager_pool()->num_used_blocks(), used_blocks);
    EXPECT_TRUE(engine.all_requests_marked());
  }
  config.enable_graph(previous_graph);
  config.disable_graph_warmup(previous_disabled);
  config.acl_graph_decode_batch_size_limit(previous_limit);
#else
  GTEST_SKIP() << "Startup graph warmup is disabled on this platform.";
#endif
}

TEST(GraphWarmupTest, ElasticEagerWarmupExercisesEveryDpAndDrainsOverlap) {
#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU)
  ScopedDecodeGraphConfigSnapshot execution_snapshot;
  ScopedVmmConfigSnapshot cache_snapshot;
  ExecutionConfig::get_instance().enable_graph(false);
  KVCacheConfig::get_instance().enable_virtual_memory(true).vmm_elastic(true);
  RecordingProfileEngine engine(/*overlap=*/true, /*dp_size=*/2);
  const auto used_blocks = engine.block_manager_pool()->num_used_blocks();
  ProfileManager::Options options;
  options.max_tokens_per_batch(20)
      .max_seqs_per_batch(4)
      .dp_size(2)
      .enable_schedule_overlap(true)
      .enable_profile_step_time(false)
      .instance_role(InstanceRole::DEFAULT);

  ProfileManager profile_manager(&engine, options);

  EXPECT_EQ(engine.observed_batches(),
            (std::vector<int32_t>{2, 4, 4, 2, 4, 4}));
  EXPECT_EQ(engine.observed_dp_ranks(),
            (std::vector<std::vector<int32_t>>{{0, 0},
                                               {0, 0, 0, 0},
                                               {0, 0, 0, 0},
                                               {1, 1},
                                               {1, 1, 1, 1},
                                               {1, 1, 1, 1}}));
  EXPECT_EQ(engine.pending_steps(), 0);
  EXPECT_EQ(engine.block_manager_pool()->num_used_blocks(), used_blocks);
  EXPECT_TRUE(engine.all_requests_marked());
#else
  GTEST_SKIP() << "Startup warmup requires a device execution platform.";
#endif
}

TEST(GraphWarmupTest, ElasticDecodeOnlyEagerWarmupIsNotSkipped) {
#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU)
  ScopedDecodeGraphConfigSnapshot execution_snapshot;
  ScopedVmmConfigSnapshot cache_snapshot;
  ExecutionConfig::get_instance().enable_graph(false);
  KVCacheConfig::get_instance().enable_virtual_memory(true).vmm_elastic(true);
  RecordingProfileEngine engine;
  ProfileManager::Options options;
  options.max_tokens_per_batch(20).max_seqs_per_batch(4).instance_role(
      InstanceRole::DECODE);

  ProfileManager profile_manager(&engine, options);

  EXPECT_EQ(engine.observed_batches(), (std::vector<int32_t>{4}));
  EXPECT_TRUE(engine.all_requests_marked());
#else
  GTEST_SKIP() << "Startup warmup requires a device execution platform.";
#endif
}

TEST(GraphWarmupTest, ElasticWarmupAllocatesDespiteStaleCapacitySnapshot) {
#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU)
  ScopedDecodeGraphConfigSnapshot execution_snapshot;
  ScopedVmmConfigSnapshot cache_snapshot;
  ExecutionConfig::get_instance().enable_graph(false);
  KVCacheConfig::get_instance().enable_virtual_memory(true).vmm_elastic(true);
  for (bool overlap : {false, true}) {
    RecordingProfileEngine engine(overlap,
                                  /*dp_size=*/2,
                                  /*stale_capacity_snapshot=*/true);
    const std::vector<size_t> used_before =
        engine.block_manager_pool()->num_used_blocks();
    ASSERT_EQ(engine.block_manager_pool()->num_free_blocks(),
              (std::vector<size_t>{0, 0}));
    ProfileManager::Options options;
    options.max_tokens_per_batch(20)
        .max_seqs_per_batch(4)
        .dp_size(2)
        .enable_schedule_overlap(overlap)
        .enable_profile_step_time(false)
        .instance_role(InstanceRole::DEFAULT);

    ProfileManager profile_manager(&engine, options);

    EXPECT_EQ(engine.observed_batches(),
              (std::vector<int32_t>{2, 4, 4, 2, 4, 4}));
    EXPECT_EQ(engine.pending_steps(), 0);
    EXPECT_EQ(engine.block_manager_pool()->num_used_blocks(), used_before);
    EXPECT_EQ(engine.block_manager_pool()->num_free_blocks(),
              (std::vector<size_t>{0, 0}));
    EXPECT_TRUE(engine.all_requests_marked());
  }
#else
  GTEST_SKIP() << "Startup warmup requires a device execution platform.";
#endif
}

TEST(GraphWarmupTest, ElasticGraphWarmupCannotBeDeferredToServing) {
#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU)
  ScopedDecodeGraphConfigSnapshot execution_snapshot;
  ScopedVmmConfigSnapshot cache_snapshot;
  auto& config = ExecutionConfig::get_instance();
  const bool previous_disabled = config.disable_graph_warmup();
  config.enable_graph(true).disable_graph_warmup(true);
  KVCacheConfig::get_instance().enable_virtual_memory(true).vmm_elastic(true);
  EXPECT_DEATH(
      {
        RecordingProfileEngine engine;
        ProfileManager::Options options;
        options.max_tokens_per_batch(4).max_seqs_per_batch(4);
        ProfileManager profile_manager(&engine, options);
      },
      "Elastic VMM requires startup graph warmup");
  config.disable_graph_warmup(previous_disabled);
#else
  GTEST_SKIP() << "Startup warmup requires a device execution platform.";
#endif
}

}  // namespace
}  // namespace xllm
