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
#include <tuple>
#include <utility>
#include <vector>

#include "core/common/options.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/request/incremental_decoder.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/kv_cache/manager/block_manager_pool.h"
#include "core/scheduler/scheduler_callbacks.h"
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
        disable_graph_warmup_(
            ExecutionConfig::get_instance().disable_graph_warmup()),
        enable_no_padding_(ExecutionConfig::get_instance()
                               .enable_graph_mode_decode_no_padding()),
        decode_batch_limit_(ExecutionConfig::get_instance()
                                .acl_graph_decode_batch_size_limit()) {}

  ~ScopedDecodeGraphConfigSnapshot() {
    ExecutionConfig::get_instance()
        .enable_graph(enable_graph_)
        .disable_graph_warmup(disable_graph_warmup_)
        .enable_graph_mode_decode_no_padding(enable_no_padding_)
        .acl_graph_decode_batch_size_limit(decode_batch_limit_);
  }

 private:
  bool enable_graph_;
  bool disable_graph_warmup_;
  bool enable_no_padding_;
  int32_t decode_batch_limit_;
};

class ScopedVmmConfigSnapshot final {
 public:
  ScopedVmmConfigSnapshot()
      : enabled_(KVCacheConfig::get_instance().enable_vmm_kvcache()),
        elastic_(KVCacheConfig::get_instance().vmm_elastic()) {}

  ~ScopedVmmConfigSnapshot() {
    KVCacheConfig::get_instance().enable_vmm_kvcache(enabled_).vmm_elastic(
        elastic_);
  }

 private:
  bool enabled_;
  bool elastic_;
};

class ScopedModelImplSnapshot final {
 public:
  ScopedModelImplSnapshot()
      : model_impl_(ModelConfig::get_instance().model_impl()) {}

  ~ScopedModelImplSnapshot() {
    ModelConfig::get_instance().model_impl(model_impl_);
  }

 private:
  std::string model_impl_;
};

Sequence make_sequence(size_t index, const std::vector<int32_t>& tokens) {
  RequestSamplingParam sampling_param;

  sampling_param.is_embeddings = false;

  StoppingChecker stopping_checker;

  SequenceParams params;
  params.seq_capacity = tokens.size() + 8;
  params.echo = false;
  params.skip_special_tokens = true;
  params.streaming = false;
  params.enable_schedule_overlap = false;
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
    int32_t num_speculative_tokens,
    bool enable_no_padding,
    int32_t max_graph_batch_size = 0) {
  DecodeGraphWarmupConfig warmup_config;
  warmup_config.num_speculative_tokens = num_speculative_tokens;
  warmup_config.enable_graph_mode_decode_no_padding = enable_no_padding;
  warmup_config.max_graph_batch_size = max_graph_batch_size;
  return warmup_config;
}

class RecordingProfileEngine final {
 public:
  explicit RecordingProfileEngine(bool overlap = false,
                                  int32_t dp_size = 1,
                                  bool linear_attention = false,
                                  InstanceRole role = InstanceRole::DEFAULT,
                                  int32_t num_speculative_tokens = 0,
                                  bool enable_prefix_cache = false)
      : overlap_(overlap), linear_attention_(linear_attention) {
    BlockManagerPool::Options options;
    options.num_blocks(/*num_blocks=*/64)
        .block_size(/*block_size=*/4)
        .enable_prefix_cache(enable_prefix_cache)
        .max_seqs_per_batch(/*max_seqs_per_batch=*/8)
        .enable_linear_state(linear_attention)
        .linear_state_chunk_stride(4)
        .linear_state_num_slots(16)
        .num_embedding_blocks(16)
        .num_speculative_tokens(num_speculative_tokens)
        .instance_is_decode(role == InstanceRole::DECODE);
    block_manager_ = std::make_unique<BlockManagerPool>(options, dp_size);
    observed_batches_.reserve(options.max_seqs_per_batch());
    observed_token_lengths_.reserve(options.max_seqs_per_batch());
    observed_kv_capacities_.reserve(options.max_seqs_per_batch());
    observed_bootstrap_embeddings_.reserve(options.max_seqs_per_batch());
    model_args_.vocab_size(128)
        .eos_token_id(2)
        .max_position_embeddings(16)
        .hidden_size(8);
    if (linear_attention) {
      model_args_.model_type("qwen3_5_moe_text")
          .layer_types({"linear_attention"})
          .max_position_embeddings(64);
    }
  }

  ForwardOutput step(BatchGroup& batches) {
    EXPECT_EQ(pending_steps_, 0);
    int32_t sequence_count = 0;
    for (Batch& batch : batches) {
      std::vector<int32_t> prefill_lengths;
      prefill_lengths.reserve(batch.get_sequences().size());
      for (Sequence* sequence : batch.get_sequences()) {
        ++sequence_count;
        all_requests_marked_ =
            all_requests_marked_ && sequence->is_graph_warmup();
        observed_token_lengths_.emplace_back(sequence->num_tokens());
        observed_kv_capacities_.emplace_back(
            sequence->kv_state().current_max_tokens_capacity());
        observed_bootstrap_embeddings_.emplace_back(
            sequence->get_mtp_bootstrap_embedding().defined());
        if (linear_attention_) {
          EXPECT_GE(sequence->get_linear_state_slot_id(), 0);
        }
        if (sequence->is_prefill_stage()) {
          prefill_lengths.emplace_back(
              static_cast<int32_t>(sequence->num_tokens()));
        } else {
          ++decode_sequences_;
        }
      }
      if (!prefill_lengths.empty()) {
        prefill_batches_.emplace_back(std::move(prefill_lengths));
      }
    }
    observed_batches_.emplace_back(sequence_count);
    pending_steps_ += overlap_ ? 1 : 0;
    return ForwardOutput();
  }

  std::vector<RawForwardOutput> collect_step_outputs(
      const BatchGroup& batches) {
    EXPECT_EQ(pending_steps_, 1);
    const std::vector<size_t> used_blocks = block_manager_->num_used_blocks();
    for (const Batch& batch : batches) {
      for (const Sequence* sequence : batch.get_sequences()) {
        EXPECT_GT(used_blocks[sequence->dp_rank()], 0);
      }
    }
    --pending_steps_;
    return std::vector<RawForwardOutput>(batches.size());
  }

  BlockManagerPool* kv_cache_manager() const { return block_manager_.get(); }

  const ModelArgs& model_args() const { return model_args_; }

  void reset_profile_markers() { all_requests_marked_ = true; }

  bool all_requests_marked() const { return all_requests_marked_; }
  int32_t pending_steps() const { return pending_steps_; }
  const std::vector<int32_t>& observed_batches() const {
    return observed_batches_;
  }
  const std::vector<size_t>& observed_token_lengths() const {
    return observed_token_lengths_;
  }
  const std::vector<size_t>& observed_kv_capacities() const {
    return observed_kv_capacities_;
  }
  const std::vector<bool>& observed_bootstrap_embeddings() const {
    return observed_bootstrap_embeddings_;
  }
  const std::vector<std::vector<int32_t>>& prefill_batches() const {
    return prefill_batches_;
  }

  int32_t decode_sequences() const { return decode_sequences_; }

 private:
  std::unique_ptr<BlockManagerPool> block_manager_;
  ModelArgs model_args_;
  bool all_requests_marked_ = true;
  bool overlap_ = false;
  bool linear_attention_ = false;
  int32_t pending_steps_ = 0;
  std::vector<int32_t> observed_batches_;
  std::vector<size_t> observed_token_lengths_;
  std::vector<size_t> observed_kv_capacities_;
  std::vector<bool> observed_bootstrap_embeddings_;
  std::vector<std::vector<int32_t>> prefill_batches_;
  int32_t decode_sequences_ = 0;
};
#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU)
class LinearProfileGraphWarmupTest
    : public ::testing::TestWithParam<std::tuple<bool, int32_t>> {
 protected:
  void SetUp() override {
    original_scheduler_ = SchedulerConfig::get_instance();
    original_execution_ = ExecutionConfig::get_instance();
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() = 4;
    SchedulerConfig::get_instance().enable_chunked_prefill() = true;
    SchedulerConfig::get_instance().enable_dp_fair_token_budget() = false;
    ExecutionConfig::get_instance().enable_graph() = false;
    ExecutionConfig::get_instance().disable_graph_warmup() = true;
  }

  void TearDown() override {
    SchedulerConfig::get_instance() = original_scheduler_;
    ExecutionConfig::get_instance() = original_execution_;
  }

  ProfileManager::Options options(InstanceRole role,
                                  int32_t token_budget = 10,
                                  int32_t sequence_budget = 4) const {
    ProfileManager::Options options;
    options.enable_graph(true)
        .instance_role(role)
        .enable_dp_fair_token_budget(false)
        .enable_chunked_prefill(true)
        .max_tokens_per_chunk_for_prefill(4)
        .max_tokens_per_batch(token_budget)
        .max_seqs_per_batch(sequence_budget)
        .enable_schedule_overlap(std::get<0>(GetParam()))
        .decode_graph_warmup_config(
            make_decode_graph_warmup_config(std::get<1>(GetParam()),
                                            /*enable_no_padding=*/false));
    return options;
  }

 private:
  SchedulerConfig original_scheduler_;
  ExecutionConfig original_execution_;
};

TEST_P(LinearProfileGraphWarmupTest, PrefillWarmupDoesNotSplitIntoChunks) {
  RecordingProfileEngine engine(
      /*overlap=*/std::get<0>(GetParam()),
      /*dp_size=*/1,
      /*linear_attention=*/true,
      InstanceRole::PREFILL,
      std::get<1>(GetParam()));
  ProfileManager profile_manager(engine.kv_cache_manager(),
                                 &engine.model_args(),
                                 options(/*role=*/InstanceRole::PREFILL),
                                 make_scheduler_callbacks(&engine));

  EXPECT_EQ(engine.prefill_batches(),
            (std::vector<std::vector<int32_t>>{{10}}));
  EXPECT_EQ(engine.decode_sequences(), 0);
  EXPECT_TRUE(engine.all_requests_marked());
  EXPECT_EQ(engine.kv_cache_manager()->num_used_blocks(),
            (std::vector<size_t>{0}));
}

TEST_P(LinearProfileGraphWarmupTest,
       AutomaticChunkKeepsFullPrefillWarmupBudget) {
  RecordingProfileEngine engine(
      /*overlap=*/std::get<0>(GetParam()),
      /*dp_size=*/3,
      /*linear_attention=*/true,
      InstanceRole::PREFILL,
      std::get<1>(GetParam()));
  ProfileManager::Options profile_options = options(InstanceRole::PREFILL);
  profile_options.dp_size(3)
      .enable_dp_fair_token_budget(true)
      .max_tokens_per_chunk_for_prefill(-1);
  ProfileManager profile_manager(engine.kv_cache_manager(),
                                 &engine.model_args(),
                                 profile_options,
                                 make_scheduler_callbacks(&engine));
  EXPECT_EQ(engine.prefill_batches(),
            (std::vector<std::vector<int32_t>>{{10}}));
}

TEST_P(LinearProfileGraphWarmupTest, PrefillUsesInjectedFairBudget) {
  // Fixture globals disable fairness; explicit options must control the shape.
  for (const bool chunked : {false, true}) {
    RecordingProfileEngine engine(
        /*overlap=*/std::get<0>(GetParam()),
        /*dp_size=*/3,
        /*linear_attention=*/true,
        InstanceRole::PREFILL,
        std::get<1>(GetParam()));
    ProfileManager::Options profile_options = options(InstanceRole::PREFILL);
    profile_options.dp_size(3)
        .enable_dp_fair_token_budget(true)
        .enable_chunked_prefill(chunked)
        .max_tokens_per_chunk_for_prefill(6);
    ProfileManager profile_manager(engine.kv_cache_manager(),
                                   &engine.model_args(),
                                   profile_options,
                                   make_scheduler_callbacks(&engine));
    const int32_t expected_tokens = chunked ? 6 : 10;
    EXPECT_EQ(engine.prefill_batches(),
              (std::vector<std::vector<int32_t>>{{expected_tokens}}));
  }
}

TEST_P(LinearProfileGraphWarmupTest, PrefillRespectsSequenceLimit) {
  RecordingProfileEngine engine(
      /*overlap=*/std::get<0>(GetParam()),
      /*dp_size=*/1,
      /*linear_attention=*/true,
      InstanceRole::PREFILL,
      std::get<1>(GetParam()));
  ProfileManager profile_manager(
      engine.kv_cache_manager(),
      &engine.model_args(),
      options(/*role=*/InstanceRole::PREFILL, /*sequence_budget=*/2),
      make_scheduler_callbacks(&engine));

  EXPECT_EQ(engine.prefill_batches(),
            (std::vector<std::vector<int32_t>>{{10}}));
}

TEST_P(LinearProfileGraphWarmupTest, EagerWarmupAcceptsMoreThanOneCheckpoint) {
  RecordingProfileEngine engine(
      /*overlap=*/std::get<0>(GetParam()),
      /*dp_size=*/1,
      /*linear_attention=*/true,
      InstanceRole::PREFILL,
      std::get<1>(GetParam()));
  ProfileManager::Options profile_options = options(InstanceRole::PREFILL);
  profile_options.enable_graph(false);
  ProfileManager profile_manager(engine.kv_cache_manager(),
                                 &engine.model_args(),
                                 profile_options,
                                 make_scheduler_callbacks(&engine));
  EXPECT_EQ(engine.prefill_batches(),
            (std::vector<std::vector<int32_t>>{{10}}));
  EXPECT_EQ(engine.kv_cache_manager()->num_used_blocks(),
            (std::vector<size_t>{0}));
}

TEST_P(LinearProfileGraphWarmupTest,
       ProfilingAcceptsUnalignedPrefixAndLongQuery) {
  RecordingProfileEngine engine(
      /*overlap=*/std::get<0>(GetParam()),
      /*dp_size=*/1,
      /*linear_attention=*/true,
      InstanceRole::PREFILL,
      std::get<1>(GetParam()));
  ProfileManager::Options profile_options = options(InstanceRole::PREFILL);
  profile_options.disable_graph_warmup(true);
  ProfileManager profile_manager(engine.kv_cache_manager(),
                                 &engine.model_args(),
                                 profile_options,
                                 make_scheduler_callbacks(&engine));
  profile_manager.run_request(10, 3);
  profile_manager.run_request(8, 0);
  EXPECT_EQ(engine.prefill_batches(),
            (std::vector<std::vector<int32_t>>{{10}, {8}}));
  EXPECT_EQ(engine.kv_cache_manager()->num_used_blocks(),
            (std::vector<size_t>{0}));
}

TEST_P(LinearProfileGraphWarmupTest, UnifiedWarmupSeedsDecodeSlots) {
  RecordingProfileEngine engine(
      /*overlap=*/std::get<0>(GetParam()),
      /*dp_size=*/1,
      /*linear_attention=*/true,
      InstanceRole::DEFAULT,
      std::get<1>(GetParam()),
      /*enable_prefix_cache=*/true);
  ProfileManager profile_manager(engine.kv_cache_manager(),
                                 &engine.model_args(),
                                 options(/*role=*/InstanceRole::DEFAULT),
                                 make_scheduler_callbacks(&engine));

  EXPECT_EQ(engine.prefill_batches(),
            (std::vector<std::vector<int32_t>>{{10}}));
  EXPECT_GT(engine.decode_sequences(), 0);
  EXPECT_TRUE(engine.all_requests_marked());
  EXPECT_EQ(engine.kv_cache_manager()->num_blocks_in_prefix_cache(),
            (std::vector<size_t>{0}));
  EXPECT_EQ(engine.kv_cache_manager()->num_used_blocks(),
            (std::vector<size_t>{0}));
}

TEST_P(LinearProfileGraphWarmupTest, DecodeOnlyWarmupAllocatesWorkingSlots) {
  RecordingProfileEngine engine(
      /*overlap=*/std::get<0>(GetParam()),
      /*dp_size=*/1,
      /*linear_attention=*/true,
      InstanceRole::DECODE,
      std::get<1>(GetParam()));
  ProfileManager profile_manager(engine.kv_cache_manager(),
                                 &engine.model_args(),
                                 options(/*role=*/InstanceRole::DECODE),
                                 make_scheduler_callbacks(&engine));

  EXPECT_TRUE(engine.prefill_batches().empty());
  EXPECT_GT(engine.decode_sequences(), 0);
  EXPECT_TRUE(engine.all_requests_marked());
  EXPECT_EQ(engine.kv_cache_manager()->num_used_blocks(),
            (std::vector<size_t>{0}));
}

INSTANTIATE_TEST_SUITE_P(OverlapAndSpeculation,
                         LinearProfileGraphWarmupTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Values(0, 2)));

#endif
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
      /*num_speculative_tokens=*/3,
      /*enable_no_padding=*/false);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/64, /*dp_size=*/1);

  EXPECT_EQ(
      plan.batch_sizes,
      (std::vector<int32_t>{
          1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64}));
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
          Options instance_options;
          instance_options.acl_graph_decode_batch_size_limit(8)
              .num_speculative_tokens(width)
              .enable_graph_mode_decode_no_padding(false)
              .speculative_algorithm(algorithm);
          const auto config = instance_options.decode_graph_warmup_config();
          const int32_t global_capacity = local_capacity * dp_size;
          const auto plan =
              build_decode_graph_warmup_plan(config, global_capacity, dp_size);
          const int32_t local_limit = std::min(local_capacity, 8);
          const auto graph_rows = [local_limit](int32_t count) {
            return std::min(
                runtime::get_decode_graph_token_bucket(count, false),
                static_cast<int64_t>(local_limit));
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
            if (algorithm == "MTP") {
              const int64_t expected = graph_rows(local);
              EXPECT_TRUE(
                  std::any_of(plan.batch_sizes.begin(),
                              plan.batch_sizes.end(),
                              [&](int32_t global) {
                                return global <= local_limit * dp_size &&
                                       graph_rows(global / dp_size) == expected;
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
      /*num_speculative_tokens=*/5,
      /*enable_no_padding=*/false,
      /*max_graph_batch_size=*/16);
  const int32_t dp_size = 4;
  const int32_t max_global_batch_size = 64;
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, max_global_batch_size, dp_size);

  EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{4, 8, 20, 32, 40, 52, 64}));

  std::vector<int64_t> warmed_buckets;
  warmed_buckets.reserve(plan.batch_sizes.size());
  for (const int32_t batch_size : plan.batch_sizes) {
    const int64_t num_tokens = static_cast<int64_t>(batch_size / dp_size) *
                               (warmup_config.num_speculative_tokens + 1);
    warmed_buckets.push_back(runtime::get_decode_graph_token_bucket(
        num_tokens, /*enable_no_padding=*/false));
  }
  for (int32_t local_batch_size = 1;
       local_batch_size <= std::min(max_global_batch_size / dp_size,
                                    warmup_config.max_graph_batch_size);
       ++local_batch_size) {
    const int64_t num_tokens = static_cast<int64_t>(local_batch_size) *
                               (warmup_config.num_speculative_tokens + 1);
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
      /*num_speculative_tokens=*/3,
      /*enable_no_padding=*/true,
      /*max_graph_batch_size=*/16);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/7, /*dp_size=*/2);

  EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{2, 4, 6, 7}));
}

TEST(DecodeGraphWarmupPlanTest, PreservesSuppliedWarmupConfig) {
  const DecodeGraphWarmupConfig warmup_config = make_decode_graph_warmup_config(
      /*num_speculative_tokens=*/3,
      /*enable_no_padding=*/false);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/16, /*dp_size=*/1);

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
  options.max_tokens_per_batch(/*value=*/4)
      .max_seqs_per_batch(/*value=*/1)
      .dp_size(/*value=*/1);
  ProfileManager profile_manager(engine.kv_cache_manager(),
                                 &engine.model_args(),
                                 options,
                                 make_scheduler_callbacks(&engine));
  engine.reset_profile_markers();

  profile_manager.run_request(/*token_length=*/4, /*prefix_length=*/0);

  EXPECT_TRUE(engine.all_requests_marked());
}

TEST(GraphWarmupTest, DecodeProfileUsesConfiguredSpeculativeGeometry) {
  ScopedDecodeGraphConfigSnapshot snapshot;
  ExecutionConfig::get_instance().enable_graph(false);
  RecordingProfileEngine engine;
  Options engine_options;
  engine_options.num_speculative_tokens(3);
  ProfileManager::Options options;
  options.max_tokens_per_batch(5)
      .max_seqs_per_batch(1)
      .dp_size(1)
      .instance_role(InstanceRole::DECODE)
      .decode_graph_warmup_config(engine_options.decode_graph_warmup_config());
  ProfileManager profile_manager(engine.kv_cache_manager(),
                                 &engine.model_args(),
                                 options,
                                 make_scheduler_callbacks(&engine));

  profile_manager.profile_decode_step_time(/*token_length=*/5,
                                           /*batch_size=*/1,
                                           /*min_context_len=*/5,
                                           /*max_context_len=*/5);

  // Five context tokens plus four decode tokens need three four-token blocks.
  EXPECT_EQ(engine.observed_kv_capacities(), (std::vector<size_t>{12, 12, 12}));
  EXPECT_EQ(engine.observed_bootstrap_embeddings(),
            (std::vector<bool>{true, true, true}));
}

TEST(GraphWarmupTest, StartupWarmupUsesInstanceModeAndRole) {
#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU)
  ScopedDecodeGraphConfigSnapshot snapshot;
  for (const bool graph : {false, true}) {
    for (const bool disabled : {false, true}) {
      for (const bool overlap : {false, true}) {
        for (const InstanceRole role : {InstanceRole::DEFAULT,
                                        InstanceRole::PREFILL,
                                        InstanceRole::DECODE,
                                        InstanceRole::MIX}) {
          SCOPED_TRACE(graph);
          SCOPED_TRACE(disabled);
          SCOPED_TRACE(overlap);
          SCOPED_TRACE(role.to_string());
          ExecutionConfig::get_instance()
              .enable_graph(!graph)
              .disable_graph_warmup(!disabled);
          RecordingProfileEngine engine(overlap);
          const auto blocks_before =
              engine.kv_cache_manager()->num_used_blocks();
          ProfileManager::Options options;
          options.enable_graph(graph)
              .disable_graph_warmup(disabled)
              .max_tokens_per_batch(4)
              .max_seqs_per_batch(2)
              .enable_schedule_overlap(overlap)
              .instance_role(role);
          ProfileManager manager(&engine, options);
          std::vector<int32_t> expected;
          expected.reserve(3);
          if (graph && !disabled) {
            if (role != InstanceRole::DECODE) {
              expected.emplace_back(1);
            }
            if (role != InstanceRole::PREFILL) {
              expected.emplace_back(2);
              expected.emplace_back(1);
            }
          } else if (Platform::is_npu() && !graph &&
                     role != InstanceRole::DECODE) {
            expected.emplace_back(1);
          }
          EXPECT_EQ(engine.observed_batches(), expected);
          EXPECT_EQ(engine.pending_steps(), 0);
          EXPECT_EQ(engine.kv_cache_manager()->num_used_blocks(),
                    blocks_before);
        }
      }
    }
  }
#else
  GTEST_SKIP() << "Startup warmup requires a device execution platform.";
#endif
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
    const auto used_blocks = engine.kv_cache_manager()->num_used_blocks();
    ProfileManager::Options options;
    options.enable_graph(true)
        .max_tokens_per_batch(4)
        .max_seqs_per_batch(4)
        .dp_size(1)
        .enable_schedule_overlap(overlap)
        .enable_profile_step_time(false)
        .enable_profile_kv_blocks(false)
        .instance_role(InstanceRole::DECODE)
        .decode_graph_warmup_config(make_decode_graph_warmup_config(
            /*num_speculative_tokens=*/0,
            /*enable_no_padding=*/false,
            /*max_graph_batch_size=*/4));
    ProfileManager profile_manager(engine.kv_cache_manager(),
                                   &engine.model_args(),
                                   options,
                                   make_scheduler_callbacks(&engine));
    EXPECT_EQ(engine.observed_batches(), (std::vector<int32_t>{4, 2, 1}));
    EXPECT_EQ(engine.pending_steps(), 0);
    EXPECT_EQ(engine.kv_cache_manager()->num_used_blocks(), used_blocks);
    EXPECT_TRUE(engine.all_requests_marked());
  }
  config.enable_graph(previous_graph);
  config.disable_graph_warmup(previous_disabled);
  config.acl_graph_decode_batch_size_limit(previous_limit);
#else
  GTEST_SKIP() << "Startup graph warmup is disabled on this platform.";
#endif
}

TEST(GraphWarmupTest, DecodeLimitUsesInstanceOptionsAcrossDpGroups) {
#if defined(USE_NPU)
  ScopedDecodeGraphConfigSnapshot snapshot;
  auto& config = ExecutionConfig::get_instance();
  config.enable_graph(true).disable_graph_warmup(false);
  for (const int32_t dp_size : {1, 2}) {
    for (const int32_t limit : {1, 2}) {
      SCOPED_TRACE(dp_size);
      SCOPED_TRACE(limit);
      config.acl_graph_decode_batch_size_limit(limit == 1 ? 2 : 1);
      RecordingProfileEngine engine(/*overlap=*/false, dp_size);
      const auto blocks_before = engine.kv_cache_manager()->num_used_blocks();
      ProfileManager::Options options;
      options.enable_graph(true)
          .max_tokens_per_batch(8)
          .max_seqs_per_batch(4)
          .dp_size(dp_size)
          .instance_role(InstanceRole::DECODE)
          .decode_graph_warmup_config(make_decode_graph_warmup_config(
              /*num_speculative_tokens=*/0,
              /*enable_no_padding=*/false,
              limit));
      ProfileManager manager(&engine, options);
      const auto& batches = engine.observed_batches();
      ASSERT_FALSE(batches.empty());
      EXPECT_EQ(*std::max_element(batches.begin(), batches.end()),
                limit * dp_size);
      EXPECT_EQ(engine.kv_cache_manager()->num_used_blocks(), blocks_before);
    }
  }
#else
  GTEST_SKIP() << "ACL decode batch limits apply only on NPU.";
#endif
}

TEST(GraphWarmupTest, PythonNoPaddingWarmsEverySequenceCount) {
#if defined(USE_NPU)
  ScopedDecodeGraphConfigSnapshot execution_snapshot;
  ScopedModelImplSnapshot model_snapshot;
  ModelConfig::get_instance().model_impl("python");
  ExecutionConfig::get_instance()
      .enable_graph(true)
      .disable_graph_warmup(false)
      .acl_graph_decode_batch_size_limit(4);
  for (const int32_t num_speculative_tokens : {0, 4}) {
    for (const bool no_padding : {false, true}) {
      for (const bool overlap : {false, true}) {
        SCOPED_TRACE(num_speculative_tokens);
        SCOPED_TRACE(no_padding);
        SCOPED_TRACE(overlap);
        RecordingProfileEngine engine(overlap);
        const std::vector<size_t> used_before =
            engine.kv_cache_manager()->num_used_blocks();
        ProfileManager::Options options;
        options.enable_graph(true)
            .max_tokens_per_batch(20)
            .max_seqs_per_batch(4)
            .dp_size(1)
            .enable_schedule_overlap(overlap)
            .enable_profile_step_time(false)
            .enable_profile_kv_blocks(false)
            .instance_role(InstanceRole::DECODE)
            .decode_graph_warmup_config(
                make_decode_graph_warmup_config(num_speculative_tokens,
                                                no_padding,
                                                /*max_graph_batch_size=*/4));

        ProfileManager profile_manager(engine.kv_cache_manager(),
                                       &engine.model_args(),
                                       options,
                                       make_scheduler_callbacks(&engine));

        std::vector<int32_t> expected = {4, 3, 2, 1};
        if (!no_padding) {
          expected = num_speculative_tokens == 0
                         ? std::vector<int32_t>{4, 2, 1}
                         : std::vector<int32_t>{4, 3, 1};
        }
        EXPECT_EQ(engine.observed_batches(), expected);
        EXPECT_EQ(engine.pending_steps(), 0);
        EXPECT_EQ(engine.kv_cache_manager()->num_used_blocks(), used_before);
      }
    }
  }
#else
  GTEST_SKIP() << "Ordinary Python ACL graph warmup requires NPU.";
#endif
}

TEST(GraphWarmupTest, VmmUsesOrdinaryEagerWarmup) {
#if defined(USE_NPU)
  ScopedDecodeGraphConfigSnapshot execution_snapshot;
  ScopedVmmConfigSnapshot cache_snapshot;
  ExecutionConfig::get_instance().enable_graph(false);
  for (bool enable_vmm : {false, true}) {
    SCOPED_TRACE(enable_vmm);
    KVCacheConfig::get_instance()
        .enable_vmm_kvcache(enable_vmm)
        .vmm_elastic(true);
    RecordingProfileEngine engine(/*overlap=*/true, /*dp_size=*/2);
    const std::vector<size_t> used_before =
        engine.kv_cache_manager()->num_used_blocks();
    ProfileManager::Options options;
    options.max_tokens_per_batch(20)
        .max_seqs_per_batch(4)
        .dp_size(2)
        .enable_schedule_overlap(true)
        .enable_profile_step_time(false)
        .instance_role(InstanceRole::DEFAULT);

    ProfileManager profile_manager(engine.kv_cache_manager(),
                                   &engine.model_args(),
                                   options,
                                   make_scheduler_callbacks(&engine));

    EXPECT_EQ(engine.observed_batches(), (std::vector<int32_t>{1}));
    EXPECT_EQ(engine.observed_token_lengths(), (std::vector<size_t>{16}));
    EXPECT_EQ(engine.pending_steps(), 0);
    EXPECT_EQ(engine.kv_cache_manager()->num_used_blocks(), used_before);
    EXPECT_TRUE(engine.all_requests_marked());
  }
#else
  GTEST_SKIP() << "Ordinary eager startup warmup requires NPU.";
#endif
}

TEST(GraphWarmupTest, VmmDecodeOnlySkipsEagerWarmup) {
#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU)
  ScopedDecodeGraphConfigSnapshot execution_snapshot;
  ScopedVmmConfigSnapshot cache_snapshot;
  ExecutionConfig::get_instance().enable_graph(false);
  KVCacheConfig::get_instance().enable_vmm_kvcache(true).vmm_elastic(true);
  RecordingProfileEngine engine(/*overlap=*/true, /*dp_size=*/2);
  const std::vector<size_t> used_before =
      engine.kv_cache_manager()->num_used_blocks();
  ProfileManager::Options options;
  options.max_tokens_per_batch(20)
      .max_seqs_per_batch(4)
      .dp_size(2)
      .enable_schedule_overlap(true)
      .enable_profile_step_time(false)
      .instance_role(InstanceRole::DECODE);

  ProfileManager profile_manager(engine.kv_cache_manager(),
                                 &engine.model_args(),
                                 options,
                                 make_scheduler_callbacks(&engine));

  EXPECT_TRUE(engine.observed_batches().empty());
  EXPECT_EQ(engine.pending_steps(), 0);
  EXPECT_EQ(engine.kv_cache_manager()->num_used_blocks(), used_before);
#else
  GTEST_SKIP() << "Startup warmup requires a device execution platform.";
#endif
}

TEST(GraphWarmupTest, VmmGraphWarmupCanBeDisabled) {
#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU)
  ScopedDecodeGraphConfigSnapshot execution_snapshot;
  ScopedVmmConfigSnapshot cache_snapshot;
  ExecutionConfig::get_instance().enable_graph(true).disable_graph_warmup(true);
  KVCacheConfig::get_instance().enable_vmm_kvcache(true).vmm_elastic(true);
  RecordingProfileEngine engine(/*overlap=*/true, /*dp_size=*/2);
  const std::vector<size_t> used_before =
      engine.kv_cache_manager()->num_used_blocks();
  ProfileManager::Options options;
  options.enable_graph(true)
      .disable_graph_warmup(true)
      .max_tokens_per_batch(20)
      .max_seqs_per_batch(4)
      .dp_size(2)
      .enable_schedule_overlap(true)
      .enable_profile_step_time(false)
      .instance_role(InstanceRole::DEFAULT);

  ProfileManager profile_manager(engine.kv_cache_manager(),
                                 &engine.model_args(),
                                 options,
                                 make_scheduler_callbacks(&engine));

  EXPECT_TRUE(engine.observed_batches().empty());
  EXPECT_EQ(engine.pending_steps(), 0);
  EXPECT_EQ(engine.kv_cache_manager()->num_used_blocks(), used_before);
#else
  GTEST_SKIP() << "Startup warmup requires a device execution platform.";
#endif
}

}  // namespace
}  // namespace xllm
