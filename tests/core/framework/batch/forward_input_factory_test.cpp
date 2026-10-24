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

#include "core/framework/batch/forward_input_factory.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/batch/batch_group.h"
#include "core/framework/block/block_manager_pool.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/kv_cache/block/block_manager_impl.h"

namespace xllm {
namespace {

RequestSamplingParam& test_sampling_params() {
  static RequestSamplingParam params;
  return params;
}

StoppingChecker& test_stopping_checker() {
  static StoppingChecker checker;
  static const bool initialized = [] {
    checker.set_max_generated_tokens(32);
    return true;
  }();
  (void)initialized;
  return checker;
}

Sequence make_sequence(
    const std::vector<int32_t>& token_ids,
    const std::string& request_id,
    bool is_graph_warmup = false,
    std::shared_ptr<const JsonObjectGrammar> grammar = nullptr) {
  SequenceParams params;
  params.seq_capacity = token_ids.size() + 8;
  params.stopping_checker = &test_stopping_checker();
  params.sampling_param = &test_sampling_params();
  params.skip_special_tokens = true;
  params.echo = false;
  params.logprobs = false;
  params.request_id = request_id;
  params.is_graph_warmup = is_graph_warmup;
  params.json_object_grammar = std::move(grammar);

  IncrementalDecoder decoder(/*prompt=*/"",
                             /*num_prompt_tokens=*/token_ids.size(),
                             /*echo=*/false,
                             /*skip_special_tokens=*/true);
  return Sequence(/*index=*/0,
                  token_ids,
                  torch::Tensor(),
                  MMData(),
                  std::move(decoder),
                  params);
}

void add_kv_block(Sequence* sequence, BlockManagerImpl* manager) {
  sequence->add_blocks(BlockType::KV, manager->allocate(1));
}

Batch make_batch(Sequence* sequence) {
  Batch batch;
  batch.add(sequence);
  return batch;
}

ForwardInputFactory make_factory(
    uint32_t dp_size,
    uint32_t cp_size = 1,
    int64_t max_tokens_per_batch = 0,
    bool enable_dp_global_json_object_active = false) {
  ForwardInputFactoryOptions options;
  options.dp_size = dp_size;
  options.cp_size = cp_size;
  options.max_tokens_per_batch = max_tokens_per_batch;
  options.enable_dp_global_json_object_active =
      enable_dp_global_json_object_active;
  return ForwardInputFactory(options);
}

class ScopedGraphMode final {
 public:
  explicit ScopedGraphMode(bool enabled)
      : execution_config_(ExecutionConfig::get_instance()),
        previous_value_(execution_config_.enable_graph()) {
    execution_config_.enable_graph(enabled);
  }

  ~ScopedGraphMode() { execution_config_.enable_graph(previous_value_); }

 private:
  ExecutionConfig& execution_config_;
  bool previous_value_;
};

class ScopedJsonObjectOutput final {
 public:
  explicit ScopedJsonObjectOutput(bool enabled)
      : previous_value_(
            ServiceConfig::get_instance().enable_json_object_output()) {
    ServiceConfig::get_instance().enable_json_object_output(enabled);
  }

  ~ScopedJsonObjectOutput() {
    ServiceConfig::get_instance().enable_json_object_output(previous_value_);
  }

 private:
  bool previous_value_;
};

TEST(ForwardInputFactoryTest, AggregatesDpMetadataAndMixedForwardTypes) {
  BlockManager::Options manager_options;
  manager_options.num_blocks(8).block_size(4);
  BlockManagerImpl prefill_manager(manager_options);
  BlockManagerImpl decode_manager(manager_options);

  Sequence prefill = make_sequence({1, 2, 3}, "prefill");
  add_kv_block(&prefill, &prefill_manager);
  Sequence decode = make_sequence({4, 5}, "decode");
  add_kv_block(&decode, &decode_manager);
  decode.kv_state().set_kv_cache_tokens_num(2);
  decode.append_token(Token(6));

  BatchGroup batches(2);
  batches[0] = make_batch(&prefill);
  batches[1] = make_batch(&decode);

  auto factory = make_factory(
      /*dp_size=*/2,
      /*cp_size=*/1,
      /*max_tokens_per_batch=*/0,
      /*enable_dp_global_json_object_active=*/true);
  std::vector<LlmForwardInput> inputs;
  bool is_graph_warmup = false;
  factory.create_inputs(batches, ModelArgs(), inputs, is_graph_warmup);

  ASSERT_EQ(inputs.size(), 2u);
  EXPECT_EQ(inputs[0].input_params.parallel.dp_global_token_nums,
            (std::vector<int32_t>{3, 1}));
  for (const auto& input : inputs) {
    EXPECT_EQ(input.input_params.parallel.dp_global_json_object_active,
              (std::vector<int32_t>{0, 0}));
    EXPECT_EQ(input.input_params.parallel.dp_global_token_nums,
              (std::vector<int32_t>{3, 1}));
    EXPECT_EQ(input.input_params.parallel.dp_global_sequence_nums,
              (std::vector<int32_t>{1, 1}));
    EXPECT_EQ(input.input_params.parallel.raw_dp_global_token_nums,
              (std::vector<int32_t>{3, 1}));
  }
  EXPECT_TRUE(inputs[0].input_params.meta.batch_forward_type.is_prefill());
  EXPECT_TRUE(inputs[1].input_params.meta.batch_forward_type.is_decode());
}

TEST(ForwardInputFactoryTest, DpMetadataIsIndependentAcrossRanksAndCalls) {
  ScopedJsonObjectOutput json_output(/*enabled=*/true);
  ScopedGraphMode graph_mode(/*enabled=*/false);
  BlockManager::Options manager_options;
  manager_options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(manager_options);

  Sequence first = make_sequence({10, 11}, "first");
  std::shared_ptr<const JsonObjectGrammar> grammar =
      std::make_shared<const JsonObjectGrammar>(
          std::vector<std::string>{"{", "}", ""},
          std::unordered_set<int32_t>{2});
  Sequence second = make_sequence({20, 21, 22},
                                  "json",
                                  /*is_graph_warmup=*/false,
                                  std::move(grammar));
  add_kv_block(&first, &manager);
  add_kv_block(&second, &manager);

  BatchGroup batches(3);
  batches[0] = make_batch(&first);
  batches[1] = make_batch(&second);
  auto factory = make_factory(
      /*dp_size=*/3,
      /*cp_size=*/1,
      /*max_tokens_per_batch=*/0,
      /*enable_dp_global_json_object_active=*/true);
  std::vector<LlmForwardInput> inputs;
  bool is_graph_warmup = false;
  factory.create_inputs(batches, ModelArgs(), inputs, is_graph_warmup);

  ASSERT_EQ(inputs.size(), 3u);
  const std::vector<int32_t> token_counts = {2, 3, 0};
  const std::vector<int32_t> sequence_counts = {1, 1, 0};
  const std::vector<int32_t> json_active = {0, 1, 0};
  const std::vector<int32_t> is_decode = {0, 0, 0};
  const std::vector<uint64_t> generations =
      inputs.front().input_params.parallel.dp_global_batch_generations;
  ASSERT_EQ(generations.size(), 3u);
  for (const auto& input : inputs) {
    const auto& parallel = input.input_params.parallel;
    EXPECT_EQ(parallel.dp_global_token_nums, token_counts);
    EXPECT_EQ(parallel.raw_dp_global_token_nums, token_counts);
    EXPECT_EQ(parallel.dp_global_sequence_nums, sequence_counts);
    EXPECT_EQ(parallel.dp_global_kv_max_seq_lens, token_counts);
    EXPECT_EQ(parallel.dp_global_json_object_active, json_active);
    EXPECT_EQ(parallel.dp_is_decode, is_decode);
    EXPECT_EQ(parallel.dp_global_batch_generations, generations);
  }

  auto& last_parallel = inputs.back().input_params.parallel;
  ASSERT_EQ(last_parallel.dp_global_token_nums.size(), 3u);
  ASSERT_EQ(last_parallel.raw_dp_global_token_nums.size(), 3u);
  ASSERT_EQ(last_parallel.dp_global_sequence_nums.size(), 3u);
  ASSERT_EQ(last_parallel.dp_global_kv_max_seq_lens.size(), 3u);
  ASSERT_EQ(last_parallel.dp_global_json_object_active.size(), 3u);
  ASSERT_EQ(last_parallel.dp_is_decode.size(), 3u);
  ASSERT_EQ(last_parallel.dp_global_batch_generations.size(), 3u);
  last_parallel.dp_global_token_nums[0] = 9;
  EXPECT_EQ(last_parallel.raw_dp_global_token_nums, token_counts);
  last_parallel.raw_dp_global_token_nums[1] = 8;
  EXPECT_EQ(last_parallel.dp_global_token_nums,
            (std::vector<int32_t>{9, 3, 0}));
  last_parallel.dp_global_sequence_nums[0] = 7;
  last_parallel.dp_global_kv_max_seq_lens[1] = 6;
  last_parallel.dp_global_json_object_active[1] = 0;
  last_parallel.dp_is_decode[2] = 1;
  last_parallel.dp_global_batch_generations[0] = 99;

  auto expect_first_rank_metadata = [&] {
    const auto& parallel = inputs.front().input_params.parallel;
    EXPECT_EQ(parallel.dp_global_token_nums, token_counts);
    EXPECT_EQ(parallel.raw_dp_global_token_nums, token_counts);
    EXPECT_EQ(parallel.dp_global_sequence_nums, sequence_counts);
    EXPECT_EQ(parallel.dp_global_kv_max_seq_lens, token_counts);
    EXPECT_EQ(parallel.dp_global_json_object_active, json_active);
    EXPECT_EQ(parallel.dp_is_decode, is_decode);
    EXPECT_EQ(parallel.dp_global_batch_generations, generations);
  };
  expect_first_rank_metadata();

  BatchGroup empty_batches(3);
  std::vector<LlmForwardInput> next_inputs;
  factory.create_inputs(
      empty_batches, ModelArgs(), next_inputs, is_graph_warmup);
  ASSERT_EQ(next_inputs.size(), 3u);
  for (const auto& input : next_inputs) {
    const auto& parallel = input.input_params.parallel;
    EXPECT_EQ(parallel.dp_global_token_nums, is_decode);
    EXPECT_EQ(parallel.raw_dp_global_token_nums, is_decode);
    EXPECT_EQ(parallel.dp_global_sequence_nums, is_decode);
    EXPECT_EQ(parallel.dp_global_kv_max_seq_lens, is_decode);
    EXPECT_EQ(parallel.dp_global_json_object_active, is_decode);
    EXPECT_EQ(parallel.dp_is_decode, is_decode);
    ASSERT_EQ(parallel.dp_global_batch_generations.size(), 3u);
    EXPECT_NE(parallel.dp_global_batch_generations[0], 99u);
  }
  expect_first_rank_metadata();
  EXPECT_EQ(last_parallel.dp_global_token_nums,
            (std::vector<int32_t>{9, 3, 0}));
  EXPECT_EQ(last_parallel.raw_dp_global_token_nums,
            (std::vector<int32_t>{2, 8, 0}));
  EXPECT_EQ(last_parallel.dp_global_sequence_nums,
            (std::vector<int32_t>{7, 1, 0}));
  EXPECT_EQ(last_parallel.dp_global_kv_max_seq_lens,
            (std::vector<int32_t>{2, 6, 0}));
  EXPECT_EQ(last_parallel.dp_global_json_object_active, is_decode);
  EXPECT_EQ(last_parallel.dp_is_decode, (std::vector<int32_t>{0, 0, 1}));
  EXPECT_EQ(last_parallel.dp_global_batch_generations[0], 99u);
}

TEST(ForwardInputFactoryTest, EmptyRanksInheritGraphDecodeMetadata) {
  BlockManager::Options manager_options;
  manager_options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(manager_options);
  Sequence decode = make_sequence({1, 2}, "decode");
  add_kv_block(&decode, &manager);
  decode.kv_state().set_kv_cache_tokens_num(2);
  decode.append_token(Token(3));

  BatchGroup batches(2);
  batches[0] = make_batch(&decode);

  ScopedGraphMode graph_mode(/*enabled=*/true);
  auto factory = make_factory(/*dp_size=*/2);
  std::vector<LlmForwardInput> inputs;
  bool is_graph_warmup = false;
  factory.create_inputs(batches, ModelArgs(), inputs, is_graph_warmup);

  ASSERT_EQ(inputs.size(), 2u);
  EXPECT_EQ(inputs[0].input_params.parallel.dp_global_token_nums,
            (std::vector<int32_t>{1, 0}));
  EXPECT_TRUE(inputs[1].input_params.meta.batch_forward_type.is_decode());
  EXPECT_EQ(inputs[0].input_params.parallel.dp_is_decode,
            (std::vector<int32_t>{1, 1}));
  EXPECT_EQ(inputs[1].input_params.parallel.dp_is_decode,
            (std::vector<int32_t>{1, 1}));
}

TEST(ForwardInputFactoryTest, PropagatesGraphWarmupAndHandlesAllEmptyGroup) {
  BlockManager::Options manager_options;
  manager_options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(manager_options);
  Sequence warmup = make_sequence({1, 2}, "warmup", /*is_graph_warmup=*/true);
  add_kv_block(&warmup, &manager);

  BatchGroup warmup_batches(2);
  warmup_batches[0] = make_batch(&warmup);
  auto factory = make_factory(/*dp_size=*/2);
  std::vector<LlmForwardInput> warmup_inputs;
  bool warmup_is_graph_warmup = false;
  factory.create_inputs(
      warmup_batches, ModelArgs(), warmup_inputs, warmup_is_graph_warmup);
  EXPECT_TRUE(warmup_is_graph_warmup);
  for (const auto& input : warmup_inputs) {
    EXPECT_TRUE(input.input_params.meta.is_graph_warmup);
  }

  BatchGroup empty_batches(2);
  std::vector<LlmForwardInput> empty_inputs;
  bool empty_is_graph_warmup = true;
  factory.create_inputs(
      empty_batches, ModelArgs(), empty_inputs, empty_is_graph_warmup);
  EXPECT_FALSE(empty_is_graph_warmup);
  EXPECT_EQ(empty_inputs[0].input_params.parallel.dp_global_token_nums,
            (std::vector<int32_t>{0, 0}));
  for (const auto& input : empty_inputs) {
    EXPECT_TRUE(input.input_params.meta.batch_forward_type.is_empty());
    EXPECT_TRUE(input.input_params.parallel.dp_is_decode.empty() ||
                input.input_params.parallel.dp_is_decode ==
                    (std::vector<int32_t>{0, 0}));
  }
}

TEST(ForwardInputFactoryTest, KeepsOneInputPerDpRankForContextParallelism) {
  BlockManager::Options manager_options;
  manager_options.num_blocks(8).block_size(4);
  BlockManagerImpl first_manager(manager_options);
  BlockManagerImpl second_manager(manager_options);
  Sequence first = make_sequence({1, 2}, "first");
  Sequence second = make_sequence({3, 4}, "second");
  add_kv_block(&first, &first_manager);
  add_kv_block(&second, &second_manager);

  BatchGroup batches(2);
  batches[0] = make_batch(&first);
  batches[1] = make_batch(&second);
  auto factory = make_factory(/*dp_size=*/2, /*cp_size=*/3);
  std::vector<LlmForwardInput> inputs;
  bool is_graph_warmup = false;
  factory.create_inputs(batches, ModelArgs(), inputs, is_graph_warmup);

  ASSERT_EQ(inputs.size(), 2u);
  EXPECT_EQ(inputs[0].input_params.meta.num_sequences, 1);
  EXPECT_EQ(inputs[1].input_params.meta.num_sequences, 1);
}

TEST(ForwardInputFactoryTest, TracksEmbeddingGenerationAcrossBatchSteps) {
  BlockManagerPool::Options manager_options;
  manager_options.num_blocks(8)
      .block_size(4)
      .num_embedding_blocks(8)
      .num_speculative_tokens(1)
      .max_seqs_per_batch(8);
  BlockManagerPool first_manager(manager_options);
  BlockManagerPool second_manager(manager_options);
  Sequence first = make_sequence({1, 2}, "request-a");
  Sequence second = make_sequence({3, 4}, "request-b");
  ASSERT_TRUE(first_manager.allocate(&first));
  ASSERT_TRUE(second_manager.allocate(&second));

  auto factory = make_factory(/*dp_size=*/1);
  BatchGroup first_batches(1);
  first_batches[0] = make_batch(&first);
  std::vector<LlmForwardInput> first_inputs;
  bool first_is_graph_warmup = false;
  factory.create_inputs(
      first_batches, ModelArgs(), first_inputs, first_is_graph_warmup);
  ASSERT_EQ(
      first_inputs[0].input_params.parallel.dp_global_batch_generations.size(),
      1u);
  const uint64_t first_generation =
      first_inputs[0].input_params.parallel.dp_global_batch_generations[0];
  EXPECT_GE(first_generation, 1u);

  BatchGroup second_batches(1);
  second_batches[0] = make_batch(&second);
  std::vector<LlmForwardInput> second_inputs;
  bool second_is_graph_warmup = false;
  factory.create_inputs(
      second_batches, ModelArgs(), second_inputs, second_is_graph_warmup);
  EXPECT_GT(
      second_inputs[0].input_params.parallel.dp_global_batch_generations[0],
      first_generation);
}

TEST(ForwardInputFactoryTest, EnforcesDeepseekV4BatchTokenBudget) {
  BlockManager::Options manager_options;
  manager_options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(manager_options);
  Sequence sequence = make_sequence({1, 2, 3}, "dsv4");
  add_kv_block(&sequence, &manager);
  BatchGroup batches(1);
  batches[0] = make_batch(&sequence);

  ModelArgs args;
  args.model_type("deepseek_v4");
  auto factory = make_factory(/*dp_size=*/1,
                              /*cp_size=*/1,
                              /*max_tokens_per_batch=*/3);
  std::vector<LlmForwardInput> inputs;
  bool is_graph_warmup = false;
  factory.create_inputs(batches, args, inputs, is_graph_warmup);
  ASSERT_EQ(inputs[0].input_params.parallel.dp_global_token_nums,
            (std::vector<int32_t>{3}));
  EXPECT_EQ(inputs[0].host_token_ids().numel(), 3);
}

TEST(ForwardInputFactoryTest, PreparationAdvancesSequenceKvCursor) {
  BlockManager::Options manager_options;
  manager_options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(manager_options);
  Sequence sequence = make_sequence({1, 2, 3}, "cursor");
  add_kv_block(&sequence, &manager);
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);

  BatchGroup batches(1);
  batches[0] = make_batch(&sequence);
  auto factory = make_factory(/*dp_size=*/1);
  std::vector<LlmForwardInput> inputs;
  bool is_graph_warmup = false;
  factory.create_inputs(batches, ModelArgs(), inputs, is_graph_warmup);
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 3u);
}

}  // namespace
}  // namespace xllm
