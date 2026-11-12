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
#include "core/framework/batch/shared_kv_cache_mapping_tracker.h"
#include "core/framework/block/block_manager_pool.h"
#include "core/framework/block/composite_block_manager.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/constrained/json_object_grammar.h"
#include "core/framework/model/model_args.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/kv_cache/block/block_manager_impl.h"
#include "core/kv_cache/block/static_kv_cache_block_pool.h"
#include "core/kv_cache/layout/static_kv_cache_layout.h"
#include "core/util/tensor_helper.h"

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

StaticKVCacheLayout make_publication_layout() {
  StaticKVCacheLayoutSpec spec;
  spec.num_slots = 64;
  spec.num_slabs = 16;
  spec.full_row_bytes = 16;
  spec.conv_row_bytes = 8;
  spec.ssm_row_bytes = 32;
  spec.checkpoint_stride = 2;
  spec.full_layer_ids = {0, 3};
  spec.linear_layer_ids = {1, 2};
  return StaticKVCacheLayout(spec);
}

BlockManager::Options make_publication_block_options() {
  BlockManager::Options options;
  options.num_blocks(64)
      .block_size(4)
      .enable_linear_state(true)
      .linear_state_num_slots(64)
      .enable_shared_kv_cache(true)
      .shared_layout(
          std::make_shared<StaticKVCacheLayout>(make_publication_layout()))
      .enable_prefix_cache(false);
  return options;
}

TEST(SharedKVCacheMappingTrackerTest,
     AbandonedPreparationRepublishesUntilConsumerAcknowledges) {
  auto pool = std::make_shared<StaticKVCacheBlockPool>(
      /*logical_capacity=*/64, make_publication_layout());
  const auto ids = pool->allocate(/*num_blocks=*/2, BlockType::KV);
  ASSERT_EQ(ids.size(), 2u);
  const std::vector<Block> blocks = {Block(ids[0], nullptr),
                                     Block(ids[1], nullptr)};
  SharedKVCacheMappingTracker mapping_tracker;
  std::vector<int32_t> initial_rows;
  std::vector<uint64_t> initial_epochs;
  mapping_tracker.begin();
  mapping_tracker.append_rows(
      pool, BlockType::KV, blocks, &initial_rows, &initial_epochs);
  ASSERT_EQ(initial_rows.size(), 6u);
  EXPECT_EQ(initial_epochs, (std::vector<uint64_t>{1, 1}));

  std::vector<int32_t> duplicate_rows;
  mapping_tracker.append_rows(pool, BlockType::KV, blocks, &duplicate_rows);
  EXPECT_TRUE(duplicate_rows.empty());
  mapping_tracker.begin();
  std::vector<int32_t> retried_rows;
  mapping_tracker.append_rows(pool, BlockType::KV, blocks, &retried_rows);
  EXPECT_EQ(retried_rows, initial_rows);
  mapping_tracker.acknowledge();
  mapping_tracker.begin();
  std::vector<int32_t> unchanged_rows;
  mapping_tracker.append_rows(pool, BlockType::KV, blocks, &unchanged_rows);
  EXPECT_TRUE(unchanged_rows.empty());

  const auto next = pool->allocate(/*num_blocks=*/1, BlockType::KV);
  ASSERT_EQ(next.size(), 1u);
  const std::vector<Block> expanded = {
      blocks[0], blocks[1], Block(next[0], nullptr)};
  std::vector<int32_t> new_rows;
  std::vector<uint64_t> new_epochs;
  mapping_tracker.append_rows(
      pool, BlockType::KV, expanded, &new_rows, &new_epochs);
  ASSERT_EQ(new_rows.size(), 3u);
  EXPECT_EQ(new_rows[0], next[0]);
  EXPECT_EQ(new_epochs, (std::vector<uint64_t>{1}));
}

TEST(SharedKVCacheMappingTrackerTest,
     ConsumersKeepIndependentFullAndLinearAllocationGenerations) {
  auto pool = std::make_shared<StaticKVCacheBlockPool>(
      /*logical_capacity=*/64, make_publication_layout());
  const auto full = pool->allocate(/*num_blocks=*/1, BlockType::KV);
  const auto linear = pool->allocate(/*num_blocks=*/1, BlockType::LINEAR);
  ASSERT_EQ(full.size(), 1u);
  ASSERT_EQ(linear.size(), 1u);
  ASSERT_EQ(full[0], linear[0]);
  const std::vector<Block> full_blocks = {Block(full[0], nullptr)};
  const std::vector<Block> linear_blocks = {Block(linear[0], nullptr)};
  SharedKVCacheMappingTracker first;
  SharedKVCacheMappingTracker second;
  std::vector<int32_t> first_full;
  std::vector<int32_t> first_linear;
  first.append_rows(pool, BlockType::KV, full_blocks, &first_full);
  first.append_rows(pool, BlockType::LINEAR, linear_blocks, &first_linear);
  first.acknowledge();
  ASSERT_EQ(first_full.size(), 3u);
  ASSERT_EQ(first_linear.size(), 5u);

  std::vector<int32_t> second_full;
  std::vector<int32_t> second_linear;
  second.append_rows(pool, BlockType::KV, full_blocks, &second_full);
  second.append_rows(pool, BlockType::LINEAR, linear_blocks, &second_linear);
  EXPECT_EQ(second_full, first_full);
  EXPECT_EQ(second_linear, first_linear);
  second.acknowledge();

  pool->release(full[0], BlockType::KV);
  pool->release(linear[0], BlockType::LINEAR);
  const auto reused_full = pool->allocate(/*num_blocks=*/1, BlockType::KV);
  const auto reused_linear =
      pool->allocate(/*num_blocks=*/1, BlockType::LINEAR);
  ASSERT_EQ(reused_full, full);
  ASSERT_EQ(reused_linear, linear);
  std::vector<int32_t> reused_full_rows;
  std::vector<int32_t> reused_linear_rows;
  std::vector<uint64_t> reused_full_epochs;
  first.begin();
  first.append_rows(
      pool, BlockType::KV, full_blocks, &reused_full_rows, &reused_full_epochs);
  first.append_rows(
      pool, BlockType::LINEAR, linear_blocks, &reused_linear_rows);
  EXPECT_EQ(reused_full_rows.size(), 3u);
  EXPECT_EQ(reused_full_epochs, (std::vector<uint64_t>{2}));
  EXPECT_EQ(reused_linear_rows.size(), 5u);

  first.acknowledge();
  first.reset();
  reused_full_rows.clear();
  first.append_rows(pool, BlockType::KV, full_blocks, &reused_full_rows);
  EXPECT_EQ(reused_full_rows.size(), 3u);
  first.acknowledge();
  auto replacement_pool = std::make_shared<StaticKVCacheBlockPool>(
      /*logical_capacity=*/64, make_publication_layout());
  const auto replacement_full =
      replacement_pool->allocate(/*num_blocks=*/1, BlockType::KV);
  ASSERT_EQ(replacement_full, full);
  reused_full_rows.clear();
  first.append_rows(
      replacement_pool, BlockType::KV, full_blocks, &reused_full_rows);
  EXPECT_EQ(reused_full_rows.size(), 3u);
}

TEST(ForwardInputFactoryTest,
     SharedMappingsRequireAcknowledgmentAndResetWithWorkerStorage) {
  const auto options = make_publication_block_options();
  CompositeBlockManager manager(build_composite_leaves(options), options);
  Sequence sequence = make_sequence({1, 2, 3}, "shared");
  sequence.add_blocks(BlockType::KV, manager.allocate_blocks(BlockType::KV, 1));
  sequence.add_blocks(BlockType::LINEAR,
                      manager.allocate_blocks(BlockType::LINEAR, 1));
  ModelArgs args;
  args.layer_types({"full_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention"});
  ForwardInputFactoryOptions factory_options;
  factory_options.enable_shared_kv_cache_deltas = true;
  ForwardInputFactory factory(factory_options);
  std::vector<LlmForwardInput> inputs;
  bool is_graph_warmup = false;
  auto prepare = [&](ForwardInputFactory& selected_factory) {
    sequence.kv_state().set_kv_cache_tokens_num(0);
    BatchGroup batches(1);
    batches[0].add(&sequence);
    selected_factory.create_inputs(batches, args, inputs, is_graph_warmup);
  };
  prepare(factory);
  ASSERT_EQ(inputs.size(), 1u);
  const auto full_rows = inputs[0].input_params.embedding.full_block_slot_rows;
  const auto linear_rows =
      inputs[0].input_params.embedding.linear_state_slot_rows;
  ASSERT_EQ(full_rows.size(), 3u);
  ASSERT_EQ(linear_rows.size(), 5u);
  const auto& attention = inputs[0].input_params.attention;
  ASSERT_TRUE(attention.host.shared_full_block_table.defined());
  EXPECT_EQ(attention.host.block_tables.item<int32_t>(), full_rows[0]);
  EXPECT_EQ(attention.host.shared_full_block_table.item<int32_t>(),
            full_rows[1]);
  EXPECT_EQ(
      tensor_to_vector<int32_t>(attention.host.shared_full_slot_mapping),
      std::vector<int32_t>(
          {full_rows[1] * 4, full_rows[1] * 4 + 1, full_rows[1] * 4 + 2}));
  const torch::Tensor final_table =
      attention.host.shared_full_block_table.clone();
  const torch::Tensor final_slots =
      attention.host.shared_full_slot_mapping.clone();
  prepare(factory);
  EXPECT_EQ(inputs[0].input_params.embedding.full_block_slot_rows, full_rows);
  EXPECT_EQ(inputs[0].input_params.embedding.linear_state_slot_rows,
            linear_rows);
  factory.acknowledge_shared_kv_cache();
  prepare(factory);
  EXPECT_TRUE(inputs[0].input_params.embedding.full_block_slot_rows.empty());
  EXPECT_TRUE(inputs[0].input_params.embedding.full_block_lease_epochs.empty());
  EXPECT_TRUE(inputs[0].input_params.embedding.linear_state_slot_rows.empty());
  EXPECT_TRUE(torch::equal(
      inputs[0].input_params.attention.host.shared_full_block_table,
      final_table));
  EXPECT_TRUE(torch::equal(
      inputs[0].input_params.attention.host.shared_full_slot_mapping,
      final_slots));
  const LlmForwardInput draft = make_llm_draft_input(inputs[0]);
  EXPECT_FALSE(
      draft.input_params.attention.host.shared_full_block_table.defined());
  EXPECT_FALSE(
      draft.input_params.attention.device.shared_full_block_table.defined());
  EXPECT_FALSE(
      draft.input_params.attention.host.shared_full_slot_mapping.defined());
  EXPECT_FALSE(
      draft.input_params.attention.device.shared_full_slot_mapping.defined());
  EXPECT_TRUE(torch::equal(draft.input_params.attention.host.block_tables,
                           inputs[0].input_params.attention.host.block_tables));
  const LlmForwardInput shared_draft =
      make_llm_draft_input(inputs[0], /*shared_full_cache=*/true);
  EXPECT_TRUE(torch::equal(
      shared_draft.input_params.attention.host.shared_full_block_table,
      final_table));
  EXPECT_TRUE(torch::equal(
      shared_draft.input_params.attention.host.shared_full_slot_mapping,
      final_slots));
  EXPECT_TRUE(shared_draft.input_params.embedding.linear_state_ids.empty());
  factory.acknowledge_shared_kv_cache();
  factory.reset_shared_kv_cache_mapping_tracker();
  prepare(factory);
  EXPECT_EQ(inputs[0].input_params.embedding.full_block_slot_rows, full_rows);
  EXPECT_EQ(inputs[0].input_params.embedding.linear_state_slot_rows,
            linear_rows);

  ForwardInputFactory snapshot_factory(ForwardInputFactoryOptions{});
  prepare(snapshot_factory);
  EXPECT_EQ(inputs[0].input_params.embedding.full_block_slot_rows, full_rows);
  snapshot_factory.acknowledge_shared_kv_cache();
  prepare(snapshot_factory);
  EXPECT_EQ(inputs[0].input_params.embedding.full_block_slot_rows, full_rows);
  EXPECT_EQ(inputs[0].input_params.embedding.linear_state_slot_rows,
            linear_rows);
}

TEST(ForwardInputFactoryTest,
     SharedMappingAcknowledgmentKeepsDpAndEmptyRankCursorsIndependent) {
  ScopedGraphMode graph_mode(true);
  const auto options = make_publication_block_options();
  CompositeBlockManager first_manager(build_composite_leaves(options), options);
  CompositeBlockManager second_manager(build_composite_leaves(options),
                                       options);
  Sequence first = make_sequence({1, 2, 3}, "first");
  Sequence second = make_sequence({4, 5, 6}, "second");
  first.add_blocks(BlockType::KV,
                   first_manager.allocate_blocks(BlockType::KV, 1));
  first.add_blocks(BlockType::LINEAR,
                   first_manager.allocate_blocks(BlockType::LINEAR, 1));
  second.add_blocks(BlockType::KV,
                    second_manager.allocate_blocks(BlockType::KV, 1));
  second.add_blocks(BlockType::LINEAR,
                    second_manager.allocate_blocks(BlockType::LINEAR, 1));
  ModelArgs args;
  args.layer_types({"full_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention"});
  ForwardInputFactoryOptions factory_options;
  factory_options.dp_size = 2;
  factory_options.cp_size = 2;
  factory_options.enable_shared_kv_cache_deltas = true;
  ForwardInputFactory factory(factory_options);
  std::vector<LlmForwardInput> inputs;
  bool is_graph_warmup = false;
  auto prepare = [&](bool include_first, bool include_second) {
    first.kv_state().set_kv_cache_tokens_num(0);
    second.kv_state().set_kv_cache_tokens_num(0);
    BatchGroup batches(2);
    if (include_first) {
      batches[0].add(&first);
    }
    if (include_second) {
      batches[1].add(&second);
    }
    factory.create_inputs(batches, args, inputs, is_graph_warmup);
  };
  prepare(/*include_first=*/true, /*include_second=*/true);
  ASSERT_EQ(inputs.size(), 2u);
  EXPECT_EQ(inputs[0].input_params.embedding.full_block_slot_rows.size(), 3u);
  EXPECT_EQ(inputs[1].input_params.embedding.full_block_slot_rows.size(), 3u);
  EXPECT_EQ(inputs[0].input_params.embedding.linear_state_slot_rows.size(), 5u);
  EXPECT_EQ(inputs[1].input_params.embedding.linear_state_slot_rows.size(), 5u);
  factory.acknowledge_shared_kv_cache();
  prepare(/*include_first=*/true, /*include_second=*/true);
  EXPECT_TRUE(inputs[0].input_params.embedding.full_block_slot_rows.empty());
  EXPECT_TRUE(inputs[1].input_params.embedding.full_block_slot_rows.empty());
  factory.acknowledge_shared_kv_cache();

  factory.reset_shared_kv_cache_mapping_tracker();
  prepare(/*include_first=*/false, /*include_second=*/true);
  EXPECT_TRUE(inputs[0].input_params.embedding.full_block_slot_rows.empty());
  EXPECT_EQ(inputs[1].input_params.embedding.full_block_slot_rows.size(), 3u);
  factory.acknowledge_shared_kv_cache();
  prepare(/*include_first=*/true, /*include_second=*/true);
  EXPECT_EQ(inputs[0].input_params.embedding.full_block_slot_rows.size(), 3u);
  EXPECT_EQ(inputs[0].input_params.embedding.linear_state_slot_rows.size(), 5u);
  EXPECT_TRUE(inputs[1].input_params.embedding.full_block_slot_rows.empty());
  EXPECT_TRUE(inputs[1].input_params.embedding.linear_state_slot_rows.empty());
}

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
