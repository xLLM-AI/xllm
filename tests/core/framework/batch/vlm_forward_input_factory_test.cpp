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

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/batch/batch_group.h"
#include "core/framework/batch/forward_input_factory.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/kv_cache/block/block_manager_impl.h"
#include "core/kv_cache/manager/block_manager_pool.h"
#include "worker.pb.h"

namespace xllm {
namespace {

using VlmInputFactory = ForwardInputFactory<VlmForwardInput>;

class VlmForwardInputFactoryTest : public ::testing::Test {
 protected:
  VlmForwardInputFactoryTest() {
    stopping_checker_.set_max_generated_tokens(/*max_generated_tokens=*/8);
  }

  Sequence make_sequence(const std::vector<int32_t>& tokens,
                         const std::string& request_id,
                         bool is_graph_warmup = false) {
    SequenceParams params;
    params.seq_capacity = tokens.size() + 8;
    params.sampling_param = &sampling_params_;
    params.stopping_checker = &stopping_checker_;
    params.request_id = request_id;
    params.is_graph_warmup = is_graph_warmup;
    params.echo = false;
    params.skip_special_tokens = true;
    params.logprobs = false;
    IncrementalDecoder decoder(/*prompt=*/"",
                               /*num_prompt_tokens=*/tokens.size(),
                               /*echo=*/false,
                               /*skip_special_tokens=*/true);
    return Sequence(/*index=*/0,
                    tokens,
                    torch::Tensor(),
                    MMData(),
                    std::move(decoder),
                    params);
  }

  void expect_packed_metadata(const VlmForwardInput& input) {
    proto::PackedForwardInput packed;
    ASSERT_TRUE(input.to_packed_proto(&packed));
    VlmForwardInput lazy;
    ASSERT_TRUE(lazy.from_packed_proto(packed));
    const VlmForwardInput decoded =
        lazy.to(torch::Device(torch::kCPU), torch::kFloat32);
    EXPECT_EQ(decoded.input_params.parallel.dp_global_batch_generations,
              input.input_params.parallel.dp_global_batch_generations);
    EXPECT_EQ(decoded.input_params.meta.is_graph_warmup,
              input.input_params.meta.is_graph_warmup);
  }

  RequestSamplingParam sampling_params_;
  StoppingChecker stopping_checker_;
};

class ScopedGraphMode final {
 public:
  explicit ScopedGraphMode(bool enabled)
      : config_(ExecutionConfig::get_instance()),
        previous_value_(config_.enable_graph()) {
    config_.enable_graph(enabled);
  }

  ~ScopedGraphMode() { config_.enable_graph(previous_value_); }

 private:
  ExecutionConfig& config_;
  bool previous_value_;
};

TEST_F(VlmForwardInputFactoryTest, PreservesMixedRankTypesAndDpMetadata) {
  BlockManager::Options manager_options;
  manager_options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(manager_options);
  Sequence prefill = make_sequence({1, 2, 3}, "prefill");
  prefill.add_blocks(BlockType::KV, manager.allocate(/*num_blocks=*/1));
  Sequence decode = make_sequence({4, 5}, "decode");
  decode.add_blocks(BlockType::KV, manager.allocate(/*num_blocks=*/1));
  decode.kv_state().set_kv_cache_tokens_num(2);
  decode.append_token(Token(6));

  BatchGroup batches(2);
  batches[0].add(&prefill);
  batches[1].add(&decode);
  ForwardInputFactoryOptions options;
  options.dp_size = 2;
  options.enable_dp_global_json_object_active = true;
  VlmInputFactory factory(options);
  std::vector<VlmForwardInput> inputs;
  factory.create_inputs(batches, ModelArgs(), inputs);

  ASSERT_EQ(inputs.size(), 2u);
  EXPECT_TRUE(inputs[0].input_params.meta.batch_forward_type.is_prefill());
  EXPECT_TRUE(inputs[1].input_params.meta.batch_forward_type.is_decode());
  EXPECT_TRUE(torch::equal(inputs[0].host_token_ids(),
                           torch::tensor({1, 2, 3}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(inputs[1].host_token_ids(),
                           torch::tensor({6}, torch::kInt32)));
  for (const auto& input : inputs) {
    const auto& parallel = input.input_params.parallel;
    EXPECT_EQ(parallel.dp_global_token_nums, (std::vector<int32_t>{3, 1}));
    EXPECT_EQ(parallel.raw_dp_global_token_nums, (std::vector<int32_t>{3, 1}));
    EXPECT_EQ(parallel.dp_global_sequence_nums, (std::vector<int32_t>{1, 1}));
    EXPECT_EQ(parallel.dp_global_kv_max_seq_lens, (std::vector<int32_t>{3, 3}));
    EXPECT_EQ(parallel.dp_global_json_object_active,
              (std::vector<int32_t>{0, 0}));
    EXPECT_EQ(parallel.dp_is_decode, (std::vector<int32_t>{0, 1}));
  }
}

TEST_F(VlmForwardInputFactoryTest, EmptyRankUsesInstanceGraphDecodeOption) {
  for (const bool enable_graph : {false, true}) {
    SCOPED_TRACE(enable_graph);
    for (const bool is_decode : {false, true}) {
      SCOPED_TRACE(is_decode);
      ScopedGraphMode graph_mode(/*enabled=*/!enable_graph);
      BlockManager::Options manager_options;
      manager_options.num_blocks(4).block_size(4);
      BlockManagerImpl manager(manager_options);
      Sequence sequence = make_sequence({1, 2}, "request");
      sequence.add_blocks(BlockType::KV, manager.allocate(/*num_blocks=*/1));
      if (is_decode) {
        sequence.kv_state().set_kv_cache_tokens_num(2);
        sequence.append_token(Token(3));
      }
      BatchGroup batches(2);
      batches[1].add(&sequence);
      ForwardInputFactoryOptions options;
      options.dp_size = 2;
      options.enable_graph = enable_graph;
      VlmInputFactory factory(options);
      options.enable_graph = !enable_graph;
      std::vector<VlmForwardInput> inputs;
      factory.create_inputs(batches, ModelArgs(), inputs);

      ASSERT_EQ(inputs.size(), 2u);
      EXPECT_FALSE(inputs[0].token_ids.defined());
      EXPECT_EQ(inputs[0].input_params.meta.batch_forward_type.is_decode(),
                is_decode);
      EXPECT_EQ(inputs[0].input_params.meta.num_sequences, 0);
      for (const auto& input : inputs) {
        EXPECT_EQ(input.input_params.parallel.dp_global_token_nums,
                  (std::vector<int32_t>{0, is_decode ? 1 : 2}));
        EXPECT_EQ(input.input_params.parallel.dp_global_sequence_nums,
                  (std::vector<int32_t>{0, 1}));
        EXPECT_EQ(input.input_params.parallel.dp_is_decode,
                  (std::vector<int32_t>{enable_graph && is_decode ? 1 : 0,
                                        is_decode ? 1 : 0}));
        EXPECT_TRUE(
            input.input_params.parallel.dp_global_json_object_active.empty());
      }
    }
  }
}

TEST_F(VlmForwardInputFactoryTest, AllEmptyGroupClearsPreviousInputs) {
  ScopedGraphMode graph_mode(/*enabled=*/false);
  ForwardInputFactoryOptions options;
  options.dp_size = 2;
  options.enable_graph = true;
  VlmInputFactory factory(options);
  std::vector<VlmForwardInput> inputs(1);
  inputs[0].token_ids = torch::tensor({42}, torch::kInt32);
  BatchGroup batches(2);
  factory.create_inputs(batches, ModelArgs(), inputs);

  ASSERT_EQ(inputs.size(), 2u);
  for (const auto& input : inputs) {
    EXPECT_FALSE(input.token_ids.defined());
    EXPECT_TRUE(input.input_params.meta.batch_forward_type.is_empty());
    EXPECT_EQ(input.input_params.parallel.dp_global_token_nums,
              (std::vector<int32_t>{0, 0}));
    EXPECT_EQ(input.input_params.parallel.dp_is_decode,
              (std::vector<int32_t>{0, 0}));
  }
}

TEST_F(VlmForwardInputFactoryTest,
       TracksRemoteBatchIdentityAtEqualTokenCounts) {
  BlockManagerPool::Options manager_options;
  manager_options.num_blocks(8)
      .block_size(4)
      .num_embedding_blocks(8)
      .num_speculative_tokens(1)
      .max_seqs_per_batch(8);
  BlockManagerPool first_manager(manager_options);
  BlockManagerPool second_manager(manager_options);
  BlockManagerPool replacement_manager(manager_options);
  Sequence first = make_sequence({1, 2}, "first");
  Sequence second = make_sequence({3, 4}, "second");
  Sequence replacement = make_sequence({3, 4}, "replacement");
  Sequence moved_slot = make_sequence({3, 4}, "replacement");
  ASSERT_TRUE(first_manager.allocate(&first));
  ASSERT_TRUE(second_manager.allocate(&second));
  ASSERT_TRUE(replacement_manager.allocate(&replacement));
  ASSERT_TRUE(second_manager.allocate(&moved_slot));
  ForwardInputFactoryOptions options;
  options.dp_size = 2;
  VlmInputFactory factory(options);
  auto prepare = [&](Sequence& remote) {
    first.kv_state().set_kv_cache_tokens_num(0);
    remote.kv_state().set_kv_cache_tokens_num(0);
    BatchGroup batches(2);
    batches[0].add(&first);
    batches[1].add(&remote);
    std::vector<VlmForwardInput> inputs;
    factory.create_inputs(batches, ModelArgs{}, inputs);
    return inputs;
  };

  auto first_inputs = prepare(second);
  ASSERT_EQ(first_inputs.size(), 2u);
  const auto generations =
      first_inputs[0].input_params.parallel.dp_global_batch_generations;
  ASSERT_EQ(generations.size(), 2u);
  EXPECT_GE(generations[0], 1u);
  EXPECT_GE(generations[1], 1u);
  auto repeated_inputs = prepare(second);
  for (const auto& input : repeated_inputs) {
    EXPECT_EQ(input.input_params.parallel.dp_global_batch_generations,
              generations);
  }

  auto replaced_inputs = prepare(replacement);
  ASSERT_EQ(replaced_inputs.size(), 2u);
  EXPECT_EQ(replaced_inputs[1].input_params.embedding.embedding_ids,
            first_inputs[1].input_params.embedding.embedding_ids);
  const auto replaced_generations =
      replaced_inputs[0].input_params.parallel.dp_global_batch_generations;
  ASSERT_EQ(replaced_generations.size(), 2u);
  EXPECT_EQ(replaced_generations[0], generations[0]);
  EXPECT_GT(replaced_generations[1], generations[1]);
  EXPECT_EQ(replaced_inputs[0].input_params.embedding.request_ids,
            first_inputs[0].input_params.embedding.request_ids);
  for (const auto& input : replaced_inputs) {
    EXPECT_EQ(input.input_params.parallel.dp_global_token_nums,
              first_inputs[0].input_params.parallel.dp_global_token_nums);
    EXPECT_EQ(input.input_params.parallel.dp_global_batch_generations,
              replaced_generations);
    expect_packed_metadata(input);
  }
  replaced_inputs[1].input_params.parallel.dp_global_batch_generations[0] = 99;
  EXPECT_EQ(
      replaced_inputs[0].input_params.parallel.dp_global_batch_generations,
      replaced_generations);
  EXPECT_EQ(first_inputs[0].input_params.parallel.dp_global_batch_generations,
            generations);

  auto moved_inputs = prepare(moved_slot);
  EXPECT_EQ(moved_inputs[1].input_params.embedding.request_ids,
            replaced_inputs[1].input_params.embedding.request_ids);
  EXPECT_NE(moved_inputs[1].input_params.embedding.embedding_ids,
            replaced_inputs[1].input_params.embedding.embedding_ids);
  EXPECT_GT(
      moved_inputs[0].input_params.parallel.dp_global_batch_generations[1],
      replaced_generations[1]);

  BatchGroup empty_batches(2);
  std::vector<VlmForwardInput> empty_inputs;
  factory.create_inputs(empty_batches, ModelArgs{}, empty_inputs);
  ASSERT_EQ(empty_inputs.size(), 2u);
  const auto empty_generations =
      empty_inputs[0].input_params.parallel.dp_global_batch_generations;
  ASSERT_EQ(empty_generations.size(), 2u);
  EXPECT_GT(empty_generations[0], replaced_generations[0]);
  EXPECT_GT(empty_generations[1], replaced_generations[1]);
  for (const auto& input : empty_inputs) {
    EXPECT_EQ(input.input_params.parallel.dp_global_batch_generations,
              empty_generations);
    expect_packed_metadata(input);
  }
  auto restored_inputs = prepare(replacement);
  for (const auto& input : restored_inputs) {
    ASSERT_EQ(input.input_params.parallel.dp_global_batch_generations.size(),
              2u);
    EXPECT_GT(input.input_params.parallel.dp_global_batch_generations[0],
              empty_generations[0]);
    EXPECT_GT(input.input_params.parallel.dp_global_batch_generations[1],
              empty_generations[1]);
  }
}

TEST_F(VlmForwardInputFactoryTest, PropagatesWarmupToEmptyRanksAndClearsIt) {
  for (const uint32_t active_rank : {0u, 1u}) {
    SCOPED_TRACE(active_rank);
    BlockManager::Options manager_options;
    manager_options.num_blocks(4).block_size(4);
    BlockManagerImpl manager(manager_options);
    Sequence warmup = make_sequence({1, 2}, "warmup", /*is_graph_warmup=*/true);
    warmup.add_blocks(BlockType::KV, manager.allocate(/*num_blocks=*/1));
    warmup.kv_state().set_kv_cache_tokens_num(2);
    warmup.append_token(Token(3));
    ForwardInputFactoryOptions options;
    options.dp_size = 2;
    options.enable_graph = true;
    VlmInputFactory factory(options);
    BatchGroup batches(2);
    batches[active_rank].add(&warmup);
    std::vector<VlmForwardInput> inputs;
    EXPECT_TRUE(factory.create_inputs(batches, ModelArgs{}, inputs));
    ASSERT_EQ(inputs.size(), 2u);
    for (const auto& input : inputs) {
      EXPECT_TRUE(input.input_params.meta.is_graph_warmup);
      EXPECT_TRUE(input.input_params.meta.batch_forward_type.is_decode());
      expect_packed_metadata(input);
    }

    Sequence request = make_sequence({4, 5}, "request");
    request.add_blocks(BlockType::KV, manager.allocate(/*num_blocks=*/1));
    BatchGroup serving_batches(2);
    serving_batches[1 - active_rank].add(&request);
    std::vector<VlmForwardInput> serving_inputs;
    EXPECT_FALSE(
        factory.create_inputs(serving_batches, ModelArgs{}, serving_inputs));
    for (const auto& input : serving_inputs) {
      EXPECT_FALSE(input.input_params.meta.is_graph_warmup);
      expect_packed_metadata(input);
    }

    BatchGroup empty_batches(2);
    EXPECT_FALSE(factory.create_inputs(empty_batches, ModelArgs{}, inputs));
    for (const auto& input : inputs) {
      EXPECT_FALSE(input.input_params.meta.is_graph_warmup);
    }
  }
}

}  // namespace
}  // namespace xllm
