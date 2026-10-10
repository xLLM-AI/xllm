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

#include "core/framework/batch/vlm_forward_input_factory.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/batch/batch_group.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/kv_cache/block/block_manager_impl.h"

namespace xllm {
namespace {

class VlmForwardInputFactoryTest : public ::testing::Test {
 protected:
  VlmForwardInputFactoryTest() {
    stopping_checker_.set_max_generated_tokens(/*max_generated_tokens=*/8);
  }

  Sequence make_sequence(const std::vector<int32_t>& tokens,
                         const std::string& request_id) {
    SequenceParams params;
    params.seq_capacity = tokens.size() + 8;
    params.sampling_param = &sampling_params_;
    params.stopping_checker = &stopping_checker_;
    params.request_id = request_id;
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
  VlmForwardInputFactoryOptions options;
  options.dp_size = 2;
  options.enable_dp_global_json_object_active = true;
  VlmForwardInputFactory factory(options);
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

TEST_F(VlmForwardInputFactoryTest, EmptyRankParticipatesInGraphDecode) {
  BlockManager::Options manager_options;
  manager_options.num_blocks(4).block_size(4);
  BlockManagerImpl manager(manager_options);
  Sequence decode = make_sequence({1, 2}, "decode");
  decode.add_blocks(BlockType::KV, manager.allocate(/*num_blocks=*/1));
  decode.kv_state().set_kv_cache_tokens_num(2);
  decode.append_token(Token(3));
  BatchGroup batches(2);
  batches[1].add(&decode);
  ScopedGraphMode graph_mode(/*enabled=*/true);
  VlmForwardInputFactoryOptions options;
  options.dp_size = 2;
  VlmForwardInputFactory factory(options);
  std::vector<VlmForwardInput> inputs;
  factory.create_inputs(batches, ModelArgs(), inputs);

  ASSERT_EQ(inputs.size(), 2u);
  EXPECT_FALSE(inputs[0].token_ids.defined());
  EXPECT_TRUE(inputs[0].input_params.meta.batch_forward_type.is_decode());
  EXPECT_EQ(inputs[0].input_params.meta.num_sequences, 0);
  for (const auto& input : inputs) {
    EXPECT_EQ(input.input_params.parallel.dp_global_token_nums,
              (std::vector<int32_t>{0, 1}));
    EXPECT_EQ(input.input_params.parallel.dp_global_sequence_nums,
              (std::vector<int32_t>{0, 1}));
    EXPECT_EQ(input.input_params.parallel.dp_is_decode,
              (std::vector<int32_t>{1, 1}));
    EXPECT_TRUE(
        input.input_params.parallel.dp_global_json_object_active.empty());
  }
}

TEST_F(VlmForwardInputFactoryTest, AllEmptyGroupClearsPreviousInputs) {
  ScopedGraphMode graph_mode(/*enabled=*/true);
  VlmForwardInputFactoryOptions options;
  options.dp_size = 2;
  VlmForwardInputFactory factory(options);
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

}  // namespace
}  // namespace xllm
