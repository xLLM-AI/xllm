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

#include "core/framework/batch/rec_forward_input_factory.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/framework/batch/rec_batch_factory.h"
#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/request/request.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/kv_cache/block/block_manager_impl.h"

namespace xllm {
namespace {

template <typename T>
class ScopedConfigValue final {
 public:
  ScopedConfigValue(T& value, T new_value) : value_(value), old_value_(value) {
    value_ = new_value;
  }

  ~ScopedConfigValue() { value_ = old_value_; }

 private:
  T& value_;
  T old_value_;
};

std::shared_ptr<Request> make_request(const std::vector<int32_t>& tokens,
                                      int32_t rank,
                                      int32_t beam_width = 1) {
  RequestSamplingParam sampling;
  sampling.beam_width = beam_width;
  StoppingChecker stopping;
  stopping.set_max_generated_tokens(/*max_generated_tokens=*/8);
  stopping.set_ignore_eos(true);
  RequestState state("prompt",
                     tokens,
                     sampling,
                     SchedulerParam{},
                     stopping,
                     /*seq_capacity=*/32,
                     /*n=*/1,
                     /*best_of=*/1,
                     /*logprobs=*/false,
                     /*stream=*/false,
                     /*echo=*/false,
                     /*skip_special_tokens=*/true,
                     /*enable_schedule_overlap=*/false,
                     /*output_func=*/nullptr,
                     /*outputs_func=*/nullptr);
  state.rec_type = RecType::kLlmRec;
  auto request = std::make_shared<Request>("request", "", "", std::move(state));
  request->sequences().front()->set_dp_rank(rank);
  return request;
}

TEST(RecForwardInputFactoryTest, RefreshesForwardTypeAfterSequenceAdvance) {
  BlockManager::Options manager_options;
  manager_options.num_blocks(4).block_size(4);
  BlockManagerImpl manager(manager_options);
  auto request = make_request({1, 2, 3}, /*rank=*/0);
  Sequence* sequence = request->sequences()[0].get();
  sequence->add_blocks(BlockType::KV, manager.allocate(/*num_blocks=*/1));
  RecBatchFactory batch_factory(/*dp_size=*/2, BatchInputType::SEQUENCE);
  auto batches = batch_factory.create_batches({request}, {sequence}, {3});
  RecForwardInputFactoryOptions options;
  options.dp_size = 2;
  RecForwardInputFactory factory(options);
  std::vector<RecForwardInput> inputs;
  factory.create_inputs(batches, ModelArgs(), inputs);

  ASSERT_EQ(inputs.size(), 2u);
  EXPECT_TRUE(inputs[0].input_params.meta.batch_forward_type.is_prefill());
  EXPECT_EQ(inputs[0].input_params.parallel.dp_global_token_nums,
            (std::vector<int32_t>{3, 0}));
  EXPECT_EQ(sequence->kv_state().kv_cache_tokens_num(), 3u);
  sequence->append_token(Token(4));
  factory.create_inputs(batches, ModelArgs(), inputs);

  ASSERT_EQ(inputs.size(), 2u);
  EXPECT_TRUE(torch::equal(inputs[0].host_token_ids(),
                           torch::tensor({4}, torch::kInt32)));
  for (const auto& input : inputs) {
    EXPECT_TRUE(input.input_params.meta.batch_forward_type.is_decode());
    EXPECT_EQ(input.input_params.parallel.dp_global_token_nums,
              (std::vector<int32_t>{1, 0}));
    EXPECT_EQ(input.input_params.parallel.raw_dp_global_token_nums,
              (std::vector<int32_t>{1, 0}));
    EXPECT_EQ(input.input_params.parallel.dp_global_sequence_nums,
              (std::vector<int32_t>{1, 0}));
    EXPECT_EQ(input.input_params.parallel.dp_is_decode,
              (std::vector<int32_t>{1, 0}));
  }
}

TEST(RecForwardInputFactoryTest, AggregatesDifferentRankSequenceCounts) {
  BlockManager::Options manager_options;
  manager_options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(manager_options);
  auto first = make_request({1, 2}, /*rank=*/0);
  auto second = make_request({3}, /*rank=*/0);
  auto third = make_request({4, 5, 6}, /*rank=*/1);
  std::vector<Sequence*> sequences{first->sequences()[0].get(),
                                   second->sequences()[0].get(),
                                   third->sequences()[0].get()};
  for (Sequence* sequence : sequences) {
    sequence->add_blocks(BlockType::KV, manager.allocate(/*num_blocks=*/1));
  }
  RecBatchFactory batch_factory(/*dp_size=*/2, BatchInputType::SEQUENCE);
  auto batches = batch_factory.create_batches(
      {first, second, third}, sequences, {2, 1, 3});
  RecForwardInputFactoryOptions options;
  options.dp_size = 2;
  RecForwardInputFactory factory(options);
  std::vector<RecForwardInput> inputs;
  factory.create_inputs(batches, ModelArgs(), inputs);

  ASSERT_EQ(inputs.size(), 2u);
  for (const auto& input : inputs) {
    EXPECT_EQ(input.input_params.parallel.dp_global_token_nums,
              (std::vector<int32_t>{3, 3}));
    EXPECT_EQ(input.input_params.parallel.dp_global_sequence_nums,
              (std::vector<int32_t>{2, 1}));
    EXPECT_EQ(input.input_params.parallel.dp_is_decode,
              (std::vector<int32_t>{0, 0}));
  }
}

TEST(RecForwardInputFactoryTest, MultiRoundInputKeepsScheduledTokenBudgets) {
  ScopedConfigValue<int32_t> decode_rounds(
      RecConfig::get_instance().max_decode_rounds(), /*new_value=*/3);
  ScopedConfigValue<int32_t> beam_width(
      BeamSearchConfig::get_instance().beam_width(), /*new_value=*/2);
  auto first = make_request({1, 2, 3}, /*rank=*/0, /*beam_width=*/2);
  auto second = make_request({4, 5, 6}, /*rank=*/0, /*beam_width=*/2);
  auto unscheduled = first->sequences()[0]->fork(/*index=*/1);
  unscheduled->finish();
  first->sequences().emplace_back(std::move(unscheduled));
  RecBatchFactory batch_factory(/*dp_size=*/1, BatchInputType::REC_MULTI_ROUND);
  auto batches = batch_factory.create_batches(
      {first, second},
      {first->sequences()[0].get(), second->sequences()[0].get()},
      {1, 2});
  RecForwardInputFactory factory(RecForwardInputFactoryOptions{});
  auto input = factory.create_input(batches[0], ModelArgs());

  EXPECT_TRUE(torch::equal(input.host_token_ids(),
                           torch::tensor({1, 4, 5}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(input.host_positions(),
                           torch::tensor({0, 0, 1}, torch::kInt32)));
  EXPECT_EQ(input.input_params.meta.num_sequences, 2);
  EXPECT_EQ(input.input_params.meta.q_max_seq_len, 2);
  EXPECT_EQ(input.input_params.meta.batch_id, batches[0].batch_id());
  ASSERT_TRUE(input.has_step_meta());
  EXPECT_EQ(input.step_meta()->batch_size, 2);
  EXPECT_EQ(input.step_meta()->beam_width, 2);
  EXPECT_EQ(input.step_meta()->total_round, 3);
  EXPECT_EQ(input.step_meta()->decode_positions_vec,
            (std::vector<int32_t>{3, 3}));
  EXPECT_TRUE(torch::equal(input.decoder_sampling_params.selected_token_idxes,
                           torch::tensor({0, 1, 2, 3}, torch::kInt32)));
}

TEST(RecForwardInputFactoryTest, EmptySingleBatchReturnsEmptyInput) {
  RecForwardInputFactory factory(RecForwardInputFactoryOptions{});
  RecBatch batch(BatchInputType::REC_MULTI_ROUND);
  auto input = factory.create_input(batch, ModelArgs());

  EXPECT_FALSE(input.token_ids.defined());
  EXPECT_FALSE(input.has_step_meta());
  EXPECT_EQ(input.input_params.meta.num_sequences, 0);
}

}  // namespace
}  // namespace xllm
