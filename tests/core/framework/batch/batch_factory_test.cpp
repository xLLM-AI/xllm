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

#include "core/framework/batch/batch_factory.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/framework/batch/batch_output_action.h"
#include "core/framework/batch/sampling_input_builder.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/multimodal/mm_batch_data.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_type.h"
#include "core/framework/request/request.h"
#include "core/framework/request/stopping_checker.h"
#include "core/kv_cache/block/block_manager_impl.h"
#include "core/layers/common/attention_metadata.h"
#include "core/util/hash_util.h"

namespace xllm {
namespace {

std::shared_ptr<Request> make_request(int32_t rank,
                                      std::vector<int32_t> prompt_tokens = {1,
                                                                            2,
                                                                            3},
                                      MMData mm_data = {},
                                      torch::Tensor input_embedding = {}) {
  RequestSamplingParam sampling;

  StoppingChecker stopping;
  stopping.set_max_generated_tokens(/*max_generated_tokens=*/8);
  stopping.set_max_context_len(/*max_context_len=*/64);
  stopping.set_ignore_eos(true);
  RequestState state("prompt",
                     std::move(prompt_tokens),
                     sampling,
                     SchedulerParam{},
                     stopping,
                     /*seq_capacity=*/64,
                     /*n=*/1,
                     /*best_of=*/1,
                     /*logprobs=*/false,
                     /*stream=*/false,
                     /*echo=*/false,
                     /*skip_special_tokens=*/true,
                     /*enable_schedule_overlap=*/false,
                     /*output_func=*/nullptr,
                     /*outputs_func=*/nullptr);
  state.mm_data = std::move(mm_data);
  state.input_embedding = std::move(input_embedding);
  auto request = std::make_shared<Request>("request", "", "", std::move(state));
  request->sequences().front()->set_dp_rank(rank);
  return request;
}

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

TEST(SamplingInputBuilderTest, MergeOffsetsSelectedRows) {
  RequestSamplingParam params;
  SamplingInputBuilder first;
  first.append(&params, 1);
  first.append(&params, 3);
  SamplingInputBuilder second;
  second.append(&params, 0);
  first.merge(std::move(second), /*token_offset=*/4);
  EXPECT_EQ(first.selected_token_indices(), (std::vector<int32_t>{1, 3, 4}));
  const auto sampling = first.build();
  EXPECT_TRUE(torch::equal(sampling.selected_token_idxes(),
                           torch::tensor({1, 3, 4}, torch::kInt)));
}

TEST(SamplingInputBuilderTest, PadsAndMergesAdjustedTokenCounts) {
  RequestSamplingParam params;
  params.frequency_penalty = 1.0;
  SamplingInputBuilder first;
  const SamplingInputBuilder::TokenCounts counts{{42, 3}};
  const SamplingInputBuilder::TokenCounts excluded{{42, 1}};
  first.append(&params, 0, &counts, &excluded);
  SamplingInputBuilder second;
  const SamplingInputBuilder::TokenCounts fully_excluded{{42, 3}};
  second.append(&params, 0, &counts, &fully_excluded);
  first.merge(std::move(second), /*token_offset=*/1);
  const auto sampling = first.build();
  EXPECT_TRUE(torch::equal(sampling.unique_token_ids(),
                           torch::tensor({{42}, {0}}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(sampling.unique_token_counts(),
                           torch::tensor({{2}, {0}}, torch::kInt)));
  EXPECT_TRUE(torch::equal(sampling.unique_token_ids_lens(),
                           torch::tensor({1, 0}, torch::kInt)));
}

TEST(BatchFactoryTest, IndependentFactoriesKeepTheirOwnDpSize) {
  BatchFactory single(/*dp_size=*/1);
  BatchFactory multiple(/*dp_size=*/3);
  auto request = make_request(/*rank=*/2);
  auto batches =
      multiple.create_batches({request->sequences().front().get()}, {2});
  ASSERT_EQ(batches.size(), 3);
  EXPECT_TRUE(batches[0].empty());
  EXPECT_TRUE(batches[1].empty());
  EXPECT_EQ(batches[2].size(), 1);
  EXPECT_EQ(single.create_batches({}, {}).size(), 1);
  EXPECT_EQ(multiple.create_batches({}, {}).size(), 3);
}

TEST(BatchFactoryTest, PreservesRankLocalSequenceOrderAndBudgets) {
  BatchFactory factory(/*dp_size=*/9);
  auto first = make_request(/*rank=*/8);
  auto second = make_request(/*rank=*/0);
  auto third = make_request(/*rank=*/8);
  auto* seq1 = first->sequences().front().get();
  auto* seq2 = second->sequences().front().get();
  auto* seq3 = third->sequences().front().get();
  auto batches = factory.create_batches({seq1, seq2, seq3}, {1, 2, 3});
  ASSERT_EQ(batches.size(), 9);
  EXPECT_EQ(batches[0].get_sequences(), (std::vector<Sequence*>{seq2}));
  EXPECT_EQ(batches[0].get_allowed_max_tokens(), (std::vector<uint32_t>{2}));
  for (size_t rank = 1; rank < 8; ++rank) {
    EXPECT_TRUE(batches[rank].empty());
    EXPECT_EQ(batches[rank].batch_id(), UNINITIALIZED_BATCH_ID);
  }
  EXPECT_EQ(batches[8].get_sequences(), (std::vector<Sequence*>{seq1, seq3}));
  EXPECT_EQ(batches[8].get_allowed_max_tokens(), (std::vector<uint32_t>{1, 3}));
  EXPECT_NE(batches[0].batch_id(), UNINITIALIZED_BATCH_ID);
  EXPECT_NE(batches[0].batch_id(), batches[8].batch_id());
}

TEST(BatchStateTest, MixedForwardTypeSurvivesAdditionalStagesAndRefreshes) {
  auto prefill = make_request(/*rank=*/0);
  auto decode = make_request(/*rank=*/0);
  auto chunked = make_request(/*rank=*/0);
  auto* prefill_sequence = prefill->sequences()[0].get();
  auto* decode_sequence = decode->sequences()[0].get();
  auto* chunked_sequence = chunked->sequences()[0].get();
  decode_sequence->kv_state().set_kv_cache_tokens_num(/*num=*/3);
  chunked_sequence->kv_state().set_kv_cache_tokens_num(/*num=*/1);
  BatchState state;
  state.reserve(/*sequence_count=*/3, /*group_count=*/0);
  state.add(prefill_sequence, /*token_budget=*/3);
  EXPECT_TRUE(state.prepare_sequence_input_data().forward_type.is_prefill());
  state.add(decode_sequence, /*token_budget=*/1);
  EXPECT_TRUE(state.prepare_sequence_input_data().forward_type.is_mixed());
  state.add(chunked_sequence, /*token_budget=*/2);
  EXPECT_TRUE(state.prepare_sequence_input_data().forward_type.is_mixed());

  // A reused batch must recompute its type after sequence KV state changes.
  for (auto* sequence : state.sequence_plan().sequences()) {
    sequence->kv_state().set_kv_cache_tokens_num(/*num=*/0);
  }
  state.refresh_forward_type(state.sequence_plan().sequences());
  EXPECT_TRUE(state.prepare_sequence_input_data().forward_type.is_prefill());
  for (auto* sequence : state.sequence_plan().sequences()) {
    sequence->kv_state().set_kv_cache_tokens_num(/*num=*/1);
  }
  state.refresh_forward_type(state.sequence_plan().sequences());
  EXPECT_TRUE(
      state.prepare_sequence_input_data().forward_type.is_chunked_prefill());
  for (auto* sequence : state.sequence_plan().sequences()) {
    sequence->kv_state().set_kv_cache_tokens_num(/*num=*/3);
  }
  state.refresh_forward_type(state.sequence_plan().sequences());
  EXPECT_TRUE(state.prepare_sequence_input_data().forward_type.is_decode());
  prefill_sequence->kv_state().set_kv_cache_tokens_num(/*num=*/0);
  state.refresh_forward_type(state.sequence_plan().sequences());
  EXPECT_TRUE(state.prepare_sequence_input_data().forward_type.is_mixed());
}

TEST(BatchSequencePlanTest, ReorderingMovesSequencesTogetherWithBudgets) {
  auto first = make_request(0);
  auto second = make_request(0);
  BatchSequencePlan plan;
  plan.add(first->sequences()[0].get(), 2);
  plan.add(second->sequences()[0].get(), 5);
  plan.reorder({1, 0});
  EXPECT_EQ(plan[0].sequence, second->sequences()[0].get());
  EXPECT_EQ(plan[0].token_budget, 5);
  EXPECT_EQ(plan[1].sequence, first->sequences()[0].get());
  EXPECT_EQ(plan[1].token_budget, 2);
}

TEST(BatchSequencePlanDeathTest, RejectsInvalidPermutation) {
  auto request = make_request(0);
  BatchSequencePlan plan;
  plan.add(request->sequences()[0].get(), 1);
  plan.add(request->sequences()[0].get(), 2);
  EXPECT_DEATH(plan.reorder({0, 0}), "must be a permutation");
  EXPECT_DEATH(plan.reorder({0, 2}), "index < size");
  EXPECT_DEATH(plan.reorder({0}), "source_indices.size");
}

TEST(BatchFactoryDeathTest, RejectsMismatchedBudgetsAndInvalidRanks) {
  BatchFactory factory(/*dp_size=*/2);
  auto request = make_request(/*rank=*/0);
  auto* sequence = request->sequences()[0].get();
  EXPECT_DEATH(factory.create_batches({sequence}, {}),
               "Each scheduled sequence requires one token budget");
  EXPECT_DEATH(factory.create_batches({sequence}, {0}), "budgets");
  const size_t overflow_budget =
      static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1;
  EXPECT_DEATH(factory.create_batches({sequence}, {overflow_budget}),
               "budgets");
  sequence->set_dp_rank(-1);
  EXPECT_DEATH(factory.create_batches({sequence}, {1}), "dp_rank");
  sequence->set_dp_rank(2);
  EXPECT_DEATH(factory.create_batches({sequence}, {1}), "dp_rank");
}

TEST(BatchFactoryDeathTest, RejectsFinishedScheduledSequences) {
  BatchFactory factory(/*dp_size=*/2);
  auto active_request = make_request(/*rank=*/0);
  auto finished_request = make_request(/*rank=*/1);
  auto* active_sequence = active_request->sequences()[0].get();
  auto* finished_sequence = finished_request->sequences()[0].get();
  finished_sequence->finish();
  EXPECT_DEATH(
      factory.create_batches({active_sequence, finished_sequence}, {1, 1}),
      "finished");

  Batch batch;
  EXPECT_DEATH(batch.add(finished_sequence), "finished");
}

TEST(BatchFactoryDeathTest, RejectsNullScheduledSequences) {
  BatchFactory factory(/*dp_size=*/1);
  EXPECT_DEATH(factory.create_batches({nullptr}, {1}), "sequence != nullptr");

  Batch batch;
  EXPECT_DEATH(batch.add(static_cast<Sequence*>(nullptr)),
               "sequence != nullptr");
}

TEST(BatchFactoryTest, MixedEmbeddingInputsKeepRankLocalOrderAcrossCalls) {
  BatchFactory factory(/*dp_size=*/9);
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);
  const auto first_embedding =
      torch::arange(12, torch::kFloat32).reshape({3, 4});
  const auto last_embedding =
      torch::arange(12, 24, torch::kFloat32).reshape({3, 4});
  const auto other_embedding = torch::full({3, 4}, 42.0F);
  auto first = make_request(/*rank=*/8,
                            /*prompt_tokens=*/{11, 12, 13},
                            /*mm_data=*/{},
                            first_embedding);
  auto plain = make_request(/*rank=*/8);
  auto other_rank = make_request(/*rank=*/0,
                                 /*prompt_tokens=*/{31, 32, 33},
                                 /*mm_data=*/{},
                                 other_embedding);
  auto last = make_request(/*rank=*/8,
                           /*prompt_tokens=*/{21, 22, 23},
                           /*mm_data=*/{},
                           last_embedding);
  const std::vector<std::shared_ptr<Request>> requests{
      plain, first, other_rank, last};
  std::vector<Sequence*> sequences;
  sequences.reserve(requests.size());
  for (const auto& request : requests) {
    auto* sequence = request->sequences()[0].get();
    sequence->add_blocks(BlockType::KV, manager.allocate(/*num_blocks=*/1));
    sequences.emplace_back(sequence);
  }
  auto batches = factory.create_batches(sequences, {3, 3, 3, 3});
  ASSERT_EQ(batches.size(), 9);
  ModelArgs args{};
  const auto input = batches[8].prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0, args);
  EXPECT_TRUE(torch::equal(input.input_params.embedding.input_embedding,
                           torch::cat({first_embedding, last_embedding})));
  EXPECT_TRUE(torch::equal(
      input.token_ids,
      torch::tensor({1, 2, 3, 11, 12, 13, 21, 22, 23}, torch::kInt32)));
  const auto other_input = batches[0].prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0, args);
  EXPECT_TRUE(torch::equal(other_input.input_params.embedding.input_embedding,
                           other_embedding));

  auto next = make_request(/*rank=*/8);
  auto* next_sequence = next->sequences()[0].get();
  next_sequence->add_blocks(BlockType::KV, manager.allocate(/*num_blocks=*/1));
  auto next_batches = factory.create_batches({next_sequence}, {3});
  ASSERT_EQ(next_batches.size(), 9);
  const auto next_input = next_batches[8].prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0, args);
  EXPECT_FALSE(next_input.input_params.embedding.input_embedding.defined());
  EXPECT_TRUE(next_batches[0].empty());
}
}  // namespace
}  // namespace xllm
