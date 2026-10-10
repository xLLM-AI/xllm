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

#include "core/framework/batch/rec_batch_factory.h"
#include "core/framework/batch/rec_batch_output_handler.h"
#include "core/framework/batch/sampling_input_builder.h"
#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/rec_model_params.h"
#include "core/framework/multimodal/mm_batch_data.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_type.h"
#include "core/framework/request/onerec_sequence.h"
#include "core/framework/request/rec_sequence.h"
#include "core/framework/request/request.h"
#include "core/framework/request/stopping_checker.h"
#include "core/kv_cache/block/block_manager_impl.h"
#include "core/layers/common/attention_metadata.h"
#include "core/runtime/rec_forward_params.h"
#include "core/util/hash_util.h"

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

std::shared_ptr<Request> make_request(int32_t rank,
                                      RecType rec_type = RecType::kNone,
                                      int32_t beam_width = 1,
                                      std::vector<int32_t> prompt_tokens = {1,
                                                                            2,
                                                                            3},
                                      MMData mm_data = {}) {
  RequestSamplingParam sampling;
  sampling.beam_width = beam_width;
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
  state.rec_type = rec_type;
  state.mm_data = std::move(mm_data);
  auto request = std::make_shared<Request>("request", "", "", std::move(state));
  request->sequences().front()->set_dp_rank(rank);
  return request;
}

MMData make_decoder_context_data() {
  MMData mm_data;
  mm_data.add(MMType::EMBEDDING,
              OneRecSequence::kDecoderContextEmbeddingName,
              torch::zeros({2, 4}));
  return mm_data;
}

TEST(SamplingInputBuilderTest,
     MergeOffsetsSelectedRowsAndSampleRowsSeparately) {
  RequestSamplingParam params;
  SamplingInputBuilder first;
  first.append(&params, 1, nullptr, nullptr, /*sample=*/false);
  first.append(&params, 3);
  SamplingInputBuilder second;
  second.append(&params, 0);
  first.merge(std::move(second), /*token_offset=*/4);
  EXPECT_EQ(first.selected_token_indices(), (std::vector<int32_t>{1, 3, 4}));
  EXPECT_EQ(first.sample_indices(), (std::vector<int32_t>{1, 2}));
  const auto sampling = first.build();
  EXPECT_TRUE(torch::equal(sampling.selected_token_idxes,
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
  EXPECT_TRUE(torch::equal(sampling.unique_token_ids,
                           torch::tensor({{42}, {0}}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(sampling.unique_token_counts,
                           torch::tensor({{2}, {0}}, torch::kInt)));
  EXPECT_TRUE(torch::equal(sampling.unique_token_ids_lens,
                           torch::tensor({1, 0}, torch::kInt)));
}

TEST(BatchFactoryTest, FactoriesKeepDomainsSeparateForTheSameInputContract) {
  BatchFactory sequence_factory(/*dp_size=*/1);
  RecBatchFactory rec_factory(/*dp_size=*/1, BatchInputType::SEQUENCE);
  auto sequence_batches = sequence_factory.create_batches({}, {}, {});
  auto rec_batches = rec_factory.create_batches({}, {}, {});
  static_assert(std::is_same_v<decltype(sequence_batches[0]), Batch&>);
  static_assert(std::is_same_v<decltype(rec_batches[0]), RecBatch&>);
  static_assert(!std::is_convertible_v<RecBatch&, Batch&>);
  EXPECT_EQ(sequence_batches[0].size(), 0);
  EXPECT_EQ(rec_batches[0].size(), 0);
  EXPECT_EQ(sequence_batches[0].input_type(), BatchInputType::SEQUENCE);
  EXPECT_EQ(rec_batches[0].input_type(), BatchInputType::SEQUENCE);
  ModelArgs args;
  const auto input =
      rec_batches[0].prepare_forward_input(args, /*thread_pool=*/nullptr);
  static_assert(std::is_same_v<std::decay_t<decltype(input)>, RecForwardInput>);
  static_assert(std::is_same_v<decltype(input.input_params), RecModelParams>);
  static_assert(!std::is_base_of_v<LlmForwardInput, RecForwardInput>);
  static_assert(!std::is_base_of_v<ModelInputParams, RecModelParams>);
  static_assert(!std::is_convertible_v<RecForwardInput, LlmForwardInput>);
  static_assert(!std::is_convertible_v<RecModelParams, ModelInputParams>);
  EXPECT_FALSE(input.token_ids.defined());
}

TEST(RecForwardInputTest, DeviceConversionPreservesRecPayloadAndSampling) {
  RecForwardInput source;
  source.token_ids = torch::tensor({7, 8}, torch::kInt32);
  source.positions = torch::tensor({2, 3}, torch::kInt32);
  source.input_params.meta.num_sequences = 1;
  source.input_params.meta.batch_id = 17;
  source.input_params.embedding.linear_state_ids = {4};
  const auto values = torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}});
  const auto indices = torch::tensor({0, 2}, torch::kInt64);
  source.input_params.features.mm_data = MMBatchData(
      MMType::EMBEDDING,
      {{"MULTI_MODAL_VALUES", values}, {"MULTI_MODAL_INDICES", indices}});
  auto& xattention = source.input_params.mutable_onerec_xattention_params();
  xattention.encoder_seq_lens = {3};
  xattention.encoder_token_ids = torch::tensor({1, 2, 3}, torch::kInt32);
  xattention.decoder_context_embedding = values;
  xattention.beam_width_tensor = torch::tensor({2}, torch::kInt32);
  source.sampling_params.selected_token_idxes =
      torch::tensor({1}, torch::kInt32);
  source.decoder_sampling_params.selected_token_idxes =
      torch::tensor({0, 1}, torch::kInt32);
  source.decoder_sampling_params.temperatures = torch::tensor({0.5F, 0.75F});
  source.step_decode = StepDecodeMeta{.batch_size = 1,
                                      .beam_width = 2,
                                      .current_round = 1,
                                      .total_round = 3,
                                      .full_kv_shape = {6, 2, 4},
                                      .decode_positions_vec = {3}};

  const auto input = source.to(torch::Device(torch::kCPU), torch::kFloat32);
  EXPECT_TRUE(input.runtime.device_tensors_ready);
  EXPECT_TRUE(torch::equal(input.token_ids, source.token_ids));
  EXPECT_TRUE(torch::equal(input.host_token_ids(), source.token_ids));
  EXPECT_TRUE(torch::equal(input.host_positions(), source.positions));
  EXPECT_EQ(input.input_params.meta.batch_id, 17);
  EXPECT_TRUE(torch::equal(input.input_params.embedding.linear_state_indices,
                           torch::tensor({4}, torch::kInt32)));
  const auto converted_values =
      input.input_params.features.mm_data.get<torch::Tensor>(
          "MULTI_MODAL_VALUES");
  const auto converted_indices =
      input.input_params.features.mm_data.get<torch::Tensor>(
          "MULTI_MODAL_INDICES");
  ASSERT_TRUE(converted_values.has_value());
  ASSERT_TRUE(converted_indices.has_value());
  EXPECT_TRUE(torch::equal(*converted_values, values));
  EXPECT_TRUE(torch::equal(*converted_indices, indices));
  EXPECT_EQ(converted_indices->scalar_type(), torch::kInt64);
  const auto* converted_xattention =
      input.input_params.onerec_xattention_params();
  ASSERT_NE(converted_xattention, nullptr);
  EXPECT_EQ(converted_xattention->encoder_seq_lens, (std::vector<int32_t>{3}));
  EXPECT_TRUE(torch::equal(converted_xattention->encoder_token_ids,
                           xattention.encoder_token_ids));
  EXPECT_TRUE(
      torch::equal(converted_xattention->decoder_context_embedding, values));
  EXPECT_TRUE(torch::equal(input.sampling_params.selected_token_idxes,
                           source.sampling_params.selected_token_idxes));
  EXPECT_TRUE(
      torch::equal(input.decoder_sampling_params.selected_token_idxes,
                   source.decoder_sampling_params.selected_token_idxes));
  EXPECT_TRUE(torch::equal(input.decoder_sampling_params.temperatures,
                           source.decoder_sampling_params.temperatures));
  ASSERT_TRUE(input.has_step_meta());
  EXPECT_EQ(input.step_meta()->batch_size, 1);
  EXPECT_EQ(input.step_meta()->beam_width, 2);
  EXPECT_EQ(input.step_meta()->current_round, 1);
  EXPECT_EQ(input.step_meta()->total_round, 3);
  EXPECT_EQ(input.step_meta()->full_kv_shape, (std::vector<int64_t>{6, 2, 4}));
  EXPECT_EQ(input.step_meta()->decode_positions_vec, (std::vector<int32_t>{3}));
}

TEST(RecModelParamsTest, DeviceConversionPreservesMultiRoundStrategy) {
  RecModelParams source;
  auto& multi_round = source.mutable_llmrec_params();
  multi_round.batch_size = 1;
  multi_round.beam_width = 2;
  multi_round.total_round = 3;
  multi_round.current_round_tensor = torch::tensor({1}, torch::kInt32);
  multi_round.full_k_caches = {torch::ones({2, 3, 4})};
  multi_round.full_v_caches = {torch::zeros({2, 3, 4})};
  multi_round.decode_positions_tensor_list = {
      torch::tensor({2, 2}, torch::kInt32)};

  const auto input = source.to(torch::Device(torch::kCPU));
  const auto* converted = input.llmrec_params();
  ASSERT_NE(converted, nullptr);
  EXPECT_FALSE(input.has_onerec_params());
  EXPECT_EQ(converted->batch_size, 1);
  EXPECT_EQ(converted->beam_width, 2);
  EXPECT_EQ(converted->total_round, 3);
  EXPECT_TRUE(torch::equal(converted->current_round_tensor,
                           multi_round.current_round_tensor));
  ASSERT_EQ(converted->full_k_caches.size(), 1);
  ASSERT_EQ(converted->full_v_caches.size(), 1);
  ASSERT_EQ(converted->decode_positions_tensor_list.size(), 1);
  EXPECT_TRUE(
      torch::equal(converted->full_k_caches[0], multi_round.full_k_caches[0]));
  EXPECT_TRUE(
      torch::equal(converted->full_v_caches[0], multi_round.full_v_caches[0]));
  EXPECT_TRUE(torch::equal(converted->decode_positions_tensor_list[0],
                           multi_round.decode_positions_tensor_list[0]));
}

TEST(RecModelParamsTest, BorrowedExecutionViewMutatesNativeOwnedState) {
  RecModelParams owner;
  owner.attention.host.kv_seq_lens = {2, 3};
  owner.embedding.request_ids = {"first", "second"};
  owner.embedding.input_embedding = torch::ones({2, 4});
  auto& strategy = owner.mutable_llmrec_params();
  strategy.full_k_caches = {torch::ones({2, 3, 4})};
  const auto* original_lengths = owner.attention.host.kv_seq_lens.data();
  const auto* original_request_ids = owner.embedding.request_ids.data();
  const void* original_embedding = owner.embedding.input_embedding.data_ptr();
  const void* original_cache = strategy.full_k_caches[0].data_ptr();
  {
    ModelInputParams params(owner);
    EXPECT_EQ(params.attention.host.kv_seq_lens.data(), original_lengths);
    EXPECT_EQ(params.embedding.request_ids.data(), original_request_ids);
    EXPECT_EQ(params.embedding.input_embedding.data_ptr(), original_embedding);
    EXPECT_EQ(params.llmrec_params()->full_k_caches[0].data_ptr(),
              original_cache);
    params.attention.host.kv_seq_lens[1] = 4;
    params.mutable_llmrec_params().current_round_tensor =
        torch::tensor({2}, torch::kInt32);
    params.attn_metadata = std::make_shared<layer::AttentionMetadata>();
    params.enable_graph = true;
  }
  EXPECT_EQ(owner.attention.host.kv_seq_lens, (std::vector<int32_t>{2, 4}));
  EXPECT_EQ(owner.attention.host.kv_seq_lens.data(), original_lengths);
  EXPECT_EQ(owner.embedding.request_ids.data(), original_request_ids);
  EXPECT_EQ(owner.embedding.input_embedding.data_ptr(), original_embedding);
  EXPECT_EQ(owner.llmrec_params()->full_k_caches[0].data_ptr(), original_cache);
  EXPECT_TRUE(torch::equal(owner.llmrec_params()->current_round_tensor,
                           torch::tensor({2}, torch::kInt32)));
  EXPECT_NE(owner.attn_metadata, nullptr);
  EXPECT_TRUE(owner.enable_graph);
}

TEST(BatchFactoryTest, RecBatchFinishesSequenceAndGroupInputs) {
  RecBatchFactory sequence_factory(/*dp_size=*/1, BatchInputType::SEQUENCE);
  auto sequence_request = make_request(/*rank=*/0, RecType::kLlmRec);
  auto sequence_batches = sequence_factory.create_batches(
      {sequence_request}, {sequence_request->sequences()[0].get()}, {1});
  EXPECT_FALSE(sequence_request->sequences()[0]->finished());
  sequence_batches[0].finish();
  EXPECT_TRUE(sequence_request->sequences()[0]->finished());

  RecBatchFactory group_factory(/*dp_size=*/1, BatchInputType::ONEREC);
  auto group_request = make_request(/*rank=*/0, RecType::kOneRec);
  auto group_batches = group_factory.create_batches(
      {group_request}, {group_request->sequences()[0].get()}, {0});
  EXPECT_FALSE(group_request->sequence_group()->finished());
  group_batches[0].finish();
  EXPECT_TRUE(group_request->sequence_group()->finished());
}

TEST(BatchFactoryTest, RecOutputsRefreshSamplingTargetsAcrossForwards) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);
  RecBatchFactory factory(/*dp_size=*/1, BatchInputType::SEQUENCE);
  auto first = make_request(/*rank=*/0, RecType::kLlmRec);
  auto second = make_request(/*rank=*/0, RecType::kLlmRec);
  auto* first_sequence = first->sequences()[0].get();
  auto* second_sequence = second->sequences()[0].get();
  first_sequence->add_blocks(BlockType::KV, manager.allocate(/*num_blocks=*/1));
  second_sequence->add_blocks(BlockType::KV,
                              manager.allocate(/*num_blocks=*/1));
  auto batches = factory.create_batches(
      {first, second}, {first_sequence, second_sequence}, {2, 3});
  auto& batch = batches[0];
  const uint64_t batch_id = batch.batch_id();
  ModelArgs args;

  // Only the second sequence completes prefill and owns an output row.
  const auto prefill_input = batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0, args);
  static_assert(
      std::is_same_v<std::decay_t<decltype(prefill_input)>, RecForwardInput>);
  EXPECT_TRUE(torch::equal(prefill_input.token_ids,
                           torch::tensor({1, 2, 1, 2, 3}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(prefill_input.sampling_params.selected_token_idxes,
                           torch::tensor({4}, torch::kInt32)));
  EXPECT_EQ(prefill_input.input_params.meta.batch_id, batch_id);
  EXPECT_FALSE(prefill_input.has_step_meta());
  SampleOutput sample_output;
  sample_output.next_tokens = torch::tensor({42}, torch::kInt);
  batch.process_sample_output(sample_output,
                              /*replace_fake_token=*/false,
                              /*force_requested_beam_result_size=*/false);
  EXPECT_EQ(first_sequence->num_generated_tokens(), 0);
  ASSERT_EQ(second_sequence->num_generated_tokens(), 1);
  EXPECT_EQ(second_sequence->tokens()[3], 42);

  // The next forward completes the first prefill while decoding the second.
  batch.refresh_forward_type();
  const auto input = batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0, args);
  EXPECT_TRUE(
      torch::equal(input.token_ids, torch::tensor({3, 42}, torch::kInt32)));
  RawForwardOutput raw_output;
  raw_output.outputs.resize(2);
  raw_output.outputs[0].tokens = {RawToken{.id = 51}};
  raw_output.outputs[1].tokens = {RawToken{.id = 43}};
  batch.process_sample_output(raw_output, /*replace_fake_token=*/false);

  ASSERT_EQ(first_sequence->num_generated_tokens(), 1);
  ASSERT_EQ(second_sequence->num_generated_tokens(), 2);
  EXPECT_EQ(first_sequence->tokens()[3], 51);
  EXPECT_EQ(second_sequence->tokens()[3], 42);
  EXPECT_EQ(second_sequence->tokens()[4], 43);
  EXPECT_EQ(batch.batch_id(), batch_id);
  EXPECT_EQ(batch.get_allowed_max_tokens(), (std::vector<uint32_t>{2, 3}));
}

TEST(BatchOutputHandlerTest, DoesNotSampleOneRecDecoderContext) {
  ScopedConfigValue<bool> prefill_only(
      RecConfig::get_instance().enable_rec_prefill_only(), true);
  ScopedConfigValue<int32_t> decode_rounds(
      RecConfig::get_instance().max_decode_rounds(), 0);
  auto request = make_request(/*rank=*/0,
                              RecType::kOneRec,
                              /*beam_width=*/1,
                              /*prompt_tokens=*/{1, 2, 3},
                              make_decoder_context_data());
  auto* sequence = request->sequences()[0].get();
  ASSERT_EQ(sequence->num_tokens(), 0u);
  BatchState state;
  state.reserve(/*sequence_count=*/1, /*group_count=*/0);
  state.add(sequence, /*token_budget=*/1);
  const auto data = state.input_data(state.sequence_plan());
  BatchOutputHandler handler;
  handler.prepare(data);
  RawForwardOutput output;
  output.outputs.resize(1);
  output.outputs[0].tokens = {RawToken{.id = 42}};
  handler.process_sample_output({data.sequences, data.sequence_groups},
                                output,
                                /*replace_fake_token=*/false);
  EXPECT_EQ(sequence->num_generated_tokens(), 0u);
}

TEST(RecBatchOutputHandlerTest,
     ContextTargetsKeepMixedOutputOrderAcrossForwards) {
  ScopedConfigValue<bool> prefill_only(
      RecConfig::get_instance().enable_rec_prefill_only(), true);
  ScopedConfigValue<int32_t> decode_rounds(
      RecConfig::get_instance().max_decode_rounds(), 0);
  ScopedConfigValue<bool> schedule_overlap(
      SchedulerConfig::get_instance().enable_schedule_overlap(), false);
  auto context_request = make_request(/*rank=*/0,
                                      RecType::kOneRec,
                                      /*beam_width=*/1,
                                      /*prompt_tokens=*/{1, 2, 3},
                                      make_decoder_context_data());
  auto partial_request = make_request(/*rank=*/0, RecType::kLlmRec);
  auto bos_request = make_request(/*rank=*/0, RecType::kOneRec);
  auto* context_sequence = context_request->sequences()[0].get();
  auto* partial_sequence = partial_request->sequences()[0].get();
  auto* bos_sequence = bos_request->sequences()[0].get();
  ASSERT_EQ(context_sequence->num_tokens(), 0u);
  ASSERT_EQ(bos_sequence->num_tokens(), 1u);
  BatchState state;
  state.reserve(/*sequence_count=*/3, /*group_count=*/0);
  state.add(context_sequence, /*token_budget=*/1);
  state.add(partial_sequence, /*token_budget=*/1);
  state.add(bos_sequence, /*token_budget=*/1);
  const auto data = state.input_data(state.sequence_plan());
  const BatchOutputData output_data{data.sequences, data.sequence_groups};
  RecBatchOutputHandler handler(BatchInputType::ONEREC);
  handler.prepare(data);

  // Input builders advance KV after the output targets have been captured.
  partial_sequence->kv_state().set_kv_cache_tokens_num(/*num=*/1);
  bos_sequence->kv_state().set_kv_cache_tokens_num(/*num=*/1);
  RawForwardOutput raw_output;
  raw_output.outputs.resize(2);
  raw_output.outputs[0].tokens = {RawToken{.id = 42}};
  raw_output.outputs[1].tokens = {RawToken{.id = 51}};
  handler.process_sample_output(output_data,
                                raw_output,
                                /*replace_fake_token=*/false);
  ASSERT_EQ(context_sequence->num_generated_tokens(), 1u);
  ASSERT_EQ(bos_sequence->num_generated_tokens(), 1u);
  EXPECT_EQ(context_sequence->tokens()[0], 42);
  EXPECT_EQ(bos_sequence->tokens()[1], 51);
  EXPECT_EQ(partial_sequence->num_generated_tokens(), 0u);

  // The former context-only sequence now owns a regular decode target.
  handler.prepare(data);
  context_sequence->kv_state().set_kv_cache_tokens_num(/*num=*/1);
  partial_sequence->kv_state().set_kv_cache_tokens_num(/*num=*/2);
  bos_sequence->kv_state().set_kv_cache_tokens_num(/*num=*/2);
  SampleOutput sample_output;
  sample_output.next_tokens = torch::tensor({43, 52}, torch::kInt);
  handler.process_sample_output(output_data,
                                sample_output,
                                /*replace_fake_token=*/false,
                                /*force_requested_beam_result_size=*/false);
  ASSERT_EQ(context_sequence->num_generated_tokens(), 2u);
  ASSERT_EQ(bos_sequence->num_generated_tokens(), 2u);
  EXPECT_EQ(context_sequence->tokens()[1], 43);
  EXPECT_EQ(bos_sequence->tokens()[2], 52);
  EXPECT_EQ(partial_sequence->num_generated_tokens(), 0u);
}

TEST(RecBatchOutputHandlerTest, ContextTargetsRequireLegacyGroupContract) {
  struct ContractCase {
    BatchInputType input_type;
    bool prefill_only;
    int32_t decode_rounds;
  };
  const ContractCase cases[] = {
      {BatchInputType::ONEREC, false, 0},
      {BatchInputType::ONEREC_XATTENTION, true, 2},
      {BatchInputType::SEQUENCE, true, 0},
  };
  ScopedConfigValue<bool> schedule_overlap(
      SchedulerConfig::get_instance().enable_schedule_overlap(), false);
  for (const auto& contract : cases) {
    ScopedConfigValue<bool> prefill_only(
        RecConfig::get_instance().enable_rec_prefill_only(),
        contract.prefill_only);
    ScopedConfigValue<int32_t> decode_rounds(
        RecConfig::get_instance().max_decode_rounds(), contract.decode_rounds);
    auto context_request = make_request(/*rank=*/0,
                                        RecType::kOneRec,
                                        /*beam_width=*/1,
                                        /*prompt_tokens=*/{1, 2, 3},
                                        make_decoder_context_data());
    auto bos_request = make_request(/*rank=*/0, RecType::kOneRec);
    auto* context_sequence = context_request->sequences()[0].get();
    auto* bos_sequence = bos_request->sequences()[0].get();
    BatchState state;
    state.reserve(/*sequence_count=*/2, /*group_count=*/0);
    state.add(context_sequence, /*token_budget=*/1);
    state.add(bos_sequence, /*token_budget=*/1);
    const auto data = state.input_data(state.sequence_plan());
    RecBatchOutputHandler handler(contract.input_type);
    handler.prepare(data);
    bos_sequence->kv_state().set_kv_cache_tokens_num(/*num=*/1);
    RawForwardOutput output;
    output.outputs.resize(1);
    output.outputs[0].tokens = {RawToken{.id = 61}};
    handler.process_sample_output({data.sequences, data.sequence_groups},
                                  output,
                                  /*replace_fake_token=*/false);
    EXPECT_EQ(context_sequence->num_generated_tokens(), 0u);
    ASSERT_EQ(bos_sequence->num_generated_tokens(), 1u);
    EXPECT_EQ(bos_sequence->tokens()[1], 61);
  }
}

TEST(BatchFactoryTest, IndependentFactoriesKeepTheirOwnDpSize) {
  BatchFactory single(/*dp_size=*/1);
  BatchFactory multiple(/*dp_size=*/3);
  auto request = make_request(/*rank=*/2);
  auto batches = multiple.create_batches(
      {request}, {request->sequences().front().get()}, {2});
  ASSERT_EQ(batches.size(), 3);
  EXPECT_TRUE(batches[0].empty());
  EXPECT_TRUE(batches[1].empty());
  EXPECT_EQ(batches[2].size(), 1);
  EXPECT_EQ(single.create_batches({}, {}, {}).size(), 1);
  EXPECT_EQ(multiple.create_batches({}, {}, {}).size(), 3);
}

TEST(BatchFactoryTest, PreservesRankLocalSequenceOrderAndBudgets) {
  BatchFactory factory(/*dp_size=*/3);
  auto first = make_request(/*rank=*/2);
  auto second = make_request(/*rank=*/0);
  auto third = make_request(/*rank=*/2);
  auto* seq1 = first->sequences().front().get();
  auto* seq2 = second->sequences().front().get();
  auto* seq3 = third->sequences().front().get();
  auto batches = factory.create_batches(
      {first, second, third}, {seq1, seq2, seq3}, {1, 2, 3});
  EXPECT_EQ(batches[0].get_sequences(), (std::vector<Sequence*>{seq2}));
  EXPECT_EQ(batches[0].get_allowed_max_tokens(), (std::vector<uint32_t>{2}));
  EXPECT_TRUE(batches[1].empty());
  EXPECT_EQ(batches[2].get_sequences(), (std::vector<Sequence*>{seq1, seq3}));
  EXPECT_EQ(batches[2].get_allowed_max_tokens(), (std::vector<uint32_t>{1, 3}));
  EXPECT_NE(batches[0].batch_id(), UNINITIALIZED_BATCH_ID);
  EXPECT_NE(batches[0].batch_id(), batches[2].batch_id());
  EXPECT_EQ(batches[1].batch_id(), UNINITIALIZED_BATCH_ID);
}

TEST(BatchFactoryTest, BeamBatchRetainsAllRequestGroupsInOrder) {
  BatchFactory factory(/*dp_size=*/1);
  auto first = make_request(/*rank=*/0);
  auto beam = make_request(/*rank=*/0, RecType::kNone, /*beam_width=*/2);
  auto batches = factory.create_batches(
      {first, beam},
      {first->sequences()[0].get(), beam->sequences()[0].get()},
      {3, 3});
  ASSERT_EQ(batches[0].sequence_groups().size(), 2);
  EXPECT_EQ(batches[0].sequence_groups()[0], first->sequence_group());
  EXPECT_EQ(batches[0].sequence_groups()[1], beam->sequence_group());
}

TEST(BatchFactoryTest, ReusingFactoryDoesNotRetainPreviousBeamGroups) {
  BatchFactory factory(/*dp_size=*/1);
  auto beam = make_request(/*rank=*/0, RecType::kNone, /*beam_width=*/2);
  auto beam_batches =
      factory.create_batches({beam}, {beam->sequences()[0].get()}, {3});
  ASSERT_EQ(beam_batches[0].sequence_groups().size(), 1);

  auto request = make_request(/*rank=*/0);
  auto batches =
      factory.create_batches({request}, {request->sequences()[0].get()}, {3});
  EXPECT_TRUE(batches[0].sequence_groups().empty());
  EXPECT_EQ(batches[0].get_sequences(),
            (std::vector<Sequence*>{request->sequences()[0].get()}));
  EXPECT_EQ(batches[0].input_type(), BatchInputType::SEQUENCE);

  auto empty_batches = factory.create_batches({}, {}, {});
  EXPECT_TRUE(empty_batches[0].empty());
  EXPECT_EQ(empty_batches[0].batch_id(), UNINITIALIZED_BATCH_ID);
}

TEST(BatchFactoryTest, FactoriesKeepIndependentInputContractsAcrossCalls) {
  RecBatchFactory onerec_factory(/*dp_size=*/2, BatchInputType::ONEREC);
  RecBatchFactory multi_round_factory(/*dp_size=*/1,
                                      BatchInputType::REC_MULTI_ROUND);
  auto onerec = make_request(/*rank=*/1, RecType::kOneRec);
  auto multi_round = make_request(/*rank=*/0, RecType::kLlmRec);

  // Interleave calls so neither factory can replace the other's configuration.
  for (int32_t iteration = 0; iteration < 2; ++iteration) {
    auto onerec_batches = onerec_factory.create_batches(
        {onerec}, {onerec->sequences()[0].get()}, {0});
    ASSERT_EQ(onerec_batches.size(), 2);
    EXPECT_TRUE(onerec_batches[0].empty());
    EXPECT_EQ(onerec_batches[1].input_type(), BatchInputType::ONEREC);
    EXPECT_EQ(onerec_batches[1].sequence_groups(),
              (std::vector<SequencesGroup*>{onerec->sequence_group()}));

    auto multi_round_batches = multi_round_factory.create_batches(
        {multi_round}, {multi_round->sequences()[0].get()}, {2});
    ASSERT_EQ(multi_round_batches.size(), 1);
    EXPECT_EQ(multi_round_batches[0].input_type(),
              BatchInputType::REC_MULTI_ROUND);
    EXPECT_TRUE(multi_round_batches[0].sequence_groups().empty());
    EXPECT_EQ(multi_round_batches[0].get_sequences(),
              (std::vector<Sequence*>{multi_round->sequences()[0].get()}));
    EXPECT_EQ(multi_round_batches[0].get_allowed_max_tokens(),
              (std::vector<uint32_t>{2}));
  }
}

TEST(BatchFactoryTest, RecSequenceInputsWithoutBeamDoNotAttachGroups) {
  for (BatchInputType type :
       {BatchInputType::SEQUENCE, BatchInputType::REC_MULTI_ROUND}) {
    RecBatchFactory factory(/*dp_size=*/2, type);
    auto request = make_request(/*rank=*/1, RecType::kLlmRec);
    auto* sequence = request->sequences()[0].get();
    auto batches = factory.create_batches({request}, {sequence}, {2});
    ASSERT_EQ(batches.size(), 2);
    EXPECT_TRUE(batches[0].empty());
    EXPECT_EQ(batches[1].input_type(), type);
    EXPECT_EQ(batches[1].get_sequences(), (std::vector<Sequence*>{sequence}));
    EXPECT_EQ(batches[1].get_allowed_max_tokens(), (std::vector<uint32_t>{2}));
    EXPECT_TRUE(batches[1].sequence_groups().empty());
  }
}

TEST(BatchFactoryTest, RecMultiRoundBuilderUsesScheduledSequencesAndBudgets) {
  ScopedConfigValue<int32_t> decode_rounds(
      RecConfig::get_instance().max_decode_rounds(), 3);
  ScopedConfigValue<int32_t> beam_width(
      BeamSearchConfig::get_instance().beam_width(), 2);
  RecBatchFactory factory(/*dp_size=*/1, BatchInputType::REC_MULTI_ROUND);
  auto first = make_request(/*rank=*/0, RecType::kLlmRec, /*beam_width=*/2);
  auto second = make_request(/*rank=*/0, RecType::kLlmRec, /*beam_width=*/2);
  // A group may own sequences that were not scheduled. Their tokens must not
  // be flattened into the input or paired with another sequence's budget.
  auto finished = first->sequences()[0]->fork(/*index=*/1);
  finished->finish();
  first->sequences().emplace_back(std::move(finished));

  auto* first_sequence = first->sequences()[0].get();
  auto* second_sequence = second->sequences()[0].get();
  auto batches = factory.create_batches(
      {first, second}, {first_sequence, second_sequence}, {1, 2});
  ASSERT_EQ(batches[0].sequence_groups().size(), 2);
  ModelArgs args;
  auto input = batches[0].prepare_forward_input(/*num_decoding_tokens=*/1,
                                                /*min_decoding_batch_size=*/0,
                                                args);
  static_assert(std::is_same_v<decltype(input), RecForwardInput>);

  EXPECT_TRUE(
      torch::equal(input.token_ids, torch::tensor({1, 1, 2}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(input.positions, torch::tensor({0, 0, 1}, torch::kInt32)));
  EXPECT_EQ(input.input_params.meta.num_sequences, 2);
  EXPECT_EQ(input.input_params.meta.q_max_seq_len, 2);
  EXPECT_EQ(input.input_params.meta.batch_id, batches[0].batch_id());
  EXPECT_TRUE(input.input_params.meta.batch_forward_type.is_prefill());
  ASSERT_TRUE(input.step_decode.has_value());
  EXPECT_EQ(input.step_decode->batch_size, 2);
  EXPECT_EQ(input.step_decode->beam_width, 2);
  EXPECT_EQ(input.step_decode->total_round, 3);
  EXPECT_EQ(input.step_decode->decode_positions_vec,
            (std::vector<int32_t>{3, 3}));
  EXPECT_TRUE(torch::equal(input.decoder_sampling_params.selected_token_idxes,
                           torch::tensor({0, 1, 2, 3}, torch::kInt32)));
  EXPECT_EQ(batches[0].get_allowed_max_tokens(), (std::vector<uint32_t>{1, 2}));
}

#if defined(USE_CUDA) || defined(USE_NPU) || defined(USE_MLU) || \
    defined(USE_MUSA)
TEST(BatchFactoryTest, OneRecBuildersPreserveEncoderAndDecoderContracts) {
  ScopedConfigValue<int32_t> decode_rounds(
      RecConfig::get_instance().max_decode_rounds(), 3);
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);
  for (BatchInputType type :
       {BatchInputType::ONEREC, BatchInputType::ONEREC_XATTENTION}) {
    RecBatchFactory factory(/*dp_size=*/1, type);
    auto request = make_request(/*rank=*/0, RecType::kOneRec, /*beam_width=*/2);
    auto* sequence = request->sequences()[0].get();
    sequence->add_blocks(BlockType::KV, manager.allocate(/*num_blocks=*/1));
    auto batches = factory.create_batches({request}, {sequence}, {1});
    ModelArgs args;
    const auto input = batches[0].prepare_forward_input(
        /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0, args);
    static_assert(
        std::is_same_v<std::decay_t<decltype(input)>, RecForwardInput>);
    const auto* onerec = input.input_params.onerec_params();
    ASSERT_NE(onerec, nullptr);
    EXPECT_EQ(onerec->encoder_seq_lens, (std::vector<int32_t>{3}));
    EXPECT_EQ(onerec->bs, 1);
    EXPECT_EQ(onerec->group_width, 1);
    EXPECT_TRUE(onerec->has_encoder_output);
    EXPECT_TRUE(onerec->is_first_prefill);
    EXPECT_EQ(input.input_params.meta.batch_id, batches[0].batch_id());
    EXPECT_EQ(input.token_ids.numel(), 1);
    EXPECT_TRUE(torch::equal(input.sampling_params.selected_token_idxes,
                             torch::tensor({0}, torch::kInt32)));
    if (type == BatchInputType::ONEREC) {
      EXPECT_FALSE(input.has_step_meta());
      EXPECT_FALSE(input.input_params.has_onerec_xattention_params());
      continue;
    }
    EXPECT_TRUE(input.input_params.has_onerec_xattention_params());
    ASSERT_TRUE(input.has_step_meta());
    EXPECT_EQ(input.step_meta()->batch_size, 1);
    EXPECT_EQ(input.step_meta()->beam_width, 2);
    EXPECT_EQ(input.step_meta()->total_round, 3);
    EXPECT_EQ(input.step_meta()->decode_positions_vec,
              (std::vector<int32_t>{1}));
    EXPECT_TRUE(torch::equal(input.decoder_sampling_params.selected_token_idxes,
                             torch::tensor({0, 1}, torch::kInt32)));
  }
}
#endif

TEST(BatchFactoryTest, RecMultiRoundOutputsFollowRankLocalRequestOrder) {
  RecBatchFactory factory(/*dp_size=*/2, BatchInputType::REC_MULTI_ROUND);
  auto first = make_request(/*rank=*/1, RecType::kLlmRec, /*beam_width=*/2);
  auto other_rank =
      make_request(/*rank=*/0, RecType::kLlmRec, /*beam_width=*/2);
  auto last = make_request(/*rank=*/1, RecType::kLlmRec, /*beam_width=*/2);
  auto batches = factory.create_batches({first, other_rank, last},
                                        {first->sequences()[0].get(),
                                         other_rank->sequences()[0].get(),
                                         last->sequences()[0].get()},
                                        {3, 3, 3});
  ASSERT_EQ(batches[1].sequence_groups().size(), 2);
  EXPECT_EQ(batches[1].sequence_groups()[0], first->sequence_group());
  EXPECT_EQ(batches[1].sequence_groups()[1], last->sequence_group());

  ForwardOutput output;
  // The final result may be wider than the requested beam width.
  output.beam_sequence_group =
      torch::arange(18, torch::kInt32).reshape({2, 3, 3});
  output.beam_search_output.out_logprobs =
      torch::tensor({-0.1f, -0.2f, -0.3f, -0.4f, -0.5f, -0.6f});
  batches[1].process_beam_sequence_group(output);

  const auto& first_result =
      RecSequence::from(*first->sequences()[0]).beam_search_result();
  const auto& last_result =
      RecSequence::from(*last->sequences()[0]).beam_search_result();
  EXPECT_TRUE(first_result.ready());
  EXPECT_EQ(first_result.beam_width(), 3);
  EXPECT_EQ(first_result.total_rounds(), 3);
  EXPECT_EQ(
      first_result.beams(),
      (std::vector<std::vector<int32_t>>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}));
  EXPECT_EQ(first_result.last_logprobs(),
            (std::vector<float>{-0.1f, -0.2f, -0.3f}));
  EXPECT_EQ(last_result.beams(),
            (std::vector<std::vector<int32_t>>{
                {9, 10, 11}, {12, 13, 14}, {15, 16, 17}}));
  EXPECT_EQ(last_result.last_logprobs(),
            (std::vector<float>{-0.4f, -0.5f, -0.6f}));
  EXPECT_FALSE(RecSequence::from(*other_rank->sequences()[0])
                   .beam_search_result()
                   .ready());
}

TEST(BatchFactoryTest, OneRecInputTypeSurvivesSequenceRefresh) {
  for (BatchInputType type :
       {BatchInputType::ONEREC, BatchInputType::ONEREC_XATTENTION}) {
    RecBatchFactory factory(/*dp_size=*/2, type);
    auto request = make_request(/*rank=*/1, RecType::kOneRec);
    auto* sequence = request->sequences()[0].get();
    // Embedding-only decoder prefill is allowed to have a zero token budget.
    auto batches = factory.create_batches({request}, {sequence}, {0});
    EXPECT_TRUE(batches[0].empty());
    EXPECT_EQ(batches[0].input_type(), type);
    EXPECT_FALSE(batches[1].empty());
    EXPECT_EQ(batches[1].size(), 1);
    EXPECT_EQ(batches[1].num_scheduled_sequences(), 0);
    EXPECT_EQ(batches[1].num_groups(), 1);
    EXPECT_EQ(batches[1].get_sequences(), (std::vector<Sequence*>{sequence}));
    const uint64_t batch_id = batches[1].batch_id();
    batches[1].refresh_sequences_from_groups();
    EXPECT_EQ(batches[1].size(), 1);
    EXPECT_TRUE(batches[1].get_allowed_max_tokens().empty());
    EXPECT_EQ(batches[1].input_type(), type);
    batches[1].set_batch_id();
    EXPECT_EQ(batches[1].batch_id(), batch_id);
  }
}

TEST(BatchFactoryTest, OneRecAccessorsFollowGroupReplacementWithoutRefresh) {
  RecBatchFactory factory(/*dp_size=*/1, BatchInputType::ONEREC);
  auto request = make_request(0, RecType::kOneRec);
  auto batches =
      factory.create_batches({request}, {request->sequences()[0].get()}, {1});
  auto replacement = request->sequences()[0]->fork(/*index=*/1);
  auto* replacement_ptr = replacement.get();
  request->sequences()[0] = std::move(replacement);
  EXPECT_EQ(batches[0][0], replacement_ptr);
  EXPECT_EQ(batches[0].get_sequences(),
            (std::vector<Sequence*>{replacement_ptr}));
  EXPECT_EQ(batches[0].num_scheduled_sequences(), 0);
  EXPECT_TRUE(batches[0].get_allowed_max_tokens().empty());
  EXPECT_FALSE(replacement_ptr->finished());
  batches[0].finish();
  EXPECT_TRUE(replacement_ptr->finished());
  EXPECT_TRUE(request->sequence_group()->finished());
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

TEST(BatchFactoryTest, EmptyRecRanksPrepareEmptyInputs) {
  ModelArgs args;
  for (BatchInputType type : {BatchInputType::SEQUENCE,
                              BatchInputType::ONEREC,
                              BatchInputType::ONEREC_XATTENTION,
                              BatchInputType::REC_MULTI_ROUND}) {
    RecBatchFactory factory(/*dp_size=*/2, type);
    auto batches = factory.create_batches({}, {}, {});
    for (auto& batch : batches) {
      auto input = batch.prepare_forward_input(/*num_decoding_tokens=*/1,
                                               /*min_decoding_batch_size=*/0,
                                               args);
      EXPECT_TRUE(input.input_params.meta.batch_forward_type.is_empty());
    }
  }
}

TEST(BatchFactoryTest, TransfersAreConsumedOnlyForActiveRanks) {
  BatchFactory factory(/*dp_size=*/2);
  auto request = make_request(/*rank=*/1);
  std::vector<std::vector<BlockTransferInfo>> transfers(2);
  uint8_t key[XXH3_128BITS_HASH_VALUE_LEN] = {};
  transfers[0].emplace_back(/*src_id=*/1, /*dst_id=*/2, key, TransferType::G2D);
  transfers[1].emplace_back(/*src_id=*/3, /*dst_id=*/4, key, TransferType::G2D);
  auto batches = factory.create_batches(
      {request}, {request->sequences()[0].get()}, {3}, &transfers);
  EXPECT_EQ(transfers[0].size(), 1);
  EXPECT_TRUE(transfers[1].empty());
  EXPECT_TRUE(batches[0].empty());
  EXPECT_FALSE(batches[1].empty());
}

TEST(BatchOutputHandlerDeathTest, RejectsBeamSourceOutsideBatch) {
  BatchFactory factory(/*dp_size=*/1);
  auto request = make_request(/*rank=*/0, RecType::kNone, /*beam_width=*/2);
  auto batches =
      factory.create_batches({request}, {request->sequences()[0].get()}, {1});
  RawForwardOutput output;
  output.src_seq_idxes = {1};
  output.out_tokens = {4};
  output.out_logprobs = {0.0F};
  EXPECT_DEATH(batches[0].process_beam_search_output(output, false),
               "data.sequences.size");
  output.src_seq_idxes = {-1};
  EXPECT_DEATH(batches[0].process_beam_search_output(output, false),
               "src_seq_idx >= 0");
}

TEST(BatchFactoryDeathTest, RejectsInvalidDpSize) {
  EXPECT_DEATH(BatchFactory(/*dp_size=*/0), "dp_size_");
  EXPECT_DEATH((RecBatchFactory(/*dp_size=*/-1, BatchInputType::ONEREC)),
               "dp_size_");
}

TEST(BatchFactoryDeathTest, RejectsUnsupportedInputTypeAtConstruction) {
  EXPECT_DEATH(
      (RecBatchFactory(/*dp_size=*/1, static_cast<BatchInputType>(-1))),
      "Unsupported batch input type");
}

TEST(BatchFactoryDeathTest, RejectsGroupOnlySequenceInput) {
  auto request = make_request(/*rank=*/0, RecType::kOneRec);
  Batch batch;
  batch.add(request->sequence_group());
  ModelArgs args;
  EXPECT_DEATH(batch.prepare_forward_input(/*num_decoding_tokens=*/1,
                                           /*min_decoding_batch_size=*/0,
                                           args),
               "Sequence input requires scheduled sequences");
  EXPECT_DEATH(batch.prepare_forward_input(args, /*thread_pool=*/nullptr),
               "Sequence input requires scheduled sequences");
}

TEST(BatchFactoryDeathTest, RejectsOneRecInputWithoutRequestGroups) {
  auto request = make_request(/*rank=*/0, RecType::kOneRec);
  RecBatch batch(BatchInputType::ONEREC);
  batch.add(request->sequences()[0].get());
  ModelArgs args;
  EXPECT_DEATH(batch.prepare_forward_input(/*num_decoding_tokens=*/1,
                                           /*min_decoding_batch_size=*/0,
                                           args),
               "OneRec input requires request groups");
}

TEST(BatchFactoryDeathTest, RejectsMismatchedBudgetsAndInvalidRanks) {
  BatchFactory factory(/*dp_size=*/2);
  auto request = make_request(/*rank=*/0);
  auto* sequence = request->sequences()[0].get();
  EXPECT_DEATH(factory.create_batches({request}, {sequence}, {}),
               "Each scheduled sequence requires one token budget");
  EXPECT_DEATH(factory.create_batches({request}, {sequence}, {0}), "budgets");
  const size_t overflow_budget =
      static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1;
  EXPECT_DEATH(factory.create_batches({request}, {sequence}, {overflow_budget}),
               "budgets");
  sequence->set_dp_rank(-1);
  EXPECT_DEATH(factory.create_batches({request}, {sequence}, {1}), "dp_rank");
  sequence->set_dp_rank(2);
  EXPECT_DEATH(factory.create_batches({request}, {sequence}, {1}), "dp_rank");
}

TEST(BatchFactoryDeathTest, RejectsInvalidSwapRankCount) {
  BatchFactory factory(/*dp_size=*/2);
  std::vector<std::vector<BlockTransferInfo>> transfers(1);
  EXPECT_DEATH(factory.create_batches({}, {}, {}, &transfers), "swap_infos");
}

}  // namespace
}  // namespace xllm
