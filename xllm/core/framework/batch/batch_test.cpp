/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "batch.h"

#include <absl/time/clock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <optional>

#include "batch_input_builder.h"
#include "common/global_flags.h"
#include "framework/block/block.h"
#include "framework/block/block_manager_impl.h"
#include "framework/model/model_args.h"
#include "framework/request/sequences_group.h"
#include "framework/request/stopping_checker.h"
#include "framework/sampling/sampling_params.h"
#include "onerec_xattention_batch_input_builder.h"
#include "platform/device.h"

namespace xllm {

template <typename T>
bool equal(const torch::Tensor& t, const std::vector<T>& d) {
  auto flatten_t = t.flatten();
  if (flatten_t.size(0) != d.size()) {
    return false;
  }
  for (int i = 0; i < d.size(); i++) {
    if (flatten_t[i].item<T>() != d[i]) {
      return false;
    }
  }
  return true;
}

RawSampleOutput make_raw_sample_output(int64_t token_id,
                                       std::optional<float> logprob,
                                       std::vector<int64_t> top_tokens = {},
                                       std::vector<float> top_logprobs = {}) {
  RawToken raw_token;
  raw_token.id = token_id;
  raw_token.logprob = logprob;
  raw_token.top_tokens = std::move(top_tokens);
  raw_token.top_logprobs = std::move(top_logprobs);

  RawSampleOutput raw_output;
  raw_output.tokens.push_back(std::move(raw_token));
  return raw_output;
}

TEST(BatchTest, Basic) {
  // use init device to trigger the loading of torch backend for different
  // devices
  //  since the allocation of pinnned memory on cpu is still backend-dependent.
  torch::Device device(Device::type_torch(), 0);
  const uint32_t n_blocks = 20;
  const uint32_t block_size = 4;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  sampling_param.frequency_penalty = 0.1;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(20);
  const size_t capacity = 100;
  SequenceParams seq_params;
  seq_params.seq_capacity = capacity;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = false;
  seq_params.enable_schedule_overlap = false;

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder fake_decoder1("", 1, false, false);
  // prepare sequences
  // sequence in prefill phase
  Sequence seq1(/*index=*/0,
                /*token_ids=*/{1, 3, 5, 7, 5, 4, 3, 2, 1},
                input_embedding,
                mm_data,
                std::move(fake_decoder1),
                seq_params);

  seq1.add_kv_blocks(manager.allocate(3));  // [1, 2, 3]

  IncrementalDecoder fake_decoder2("", 2, false, false);
  // seq in decode phase
  Sequence seq2(/*index=*/0,
                /*token_ids=*/{2, 4, 6, 8, 6, 4, 2},
                input_embedding,
                mm_data,
                std::move(fake_decoder2),
                seq_params);
  seq2.add_kv_blocks(manager.allocate(4));  // [4, 5, 6, 7]
  seq2.kv_state().incr_kv_cache_tokens_num(/*size=*/7);
  seq2.append_token(100);

  IncrementalDecoder fake_decoder3("", 3, false, false);
  // seq in decode phase
  Sequence seq3(
      /*index=*/0,
      /*token_ids=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19},
      input_embedding,
      mm_data,
      std::move(fake_decoder3),
      seq_params);
  seq3.add_kv_blocks(manager.allocate(5));  // [8, 9, 10, 11, 12]
  seq3.kv_state().incr_kv_cache_tokens_num(/*size=*/15);
  seq3.append_token(200);

  IncrementalDecoder fake_decoder4("", 4, false, false);
  Sequence seq4(/*index=*/0,
                /*token_ids=*/{1, 3, 5, 7, 5, 4, 3, 2, 1},
                input_embedding,
                mm_data,
                std::move(fake_decoder4),
                seq_params);
  seq4.kv_state().add_kv_blocks(manager.allocate(3));  // [13, 14, 15]
  seq4.kv_state().incr_kv_cache_tokens_num(/*size=*/4);

  // define outputs
  Batch batch({&seq1, &seq2, &seq3});
  // allowed chunk size 4
  batch.add(&seq4, 4);
  ForwardInput forward_input = batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_bach_size=*/0, ModelArgs());

  // check num tokens in kv cache
  EXPECT_EQ(seq1.kv_state().kv_cache_tokens_num(), 9);
  EXPECT_EQ(seq2.kv_state().kv_cache_tokens_num(), 8);
  EXPECT_EQ(seq3.kv_state().kv_cache_tokens_num(), 16);
  EXPECT_EQ(seq4.kv_state().kv_cache_tokens_num(), 8);

  // clang-format off
  // check the flatten token ids
  const std::vector<int32_t> expcted_tokens = {
      /*seq1*/ 1, 3, 5, 7, 5, 4, 3, 2, 1, 
      /*seq2*/ 100, 
      /*seq3*/ 200,
      /*seq4*/ 5, 4, 3, 2};
  EXPECT_TRUE(equal(forward_input.token_ids, expcted_tokens));

  // check the flatten positions
  const std::vector<int32_t> expected_pos = {
    /*seq1*/ 0, 1, 2, 3, 4, 5, 6, 7, 8,
    /*seq2*/ 7, 
    /*seq3*/ 15,
    /*seq4*/ 4, 5, 6, 7};
  EXPECT_TRUE(equal(forward_input.positions, expected_pos));

  // check the input parameters
  const ModelInputParams& input_params = forward_input.input_params;
  EXPECT_TRUE(input_params.batch_forward_type.is_mixed());
  EXPECT_EQ(input_params.num_sequences, 4);
  EXPECT_EQ(input_params.q_max_seq_len, 9);
  EXPECT_EQ(input_params.kv_max_seq_len, 16);

#if defined(USE_NPU)
  const std::vector<int32_t> q_seq_lens = {9, 1, 1, 4};
#else
  const std::vector<int32_t> q_seq_lens = {0, 9, 10, 11, 15};
#endif
  EXPECT_TRUE(equal(input_params.q_seq_lens, q_seq_lens));

//  seq4's kv_seq_len = q_len + num_cached_tokens (q_len<=max_allowed_tokens)
#if defined(USE_NPU)
  const std::vector<int32_t> kv_seq_lens = {9, 8, 16, 8};
#else
  const std::vector<int32_t> kv_seq_lens = {0, 9, 17, 33, 41};
#endif
  EXPECT_TRUE(equal(input_params.kv_seq_lens, kv_seq_lens));

  const std::vector<int32_t> new_cache_slots = {
    /*seq1*/ 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    /*seq2*/ 23, 
    /*seq3*/ 47,
    /*seq4*/ 56,57,58,59
    };
  EXPECT_TRUE(equal(input_params.new_cache_slots, new_cache_slots));

  const std::vector<int32_t> block_tables = {
    /*seq1*/ 1, 2, 3,  0,  0,
    /*seq2*/ 4, 5, 6,  7,  0,
    /*seq3*/ 8, 9, 10, 11, 12,
    /*seq4*/ 13, 14, 15, 0, 0};
  EXPECT_TRUE(equal(input_params.block_tables, block_tables));

  // const std::vector<int32_t> last_token_idxes = {8, 9, 10};
  // EXPECT_TRUE(equal(input_params.last_token_idxes, last_token_idxes));

  const auto& sampling_params = forward_input.sampling_params;
  const std::vector<int64_t> unique_ids = {
    /*seq1*/   2,  4,  7,  5,  3,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    /*seq2*/ 100,  8,  6,  4,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
    /*seq3*/ 200,  19, 17, 1,  2,  15, 3,  4,  5,  6,  7,  8,  9, 10, 11, 13
    };
  // seq4 has no sampling parameters
  EXPECT_TRUE(equal(sampling_params.unique_token_ids, unique_ids));

  // const std::vector<int32_t> unique_counts = {
  //   /*seq1*/  1,  1,  1,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
  //   /*seq2*/  1,  1,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
  //   /*seq3*/  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1
  // };
  // // seq4 has no sampling parameters
  // EXPECT_TRUE(equal(sampling_params.unique_token_counts, unique_counts));

  const std::vector<int32_t> token_ids_lens = {6, 5, 16};
  EXPECT_TRUE(equal(sampling_params.unique_token_ids_lens, token_ids_lens));

  // The model's output hidden_states shape is [num_tokens, hidden_size].
  // selected_token_idxes is used to select the required corresponding token's hidden_state from it, perform logits calculation, and obtain the probability that the token belongs to each class.
  // The logits_processor's output logits shape is [num_selected_tokens, vocab_size]. sample_idxes is used to select the required token's corresponding logits from it.
  // Generally, num_selected_tokens equals num_sample_tokens.
  const std::vector<int32_t> expected_selected_token_idxes = {8,9,10};
  // seq4 has no sampling parameters
  EXPECT_EQ(sampling_params.selected_token_idxes.size(0), batch.size()-1);
  EXPECT_TRUE(equal(sampling_params.selected_token_idxes, expected_selected_token_idxes));
  const std::vector<int32_t> expected_sample_idxes = {0,1,2};
  EXPECT_TRUE(equal(sampling_params.sample_idxes, expected_sample_idxes));

  // clang-format on
}

TEST(BatchTest, SampleRequestInjectsAllMatchedSlots) {
  torch::Device device(Device::type_torch(), 0);
  const uint32_t n_blocks = 8;
  const uint32_t block_size = 4;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(1);
  const size_t capacity = 32;
  std::vector<SampleSlot> sample_slots;

  SampleSlot first_slot;
  first_slot.request_id = "sample-req";
  first_slot.sample_id = 0;
  first_slot.token_position = 2;
  sample_slots.push_back(first_slot);

  SampleSlot second_slot;
  second_slot.request_id = "sample-req";
  second_slot.sample_id = 1;
  second_slot.token_position = 5;
  sample_slots.push_back(second_slot);

  SequenceParams seq_params;
  seq_params.seq_capacity = capacity;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.sample_slots = &sample_slots;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = true;
  seq_params.enable_schedule_overlap = false;

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder decoder("", 1, false, false);
  Sequence seq(/*index=*/0,
               /*token_ids=*/{1, 10, 11, 12, 13, 14},
               input_embedding,
               mm_data,
               std::move(decoder),
               seq_params);
  seq.add_kv_blocks(manager.allocate(2));

  Batch batch({&seq});
  ForwardInput forward_input = batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_bach_size=*/0, ModelArgs());

  const auto& sampling_params_out = forward_input.sampling_params;
  const std::vector<int32_t> expected_selected_token_idxes = {1, 4};
  const std::vector<int32_t> expected_sample_idxes = {0, 1};
  EXPECT_TRUE(equal(sampling_params_out.selected_token_idxes,
                    expected_selected_token_idxes));
  EXPECT_TRUE(equal(sampling_params_out.sample_idxes, expected_sample_idxes));
  EXPECT_EQ(sampling_params_out.selected_token_idxes.size(0), 2);
}

TEST(BatchTest, SampleRequestKeepsThreadedRawBuilderOffsetsStable) {
  torch::Device device(Device::type_torch(), 0);
  const uint32_t n_blocks = 8;
  const uint32_t block_size = 4;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(1);
  const size_t capacity = 32;

  std::vector<SampleSlot> sample_slots_seq1;
  SampleSlot seq1_slot0;
  seq1_slot0.request_id = "sample-req-1";
  seq1_slot0.sample_id = 0;
  seq1_slot0.token_position = 2;
  sample_slots_seq1.push_back(seq1_slot0);
  SampleSlot seq1_slot1;
  seq1_slot1.request_id = "sample-req-1";
  seq1_slot1.sample_id = 1;
  seq1_slot1.token_position = 4;
  sample_slots_seq1.push_back(seq1_slot1);

  SequenceParams seq1_params;
  seq1_params.seq_capacity = capacity;
  seq1_params.stopping_checker = &stopping_checker;
  seq1_params.sampling_param = &sampling_param;
  seq1_params.sample_slots = &sample_slots_seq1;
  seq1_params.skip_special_tokens = true;
  seq1_params.echo = false;
  seq1_params.logprobs = true;
  seq1_params.enable_schedule_overlap = false;

  std::vector<SampleSlot> sample_slots_seq2;
  SampleSlot seq2_slot0;
  seq2_slot0.request_id = "sample-req-2";
  seq2_slot0.sample_id = 0;
  seq2_slot0.token_position = 1;
  sample_slots_seq2.push_back(seq2_slot0);
  SampleSlot seq2_slot1;
  seq2_slot1.request_id = "sample-req-2";
  seq2_slot1.sample_id = 1;
  seq2_slot1.token_position = 3;
  sample_slots_seq2.push_back(seq2_slot1);

  SequenceParams seq2_params = seq1_params;
  seq2_params.sample_slots = &sample_slots_seq2;

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder decoder1("", 1, false, false);
  Sequence seq1(/*index=*/0,
                /*token_ids=*/{1, 21, 22, 23},
                input_embedding,
                mm_data,
                std::move(decoder1),
                seq1_params);
  seq1.add_kv_blocks(manager.allocate(1));

  IncrementalDecoder decoder2("", 2, false, false);
  Sequence seq2(/*index=*/1,
                /*token_ids=*/{1, 31, 32},
                input_embedding,
                mm_data,
                std::move(decoder2),
                seq2_params);
  seq2.add_kv_blocks(manager.allocate(1));

  std::vector<Sequence*> sequences = {&seq1, &seq2};
  std::vector<uint32_t> allowed_max_tokens = {
      std::numeric_limits<uint32_t>::max(),
      std::numeric_limits<uint32_t>::max()};
  std::vector<torch::Tensor> input_embeddings_vec;
  std::vector<MMData> mm_data_vec;
  ThreadPool thread_pool(2);
  ModelArgs args;
  BatchInputBuilder builder(sequences,
                            allowed_max_tokens,
                            input_embeddings_vec,
                            mm_data_vec,
                            /*swap_block_transfer_infos=*/nullptr,
                            /*batch_id=*/1,
                            &args,
                            BatchForwardType::PREFILL,
                            /*cp_size=*/1,
                            &thread_pool);

  RawForwardInput raw_forward_input = builder.build_raw_forward_input();

  const std::vector<int32_t> expected_selected_token_idxes = {1, 3, 4, 6};
  const std::vector<int32_t> expected_sample_idxes = {0, 1, 2, 3};
  EXPECT_EQ(raw_forward_input.sampling_params.size(), 4);
  EXPECT_EQ(raw_forward_input.selected_token_idxes,
            expected_selected_token_idxes);
  EXPECT_EQ(raw_forward_input.sample_idxes, expected_sample_idxes);
}

TEST(BatchTest, OneRecXAttentionBuildsCrossBlockCacheMetadata) {
  const uint32_t block_size = 4;
  BlockManager::Options options;
  options.num_blocks(8).block_size(block_size).enable_prefix_cache(false);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(1);
  SequenceParams seq_params;
  seq_params.seq_capacity = 16;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.enable_schedule_overlap = false;
  seq_params.rec_type = RecType::kOneRec;

  std::string prompt = "onerec";
  std::vector<int32_t> encoder_tokens = {11, 12, 13, 14, 15};
  torch::Tensor input_embedding;
  MMData mm_data;
  SequencesGroup group(
      prompt, encoder_tokens, input_embedding, mm_data, seq_params);
  auto blocks = manager.allocate(2);
  ASSERT_EQ(blocks.size(), 2u);
  const std::vector<int32_t> expected_block_ids = {blocks[0].id(),
                                                   blocks[1].id()};
  group.sequences()[0]->add_kv_blocks(blocks);
  const std::vector<int32_t> expected_slots =
      group.sequences()[0]->kv_state().kv_cache_slots(0, encoder_tokens.size());

  std::vector<SequencesGroup*> groups = {&group};
  std::vector<uint32_t> allowed_max_tokens = {
      std::numeric_limits<uint32_t>::max()};
  std::vector<torch::Tensor> input_embeddings = {input_embedding};
  std::vector<MMData> mm_data_vec = {mm_data};
  ModelArgs args;
  MPMCThreadPool thread_pool(1);
  OneRecBatchInputBuilderCache perf_cache;
  OneRecXAttentionBatchInputBuilder builder(groups,
                                            allowed_max_tokens,
                                            input_embeddings,
                                            mm_data_vec,
                                            nullptr,
                                            1,
                                            &args,
                                            BatchForwardType::PREFILL,
                                            &thread_pool,
                                            &perf_cache);

  ForwardInput input = builder.build_rec_forward_input(1, 0);
  const auto* params = input.input_params.onerec_xattention_params();
  ASSERT_NE(params, nullptr);
  EXPECT_TRUE(equal<int64_t>(params->cross_attn_kv_cu_seq_lens, {5}));
  EXPECT_TRUE(
      equal<int32_t>(params->cross_attn_block_tables, expected_block_ids));
  EXPECT_TRUE(
      equal<int32_t>(params->cross_attn_new_cache_slots, expected_slots));
}

TEST(BatchTest, SampleRequestProcessesAllMatchedRawOutputs) {
  torch::Device device(Device::type_torch(), 0);
  const uint32_t n_blocks = 8;
  const uint32_t block_size = 4;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  sampling_param.logprobs = true;
  sampling_param.top_logprobs = 2;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(1);

  std::vector<SampleSlot> sample_slots;
  SampleSlot slot0;
  slot0.request_id = "sample-req";
  slot0.sample_id = 0;
  slot0.token_position = 2;
  sample_slots.push_back(slot0);
  SampleSlot slot1 = slot0;
  slot1.sample_id = 1;
  slot1.token_position = 5;
  sample_slots.push_back(slot1);

  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.sample_slots = &sample_slots;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = true;
  seq_params.enable_schedule_overlap = false;

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder decoder("", 1, false, false);
  Sequence seq(/*index=*/0,
               /*token_ids=*/{1, 10, 11, 12, 13, 14},
               input_embedding,
               mm_data,
               std::move(decoder),
               seq_params);
  seq.add_kv_blocks(manager.allocate(2));

  Batch batch({&seq});
  batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_bach_size=*/0, ModelArgs());

  RawForwardOutput raw_output;
  raw_output.outputs.push_back(
      make_raw_sample_output(101, -0.10f, {101, 111}, {-0.10f, -1.0f}));
  raw_output.outputs.push_back(
      make_raw_sample_output(202, -0.20f, {202, 212}, {-0.20f, -2.0f}));
  batch.process_sample_output(raw_output, /*replace_fake_token=*/false);

  const size_t prompt_tokens = seq.num_prompt_tokens();
  EXPECT_EQ(seq.num_generated_tokens(), 2);
  EXPECT_EQ(seq.tokens()[prompt_tokens], 101);
  EXPECT_EQ(seq.tokens()[prompt_tokens + 1], 202);

  const auto& logprobs = seq.logprob_state()->get_logprobs();
  ASSERT_TRUE(logprobs[prompt_tokens].has_value());
  ASSERT_TRUE(logprobs[prompt_tokens + 1].has_value());
  EXPECT_FLOAT_EQ(logprobs[prompt_tokens].value(), -0.10f);
  EXPECT_FLOAT_EQ(logprobs[prompt_tokens + 1].value(), -0.20f);

  const auto& top_tokens = seq.logprob_state()->get_top_tokens();
  ASSERT_EQ(top_tokens[prompt_tokens].size(), 2);
  ASSERT_EQ(top_tokens[prompt_tokens + 1].size(), 2);
  EXPECT_EQ(top_tokens[prompt_tokens][0], 101);
  EXPECT_EQ(top_tokens[prompt_tokens + 1][0], 202);
}

TEST(BatchTest, SampleRequestDistributesRawOutputsAcrossSequences) {
  torch::Device device(Device::type_torch(), 0);
  const uint32_t n_blocks = 8;
  const uint32_t block_size = 4;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  sampling_param.logprobs = true;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(1);

  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = true;
  seq_params.enable_schedule_overlap = false;

  std::vector<SampleSlot> sample_slots_seq1;
  SampleSlot seq1_slot;
  seq1_slot.request_id = "sample-req-1";
  seq1_slot.sample_id = 0;
  seq1_slot.token_position = 2;
  sample_slots_seq1.push_back(seq1_slot);
  seq_params.sample_slots = &sample_slots_seq1;

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder decoder1("", 1, false, false);
  Sequence seq1(/*index=*/0,
                /*token_ids=*/{1, 21, 22, 23},
                input_embedding,
                mm_data,
                std::move(decoder1),
                seq_params);
  seq1.add_kv_blocks(manager.allocate(1));

  std::vector<SampleSlot> sample_slots_seq2;
  SampleSlot seq2_slot0;
  seq2_slot0.request_id = "sample-req-2";
  seq2_slot0.sample_id = 0;
  seq2_slot0.token_position = 1;
  sample_slots_seq2.push_back(seq2_slot0);
  SampleSlot seq2_slot1 = seq2_slot0;
  seq2_slot1.sample_id = 1;
  seq2_slot1.token_position = 3;
  sample_slots_seq2.push_back(seq2_slot1);
  seq_params.sample_slots = &sample_slots_seq2;

  IncrementalDecoder decoder2("", 2, false, false);
  Sequence seq2(/*index=*/1,
                /*token_ids=*/{1, 31, 32},
                input_embedding,
                mm_data,
                std::move(decoder2),
                seq_params);
  seq2.add_kv_blocks(manager.allocate(1));

  Batch batch({&seq1, &seq2});
  batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_bach_size=*/0, ModelArgs());

  RawForwardOutput raw_output;
  raw_output.outputs.push_back(make_raw_sample_output(111, -0.11f));
  raw_output.outputs.push_back(make_raw_sample_output(221, -0.21f));
  raw_output.outputs.push_back(make_raw_sample_output(222, -0.22f));
  batch.process_sample_output(raw_output, /*replace_fake_token=*/false);

  const size_t seq1_prompt_tokens = seq1.num_prompt_tokens();
  EXPECT_EQ(seq1.num_generated_tokens(), 1);
  EXPECT_EQ(seq1.tokens()[seq1_prompt_tokens], 111);

  const size_t seq2_prompt_tokens = seq2.num_prompt_tokens();
  EXPECT_EQ(seq2.num_generated_tokens(), 2);
  EXPECT_EQ(seq2.tokens()[seq2_prompt_tokens], 221);
  EXPECT_EQ(seq2.tokens()[seq2_prompt_tokens + 1], 222);
}

TEST(BatchTest, SampleRequestFallsBackToEmptyPlaceholderOnPartialRawOutputs) {
  torch::Device device(Device::type_torch(), 0);
  const uint32_t n_blocks = 8;
  const uint32_t block_size = 4;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  sampling_param.logprobs = true;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(1);

  std::vector<SampleSlot> sample_slots;
  SampleSlot slot0;
  slot0.request_id = "sample-req";
  slot0.sample_id = 0;
  slot0.token_position = 2;
  sample_slots.push_back(slot0);
  SampleSlot slot1 = slot0;
  slot1.sample_id = 1;
  slot1.token_position = 4;
  sample_slots.push_back(slot1);

  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.sample_slots = &sample_slots;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = true;
  seq_params.enable_schedule_overlap = false;

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder decoder("", 1, false, false);
  Sequence seq(/*index=*/0,
               /*token_ids=*/{1, 10, 11, 12, 13},
               input_embedding,
               mm_data,
               std::move(decoder),
               seq_params);
  seq.add_kv_blocks(manager.allocate(2));

  Batch batch({&seq});
  batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_bach_size=*/0, ModelArgs());

  RawForwardOutput raw_output;
  raw_output.outputs.push_back(make_raw_sample_output(301, -0.30f));
  batch.process_sample_output(raw_output, /*replace_fake_token=*/false);

  const size_t prompt_tokens = seq.num_prompt_tokens();
  EXPECT_EQ(seq.num_generated_tokens(), 2);
  EXPECT_EQ(seq.tokens()[prompt_tokens], 301);
  EXPECT_EQ(seq.tokens()[prompt_tokens + 1], seq.tokens()[0]);

  const auto& logprobs = seq.logprob_state()->get_logprobs();
  ASSERT_TRUE(logprobs[prompt_tokens].has_value());
  EXPECT_FALSE(logprobs[prompt_tokens + 1].has_value());
}

TEST(BatchTest, ProcessBeamSequenceGroupUsesFinalResultWidthForLogprobs) {
  torch::Device device(torch::kCPU);
  const uint32_t n_blocks = 8;
  const uint32_t block_size = 4;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  sampling_param.logprobs = true;
  sampling_param.beam_width = 2;
  sampling_param.num_return_sequences = 4;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(1);

  SequenceParams seq_params;
  seq_params.seq_capacity = 16;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = true;
  seq_params.enable_schedule_overlap = false;
  seq_params.rec_type = RecType::kLlmRec;

  std::string prompt = "prompt";
  std::vector<int32_t> prompt_tokens = {1};
  torch::Tensor input_embedding;
  MMData mm_data;

  SequencesGroup group1(
      prompt, prompt_tokens, input_embedding, mm_data, seq_params);
  group1.sequences()[0]->add_kv_blocks(manager.allocate(1));

  SequencesGroup group2(
      prompt, prompt_tokens, input_embedding, mm_data, seq_params);
  group2.sequences()[0]->add_kv_blocks(manager.allocate(1));

  Batch batch;
  batch.add(&group1);
  batch.add(&group2);

  ForwardOutput output;
  output.beam_sequence_group =
      torch::tensor({{{11, 12, 13}, {21, 22, 23}, {31, 32, 33}, {41, 42, 43}},
                     {{51, 52, 53}, {61, 62, 63}, {71, 72, 73}, {81, 82, 83}}},
                    torch::TensorOptions().dtype(torch::kInt32).device(device));
  output.beam_search_output.out_logprobs = torch::tensor(
      {0.11f, 0.21f, 0.31f, 0.41f, 0.51f, 0.61f, 0.71f, 0.81f},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));

  batch.process_beam_sequence_group(output);

  ASSERT_TRUE(group1.sequences()[0]->has_beam_result());
  ASSERT_TRUE(group2.sequences()[0]->has_beam_result());

  EXPECT_EQ(group1.sequences()[0]->beam_width_cached(), 4);
  EXPECT_EQ(group2.sequences()[0]->beam_width_cached(), 4);

  const auto& group1_logprobs = group1.sequences()[0]->beam_last_logprobs();
  const auto& group2_logprobs = group2.sequences()[0]->beam_last_logprobs();
  ASSERT_EQ(group1_logprobs.size(), 4);
  ASSERT_EQ(group2_logprobs.size(), 4);
  EXPECT_FLOAT_EQ(group1_logprobs[0], 0.11f);
  EXPECT_FLOAT_EQ(group1_logprobs[3], 0.41f);
  EXPECT_FLOAT_EQ(group2_logprobs[0], 0.51f);
  EXPECT_FLOAT_EQ(group2_logprobs[3], 0.81f);
}

TEST(BatchTest, KeepTargetsForOverlapReplacement) {
  const bool old_enable_schedule_overlap = FLAGS_enable_schedule_overlap;
  FLAGS_enable_schedule_overlap = true;

  torch::Device device(Device::type_torch(), 0);
  const uint32_t n_blocks = 8;
  const uint32_t block_size = 4;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(1);

  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = false;
  seq_params.enable_schedule_overlap = true;

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder decoder("", 1, false, false);
  Sequence seq(/*index=*/0,
               /*token_ids=*/{1, 10, 11},
               input_embedding,
               mm_data,
               std::move(decoder),
               seq_params);
  seq.add_kv_blocks(manager.allocate(1));
  seq.kv_state().incr_kv_cache_tokens_num(seq.num_prompt_tokens() - 1);

  Batch batch({&seq});
  batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_bach_size=*/0, ModelArgs());

  RawForwardOutput fake_output;
  fake_output.outputs.push_back(make_raw_sample_output(-1, std::nullopt));
  batch.process_sample_output(fake_output, /*replace_fake_token=*/false);
  EXPECT_EQ(seq.tokens()[seq.num_prompt_tokens()], -1);
  EXPECT_FALSE(seq.finished());

  RawForwardOutput real_output;
  real_output.outputs.push_back(make_raw_sample_output(101, -0.1f));
  batch.process_sample_output(real_output, /*replace_fake_token=*/true);

  EXPECT_EQ(seq.tokens()[seq.num_prompt_tokens()], 101);
  EXPECT_TRUE(seq.finished());

  FLAGS_enable_schedule_overlap = old_enable_schedule_overlap;
}

TEST(BatchTest, OverlapMTPReplacementSkipsPreemptedSequenceWithoutKVBlocks) {
  const bool old_enable_schedule_overlap = FLAGS_enable_schedule_overlap;
  FLAGS_enable_schedule_overlap = true;

  torch::Device device(Device::type_torch(), 0);
  const uint32_t n_blocks = 8;
  const uint32_t block_size = 4;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(2);

  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = false;
  seq_params.enable_schedule_overlap = true;

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder decoder("", 1, false, false);
  Sequence seq(/*index=*/0,
               /*token_ids=*/{1, 10, 11},
               input_embedding,
               mm_data,
               std::move(decoder),
               seq_params);
  seq.add_kv_blocks(manager.allocate(1));
  seq.kv_state().incr_kv_cache_tokens_num(seq.num_prompt_tokens() - 1);

  Batch batch({&seq});
  batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_bach_size=*/0, ModelArgs());

  RawForwardOutput fake_output;
  fake_output.outputs.push_back(make_raw_sample_output(-1, std::nullopt));

  batch.process_sample_output(fake_output, /*replace_fake_token=*/false);
  EXPECT_EQ(seq.num_generated_tokens(), 1);
  EXPECT_EQ(seq.tokens()[seq.num_prompt_tokens()], -1);

  seq.reset();
  EXPECT_EQ(seq.kv_state().num_kv_blocks(), 0);

  RawSampleOutput real_sample_output;
  RawToken real_token_0;
  real_token_0.id = 101;
  real_sample_output.tokens.push_back(real_token_0);
  RawToken real_token_1;
  real_token_1.id = 102;
  real_sample_output.tokens.push_back(real_token_1);
  RawForwardOutput real_output;
  real_output.outputs.push_back(std::move(real_sample_output));

  EXPECT_NO_FATAL_FAILURE(
      batch.process_sample_output(real_output, /*replace_fake_token=*/true));
  EXPECT_EQ(seq.num_generated_tokens(), 1);
  EXPECT_EQ(seq.tokens()[seq.num_prompt_tokens()], 101);

  FLAGS_enable_schedule_overlap = old_enable_schedule_overlap;
}

TEST(BatchTest, DPBalanceShuffle) {
  Batch batch;
  std::vector<uint32_t> kv_cache_tokens_num = {
      99, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
      17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
      34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48};
  auto shifted_indices = batch.cal_seq_exchange_index_test(kv_cache_tokens_num);
  // shifted_indices are expected as
  // {48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31,
  //  30, 29, 28, 27, 26, 25,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
  //  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 99}
  EXPECT_EQ(shifted_indices[0], 48);
  EXPECT_EQ(shifted_indices[48], 0);
  EXPECT_EQ(shifted_indices[1], 24);
  EXPECT_EQ(shifted_indices[47], 1);
  EXPECT_EQ(shifted_indices[2], 25);
}

}  // namespace xllm
