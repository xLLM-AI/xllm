/* Copyright 2025-2026 The xLLM Authors.
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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "batch_input_builder.h"
#include "core/framework/config/scheduler_config.h"
#include "framework/block/block.h"
#include "framework/block/block_manager_impl.h"
#include "framework/block/block_manager_pool.h"
#include "framework/block/block_utils.h"
#include "framework/block/composite_block_manager.h"
#include "framework/config/beam_search_config.h"
#include "framework/config/service_config.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/prefix_cache/block_hasher.h"
#include "framework/request/stopping_checker.h"
#include "framework/sampling/rejection_sampler.h"
#include "framework/sampling/sampling_params.h"
#include "framework/tokenizer/tokenizer.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "runtime/forward_shared_memory_manager.h"
#include "util/tensor_helper.h"

namespace xllm {

class BatchInputBuilderTestPeer final {
 public:
  static TransferKVInfo build_step_transfer_info(
      const TransferKVInfo& full_info,
      Sequence* sequence,
      uint32_t seq_len,
      uint32_t kv_split_size = 1) {
    return BatchInputBuilder::build_step_transfer_info(
        full_info, sequence, seq_len, kv_split_size);
  }
};

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
                                       std::vector<float> top_logprobs = {},
                                       std::vector<float> embeddings = {}) {
  RawToken raw_token;
  raw_token.id = token_id;
  raw_token.logprob = logprob;
  raw_token.top_tokens = std::move(top_tokens);
  raw_token.top_logprobs = std::move(top_logprobs);
  raw_token.embeddings = std::move(embeddings);

  RawSampleOutput raw_output;
  raw_output.tokens.push_back(std::move(raw_token));
  return raw_output;
}

namespace {

KVTransferMapping make_mapping(BlockType block_type,
                               const std::vector<uint64_t>& remote_ids) {
  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(block_type);
  mapping.remote_ids = remote_ids;
  return mapping;
}

TransferKVInfo make_info(const std::vector<uint64_t>& remote_ids,
                         BlockType block_type = BlockType::KV) {
  TransferKVInfo info;
  info.request_id = "req_0";
  info.dp_rank = 3;
  info.mappings.emplace_back(make_mapping(block_type, remote_ids));
  return info;
}

XTensorLayerOffsets make_offsets(const std::vector<uint64_t>& k_offsets,
                                 const std::vector<uint64_t>& v_offsets) {
  XTensorLayerOffsets offsets;
  offsets.k_offsets = k_offsets;
  offsets.v_offsets = v_offsets;
  return offsets;
}

const KVTransferMapping& find_mapping(const TransferKVInfo& info,
                                      BlockType block_type) {
  const int32_t group_id = cache_group_id(block_type);
  for (const KVTransferMapping& mapping : info.mappings) {
    if (mapping.group_id == group_id) {
      return mapping;
    }
  }
  ADD_FAILURE() << "Missing mapping for group_id=" << group_id;
  static const KVTransferMapping empty_mapping;
  return empty_mapping;
}

std::vector<uint64_t> block_ids(const std::vector<Block>& blocks) {
  std::vector<uint64_t> ids;
  ids.reserve(blocks.size());
  for (const Block& block : blocks) {
    if (block.is_valid()) {
      ids.emplace_back(static_cast<uint64_t>(block.id()));
    }
  }
  return ids;
}

void expect_mapping(const TransferKVInfo& info,
                    BlockType block_type,
                    const std::vector<uint64_t>& local_ids,
                    const std::vector<uint64_t>& remote_ids) {
  const KVTransferMapping& mapping = find_mapping(info, block_type);
  EXPECT_EQ(mapping.local_ids, local_ids);
  EXPECT_EQ(mapping.remote_ids, remote_ids);
  EXPECT_EQ(info.request_id, "req_0");
  EXPECT_EQ(info.dp_rank, 3);
}

LinearStatePrefixHash compute_linear_state_prefix_hash_for_test(
    const std::vector<int32_t>& token_ids,
    std::vector<Block>& blocks,
    size_t boundary_tokens) {
  LinearStatePrefixHash hash{};
  if (blocks.empty()) {
    return hash;
  }
  uint32_t block_size = blocks[0].size();
  if (block_size == 0 || boundary_tokens % block_size != 0) {
    return hash;
  }
  size_t boundary_blocks = boundary_tokens / block_size;
  if (boundary_blocks == 0 || boundary_tokens > token_ids.size()) {
    return hash;
  }
  const uint8_t* previous_hash = nullptr;
  for (size_t block_idx = 0; block_idx < boundary_blocks; ++block_idx) {
    xxh3_128bits_hash(previous_hash,
                      Slice<int32_t>(token_ids).slice(
                          block_idx * block_size, (block_idx + 1) * block_size),
                      hash.data());
    previous_hash = hash.data();
  }
  return hash;
}

Sequence make_basic_sequence(const std::vector<int32_t>& prompt_token_ids) {
  static RequestSamplingParam sampling_param;
  static StoppingChecker stopping_checker;

  SequenceParams seq_params;
  seq_params.seq_capacity = prompt_token_ids.size() + 8;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = false;
  seq_params.enable_schedule_overlap = false;

  IncrementalDecoder decoder(/*prompt=*/"",
                             /*num_prompt_tokens=*/prompt_token_ids.size(),
                             /*echo=*/false,
                             /*skip_special_tokens=*/true);
  return Sequence(/*index=*/0,
                  prompt_token_ids,
                  /*input_embedding=*/torch::Tensor(),
                  /*mm_data=*/MMData(),
                  decoder,
                  seq_params);
}

Sequence make_overlap_sequence(const std::vector<int32_t>& prompt_token_ids,
                               size_t seq_capacity,
                               RequestSamplingParam* sampling_param,
                               StoppingChecker* stopping_checker,
                               const std::string& request_id = "",
                               size_t sequence_index = 0) {
  SequenceParams seq_params;
  seq_params.seq_capacity = seq_capacity;
  seq_params.stopping_checker = stopping_checker;
  seq_params.sampling_param = sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = false;
  seq_params.enable_schedule_overlap = true;
  seq_params.request_id = request_id;

  IncrementalDecoder decoder(/*prompt=*/"",
                             /*num_prompt_tokens=*/prompt_token_ids.size(),
                             /*echo=*/false,
                             /*skip_special_tokens=*/true);
  return Sequence(/*index=*/sequence_index,
                  prompt_token_ids,
                  /*input_embedding=*/torch::Tensor(),
                  /*mm_data=*/MMData(),
                  decoder,
                  seq_params);
}

class ScopedPrefillChunkStride final {
 public:
  explicit ScopedPrefillChunkStride(int32_t chunk_stride)
      : previous_(SchedulerConfig::get_instance()
                      .max_tokens_per_chunk_for_prefill()) {
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(
        chunk_stride);
  }

  ~ScopedPrefillChunkStride() {
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(previous_);
  }

 private:
  int32_t previous_;
};

class ScopedJsonObjectOutput final {
 public:
  explicit ScopedJsonObjectOutput(bool enabled)
      : previous_(ServiceConfig::get_instance().enable_json_object_output()) {
    ServiceConfig::get_instance().enable_json_object_output(enabled);
  }

  ~ScopedJsonObjectOutput() {
    ServiceConfig::get_instance().enable_json_object_output(previous_);
  }

 private:
  bool previous_;
};

}  // namespace

TEST(BatchInputBuilderTest, FirstChunkUsesRemotePrefix) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(16);
  BlockManagerImpl manager(options);
  std::vector<Block> blocks = manager.allocate(2);
  Sequence sequence = make_basic_sequence({1});
  sequence.add_blocks(BlockType::KV, blocks);
  const TransferKVInfo full_info = make_info({100, 101, 102, 103, 104});

  const TransferKVInfo info =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_info, &sequence, /*seq_len=*/32);

  expect_mapping(info, BlockType::KV, block_ids(blocks), {100, 101});
  EXPECT_EQ(sequence.kv_state().next_transfer_block_idx(), 2u);
}

TEST(BatchInputBuilderTest, LaterChunkUsesLogicalOffset) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(16);
  BlockManagerImpl manager(options);
  std::vector<Block> blocks = manager.allocate(4);
  Sequence sequence = make_basic_sequence({1});
  sequence.add_blocks(BlockType::KV, blocks);
  sequence.kv_state().set_next_transfer_block_idx(2);
  const TransferKVInfo full_info = make_info({100, 101, 102, 103, 104});

  const TransferKVInfo info =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_info, &sequence, /*seq_len=*/64);

  const std::vector<uint64_t> ids = block_ids(blocks);
  expect_mapping(info, BlockType::KV, {ids[2], ids[3]}, {102, 103});
  EXPECT_EQ(sequence.kv_state().next_transfer_block_idx(), 4u);
}

TEST(BatchInputBuilderTest, PartialBoundaryRepeatsDirtyBlocks) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(16);
  BlockManagerImpl manager(options);
  std::vector<Block> blocks = manager.allocate(3);
  Sequence sequence = make_basic_sequence({1});
  sequence.add_blocks(BlockType::KV, blocks);
  const TransferKVInfo full_info = make_info({100, 101, 102});

  const TransferKVInfo info =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_info, &sequence, /*seq_len=*/33);

  expect_mapping(info, BlockType::KV, block_ids(blocks), {100, 101, 102});
  EXPECT_EQ(sequence.kv_state().next_transfer_block_idx(), 2u);
}

TEST(BatchInputBuilderTest, SharedPrefixUsesLogicalMapping) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(16);
  BlockManagerImpl manager(options);
  std::vector<Block> blocks = manager.allocate(5);
  Sequence sequence = make_basic_sequence({1});
  sequence.add_blocks(BlockType::KV, blocks);
  const TransferKVInfo full_info = make_info({100, 101, 102, 103, 104});

  const TransferKVInfo info =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_info, &sequence, /*seq_len=*/80);

  expect_mapping(
      info, BlockType::KV, block_ids(blocks), {100, 101, 102, 103, 104});
  EXPECT_EQ(sequence.kv_state().next_transfer_block_idx(), 5u);
}

TEST(BatchInputBuilderTest, FlatRemoteSharedPrefixUsesTrimmedRemoteMapping) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(16);
  BlockManagerImpl manager(options);
  std::vector<Block> blocks = manager.allocate(5);
  Sequence sequence = make_basic_sequence({1});
  sequence.add_blocks(BlockType::KV, blocks);
  sequence.kv_state().advance_transfer_block_idx(2);

  TransferKVInfo full_info = make_info({102, 103, 104});
  full_info.mappings[0].remote_shared_num = 2;
  full_info.dst_xtensor_layer_offsets = {
      make_offsets({1002, 1003, 1004}, {2002, 2003, 2004})};

  const TransferKVInfo info =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_info, &sequence, /*seq_len=*/80);

  const std::vector<uint64_t> ids = block_ids(blocks);
  const std::vector<uint64_t> expected_local(ids.begin() + 2, ids.end());
  expect_mapping(info, BlockType::KV, expected_local, {102, 103, 104});
  EXPECT_EQ(find_mapping(info, BlockType::KV).remote_shared_num, 2u);
  EXPECT_EQ(sequence.kv_state().next_transfer_block_idx(), 5u);
  ASSERT_EQ(info.dst_xtensor_layer_offsets.size(), 1u);
  EXPECT_EQ(info.dst_xtensor_layer_offsets[0].k_offsets,
            (std::vector<uint64_t>{1002, 1003, 1004}));
  EXPECT_EQ(info.dst_xtensor_layer_offsets[0].v_offsets,
            (std::vector<uint64_t>{2002, 2003, 2004}));
}

TEST(BatchInputBuilderTest, GroupRemoteSharedPrefixKeepsFullRemoteMapping) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(16);
  BlockManagerImpl manager(options);
  std::vector<Block> blocks = manager.allocate(5);
  Sequence sequence = make_basic_sequence({1});
  sequence.add_blocks(BlockType::C4, blocks);
  sequence.kv_state().advance_group_transfer_block_idx(BlockType::C4, 2);

  TransferKVInfo full_info =
      make_info({100, 101, 102, 103, 104}, BlockType::C4);
  full_info.mappings[0].remote_shared_num = 2;

  const TransferKVInfo info =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_info, &sequence, /*seq_len=*/80);

  const std::vector<uint64_t> ids = block_ids(blocks);
  const std::vector<uint64_t> expected_local(ids.begin() + 2, ids.end());
  expect_mapping(info, BlockType::C4, expected_local, {102, 103, 104});
  EXPECT_EQ(find_mapping(info, BlockType::C4).remote_shared_num, 2u);
  EXPECT_EQ(sequence.kv_state().next_group_transfer_block_idx(BlockType::C4),
            5u);
}

TEST(BatchInputBuilderTest, RemoteSWASentinelsAreNotTransferred) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(16);
  BlockManagerImpl manager(options);
  std::vector<Block> blocks = manager.allocate(4);
  Sequence sequence = make_basic_sequence({1});
  sequence.add_blocks(BlockType::SWA, blocks);

  const uint64_t invalid_id = std::numeric_limits<uint64_t>::max();
  const TransferKVInfo full_info =
      make_info({invalid_id, invalid_id, 102, 103}, BlockType::SWA);

  const TransferKVInfo info =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_info, &sequence, /*seq_len=*/64);

  const std::vector<uint64_t> ids = block_ids(blocks);
  expect_mapping(info, BlockType::SWA, {ids[2], ids[3]}, {102, 103});
  EXPECT_EQ(sequence.kv_state().next_group_transfer_block_idx(BlockType::SWA),
            4u);
}

TEST(BatchInputBuilderTest, SharedPrefixSlicesXTensorOffsets) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(16);
  BlockManagerImpl manager(options);
  std::vector<Block> blocks = manager.allocate(4);
  Sequence sequence = make_basic_sequence({1});
  sequence.add_blocks(BlockType::KV, blocks);
  sequence.kv_state().set_next_transfer_block_idx(2);
  TransferKVInfo full_info = make_info({100, 101, 102, 103, 104});
  full_info.dst_xtensor_layer_offsets = {
      make_offsets({1000, 1001, 1002, 1003, 1004},
                   {2000, 2001, 2002, 2003, 2004}),
      make_offsets({3000, 3001, 3002, 3003, 3004},
                   {4000, 4001, 4002, 4003, 4004})};
  const TransferKVInfo info =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_info, &sequence, /*seq_len=*/64);

  const std::vector<uint64_t> ids = block_ids(blocks);
  expect_mapping(info, BlockType::KV, {ids[2], ids[3]}, {102, 103});
  EXPECT_EQ(sequence.kv_state().next_transfer_block_idx(), 4u);
  ASSERT_EQ(info.dst_xtensor_layer_offsets.size(), 2u);
  EXPECT_EQ(info.dst_xtensor_layer_offsets[0].k_offsets,
            (std::vector<uint64_t>{1002, 1003}));
  EXPECT_EQ(info.dst_xtensor_layer_offsets[0].v_offsets,
            (std::vector<uint64_t>{2002, 2003}));
  EXPECT_EQ(info.dst_xtensor_layer_offsets[1].k_offsets,
            (std::vector<uint64_t>{3002, 3003}));
  EXPECT_EQ(info.dst_xtensor_layer_offsets[1].v_offsets,
            (std::vector<uint64_t>{4002, 4003}));
}

TEST(BatchInputBuilderTest, PartialBoundaryRepeatsXTensorOffsets) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(16);
  BlockManagerImpl manager(options);
  std::vector<Block> blocks = manager.allocate(3);
  Sequence sequence = make_basic_sequence({1});
  sequence.add_blocks(BlockType::KV, blocks);
  TransferKVInfo full_info = make_info({100, 101, 102});
  full_info.dst_xtensor_layer_offsets = {
      make_offsets({1000, 1001, 1002}, {2000, 2001, 2002}),
      make_offsets({3000, 3001, 3002}, {4000, 4001, 4002})};
  const TransferKVInfo info =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_info, &sequence, /*seq_len=*/33);

  expect_mapping(info, BlockType::KV, block_ids(blocks), {100, 101, 102});
  EXPECT_EQ(sequence.kv_state().next_transfer_block_idx(), 2u);
  ASSERT_EQ(info.dst_xtensor_layer_offsets.size(), 2u);
  EXPECT_EQ(info.dst_xtensor_layer_offsets[0].k_offsets,
            (std::vector<uint64_t>{1000, 1001, 1002}));
  EXPECT_EQ(info.dst_xtensor_layer_offsets[0].v_offsets,
            (std::vector<uint64_t>{2000, 2001, 2002}));
  EXPECT_EQ(info.dst_xtensor_layer_offsets[1].k_offsets,
            (std::vector<uint64_t>{3000, 3001, 3002}));
  EXPECT_EQ(info.dst_xtensor_layer_offsets[1].v_offsets,
            (std::vector<uint64_t>{4000, 4001, 4002}));
}

TEST(BatchInputBuilderTest, RemoteCoverageShortageDies) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(16);
  BlockManagerImpl manager(options);
  std::vector<Block> blocks = manager.allocate(3);
  Sequence sequence = make_basic_sequence({1});
  sequence.add_blocks(BlockType::KV, blocks);
  const TransferKVInfo full_info = make_info({100, 101});

  EXPECT_DEATH(
      {
        const TransferKVInfo info =
            BatchInputBuilderTestPeer::build_step_transfer_info(
                full_info, &sequence, /*seq_len=*/48);
        (void)info;
      },
      "remote");
}

TEST(BatchInputBuilderTest, DSV4FirstChunkSlicesFullRemoteAllocation) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(16);
  BlockManagerImpl manager(options);
  std::vector<Block> blocks = manager.allocate(2);
  Sequence sequence = make_basic_sequence({1});
  sequence.add_blocks(BlockType::C4, blocks);
  const TransferKVInfo full_info =
      make_info({100, 101, 102, 103}, BlockType::C4);

  const TransferKVInfo info =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_info, &sequence, /*seq_len=*/32);

  expect_mapping(info, BlockType::C4, block_ids(blocks), {100, 101});
  EXPECT_EQ(sequence.kv_state().next_group_transfer_block_idx(BlockType::C4),
            2u);
}

TEST(BatchInputBuilderTest, DSV4LaterChunkSkipsExpiredSWABlocks) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(16);
  BlockManagerImpl manager(options);
  std::vector<Block> live_blocks = manager.allocate(2);
  std::vector<Block> blocks(2);
  blocks.insert(blocks.end(), live_blocks.begin(), live_blocks.end());
  Sequence sequence = make_basic_sequence({1});
  sequence.add_blocks(BlockType::SWA, blocks);
  sequence.kv_state().advance_group_transfer_block_idx(BlockType::SWA, 2);
  const TransferKVInfo full_info =
      make_info({100, 101, 102, 103}, BlockType::SWA);

  const TransferKVInfo info =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_info, &sequence, /*seq_len=*/64);

  expect_mapping(info, BlockType::SWA, block_ids(live_blocks), {102, 103});
  EXPECT_EQ(sequence.kv_state().next_group_transfer_block_idx(BlockType::SWA),
            4u);
}

TEST(BatchInputBuilderTest, DSV4PartialBlockIsRepeatedOnNextChunk) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(16);
  BlockManagerImpl manager(options);
  std::vector<Block> blocks = manager.allocate(3);
  Sequence sequence = make_basic_sequence({1});
  sequence.add_blocks(BlockType::C4, blocks);
  const TransferKVInfo full_info = make_info({100, 101, 102}, BlockType::C4);

  const TransferKVInfo info =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_info, &sequence, /*seq_len=*/33);

  expect_mapping(info, BlockType::C4, block_ids(blocks), {100, 101, 102});
  EXPECT_EQ(sequence.kv_state().next_group_transfer_block_idx(BlockType::C4),
            2u);
}

TEST(BatchTest, ProcessSampleOutputStoresMtpBootstrapEmbedding) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);

  Sequence sequence = make_basic_sequence({1, 2, 3});
  sequence.add_blocks(BlockType::KV, manager.allocate(1));

  Batch batch(&sequence);
  (void)batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0, ModelArgs());

  SampleOutput output;
  output.next_tokens = torch::tensor({42}, torch::kInt);
  output.embeddings = torch::tensor({{3.0f, 4.0f}});

  batch.process_sample_output(output, /*replace_fake_token=*/false);

  torch::Tensor stored = sequence.get_mtp_bootstrap_embedding();
  ASSERT_TRUE(stored.defined());
  EXPECT_TRUE(torch::equal(stored, output.embeddings[0]));

  output.embeddings.fill_(9.0f);
  EXPECT_TRUE(torch::equal(sequence.get_mtp_bootstrap_embedding(),
                           torch::tensor({3.0f, 4.0f})));
}

TEST(BatchTest, ProcessRawOutputStoresMtpBootstrapEmbedding) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);

  Sequence sequence = make_basic_sequence({1, 2, 3});
  sequence.add_blocks(BlockType::KV, manager.allocate(1));

  Batch batch(&sequence);
  (void)batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0, ModelArgs());

  RawForwardOutput raw_output;
  raw_output.outputs.push_back(make_raw_sample_output(
      42, std::nullopt, {}, {}, /*embeddings=*/{3.0f, 4.0f}));

  batch.process_sample_output(raw_output, /*replace_fake_token=*/false);

  torch::Tensor stored = sequence.get_mtp_bootstrap_embedding();
  ASSERT_TRUE(stored.defined());
  EXPECT_TRUE(torch::equal(stored, torch::tensor({3.0f, 4.0f})));
}

TEST(SequenceTest, JsonObjectCommitAdvancesGrammarOnce) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  std::shared_ptr<const JsonObjectGrammar> grammar =
      std::make_shared<const JsonObjectGrammar>(
          std::vector<std::string>{"{", "}", ""},
          std::unordered_set<int32_t>{2});

  SequenceParams seq_params;
  seq_params.seq_capacity = 16;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.request_id = "req-valid-commit";
  seq_params.json_object_grammar = grammar;
  IncrementalDecoder decoder("", 2, false, false);
  Sequence sequence(/*index=*/0,
                    /*prompt_token_ids=*/{10, 11},
                    torch::Tensor(),
                    MMData(),
                    std::move(decoder),
                    seq_params);
  sequence.add_blocks(BlockType::KV, manager.allocate(1));
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  const size_t original_num_tokens = sequence.num_tokens();

  sequence.append_token(Token(/*id=*/0));

  EXPECT_EQ(sequence.num_tokens(), original_num_tokens + 1);
  EXPECT_EQ(sequence.tokens()[sequence.num_prompt_tokens()], 0);
  const JsonObjectGrammarState* state = sequence.json_object_state();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->snapshot().token_ids, std::vector<int32_t>({0}));
}

TEST(SequenceTest, JsonObjectOverlapCommitAdvancesGrammarOnce) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  std::shared_ptr<const JsonObjectGrammar> grammar =
      std::make_shared<const JsonObjectGrammar>(
          std::vector<std::string>{"{", "}", ""},
          std::unordered_set<int32_t>{2});

  SequenceParams seq_params;
  seq_params.seq_capacity = 16;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.enable_schedule_overlap = true;
  seq_params.request_id = "req-valid-overlap-commit";
  seq_params.json_object_grammar = grammar;
  IncrementalDecoder decoder("", 2, false, false);
  Sequence sequence(/*index=*/0,
                    /*prompt_token_ids=*/{10, 11},
                    torch::Tensor(),
                    MMData(),
                    std::move(decoder),
                    seq_params);
  sequence.add_blocks(BlockType::KV, manager.allocate(1));
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());

  sequence.append_token(Token(/*id=*/-1));
  const size_t placeholder_num_tokens = sequence.num_tokens();
  const JsonObjectGrammarState* placeholder_state =
      sequence.json_object_state();
  ASSERT_NE(placeholder_state, nullptr);
  EXPECT_TRUE(placeholder_state->snapshot().token_ids.empty());

  sequence.update_last_step_token(Token(/*id=*/0), /*token_offset=*/0);

  EXPECT_EQ(sequence.num_tokens(), placeholder_num_tokens);
  EXPECT_EQ(sequence.tokens()[sequence.num_prompt_tokens()], 0);
  const JsonObjectGrammarState* committed_state = sequence.json_object_state();
  ASSERT_NE(committed_state, nullptr);
  EXPECT_EQ(committed_state->snapshot().token_ids, std::vector<int32_t>({0}));
}

TEST(SequenceTest, RestoresJsonObjectStateForBeamCandidate) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  std::shared_ptr<const JsonObjectGrammar> grammar =
      std::make_shared<const JsonObjectGrammar>(
          std::vector<std::string>{"{", " {", "\"a\"", ":", "1"},
          std::unordered_set<int32_t>{});

  SequenceParams seq_params;
  seq_params.seq_capacity = 16;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.request_id = "req-beam-restore";
  seq_params.json_object_grammar = grammar;
  IncrementalDecoder decoder("", 2, false, false);
  Sequence sequence(/*index=*/0,
                    /*prompt_token_ids=*/{10, 11},
                    torch::Tensor(),
                    MMData(),
                    std::move(decoder),
                    seq_params);
  sequence.add_blocks(BlockType::KV, manager.allocate(1));
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  sequence.append_token(Token(/*id=*/0));

  JsonObjectGrammarSnapshot candidate_snapshot =
      sequence.json_object_state()->snapshot();
  candidate_snapshot.token_ids.back() = 1;
  ASSERT_TRUE(sequence.restore_json_object_state(candidate_snapshot));
  EXPECT_EQ(sequence.json_object_state()->snapshot().token_ids,
            std::vector<int32_t>({1}));
}

TEST(SequenceTest, JsonObjectCommitMismatchFailsBeforeTokenMutation) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  std::shared_ptr<const JsonObjectGrammar> grammar =
      std::make_shared<const JsonObjectGrammar>(
          std::vector<std::string>{"{", "}", ""},
          std::unordered_set<int32_t>{2});

  SequenceParams seq_params;
  seq_params.seq_capacity = 16;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.request_id = "req-commit";
  seq_params.json_object_grammar = grammar;
  IncrementalDecoder decoder("", 2, false, false);
  Sequence sequence(/*index=*/0,
                    /*prompt_token_ids=*/{10, 11},
                    torch::Tensor(),
                    MMData(),
                    std::move(decoder),
                    seq_params);
  sequence.add_blocks(BlockType::KV, manager.allocate(1));
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  const size_t original_num_tokens = sequence.num_tokens();

  EXPECT_NO_FATAL_FAILURE(sequence.append_token(Token(/*id=*/1)));

  EXPECT_EQ(sequence.num_tokens(), original_num_tokens);
  ASSERT_TRUE(sequence.error_status().has_value());
  EXPECT_TRUE(sequence.finished());
  const JsonObjectGrammarState* state = sequence.json_object_state();
  ASSERT_NE(state, nullptr);
  EXPECT_TRUE(state->snapshot().token_ids.empty());
}

TEST(SequenceTest, JsonObjectOverlapMismatchKeepsPlaceholderToken) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  std::shared_ptr<const JsonObjectGrammar> grammar =
      std::make_shared<const JsonObjectGrammar>(
          std::vector<std::string>{"{", "}", ""},
          std::unordered_set<int32_t>{2});

  SequenceParams seq_params;
  seq_params.seq_capacity = 16;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.enable_schedule_overlap = true;
  seq_params.request_id = "req-overlap-commit";
  seq_params.json_object_grammar = grammar;
  IncrementalDecoder decoder("", 2, false, false);
  Sequence sequence(/*index=*/0,
                    /*prompt_token_ids=*/{10, 11},
                    torch::Tensor(),
                    MMData(),
                    std::move(decoder),
                    seq_params);
  sequence.add_blocks(BlockType::KV, manager.allocate(1));
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  sequence.append_token(Token(/*id=*/-1));
  const size_t original_num_tokens = sequence.num_tokens();

  EXPECT_NO_FATAL_FAILURE(
      sequence.update_last_step_token(Token(/*id=*/1), /*token_offset=*/0));

  EXPECT_EQ(sequence.num_tokens(), original_num_tokens);
  EXPECT_EQ(sequence.tokens()[sequence.num_prompt_tokens()], -1);
  ASSERT_TRUE(sequence.error_status().has_value());
  EXPECT_TRUE(sequence.finished());
  const JsonObjectGrammarState* state = sequence.json_object_state();
  ASSERT_NE(state, nullptr);
  EXPECT_TRUE(state->snapshot().token_ids.empty());
}

TEST(BatchTest, JsonObjectCommitMismatchSkipsRemainingSampleSlotsAndEmbedding) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);
  RequestSamplingParam sampling_param;
  sampling_param.logprobs = true;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  std::shared_ptr<const JsonObjectGrammar> grammar =
      std::make_shared<const JsonObjectGrammar>(
          std::vector<std::string>{"{", "x", ""},
          std::unordered_set<int32_t>{2});
  std::vector<SampleSlot> sample_slots;
  SampleSlot first_slot;
  first_slot.request_id = "sample-req";
  first_slot.sample_id = 0;
  first_slot.token_position = 2;
  sample_slots.emplace_back(first_slot);
  SampleSlot second_slot;
  second_slot.request_id = "sample-req";
  second_slot.sample_id = 1;
  second_slot.token_position = 4;
  sample_slots.emplace_back(second_slot);

  SequenceParams seq_params;
  seq_params.seq_capacity = 16;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.sample_slots = &sample_slots;
  seq_params.request_id = "req-sample-slots";
  seq_params.json_object_grammar = grammar;
  IncrementalDecoder decoder("", 4, false, false);
  Sequence sequence(/*index=*/0,
                    /*prompt_token_ids=*/{10, 11, 12, 13},
                    torch::Tensor(),
                    MMData(),
                    std::move(decoder),
                    seq_params);
  sequence.add_blocks(BlockType::KV, manager.allocate(1));
  Batch batch(&sequence);
  (void)batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0, ModelArgs());
  const size_t original_num_tokens = sequence.num_tokens();

  RawForwardOutput raw_output;
  raw_output.outputs = {
      make_raw_sample_output(
          /*token_id=*/1, -0.1f, {}, {}, /*embeddings=*/{3.0f, 4.0f}),
      make_raw_sample_output(/*token_id=*/0, -0.2f)};
  EXPECT_NO_FATAL_FAILURE(
      batch.process_sample_output(raw_output, /*replace_fake_token=*/false));

  EXPECT_EQ(sequence.num_tokens(), original_num_tokens);
  EXPECT_FALSE(sequence.get_mtp_bootstrap_embedding().defined());
  ASSERT_TRUE(sequence.error_status().has_value());
  EXPECT_TRUE(sequence.finished());
}

TEST(BatchTest, JsonObjectErrorDiscardsFailedMmEmbeddingWithoutRowShift) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);
  RequestSamplingParam sampling_param;
  sampling_param.is_embeddings = true;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(1);

  SequenceParams failed_params;
  failed_params.seq_capacity = 8;
  failed_params.stopping_checker = &stopping_checker;
  failed_params.sampling_param = &sampling_param;
  failed_params.request_id = "req-mm-failed";
  SequenceParams healthy_params = failed_params;
  healthy_params.request_id = "req-mm-healthy";

  MMItemVec failed_items = {MMDataItem(MMType::IMAGE)};
  MMItemVec healthy_items = {MMDataItem(MMType::IMAGE)};
  IncrementalDecoder failed_decoder("", 1, false, false);
  Sequence failed(/*index=*/0,
                  /*prompt_token_ids=*/{10},
                  torch::Tensor(),
                  MMData(MMType::IMAGE, failed_items),
                  std::move(failed_decoder),
                  failed_params);
  IncrementalDecoder healthy_decoder("", 1, false, false);
  Sequence healthy(/*index=*/0,
                   /*prompt_token_ids=*/{20},
                   torch::Tensor(),
                   MMData(MMType::IMAGE, healthy_items),
                   std::move(healthy_decoder),
                   healthy_params);
  failed.add_blocks(BlockType::KV, manager.allocate(1));
  healthy.add_blocks(BlockType::KV, manager.allocate(1));
  Batch batch({&failed, &healthy});
  (void)batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0, ModelArgs());

  RawForwardOutput raw_output;
  raw_output.outputs = {make_raw_sample_output(/*token_id=*/0, std::nullopt),
                        make_raw_sample_output(/*token_id=*/0, std::nullopt)};
  raw_output.outputs[0].mm_embeddings = {torch::tensor({1.0f, 2.0f})};
  raw_output.outputs[1].mm_embeddings = {torch::tensor({3.0f, 4.0f})};
  raw_output.json_object_errors = {
      {failed.sample_sequence_id(), "missing prior sampled output row"}};
  batch.process_sample_output(raw_output, /*replace_fake_token=*/false);

  ASSERT_TRUE(failed.error_status().has_value());
  EXPECT_TRUE(healthy.finished());
  Tokenizer tokenizer;
  SequenceOutput output = healthy.generate_output(tokenizer);
  ASSERT_TRUE(output.mm_embeddings.has_value());
  ASSERT_EQ(output.mm_embeddings->size(), 1u);
  EXPECT_TRUE(torch::equal(output.mm_embeddings->at(0).embedding,
                           raw_output.outputs[1].mm_embeddings[0]));
}

TEST(BatchTest, DecodeForwardInputConsumesMtpBootstrap) {
  BlockManagerPool::Options options;
  options.num_blocks(8).block_size(4).enable_disagg_pd(true);
  options.max_seqs_per_batch(1024);
  options.num_speculative_tokens(1);
  options.num_embedding_blocks(8);
  BlockManagerPool manager(options, /*dp_size=*/1);

  Sequence sequence = make_basic_sequence({1, 2, 3});
  ASSERT_TRUE(manager.allocate(&sequence));
  ASSERT_GE(sequence.get_embedding_block_id(), 0);
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  sequence.append_token(Token(42));

  torch::Tensor embedding = torch::tensor({3.0f, 4.0f});
  sequence.update_mtp_bootstrap_embedding(embedding);

  Batch batch(&sequence);
  ForwardInput forward_input = batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0, ModelArgs());

  const auto& embed_params = forward_input.input_params.embedding;
  EXPECT_EQ(embed_params.mtp_bootstrap_row_idxes, std::vector<int32_t>{0});
  ASSERT_TRUE(embed_params.mtp_bootstrap_embeddings.defined());
  EXPECT_TRUE(torch::equal(embed_params.mtp_bootstrap_embeddings,
                           embedding.unsqueeze(0)));
  EXPECT_FALSE(sequence.get_mtp_bootstrap_embedding().defined());
}

TEST(BatchTest, DecodeForwardInputMapsSparseMtpBootstrapRows) {
  BlockManagerPool::Options options;
  options.num_blocks(8).block_size(4).enable_disagg_pd(true);
  options.max_seqs_per_batch(1024);
  options.num_speculative_tokens(1);
  options.num_embedding_blocks(8);
  BlockManagerPool manager(options, /*dp_size=*/1);

  Sequence first = make_basic_sequence({1, 2, 3});
  ASSERT_TRUE(manager.allocate(&first));
  first.kv_state().set_kv_cache_tokens_num(first.num_prompt_tokens());
  first.append_token(Token(41));

  Sequence second = make_basic_sequence({4, 5, 6});
  ASSERT_TRUE(manager.allocate(&second));
  second.kv_state().set_kv_cache_tokens_num(second.num_prompt_tokens());
  second.append_token(Token(42));

  torch::Tensor embedding = torch::tensor({3.0f, 4.0f});
  second.update_mtp_bootstrap_embedding(embedding);

  Batch batch({&first, &second});
  ForwardInput forward_input = batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0, ModelArgs());

  const auto& embed_params = forward_input.input_params.embedding;
  EXPECT_EQ(embed_params.mtp_bootstrap_row_idxes, std::vector<int32_t>{1});
  ASSERT_TRUE(embed_params.mtp_bootstrap_embeddings.defined());
  EXPECT_TRUE(torch::equal(embed_params.mtp_bootstrap_embeddings,
                           embedding.unsqueeze(0)));
}

TEST(BatchTest, Basic) {
  // use init device to trigger the loading of torch backend for different
  // devices
  //  since the allocation of pinnned memory on cpu is still backend-dependent.
  torch::Device device(Platform::type_torch(), 0);
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

  seq1.add_blocks(BlockType::KV, manager.allocate(3));  // [1, 2, 3]

  IncrementalDecoder fake_decoder2("", 2, false, false);
  // seq in decode phase
  Sequence seq2(/*index=*/0,
                /*token_ids=*/{2, 4, 6, 8, 6, 4, 2},
                input_embedding,
                mm_data,
                std::move(fake_decoder2),
                seq_params);
  seq2.add_blocks(BlockType::KV, manager.allocate(4));  // [4, 5, 6, 7]
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
  seq3.add_blocks(BlockType::KV, manager.allocate(5));  // [8, 9, 10, 11, 12]
  seq3.kv_state().incr_kv_cache_tokens_num(/*size=*/15);
  seq3.append_token(200);

  IncrementalDecoder fake_decoder4("", 4, false, false);
  Sequence seq4(/*index=*/0,
                /*token_ids=*/{1, 3, 5, 7, 5, 4, 3, 2, 1},
                input_embedding,
                mm_data,
                std::move(fake_decoder4),
                seq_params);
  seq4.kv_state().add_blocks(BlockType::KV,
                             manager.allocate(3));  // [13, 14, 15]
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
  EXPECT_TRUE(input_params.meta.batch_forward_type.is_mixed());
  EXPECT_EQ(input_params.meta.num_sequences, 4);
  EXPECT_EQ(input_params.meta.q_max_seq_len, 9);
  EXPECT_EQ(input_params.meta.kv_max_seq_len, 16);
  EXPECT_EQ(input_params.embedding.embedding_ids, std::vector<int32_t>({-1, -1, -1}));
  EXPECT_EQ(input_params.embedding.linear_state_ids,
            std::vector<int32_t>({-1, -1, -1, -1}));

#if defined(USE_NPU)
  const std::vector<int32_t> q_seq_lens = {9, 1, 1, 4};
#else
  const std::vector<int32_t> q_seq_lens = {0, 9, 10, 11, 15};
#endif
  EXPECT_TRUE(equal(input_params.attention.device.q_seq_lens, q_seq_lens));

//  seq4's kv_seq_len = q_len + num_cached_tokens (q_len<=max_allowed_tokens)
#if defined(USE_NPU)
  const std::vector<int32_t> kv_seq_lens = {9, 8, 16, 8};
#else
  const std::vector<int32_t> kv_seq_lens = {0, 9, 17, 33, 41};
#endif
  EXPECT_TRUE(equal(input_params.attention.device.kv_seq_lens, kv_seq_lens));

  const std::vector<int32_t> new_cache_slots = {
    /*seq1*/ 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    /*seq2*/ 23, 
    /*seq3*/ 47,
    /*seq4*/ 56,57,58,59
    };
  EXPECT_TRUE(equal(input_params.attention.device.new_cache_slots, new_cache_slots));

  const std::vector<int32_t> block_tables = {
    /*seq1*/ 1, 2, 3,  0,  0,
    /*seq2*/ 4, 5, 6,  7,  0,
    /*seq3*/ 8, 9, 10, 11, 12,
    /*seq4*/ 13, 14, 15, 0, 0};
  EXPECT_TRUE(equal(input_params.attention.device.block_tables, block_tables));

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
  torch::Device device(Platform::type_torch(), 0);
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
  seq.add_blocks(BlockType::KV, manager.allocate(2));

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

TEST(BatchTest, JsonObjectSampleSequenceIdsFollowSamplingRows) {
  ScopedJsonObjectOutput enabled(/*enabled=*/true);
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  std::shared_ptr<const JsonObjectGrammar> grammar =
      std::make_shared<const JsonObjectGrammar>(
          std::vector<std::string>{"{", "}", "\"", "a", ":", "1", ""},
          std::unordered_set<int32_t>{6});

  SequenceParams first_params;
  first_params.seq_capacity = 16;
  first_params.stopping_checker = &stopping_checker;
  first_params.sampling_param = &sampling_param;
  first_params.request_id = "req-shared";
  first_params.json_object_grammar = grammar;
  first_params.enable_schedule_overlap = true;

  SequenceParams second_params = first_params;
  second_params.request_id = "req-shared";
  second_params.json_object_grammar.reset();

  IncrementalDecoder first_decoder("", 2, false, false);
  Sequence first(/*index=*/0,
                 /*token_ids=*/{10, 11},
                 torch::Tensor(),
                 MMData(),
                 std::move(first_decoder),
                 first_params);
  first.add_blocks(BlockType::KV, manager.allocate(1));

  IncrementalDecoder second_decoder("", 2, false, false);
  Sequence second(/*index=*/1,
                  /*token_ids=*/{20, 21},
                  torch::Tensor(),
                  MMData(),
                  std::move(second_decoder),
                  second_params);
  second.add_blocks(BlockType::KV, manager.allocate(1));

  Batch batch({&second, &first});
  ForwardInput forward_input = batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_bach_size=*/0, ModelArgs());

  EXPECT_EQ(forward_input.sample_sequence_ids,
            std::vector<std::string>({"req-shared#1", "req-shared#0"}));
  EXPECT_EQ(forward_input.sample_prior_output_rows,
            std::vector<int32_t>({-1, -1}));
  ASSERT_EQ(forward_input.json_object_states.size(), 2u);
  ASSERT_TRUE(forward_input.sampling_params.filter_bitmask.defined());
  EXPECT_FALSE(forward_input.sampling_params.filter_mask.defined());
  EXPECT_EQ(forward_input.sampling_params.filter_bitmask.size(0), 2);
  EXPECT_EQ(forward_input.sampling_params.filter_bitmask.size(1), 1);
  EXPECT_EQ(forward_input.sampling_params.filter_bitmask.index({0, 0})
                .item<int32_t>(),
            -1);
  EXPECT_NE(forward_input.sampling_params.filter_bitmask.index({1, 0})
                .item<int32_t>(),
            -1);

  RawForwardOutput fake_output;
  fake_output.outputs = {make_raw_sample_output(-1, std::nullopt),
                         make_raw_sample_output(-2, std::nullopt)};
  batch.process_sample_output(fake_output, /*replace_fake_token=*/false);
  ForwardInput next_input = batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_bach_size=*/0, ModelArgs());
  EXPECT_EQ(next_input.sample_prior_output_rows, std::vector<int32_t>({0, 1}));
}

TEST(BatchTest, ReorderedMtpAcceptedRowsCommitToOwningSequences) {
  ScopedJsonObjectOutput enabled(/*enabled=*/true);
  BlockManager::Options options;
  options.num_blocks(8).block_size(16);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(16);
  std::shared_ptr<const JsonObjectGrammar> grammar =
      std::make_shared<const JsonObjectGrammar>(
          std::vector<std::string>{"{",
                                   "\"a\"",
                                   ":",
                                   "1",
                                   "}",
                                   "stop",
                                   "plain-6",
                                   "plain-7",
                                   "plain-8",
                                   "]",
                                   "plain-10",
                                   "plain-11",
                                   "plain-12",
                                   "plain-13",
                                   "plain-14",
                                   "plain-15"},
          std::unordered_set<int32_t>{5});

  SequenceParams constrained_params;
  constrained_params.seq_capacity = 16;
  constrained_params.stopping_checker = &stopping_checker;
  constrained_params.sampling_param = &sampling_param;
  constrained_params.request_id = "req-json-chain";
  constrained_params.json_object_grammar = grammar;
  constrained_params.json_reasoning_enabled = false;
  constrained_params.enable_schedule_overlap = true;

  SequenceParams plain_params = constrained_params;
  plain_params.request_id = "req-plain-chain";
  plain_params.json_object_grammar.reset();

  IncrementalDecoder constrained_decoder("", 2, false, false);
  Sequence constrained(/*index=*/0,
                       /*prompt_token_ids=*/{100, 101},
                       torch::Tensor(),
                       MMData(),
                       std::move(constrained_decoder),
                       constrained_params);
  constrained.add_blocks(BlockType::KV, manager.allocate(1));

  IncrementalDecoder plain_decoder("", 2, false, false);
  Sequence plain(/*index=*/1,
                 /*prompt_token_ids=*/{200, 201},
                 torch::Tensor(),
                 MMData(),
                 std::move(plain_decoder),
                 plain_params);
  plain.add_blocks(BlockType::KV, manager.allocate(1));

  Batch batch({&plain, &constrained});
  ForwardInput first_input = batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0, ModelArgs());
  EXPECT_EQ(
      first_input.sample_sequence_ids,
      std::vector<std::string>({"req-plain-chain#1", "req-json-chain#0"}));
  EXPECT_TRUE(equal(first_input.sampling_params.selected_token_idxes,
                    std::vector<int32_t>({1, 3})));
  ASSERT_EQ(first_input.json_object_states.size(), 2u);
  EXPECT_FALSE(first_input.json_object_states[0].initialized());
  EXPECT_TRUE(first_input.json_object_states[1].initialized());

  RawForwardOutput fake_output;
  fake_output.outputs = {make_raw_sample_output(-1, std::nullopt),
                         make_raw_sample_output(-2, std::nullopt)};
  batch.process_sample_output(fake_output, /*replace_fake_token=*/false);
  ForwardInput commit_input = batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0, ModelArgs());
  EXPECT_EQ(
      commit_input.sample_sequence_ids,
      std::vector<std::string>({"req-plain-chain#1", "req-json-chain#0"}));
  EXPECT_EQ(commit_input.sample_prior_output_rows,
            std::vector<int32_t>({0, 1}));

  const torch::Tensor draft_token_ids =
      torch::tensor({{10, 11, 12, 13, 14}, {0, 1, 9, 3, 4}}, torch::kInt64);
  const torch::Tensor target_token_ids =
      torch::tensor({{10, 11, 12, 13, 14}, {0, 1, 2, 3, 4}}, torch::kInt64);
  torch::Tensor target_logits = torch::zeros({2, 6, 16}, torch::kFloat32);
  for (int64_t row = 0; row < target_token_ids.size(0); ++row) {
    for (int64_t step = 0; step < target_token_ids.size(1); ++step) {
      target_logits.index_put_(
          {row, step, target_token_ids.index({row, step}).item<int64_t>()},
          20.0F);
    }
  }
  const torch::Tensor bonus_token_ids =
      torch::tensor({{15}, {5}}, torch::kInt64);
  const torch::Tensor do_sample = torch::tensor({false, false}, torch::kBool);
  RejectionSampler rejection_sampler(do_sample,
                                     /*all_random_sample=*/false,
                                     /*all_greedy_sample=*/true,
                                     /*logprobs=*/false,
                                     /*max_top_logprobs=*/0);
  const SampleOutput accepted_output =
      rejection_sampler.forward(draft_token_ids,
                                torch::zeros({2, 5}, torch::kFloat32),
                                target_logits,
                                bonus_token_ids,
                                /*mask_out_rejected_tokens=*/true);
  EXPECT_TRUE(torch::equal(
      accepted_output.next_tokens,
      torch::tensor({{10, 11, 12, 13, 14, 15}, {0, 1, 2, -1, -1, -1}},
                    torch::kInt64)));

  JsonObjectGrammarState expected_constrained_state = grammar->initial_state();
  ASSERT_TRUE(expected_constrained_state.accept_token(0));
  ASSERT_TRUE(expected_constrained_state.accept_token(1));
  EXPECT_FALSE(expected_constrained_state.can_accept_token(9));
  ASSERT_TRUE(expected_constrained_state.accept_token(2));

  const torch::Tensor undefined;
  proto::ForwardOutput proto_output;
  forward_output_to_proto(accepted_output.next_tokens,
                          undefined,
                          undefined,
                          undefined,
                          undefined,
                          /*mm_embeddings=*/{},
                          undefined,
                          /*prepared_token=*/-1,
                          undefined,
                          undefined,
                          undefined,
                          /*dit_images=*/{},
                          /*dit_text_output=*/{},
                          /*json_object_errors=*/{},
                          &proto_output);
  RawForwardOutput raw_output;
  proto_to_forward_output(proto_output, raw_output);
  ASSERT_EQ(raw_output.outputs.size(), 2u);
  EXPECT_EQ(raw_output.outputs[0].tokens.size(), 6u);
  EXPECT_EQ(raw_output.outputs[0].tokens.back().id, 15);
  EXPECT_EQ(raw_output.outputs[1].tokens.size(), 3u);
  EXPECT_EQ(raw_output.outputs[1].tokens.back().id, 2);

  EXPECT_NO_FATAL_FAILURE(
      batch.process_sample_output(raw_output, /*replace_fake_token=*/true));
  ASSERT_FALSE(plain.error_status().has_value());
  ASSERT_FALSE(constrained.error_status().has_value());
  ASSERT_EQ(plain.num_generated_tokens(), 6u);
  ASSERT_EQ(constrained.num_generated_tokens(), 3u);
  for (size_t token_idx = 0; token_idx < 6; ++token_idx) {
    EXPECT_EQ(plain.tokens()[plain.num_prompt_tokens() + token_idx],
              10 + token_idx);
  }
  EXPECT_EQ(constrained.tokens()[constrained.num_prompt_tokens()], 0);
  EXPECT_EQ(constrained.tokens()[constrained.num_prompt_tokens() + 1], 1);
  EXPECT_EQ(constrained.tokens()[constrained.num_prompt_tokens() + 2], 2);
  ASSERT_NE(constrained.json_object_state(), nullptr);
  EXPECT_EQ(constrained.json_object_state()->fingerprint(),
            expected_constrained_state.fingerprint());
}

TEST(BatchTest, JsonObjectMetadataIsSkippedWhenDisabled) {
  ScopedJsonObjectOutput disabled(/*enabled=*/false);
  BlockManager::Options options;
  options.num_blocks(4).block_size(4);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  std::shared_ptr<const JsonObjectGrammar> grammar =
      std::make_shared<const JsonObjectGrammar>(
          std::vector<std::string>{"{", "}", "\"", "a", ":", "1", ""},
          std::unordered_set<int32_t>{6});

  SequenceParams sequence_params;
  sequence_params.seq_capacity = 16;
  sequence_params.stopping_checker = &stopping_checker;
  sequence_params.sampling_param = &sampling_param;
  sequence_params.request_id = "req-disabled";
  sequence_params.json_object_grammar = grammar;
  sequence_params.enable_schedule_overlap = true;

  IncrementalDecoder decoder("", 2, false, false);
  Sequence sequence(/*index=*/0,
                    /*token_ids=*/{10, 11},
                    torch::Tensor(),
                    MMData(),
                    std::move(decoder),
                    sequence_params);
  sequence.add_blocks(BlockType::KV, manager.allocate(1));

  Batch batch({&sequence});
  ForwardInput forward_input = batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_bach_size=*/0, ModelArgs());

  EXPECT_TRUE(forward_input.json_object_states.empty());
  EXPECT_TRUE(forward_input.json_object_state_snapshots.empty());
  EXPECT_TRUE(forward_input.sample_sequence_ids.empty());
  EXPECT_TRUE(forward_input.sample_prior_output_rows.empty());
  EXPECT_FALSE(forward_input.sampling_params.filter_bitmask.defined());
  EXPECT_FALSE(forward_input.sampling_params.filter_mask.defined());
}

TEST(BatchTest, ChunkedPDTransferUsesStepWindow) {
  torch::Device device(Platform::type_torch(), 0);
  const uint32_t n_blocks = 8;
  const uint32_t block_size = 4;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = false;
  seq_params.enable_schedule_overlap = false;
  seq_params.request_id = "req-1";

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder decoder("", 1, false, false);
  Sequence seq(/*index=*/0,
               /*token_ids=*/{1, 2, 3, 4, 5, 6, 7, 8},
               input_embedding,
               mm_data,
               std::move(decoder),
               seq_params);
  seq.add_blocks(BlockType::KV, manager.allocate(2));

  TransferKVInfo info;
  info.request_id = "req-1";
  info.mappings.emplace_back(make_mapping(BlockType::KV, {100, 101, 102, 103}));
  info.dp_rank = 0;
  info.remote_instance_info.dp_size = 1;
  seq.kv_state().set_transfer_kv_info(std::move(info));

  std::vector<Sequence*> sequences = {&seq};
  std::vector<uint32_t> budgets = {8};
  BatchInputBuilder builder(sequences,
                            budgets,
                            {},
                            {},
                            nullptr,
                            0,
                            nullptr,
                            BatchForwardType::PREFILL);

  ForwardInput input =
      builder.build_forward_input(/*num_decoding_tokens=*/1,
                                  /*min_decoding_batch_size=*/0);

  ASSERT_EQ(input.transfer_kv_infos.size(), 1u);
  const KVTransferMapping& mapping =
      find_mapping(input.transfer_kv_infos[0], BlockType::KV);
  EXPECT_EQ(mapping.local_ids, (std::vector<uint64_t>{1, 2}));
  EXPECT_EQ(mapping.remote_ids, (std::vector<uint64_t>{100, 101}));
  EXPECT_EQ(seq.kv_state().next_transfer_block_idx(), 2u);
}

TEST(BatchTest, PrefixCacheTransferIgnoresKvCacheCursor) {
  torch::Device device(Platform::type_torch(), 0);
  const uint32_t block_size = 4;
  BlockManager::Options options;
  options.num_blocks(8).block_size(block_size);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = false;
  seq_params.enable_schedule_overlap = false;
  seq_params.request_id = "req-prefix";

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder decoder("", 1, false, false);
  Sequence seq(/*index=*/0,
               /*token_ids=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
               input_embedding,
               mm_data,
               std::move(decoder),
               seq_params);
  seq.add_blocks(BlockType::KV, manager.allocate(3));
  seq.kv_state().set_kv_cache_tokens_num(8);

  TransferKVInfo info;
  info.request_id = "req-prefix";
  info.mappings.emplace_back(make_mapping(BlockType::KV, {100, 101, 102}));
  info.dp_rank = 0;
  info.remote_instance_info.dp_size = 1;
  seq.kv_state().set_transfer_kv_info(std::move(info));

  std::vector<Sequence*> sequences = {&seq};
  std::vector<uint32_t> budgets = {2};
  BatchInputBuilder builder(sequences,
                            budgets,
                            {},
                            {},
                            nullptr,
                            0,
                            nullptr,
                            BatchForwardType::PREFILL);

  ForwardInput input =
      builder.build_forward_input(/*num_decoding_tokens=*/1,
                                  /*min_decoding_batch_size=*/0);

  ASSERT_EQ(input.transfer_kv_infos.size(), 1u);
  const KVTransferMapping& mapping =
      find_mapping(input.transfer_kv_infos[0], BlockType::KV);
  EXPECT_EQ(mapping.local_ids, (std::vector<uint64_t>{1, 2, 3}));
  EXPECT_EQ(mapping.remote_ids, (std::vector<uint64_t>{100, 101, 102}));
  EXPECT_EQ(seq.kv_state().next_transfer_block_idx(), 2u);
}

TEST(BatchTest, ForwardInputPreservesTransferInfoAndBatchId) {
  const uint64_t batch_id = 42;
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = false;
  seq_params.enable_schedule_overlap = false;
  seq_params.request_id = "req-1";

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder decoder("", 1, false, false);
  Sequence seq(/*index=*/0,
               /*token_ids=*/{1, 2, 3, 4, 5, 6, 7, 8},
               input_embedding,
               mm_data,
               std::move(decoder),
               seq_params);
  seq.add_blocks(BlockType::KV, manager.allocate(2));

  TransferKVInfo info;
  info.request_id = "req-1";
  info.mappings.emplace_back(make_mapping(BlockType::KV, {100, 101, 102, 103}));
  info.dp_rank = 0;
  info.remote_instance_info.dp_size = 1;
  seq.kv_state().set_transfer_kv_info(std::move(info));

  std::vector<Sequence*> sequences = {&seq};
  std::vector<uint32_t> budgets = {8};
  BatchInputBuilder builder(sequences,
                            budgets,
                            {},
                            {},
                            nullptr,
                            batch_id,
                            nullptr,
                            BatchForwardType::PREFILL);

  ForwardInput input =
      builder.build_forward_input(/*num_decoding_tokens=*/1,
                                  /*min_decoding_batch_size=*/0);

  ASSERT_EQ(input.transfer_kv_infos.size(), 1u);
  const KVTransferMapping& mapping =
      find_mapping(input.transfer_kv_infos[0], BlockType::KV);
  EXPECT_EQ(mapping.local_ids, (std::vector<uint64_t>{1, 2}));
  EXPECT_EQ(mapping.remote_ids, (std::vector<uint64_t>{100, 101}));
  EXPECT_EQ(input.input_params.meta.batch_id, batch_id);
}

TEST(BatchTest, ForwardInputPackedRoundTripPreservesTransportFields) {
  const uint64_t batch_id = 43;
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  sampling_param.logprobs = true;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = true;
  seq_params.enable_schedule_overlap = false;
  seq_params.request_id = "req-packed";

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder decoder("", 1, false, false);
  Sequence seq(/*index=*/0,
               /*token_ids=*/{1, 2, 3, 4},
               input_embedding,
               mm_data,
               std::move(decoder),
               seq_params);
  seq.add_blocks(BlockType::KV, manager.allocate(1));

  TransferKVInfo info;
  info.request_id = "req-packed";
  info.mappings.emplace_back(make_mapping(BlockType::KV, {100, 101}));
  info.dp_rank = 1;
  info.remote_instance_info.dp_size = 2;
  seq.kv_state().set_transfer_kv_info(std::move(info));

  std::vector<Sequence*> sequences = {&seq};
  std::vector<uint32_t> budgets = {4};
  BatchInputBuilder builder(sequences,
                            budgets,
                            {},
                            {},
                            nullptr,
                            batch_id,
                            nullptr,
                            BatchForwardType::PREFILL);

  ForwardInput input =
      builder.build_forward_input(/*num_decoding_tokens=*/1,
                                  /*min_decoding_batch_size=*/0);
  input.input_params.parallel.dp_global_batch_generations = {3, 7};
  input.input_params.embedding.mtp_bootstrap_row_idxes = {0};
  input.input_params.embedding.mtp_bootstrap_embeddings =
      torch::tensor({{3.0f, 4.0f}});
  input.sample_sequence_ids = {"req-packed#0"};
  input.sample_prior_output_rows = {3};
  input.sampling_params.filter_bitmask =
      torch::tensor({{static_cast<int32_t>(-1)}},
                    torch::TensorOptions().dtype(torch::kInt32));
  bool is_creator = false;
  auto shm_name =
      ForwardSharedMemoryManager::create_unique_name("batch_test_forward_input",
                                                     /*dp_group=*/0,
                                                     ForwardType::RAW_INPUT,
                                                     /*rank=*/0);
  ForwardSharedMemoryManager writer_manager(
      shm_name, 1 << 20, is_creator, ForwardType::RAW_INPUT);
  bool is_reader_creator = false;
  ForwardSharedMemoryManager reader_manager(
      shm_name, 1 << 20, is_reader_creator, ForwardType::RAW_INPUT);

  ASSERT_TRUE(writer_manager.input_write(input));

  ForwardInput round_trip;
  reader_manager.input_read(round_trip, torch::Device(torch::kCPU));

  EXPECT_EQ(round_trip.input_params.meta.batch_id, batch_id);
  EXPECT_TRUE(equal(round_trip.token_ids, std::vector<int32_t>({1, 2, 3, 4})));
  ASSERT_EQ(round_trip.transfer_kv_infos.size(), 1u);
  const KVTransferMapping& mapping =
      find_mapping(round_trip.transfer_kv_infos[0], BlockType::KV);
  EXPECT_EQ(mapping.local_ids, (std::vector<uint64_t>{1}));
  EXPECT_EQ(mapping.remote_ids, (std::vector<uint64_t>{100}));
  EXPECT_EQ(round_trip.input_params.embedding.mtp_bootstrap_row_idxes,
            std::vector<int32_t>{0});
  EXPECT_EQ(round_trip.input_params.parallel.dp_global_batch_generations,
            (std::vector<uint64_t>{3, 7}));
  EXPECT_EQ(round_trip.sample_sequence_ids,
            std::vector<std::string>{"req-packed#0"});
  EXPECT_EQ(round_trip.sample_prior_output_rows, std::vector<int32_t>{3});
  ASSERT_TRUE(round_trip.sampling_params.filter_bitmask.defined());
  EXPECT_TRUE(
      torch::equal(round_trip.sampling_params.filter_bitmask.to(torch::kCPU),
                   input.sampling_params.filter_bitmask));
  ASSERT_TRUE(
      round_trip.input_params.embedding.mtp_bootstrap_embeddings.defined());
  EXPECT_TRUE(torch::equal(
      round_trip.input_params.embedding.mtp_bootstrap_embeddings.to(
          torch::kCPU),
      torch::tensor({{3.0f, 4.0f}})));
}

TEST(BatchTest, ForwardOutputProtoRoundTripPreservesJsonObjectErrors) {
  const torch::Tensor undefined;
  const std::vector<JsonObjectOutputError> errors = {
      {"req-error#0", "missing prior sampled output row"}};
  proto::ForwardOutput proto_output;
  forward_output_to_proto(undefined,
                          undefined,
                          undefined,
                          undefined,
                          undefined,
                          /*mm_embeddings=*/{},
                          undefined,
                          /*prepared_layer_id=*/-1,
                          undefined,
                          undefined,
                          undefined,
                          /*dit_images=*/{},
                          /*dit_text_output=*/{},
                          errors,
                          &proto_output);

  RawForwardOutput round_trip;
  proto_to_forward_output(proto_output, round_trip);

  ASSERT_EQ(round_trip.json_object_errors.size(), 1u);
  EXPECT_EQ(round_trip.json_object_errors[0].sample_sequence_id, "req-error#0");
  EXPECT_EQ(round_trip.json_object_errors[0].message,
            "missing prior sampled output row");
}

TEST(BatchTest, ForwardOutputShmRoundTripPreservesJsonObjectErrors) {
  bool is_creator = false;
  const std::string shm_name = ForwardSharedMemoryManager::create_unique_name(
      "batch_test_forward_output",
      /*dp_group=*/0,
      ForwardType::RAW_OUTPUT,
      /*rank=*/0);
  ForwardSharedMemoryManager writer_manager(
      shm_name, 1 << 20, is_creator, ForwardType::RAW_OUTPUT);
  bool is_reader_creator = false;
  ForwardSharedMemoryManager reader_manager(
      shm_name, 1 << 20, is_reader_creator, ForwardType::RAW_OUTPUT);
  const torch::Tensor undefined;
  const std::vector<JsonObjectOutputError> errors = {
      {"req-error#0", "prior token violates json_object grammar"}};

  ASSERT_TRUE(writer_manager.raw_output_write(undefined,
                                              undefined,
                                              undefined,
                                              undefined,
                                              undefined,
                                              /*mm_embeddings=*/{},
                                              /*dit_images=*/{},
                                              /*dit_text_output=*/{},
                                              undefined,
                                              /*prepared_layer_id=*/-1,
                                              undefined,
                                              undefined,
                                              undefined,
                                              errors));

  RawForwardOutput round_trip;
  reader_manager.raw_output_read(round_trip);

  ASSERT_EQ(round_trip.json_object_errors.size(), 1u);
  EXPECT_EQ(round_trip.json_object_errors[0].sample_sequence_id, "req-error#0");
  EXPECT_EQ(round_trip.json_object_errors[0].message,
            "prior token violates json_object grammar");
}

TEST(BatchTest, ForwardInputBlockCopyKernelFieldsMatchExpectedLayout) {
  const bool old_enable_block_copy_kernel =
      ::xllm::BeamSearchConfig::get_instance().enable_block_copy_kernel();
  ::xllm::BeamSearchConfig::get_instance().enable_block_copy_kernel(true);

  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = false;
  seq_params.enable_schedule_overlap = false;

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder forward_decoder("", 1, false, false);
  Sequence forward_seq(/*index=*/0,
                       /*token_ids=*/{1, 2, 3, 4},
                       input_embedding,
                       mm_data,
                       std::move(forward_decoder),
                       seq_params);
  forward_seq.add_blocks(BlockType::KV, manager.allocate(1));

  std::vector<BlockTransferInfo> forward_swap_blocks = {
      BlockTransferInfo(7, 10),
      BlockTransferInfo(7, 11),
      BlockTransferInfo(8, 12)};
  std::vector<uint32_t> budgets = {4};

  std::vector<Sequence*> forward_sequences = {&forward_seq};
  BatchInputBuilder forward_builder(forward_sequences,
                                    budgets,
                                    {},
                                    {},
                                    &forward_swap_blocks,
                                    /*batch_id=*/1,
                                    nullptr,
                                    BatchForwardType::PREFILL);
  ForwardInput forward_input =
      forward_builder.build_forward_input(/*num_decoding_tokens=*/1,
                                          /*min_decoding_batch_size=*/0);

  EXPECT_TRUE(equal(forward_input.input_params.block_copy.src_block_indices,
                    std::vector<int32_t>({7, 8})));
  EXPECT_TRUE(equal(forward_input.input_params.block_copy.dst_block_indices,
                    std::vector<int32_t>({10, 11, 12})));
  EXPECT_TRUE(equal(forward_input.input_params.block_copy.cum_sum,
                    std::vector<int32_t>({2, 3})));

#if defined(USE_CUDA)
  EXPECT_EQ(forward_input.input_params.block_copy.swap_blocks.size(),
            forward_swap_blocks.size());
#else
  EXPECT_TRUE(forward_input.input_params.block_copy.swap_blocks.empty());
#endif

  ::xllm::BeamSearchConfig::get_instance().enable_block_copy_kernel(
      old_enable_block_copy_kernel);
}

TEST(BatchTest, KVCacheEmptySupportsLinearOnlyAndFullOnlyLayouts) {
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto conv_cache = torch::zeros({2, 4, 3}, options);
  auto ssm_cache = torch::zeros({2, 1, 4, 4}, options);
  KVCache linear_only_cache(
      LinearAttentionKVCacheTensors{conv_cache, ssm_cache});
  EXPECT_FALSE(linear_only_cache.empty());

  auto key_cache = torch::zeros({2, 4, 1, 8}, options);
  auto value_cache = torch::zeros({2, 4, 1, 8}, options);
  KVCache full_only_cache(KVCacheTensors{key_cache, value_cache});
  EXPECT_FALSE(full_only_cache.empty());

  KVCache empty_cache;
  EXPECT_TRUE(empty_cache.empty());
}

TEST(BatchTest, SampleRequestKeepsThreadedForwardBuilderOffsetsStable) {
  torch::Device device(Platform::type_torch(), 0);
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
  seq1.add_blocks(BlockType::KV, manager.allocate(1));

  IncrementalDecoder decoder2("", 2, false, false);
  Sequence seq2(/*index=*/1,
                /*token_ids=*/{1, 31, 32},
                input_embedding,
                mm_data,
                std::move(decoder2),
                seq2_params);
  seq2.add_blocks(BlockType::KV, manager.allocate(1));

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

  ForwardInput forward_input =
      builder.build_forward_input(/*num_decoding_tokens=*/1,
                                  /*min_decoding_batch_size=*/0);

  const std::vector<int32_t> expected_selected_token_idxes = {1, 3, 4, 6};
  const std::vector<int32_t> expected_sample_idxes = {0, 1, 2, 3};
  EXPECT_TRUE(equal(forward_input.sampling_params.selected_token_idxes,
                    expected_selected_token_idxes));
  EXPECT_TRUE(
      equal(forward_input.sampling_params.sample_idxes, expected_sample_idxes));
  ASSERT_EQ(forward_input.input_params.embedding.embedding_ids.size(),
            sequences.size());
  ASSERT_EQ(forward_input.input_params.embedding.linear_state_ids.size(),
            sequences.size());
  EXPECT_EQ(forward_input.input_params.embedding.embedding_ids,
            std::vector<int32_t>({-1, -1}));
  EXPECT_EQ(forward_input.input_params.embedding.linear_state_ids,
            std::vector<int32_t>({-1, -1}));
}

TEST(BatchTest, DecodeMinBatchSizeDoesNotPadTransportState) {
  torch::Device device(Platform::type_torch(), 0);
  const uint32_t n_blocks = 8;
  const uint32_t block_size = 4;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(20);
  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder decoder("", 1, false, false);
  Sequence seq(/*index=*/0,
               /*token_ids=*/{1, 2, 3},
               input_embedding,
               mm_data,
               std::move(decoder),
               seq_params);
  seq.add_blocks(BlockType::KV, manager.allocate(1));
  seq.kv_state().incr_kv_cache_tokens_num(/*size=*/3);
  seq.append_token(4);

  std::vector<Sequence*> sequences = {&seq};
  std::vector<uint32_t> allowed_max_tokens = {1};
  std::vector<torch::Tensor> input_embeddings_vec;
  std::vector<MMData> mm_data_vec;
  ModelArgs args;
  BatchInputBuilder builder(sequences,
                            allowed_max_tokens,
                            input_embeddings_vec,
                            mm_data_vec,
                            /*swap_block_transfer_infos=*/nullptr,
                            /*batch_id=*/1,
                            &args,
                            BatchForwardType::DECODE);

  ForwardInput forward_input =
      builder.build_forward_input(/*num_decoding_tokens=*/1,
                                  /*min_decoding_batch_size=*/3);

  EXPECT_EQ(forward_input.input_params.meta.num_sequences, 1);
  EXPECT_EQ(forward_input.input_params.embedding.linear_state_ids,
            std::vector<int32_t>({-1}));
  EXPECT_EQ(forward_input.input_params.embedding.embedding_ids,
            std::vector<int32_t>({-1}));
}

TEST(BatchTest, DecodeEmbeddingAndLinearStateIdsAreIndependentSlots) {
  const uint32_t n_blocks = 8;
  const uint32_t block_size = 4;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(20);
  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder decoder("", 1, false, false);
  Sequence seq(/*index=*/0,
               /*token_ids=*/{1, 2, 3},
               input_embedding,
               mm_data,
               std::move(decoder),
               seq_params);
  seq.add_blocks(BlockType::KV, manager.allocate(1));
  seq.kv_state().incr_kv_cache_tokens_num(/*size=*/3);
  seq.append_token(4);

  // EMBEDDING and LINEAR are now fully decoupled slots: the embedding-row id
  // (embedding_ids) and the recurrent-state id (linear_state_ids) come from
  // separate blocks and no longer share a value.
  auto embedding_block = manager.allocate(1);
  ASSERT_EQ(embedding_block.size(), 1u);
  const int32_t expected_embedding_id = embedding_block[0].id();
  seq.add_blocks(BlockType::EMBEDDING, embedding_block);

  auto linear_block = manager.allocate(1);
  ASSERT_EQ(linear_block.size(), 1u);
  const int32_t expected_linear_id = linear_block[0].id();
  seq.add_blocks(BlockType::LINEAR, linear_block);
  ASSERT_NE(expected_embedding_id, expected_linear_id);

  std::vector<Sequence*> sequences = {&seq};
  std::vector<uint32_t> allowed_max_tokens = {1};
  std::vector<torch::Tensor> input_embeddings_vec;
  std::vector<MMData> mm_data_vec;
  ModelArgs args;
  args.layer_types({"linear_attention"});
  BatchInputBuilder builder(sequences,
                            allowed_max_tokens,
                            input_embeddings_vec,
                            mm_data_vec,
                            /*swap_block_transfer_infos=*/nullptr,
                            /*batch_id=*/1,
                            &args,
                            BatchForwardType::DECODE);

  ForwardInput forward_input =
      builder.build_forward_input(/*num_decoding_tokens=*/1,
                                  /*min_decoding_batch_size=*/0);

  ASSERT_EQ(forward_input.input_params.embedding.embedding_ids.size(), 1u);
  ASSERT_EQ(forward_input.input_params.embedding.linear_state_ids.size(), 1u);
  EXPECT_EQ(forward_input.input_params.embedding.embedding_ids[0],
            expected_embedding_id);
  EXPECT_EQ(forward_input.input_params.embedding.linear_state_ids[0],
            expected_linear_id);
}

TEST(BatchTest, LinearRestoreSourceStaysPinnedForBatchLifetime) {
  // Linear-state checkpoints are hashed per chunk-end boundary. This test's
  // save/restore boundaries (16, 20) are multiples of the KV block_size (4),
  // so set the chunk stride to block_size to keep them chunk-aligned and the
  // per-chunk hash chain identical to the per-block helper below.
  ScopedPrefillChunkStride chunk_stride(/*chunk_stride=*/4);

  torch::Device device(Platform::type_torch(), 0);
  const uint32_t n_blocks = 22;
  const uint32_t block_size = 4;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = false;
  seq_params.enable_schedule_overlap = false;

  torch::Tensor input_embedding;
  MMData mm_data;
  const std::vector<int32_t> aligned_tokens = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  IncrementalDecoder aligned_decoder("", 20, false, false);
  Sequence aligned_seq(/*index=*/0,
                       aligned_tokens,
                       input_embedding,
                       mm_data,
                       std::move(aligned_decoder),
                       seq_params);
  std::vector<Block> aligned_blocks = manager.allocate(5);
  aligned_seq.add_blocks(BlockType::KV, aligned_blocks);

  IncrementalDecoder restore_decoder("", 20, false, false);
  Sequence restore_seq(/*index=*/1,
                       aligned_tokens,
                       input_embedding,
                       mm_data,
                       std::move(restore_decoder),
                       seq_params);
  std::vector<Block> restore_blocks = manager.allocate(5);
  restore_seq.add_blocks(BlockType::KV, restore_blocks);
  restore_seq.kv_state().incr_kv_cache_tokens_num(/*size=*/16);
  // A sequence that restores at a chunk boundary carries a mounted restore
  // source: production mounts it in allocate_shared_for_sequence (class A) when
  // the reused KV prefix maps onto a committed linear-state checkpoint. This
  // hand-built sequence bypasses the block manager, so mount an explicit source
  // slot to reproduce that invariant -- the builder keys the restore emission
  // off its presence.
  std::vector<Block> restore_src_blocks = manager.allocate(1);
  restore_seq.set_linear_restore_src_block(std::move(restore_src_blocks[0]));

  const std::vector<int32_t> off_boundary_tokens = {21, 22, 23, 24, 25, 26, 27,
                                                    28, 29, 30, 31, 32, 33, 34,
                                                    35, 36, 37, 38, 39, 40};
  IncrementalDecoder off_boundary_decoder("", 20, false, false);
  Sequence off_boundary_seq(/*index=*/2,
                            off_boundary_tokens,
                            input_embedding,
                            mm_data,
                            std::move(off_boundary_decoder),
                            seq_params);
  off_boundary_seq.add_blocks(BlockType::KV, manager.allocate(5));

  const std::vector<int32_t> decode_tokens = {
      41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56};
  IncrementalDecoder decode_decoder("", 16, false, false);
  Sequence decode_seq(/*index=*/3,
                      decode_tokens,
                      input_embedding,
                      mm_data,
                      std::move(decode_decoder),
                      seq_params);
  decode_seq.add_blocks(BlockType::KV, manager.allocate(5));
  decode_seq.kv_state().incr_kv_cache_tokens_num(/*size=*/16);
  decode_seq.append_token(57);

  std::optional<Batch> batch;
  batch.emplace();
  batch->add(&aligned_seq, /*allowed_max_token=*/16);
  batch->add(&restore_seq, /*allowed_max_token=*/4);
  batch->add(&off_boundary_seq, /*allowed_max_token=*/15);
  batch->add(&decode_seq, /*allowed_max_token=*/1);

  ModelArgs args;
  args.layer_types({"linear_attention"});
  ForwardInput forward_input = batch->prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_bach_size=*/0, args);

  const auto& cache_ops = forward_input.input_params.linear_state_cache_ops;
  ASSERT_EQ(cache_ops.size(), 4u);
  // The save decision now lands on the sequence (pending save), not the cache
  // op; the LINEAR leaf executes it at the next step. The restore no longer
  // rides a hash on the cache op: the builder resolves the checkpoint to a
  // source slot and sets restore_requested, mirroring KV's resolved swap
  // descriptor (the worker's copy-in keys off that bit + src slot).
  const LinearStatePrefixHash aligned_save_expected =
      compute_linear_state_prefix_hash_for_test(
          aligned_tokens, aligned_blocks, /*boundary_tokens=*/16);
  EXPECT_FALSE(is_zero_prefix_hash(aligned_save_expected));
  std::optional<XXH3Key> aligned_save = aligned_seq.take_pending_linear_save();
  ASSERT_TRUE(aligned_save.has_value());
  EXPECT_EQ(*aligned_save, XXH3Key(aligned_save_expected.data()));

  // restore_seq restores at boundary 16 (the same hash aligned_seq saved) and
  // saves at boundary 20.
  EXPECT_TRUE(cache_ops[1].restore_requested);
  EXPECT_GE(cache_ops[1].restore_src_slot_id, 0);
  const LinearStatePrefixHash restore_save_expected =
      compute_linear_state_prefix_hash_for_test(
          aligned_tokens, restore_blocks, /*boundary_tokens=*/20);
  std::optional<XXH3Key> restore_save = restore_seq.take_pending_linear_save();
  ASSERT_TRUE(restore_save.has_value());
  EXPECT_EQ(*restore_save, XXH3Key(restore_save_expected.data()));

  // off_boundary_seq and decode_seq neither restore nor save.
  EXPECT_FALSE(cache_ops[2].restore_requested);
  EXPECT_FALSE(off_boundary_seq.has_pending_linear_save());
  EXPECT_FALSE(cache_ops[3].restore_requested);
  EXPECT_FALSE(decode_seq.has_pending_linear_save());

  // The host Block handle, not device state, keeps the full pool pinned.
  EXPECT_TRUE(manager.allocate(1).empty());
  batch.reset();
  EXPECT_EQ(manager.allocate(1).size(), 1u);
}

TEST(BatchTest, UnusedLinearRestoreSourceIsReleasedDuringBuild) {
  ScopedPrefillChunkStride chunk_stride(/*chunk_stride=*/4);

  BlockManager::Options options;
  options.num_blocks(/*num_blocks=*/3).block_size(/*block_size=*/4);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(1);
  SequenceParams seq_params;
  seq_params.seq_capacity = 4;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;

  IncrementalDecoder decoder("", 3, false, false);
  Sequence sequence(/*index=*/0,
                    /*token_ids=*/{1, 2, 3},
                    /*input_embedding=*/torch::Tensor(),
                    /*mm_data=*/MMData(),
                    std::move(decoder),
                    seq_params);
  sequence.add_blocks(BlockType::KV, manager.allocate(1));
  sequence.kv_state().incr_kv_cache_tokens_num(/*size=*/2);
  std::vector<Block> restore_sources = manager.allocate(1);
  ASSERT_EQ(restore_sources.size(), 1u);
  sequence.set_linear_restore_src_block(std::move(restore_sources[0]));

  std::optional<Batch> batch;
  batch.emplace();
  batch->add(&sequence, /*allowed_max_token=*/1);
  ModelArgs args;
  args.layer_types({"linear_attention"});
  ForwardInput forward_input = batch->prepare_forward_input(
      /*num_decoding_tokens=*/0, /*min_decoding_bach_size=*/0, args);

  ASSERT_EQ(forward_input.input_params.linear_state_cache_ops.size(), 1u);
  EXPECT_FALSE(
      forward_input.input_params.linear_state_cache_ops[0].restore_requested);
  EXPECT_EQ(manager.allocate(1).size(), 1u);
}

TEST(BatchTest, ThreadedBatchPinsEveryLinearRestoreSourceUntilRelease) {
  ScopedPrefillChunkStride chunk_stride(/*chunk_stride=*/4);

  constexpr size_t kQueryTokensPerSequence = 32768;
  constexpr size_t kCachedTokensPerSequence = 4;
  const size_t sequence_tokens =
      kQueryTokensPerSequence + kCachedTokensPerSequence;
  BlockManager::Options options;
  options.num_blocks(/*num_blocks=*/5)
      .block_size(static_cast<uint32_t>(sequence_tokens));
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(1);
  SequenceParams seq_params;
  seq_params.seq_capacity = sequence_tokens + 1;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;

  std::vector<int32_t> first_tokens(sequence_tokens, 1);
  std::vector<int32_t> second_tokens(sequence_tokens, 2);
  IncrementalDecoder first_decoder("", sequence_tokens, false, false);
  Sequence first_sequence(/*index=*/0,
                          first_tokens,
                          /*input_embedding=*/torch::Tensor(),
                          /*mm_data=*/MMData(),
                          std::move(first_decoder),
                          seq_params);
  IncrementalDecoder second_decoder("", sequence_tokens, false, false);
  Sequence second_sequence(/*index=*/1,
                           second_tokens,
                           /*input_embedding=*/torch::Tensor(),
                           /*mm_data=*/MMData(),
                           std::move(second_decoder),
                           seq_params);
  first_sequence.add_blocks(BlockType::KV, manager.allocate(1));
  second_sequence.add_blocks(BlockType::KV, manager.allocate(1));
  first_sequence.kv_state().incr_kv_cache_tokens_num(
      /*size=*/kCachedTokensPerSequence);
  second_sequence.kv_state().incr_kv_cache_tokens_num(
      /*size=*/kCachedTokensPerSequence);

  std::vector<Block> first_restore_source = manager.allocate(1);
  std::vector<Block> second_restore_source = manager.allocate(1);
  ASSERT_EQ(first_restore_source.size(), 1u);
  ASSERT_EQ(second_restore_source.size(), 1u);
  first_sequence.set_linear_restore_src_block(
      std::move(first_restore_source[0]));
  second_sequence.set_linear_restore_src_block(
      std::move(second_restore_source[0]));

  std::optional<Batch> batch;
  batch.emplace();
  batch->add(&first_sequence);
  batch->add(&second_sequence);
  ThreadPool thread_pool(/*num_threads=*/2);
  ModelArgs args;
  args.layer_types({"linear_attention"});
  ForwardInput forward_input = batch->prepare_forward_input(args, &thread_pool);

  const auto& cache_ops = forward_input.input_params.linear_state_cache_ops;
  ASSERT_EQ(cache_ops.size(), 2u);
  EXPECT_TRUE(cache_ops[0].restore_requested);
  EXPECT_TRUE(cache_ops[1].restore_requested);
  EXPECT_TRUE(manager.allocate(1).empty());
  batch.reset();
  EXPECT_EQ(manager.allocate(2).size(), 2u);
}

TEST(BatchTest, SharedMemoryRoundTripPreservesLinearStateIds) {
  ForwardInput forward_input;
  auto int_options = torch::TensorOptions()
                         .dtype(torch::kInt)
                         .device(torch::kCPU)
                         .pinned_memory(true);
  forward_input.token_ids =
      torch::tensor(std::vector<int32_t>({1, 2}), int_options);
  forward_input.token_ids_host = forward_input.token_ids;
  forward_input.positions =
      torch::tensor(std::vector<int32_t>({0, 0}), int_options);
  forward_input.positions_host = forward_input.positions;
  forward_input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  forward_input.input_params.meta.num_sequences = 2;
  forward_input.input_params.meta.kv_max_seq_len = 1;
  forward_input.input_params.meta.q_max_seq_len = 1;
  forward_input.input_params.attention.host.kv_seq_lens = {1, 1};
  forward_input.input_params.attention.host.q_seq_lens = {1, 1};
  forward_input.input_params.attention.host.q_cu_seq_lens = {1, 2};
  forward_input.input_params.attention.host.kv_cache_tokens_nums = {0, 0};
  forward_input.input_params.attention.host.new_cache_slots = {0, 0};
  forward_input.input_params.attention.device.kv_seq_lens =
      torch::tensor(std::vector<int32_t>({1, 1}), int_options);
  forward_input.input_params.attention.device.q_seq_lens =
      torch::tensor(std::vector<int32_t>({1, 1}), int_options);
  forward_input.input_params.attention.device.q_cu_seq_lens =
      torch::tensor(std::vector<int32_t>({1, 2}), int_options);
  forward_input.input_params.attention.device.kv_cache_tokens_nums =
      torch::tensor(std::vector<int32_t>({0, 0}), int_options);
  forward_input.input_params.attention.device.new_cache_slots =
      torch::tensor(std::vector<int32_t>({0, 0}), int_options);
  forward_input.input_params.attention.device.block_tables = create_2d_tensor(
      std::vector<std::vector<int32_t>>{{0}, {0}}, torch::kInt);
  forward_input.input_params.attention.host.block_tables =
      forward_input.input_params.attention.device.block_tables;
  forward_input.input_params.embedding.linear_state_ids = {4, 6};

  TransferKVInfo transfer_info;
  transfer_info.request_id = "dsv4-round-trip";
  KVTransferMapping transfer_mapping;
  transfer_mapping.group_id = cache_group_id(BlockType::C128);
  transfer_mapping.local_ids = {11, 12};
  transfer_mapping.remote_ids = {101, 102};
  transfer_info.mappings.emplace_back(std::move(transfer_mapping));
  forward_input.transfer_kv_infos.emplace_back(std::move(transfer_info));

  bool is_creator = false;
  auto shm_name =
      ForwardSharedMemoryManager::create_unique_name("batch_test_linear_state",
                                                     /*dp_group=*/0,
                                                     ForwardType::RAW_INPUT,
                                                     /*rank=*/0);
  ForwardSharedMemoryManager writer_manager(
      shm_name, 1 << 20, is_creator, ForwardType::RAW_INPUT);
  bool is_reader_creator = false;
  ForwardSharedMemoryManager reader_manager(
      shm_name, 1 << 20, is_reader_creator, ForwardType::RAW_INPUT);
  ASSERT_TRUE(writer_manager.input_write(forward_input));

  ForwardInput from_shm;
  reader_manager.input_read(from_shm, torch::Device(torch::kCPU));
  EXPECT_EQ(from_shm.input_params.embedding.linear_state_ids,
            std::vector<int32_t>({4, 6}));
  ASSERT_EQ(from_shm.transfer_kv_infos.size(), 1u);
  EXPECT_EQ(from_shm.transfer_kv_infos[0].request_id, "dsv4-round-trip");
  ASSERT_EQ(from_shm.transfer_kv_infos[0].mappings.size(), 1u);
  const KVTransferMapping& from_shm_mapping =
      from_shm.transfer_kv_infos[0].mappings[0];
  EXPECT_EQ(from_shm_mapping.group_id, cache_group_id(BlockType::C128));
  EXPECT_EQ(from_shm_mapping.local_ids, (std::vector<uint64_t>{11, 12}));
  EXPECT_EQ(from_shm_mapping.remote_ids, (std::vector<uint64_t>{101, 102}));

  forward_input.input_params.embedding.linear_state_ids.clear();
  ASSERT_TRUE(writer_manager.input_write(forward_input));

  ForwardInput legacy_from_shm;
  reader_manager.input_read(legacy_from_shm, torch::Device(torch::kCPU));
  EXPECT_EQ(legacy_from_shm.input_params.embedding.linear_state_ids,
            std::vector<int32_t>({-1, -1}));
}

TEST(BatchTest, SharedMemoryRoundTripPreservesEmptyRankTensors) {
  ForwardInput forward_input;
  auto int_options = torch::TensorOptions()
                         .dtype(torch::kInt)
                         .device(torch::kCPU)
                         .pinned_memory(true);
  forward_input.token_ids = torch::empty({0}, int_options);
  forward_input.token_ids_host = forward_input.token_ids;
  forward_input.positions = torch::empty({0}, int_options);
  forward_input.positions_host = forward_input.positions;
  forward_input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  forward_input.input_params.meta.num_sequences = 0;
  forward_input.input_params.attention.device.q_seq_lens =
      torch::empty({0}, int_options);
  forward_input.input_params.attention.device.kv_seq_lens =
      torch::empty({0}, int_options);

  bool is_creator = false;
  auto shm_name =
      ForwardSharedMemoryManager::create_unique_name("batch_test_empty_rank",
                                                     /*dp_group=*/0,
                                                     ForwardType::RAW_INPUT,
                                                     /*rank=*/0);
  ForwardSharedMemoryManager writer_manager(
      shm_name, 1 << 20, is_creator, ForwardType::RAW_INPUT);
  bool is_reader_creator = false;
  ForwardSharedMemoryManager reader_manager(
      shm_name, 1 << 20, is_reader_creator, ForwardType::RAW_INPUT);
  ASSERT_TRUE(writer_manager.input_write(forward_input));

  ForwardInput from_shm;
  reader_manager.input_read(from_shm, torch::Device(torch::kCPU));

  EXPECT_TRUE(from_shm.token_ids.defined());
  EXPECT_EQ(from_shm.token_ids.numel(), 0);
  EXPECT_EQ(from_shm.token_ids.dim(), 1);
  EXPECT_TRUE(from_shm.positions.defined());
  EXPECT_EQ(from_shm.positions.numel(), 0);
  EXPECT_EQ(from_shm.positions.dim(), 1);
  EXPECT_TRUE(from_shm.input_params.attention.device.q_seq_lens.defined());
  EXPECT_EQ(from_shm.input_params.attention.device.q_seq_lens.numel(), 0);
  EXPECT_TRUE(from_shm.input_params.attention.device.kv_seq_lens.defined());
  EXPECT_EQ(from_shm.input_params.attention.device.kv_seq_lens.numel(), 0);
}

TEST(BatchTest, SampleRequestProcessesAllMatchedRawOutputs) {
  torch::Device device(Platform::type_torch(), 0);
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
  seq.add_blocks(BlockType::KV, manager.allocate(2));

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
  torch::Device device(Platform::type_torch(), 0);
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
  seq1.add_blocks(BlockType::KV, manager.allocate(1));

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
  seq2.add_blocks(BlockType::KV, manager.allocate(1));

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
  torch::Device device(Platform::type_torch(), 0);
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
  seq.add_blocks(BlockType::KV, manager.allocate(2));

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

TEST(BatchTest, KeepTargetsForOverlapReplacement) {
  const bool old_enable_schedule_overlap =
      SchedulerConfig::get_instance().enable_schedule_overlap();
  SchedulerConfig::get_instance().enable_schedule_overlap(true);

  torch::Device device(Platform::type_torch(), 0);
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
  seq.add_blocks(BlockType::KV, manager.allocate(1));
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

  SchedulerConfig::get_instance().enable_schedule_overlap(
      old_enable_schedule_overlap);
}

TEST(BatchTest, JsonObjectErrorFailsOnlyOwningRequestWithoutTokenCommit) {
  const bool old_enable_schedule_overlap =
      SchedulerConfig::get_instance().enable_schedule_overlap();
  SchedulerConfig::get_instance().enable_schedule_overlap(true);

  BlockManager::Options options;
  options.num_blocks(16).block_size(4);
  BlockManagerImpl manager(options);
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);

  Sequence failed = make_overlap_sequence({1, 10, 11},
                                          /*seq_capacity=*/32,
                                          &sampling_param,
                                          &stopping_checker,
                                          "req-error",
                                          /*sequence_index=*/0);
  Sequence failed_peer = make_overlap_sequence({1, 20, 21},
                                               /*seq_capacity=*/32,
                                               &sampling_param,
                                               &stopping_checker,
                                               "req-error",
                                               /*sequence_index=*/1);
  Sequence healthy = make_overlap_sequence({1, 30, 31},
                                           /*seq_capacity=*/32,
                                           &sampling_param,
                                           &stopping_checker,
                                           "req-healthy",
                                           /*sequence_index=*/0);
  for (Sequence* sequence :
       std::vector<Sequence*>{&failed, &failed_peer, &healthy}) {
    sequence->add_blocks(BlockType::KV, manager.allocate(1));
    sequence->kv_state().incr_kv_cache_tokens_num(
        sequence->num_prompt_tokens() - 1);
  }

  Batch batch({&failed, &failed_peer, &healthy});
  (void)batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_bach_size=*/0, ModelArgs());

  RawForwardOutput fake_output;
  fake_output.outputs = {make_raw_sample_output(-1, std::nullopt),
                         make_raw_sample_output(-2, std::nullopt),
                         make_raw_sample_output(-3, std::nullopt)};
  batch.process_sample_output(fake_output, /*replace_fake_token=*/false);

  RawForwardOutput real_output;
  real_output.outputs = {make_raw_sample_output(101, std::nullopt),
                         make_raw_sample_output(102, std::nullopt),
                         make_raw_sample_output(103, std::nullopt)};
  real_output.json_object_errors = {
      {failed.sample_sequence_id(), "missing prior sampled output row"}};
  batch.process_sample_output(real_output, /*replace_fake_token=*/true);

  EXPECT_EQ(failed.num_generated_tokens(), 1u);
  EXPECT_EQ(failed.tokens()[failed.num_prompt_tokens()], -1);
  ASSERT_TRUE(failed.error_status().has_value());
  EXPECT_FALSE(failed.error_status()->ok());
  EXPECT_TRUE(failed.finished());

  EXPECT_EQ(failed_peer.num_generated_tokens(), 1u);
  EXPECT_EQ(failed_peer.tokens()[failed_peer.num_prompt_tokens()], -2);
  ASSERT_TRUE(failed_peer.error_status().has_value());
  EXPECT_FALSE(failed_peer.error_status()->ok());
  EXPECT_TRUE(failed_peer.finished());

  EXPECT_EQ(healthy.num_generated_tokens(), 1u);
  EXPECT_EQ(healthy.tokens()[healthy.num_prompt_tokens()], 103);
  EXPECT_FALSE(healthy.error_status().has_value());
  EXPECT_FALSE(healthy.finished());

  SchedulerConfig::get_instance().enable_schedule_overlap(
      old_enable_schedule_overlap);
}

TEST(BatchTest, JsonObjectErrorForUnknownSequenceDies) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);
  Sequence sequence = make_basic_sequence({1, 2, 3});
  sequence.add_blocks(BlockType::KV, manager.allocate(1));
  Batch batch(&sequence);
  (void)batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_bach_size=*/0, ModelArgs());

  RawForwardOutput raw_output;
  raw_output.json_object_errors = {
      {"unknown-request#0", "missing prior sampled output row"}};
  EXPECT_DEATH(
      batch.process_sample_output(raw_output, /*replace_fake_token=*/false),
      "unknown sampled sequence id");
}

TEST(BatchTest, JsonObjectErrorFailsUnscheduledSiblingSequence) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  RequestState request_state(
      "prompt",
      std::vector<int32_t>{1, 2, 3},
      sampling_param,
      SchedulerParam{},
      stopping_checker,
      /*seq_capacity=*/32,
      /*n=*/2,
      /*best_of=*/2,
      /*logprobs=*/false,
      /*stream=*/false,
      /*echo=*/false,
      /*skip_special_tokens=*/true,
      /*enable_schedule_overlap=*/true,
      [](const RequestOutput&) { return true; },
      OutputsFunc{});
  Request request("req-shared", "", "", request_state);
  ASSERT_TRUE(request.expand_sequences(/*share_prefix=*/false));
  ASSERT_EQ(request.sequences().size(), 2u);
  for (const auto& sequence : request.sequences()) {
    sequence->add_blocks(BlockType::KV, manager.allocate(1));
    sequence->kv_state().incr_kv_cache_tokens_num(
        sequence->num_prompt_tokens() - 1);
  }

  Sequence* scheduled_sequence = request.sequences()[0].get();
  Batch batch(scheduled_sequence);
  (void)batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_bach_size=*/0, ModelArgs());
  RawForwardOutput raw_output;
  raw_output.json_object_errors = {{scheduled_sequence->sample_sequence_id(),
                                    "missing prior sampled output row"}};

  batch.process_sample_output(raw_output, /*replace_fake_token=*/false);

  EXPECT_TRUE(request.finished());
  ASSERT_TRUE(request.error_status().has_value());
  for (const auto& sequence : request.sequences()) {
    EXPECT_TRUE(sequence->finished());
    ASSERT_TRUE(sequence->error_status().has_value());
    EXPECT_EQ(sequence->error_status()->message(),
              request.error_status()->message());
    EXPECT_EQ(sequence->num_generated_tokens(), 0u);
  }
}

TEST(BatchTest, OverlapMTPReplacementSkipsPreemptedSequenceWithoutKVBlocks) {
  const bool old_enable_schedule_overlap =
      SchedulerConfig::get_instance().enable_schedule_overlap();
  SchedulerConfig::get_instance().enable_schedule_overlap(true);

  torch::Device device(Platform::type_torch(), 0);
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
  seq.add_blocks(BlockType::KV, manager.allocate(1));
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
  EXPECT_EQ(seq.kv_state().num_blocks(BlockType::KV), 0);

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

  SchedulerConfig::get_instance().enable_schedule_overlap(
      old_enable_schedule_overlap);
}

TEST(BatchTest, OverlapMTPReplacementKeepsCompositeKvBlocks) {
  const bool old_enable_schedule_overlap =
      SchedulerConfig::get_instance().enable_schedule_overlap();
  SchedulerConfig::get_instance().enable_schedule_overlap(true);

  const uint32_t base_block_size = 128;
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 128;
  const uint32_t max_seqs_per_batch = 4;

  BlockManager::Options options;
  options.num_blocks(base_num_blocks)
      .block_size(base_block_size)
      .sliding_window_size(window_size)
      .swa_blocks_per_seq(static_cast<uint32_t>(
          get_swa_blocks_per_seq(window_size, base_block_size)))
      .max_tokens_per_batch(1280)
      .max_seqs_per_batch(max_seqs_per_batch)
      .manager_types({1, 0, 0})
      .compress_ratios({0, 4, 128});
  CompositeBlockManager manager(build_composite_leaves(options));

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(8);

  Sequence seq = make_overlap_sequence(
      {1, 10, 11}, /*seq_capacity=*/128, &sampling_param, &stopping_checker);
  ASSERT_TRUE(manager.allocate_sequence(&seq, seq.num_prompt_tokens()));
  ASSERT_EQ(seq.kv_state().num_blocks(BlockType::KV), 0u);
  ASSERT_GT(seq.kv_state().current_max_tokens_capacity(), 0u);
  seq.kv_state().incr_kv_cache_tokens_num(seq.num_prompt_tokens() - 1);

  Batch batch({&seq});
  batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_bach_size=*/0, ModelArgs());

  RawForwardOutput fake_output;
  RawSampleOutput fake_sample_output;
  RawToken fake_token;
  fake_token.id = -1;
  fake_sample_output.tokens.push_back(fake_token);
  fake_output.outputs.push_back(std::move(fake_sample_output));
  batch.process_sample_output(fake_output, /*replace_fake_token=*/false);

  RawForwardOutput real_output;
  RawSampleOutput real_sample_output;
  RawToken real_token_0;
  real_token_0.id = 101;
  real_sample_output.tokens.push_back(real_token_0);
  RawToken real_token_1;
  real_token_1.id = 102;
  real_sample_output.tokens.push_back(real_token_1);
  real_output.outputs.push_back(std::move(real_sample_output));
  batch.process_sample_output(real_output, /*replace_fake_token=*/true);

  ASSERT_EQ(seq.num_generated_tokens(), 2);
  EXPECT_EQ(seq.tokens()[seq.num_prompt_tokens()], 101);
  EXPECT_EQ(seq.tokens()[seq.num_prompt_tokens() + 1], 102);
  EXPECT_EQ(seq.kv_state().kv_cache_tokens_num(), seq.num_prompt_tokens() + 1);

  SchedulerConfig::get_instance().enable_schedule_overlap(
      old_enable_schedule_overlap);
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
