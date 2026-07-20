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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <iostream>
#include <optional>
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
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/prefix_cache/block_hasher.h"
#include "framework/request/stopping_checker.h"
#include "framework/sampling/sampling_params.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "runtime/cp_input_partition.h"
#include "runtime/forward_shared_memory_manager.h"
#include "util/tensor_helper.h"

namespace xllm {

class BatchInputBuilderTestPeer final {
 public:
  static TransferKVInfo build_step_transfer_info(
      const TransferKVInfo& full_info,
      const std::vector<uint64_t>& local_block_ids,
      size_t next_transfer_block_idx,
      uint32_t seq_len,
      uint32_t block_size,
      size_t* advanced_transfer_block_idx) {
    return BatchInputBuilder::build_step_transfer_info(
        full_info,
        local_block_ids,
        next_transfer_block_idx,
        seq_len,
        block_size,
        advanced_transfer_block_idx);
  }

  static KVBlockTransferGroup build_group_step_transfer(
      const KVBlockTransferGroup& full_group,
      const std::vector<int32_t>& local_block_ids,
      size_t next_transfer_block_idx,
      uint32_t seq_len,
      uint32_t block_size,
      size_t* advanced_transfer_block_idx) {
    return BatchInputBuilder::build_group_step_transfer(
        full_group,
        local_block_ids,
        next_transfer_block_idx,
        seq_len,
        block_size,
        advanced_transfer_block_idx);
  }

  // Run the two build phases separately so a benchmark can attribute time to
  // the parallelized sequence processing vs. the serial tensor assembly.
  static ForwardInput run_phases_timed(BatchInputBuilder& builder,
                                       double* process_ms,
                                       double* finalize_ms) {
    const auto t0 = std::chrono::steady_clock::now();
    builder.process_sequences();
    const auto t1 = std::chrono::steady_clock::now();
    builder.padding_decode_batch_size(/*num_decoding_tokens=*/1,
                                      /*min_decoding_batch_size=*/0);
    ForwardInput out = builder.state_to_forward_input();
    const auto t2 = std::chrono::steady_clock::now();
    *process_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    *finalize_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    return out;
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

TransferKVInfo make_info(const std::vector<uint64_t>& remote_block_ids) {
  TransferKVInfo info;
  info.request_id = "req_0";
  info.remote_blocks_ids = remote_block_ids;
  info.dp_rank = 3;
  return info;
}

XTensorLayerOffsets make_offsets(const std::vector<uint64_t>& k_offsets,
                                 const std::vector<uint64_t>& v_offsets) {
  XTensorLayerOffsets offsets;
  offsets.k_offsets = k_offsets;
  offsets.v_offsets = v_offsets;
  return offsets;
}

void expect_blocks(const TransferKVInfo& info,
                   const std::vector<uint64_t>& local_block_ids,
                   const std::vector<uint64_t>& remote_block_ids) {
  EXPECT_EQ(info.local_blocks_ids, local_block_ids);
  EXPECT_EQ(info.remote_blocks_ids, remote_block_ids);
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

KVBlockTransferGroup make_dsv4_group(
    const std::vector<uint64_t>& remote_block_ids) {
  KVBlockTransferGroup group;
  group.group_id = cache_group_id(BlockType::C4);
  group.remote_blocks_ids = remote_block_ids;
  return group;
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
                               StoppingChecker* stopping_checker) {
  SequenceParams seq_params;
  seq_params.seq_capacity = seq_capacity;
  seq_params.stopping_checker = stopping_checker;
  seq_params.sampling_param = sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = false;
  seq_params.enable_schedule_overlap = true;

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

// Build `batch_size` single-token decode sequences backed by `manager`, stored
// in `storage` (a std::deque keeps element addresses stable so the returned
// pointers stay valid). Each sequence caches 3 prompt tokens and appends one
// decode token, so every batch row contributes exactly one query token.
std::vector<Sequence*> build_decode_smoke_sequences(
    int32_t batch_size,
    BlockManagerImpl& manager,
    std::deque<Sequence>& storage) {
  std::vector<Sequence*> sequences;
  sequences.reserve(batch_size);
  for (int32_t i = 0; i < batch_size; ++i) {
    storage.push_back(make_basic_sequence({1, 2, 3}));
    Sequence& seq = storage.back();
    seq.add_blocks(BlockType::KV, manager.allocate(1));
    seq.kv_state().incr_kv_cache_tokens_num(/*size=*/3);
    seq.append_token(4);
    sequences.push_back(&seq);
  }
  return sequences;
}

// Build `batch_size` prefill sequences of `prompt_len` tokens each. This makes
// per-sequence work in process_sequences() non-trivial (each row flattens
// prompt_len tokens/positions and fills prompt_len cache slots), which is the
// regime where multithreaded processing is expected to pay off -- in contrast
// to the 1-token decode builder above.
std::vector<Sequence*> build_prefill_smoke_sequences(
    int32_t batch_size,
    int32_t prompt_len,
    BlockManagerImpl& manager,
    std::deque<Sequence>& storage) {
  const int32_t block_size = 4;
  const int32_t blocks_per_seq = (prompt_len + block_size - 1) / block_size;
  std::vector<int32_t> prompt(prompt_len);
  for (int32_t i = 0; i < prompt_len; ++i) {
    prompt[i] = (i % 97) + 1;
  }
  std::vector<Sequence*> sequences;
  sequences.reserve(batch_size);
  for (int32_t i = 0; i < batch_size; ++i) {
    storage.push_back(make_basic_sequence(prompt));
    Sequence& seq = storage.back();
    seq.add_blocks(BlockType::KV, manager.allocate(blocks_per_seq));
    sequences.push_back(&seq);
  }
  return sequences;
}

// Run `iters` builds of a fresh decode batch of `batch_size` sequences and
// return the average wall-clock cost of build_forward_input in milliseconds.
// Sequences are rebuilt every iteration because build_forward_input mutates
// per-sequence KV-cache state. One untimed warmup build runs first to absorb
// one-time lazy initialization (torch allocators/kernels) that would otherwise
// be charged entirely to whichever batch runs first. When `thread_pool` is
// non-null and the batch is at least as large as the pool, the builder takes
// its multithreaded path.
double average_build_ms(int32_t batch_size,
                        int32_t iters,
                        ThreadPool* thread_pool) {
  const uint32_t block_size = 4;
  double total_ms = 0.0;
  for (int32_t iter = -1; iter < iters; ++iter) {
    const bool is_warmup = iter < 0;
    BlockManager::Options options;
    options.num_blocks(batch_size + 16).block_size(block_size);
    BlockManagerImpl manager(options);

    std::deque<Sequence> storage;
    std::vector<Sequence*> sequences =
        build_decode_smoke_sequences(batch_size, manager, storage);

    std::vector<uint32_t> allowed_max_tokens(batch_size, 1u);
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
                              BatchForwardType::DECODE,
                              /*cp_size=*/1,
                              thread_pool);

    const auto start = std::chrono::steady_clock::now();
    ForwardInput forward_input =
        builder.build_forward_input(/*num_decoding_tokens=*/1,
                                    /*min_decoding_batch_size=*/0);
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - start)
                                  .count();
    if (!is_warmup) {
      total_ms += elapsed_ms;
    }

    EXPECT_EQ(forward_input.input_params.meta.num_sequences, batch_size);
    EXPECT_EQ(forward_input.token_ids.numel(), batch_size);
  }
  return total_ms / iters;
}

struct PhaseTiming {
  double total_ms = 0.0;
  double process_ms = 0.0;
  double finalize_ms = 0.0;
};

// Like average_build_ms, but splits each build into the parallelized
// process_sequences() phase and the serial state_to_forward_input() phase so
// the benchmark can show which phase dominates. One untimed warmup runs first.
PhaseTiming average_phase_ms(int32_t batch_size,
                             int32_t iters,
                             ThreadPool* thread_pool) {
  const uint32_t block_size = 4;
  PhaseTiming acc;
  for (int32_t iter = -1; iter < iters; ++iter) {
    const bool is_warmup = iter < 0;
    BlockManager::Options options;
    options.num_blocks(batch_size + 16).block_size(block_size);
    BlockManagerImpl manager(options);

    std::deque<Sequence> storage;
    std::vector<Sequence*> sequences =
        build_decode_smoke_sequences(batch_size, manager, storage);

    std::vector<uint32_t> allowed_max_tokens(batch_size, 1u);
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
                              BatchForwardType::DECODE,
                              /*cp_size=*/1,
                              thread_pool);

    double process_ms = 0.0;
    double finalize_ms = 0.0;
    ForwardInput forward_input = BatchInputBuilderTestPeer::run_phases_timed(
        builder, &process_ms, &finalize_ms);
    if (!is_warmup) {
      acc.process_ms += process_ms;
      acc.finalize_ms += finalize_ms;
      acc.total_ms += process_ms + finalize_ms;
    }

    EXPECT_EQ(forward_input.input_params.meta.num_sequences, batch_size);
    EXPECT_EQ(forward_input.token_ids.numel(), batch_size);
  }
  acc.total_ms /= iters;
  acc.process_ms /= iters;
  acc.finalize_ms /= iters;
  return acc;
}

// Average build_forward_input cost for a PREFILL batch of `batch_size`
// sequences, each with `prompt_len` prompt tokens. Unlike the decode helper,
// per-sequence work here scales with prompt_len, so the parallelized
// process_sequences() phase has real work to distribute. One untimed warmup.
double average_prefill_build_ms(int32_t batch_size,
                                int32_t prompt_len,
                                int32_t iters,
                                ThreadPool* thread_pool) {
  const uint32_t block_size = 4;
  const int32_t blocks_per_seq = (prompt_len + block_size - 1) / block_size;
  double total_ms = 0.0;
  for (int32_t iter = -1; iter < iters; ++iter) {
    const bool is_warmup = iter < 0;
    BlockManager::Options options;
    options.num_blocks(batch_size * blocks_per_seq + 16).block_size(block_size);
    BlockManagerImpl manager(options);

    std::deque<Sequence> storage;
    std::vector<Sequence*> sequences =
        build_prefill_smoke_sequences(batch_size, prompt_len, manager, storage);

    std::vector<uint32_t> allowed_max_tokens(batch_size,
                                             static_cast<uint32_t>(prompt_len));
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
                              BatchForwardType::PREFILL,
                              /*cp_size=*/1,
                              thread_pool);

    const auto start = std::chrono::steady_clock::now();
    ForwardInput forward_input =
        builder.build_forward_input(/*num_decoding_tokens=*/1,
                                    /*min_decoding_batch_size=*/0);
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - start)
                                  .count();
    if (!is_warmup) {
      total_ms += elapsed_ms;
    }

    EXPECT_EQ(forward_input.input_params.meta.num_sequences, batch_size);
    EXPECT_EQ(forward_input.token_ids.numel(), batch_size * prompt_len);
  }
  return total_ms / iters;
}

}  // namespace

TEST(BatchInputBuilderTest, FirstChunkUsesRemotePrefix) {
  const TransferKVInfo full_info = make_info({100, 101, 102, 103, 104});
  size_t advanced_transfer_block_idx = 0;

  const TransferKVInfo info =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_info,
          /*local_block_ids=*/{10, 11},
          /*next_transfer_block_idx=*/0,
          /*seq_len=*/32,
          /*block_size=*/16,
          &advanced_transfer_block_idx);

  expect_blocks(info, {10, 11}, {100, 101});
  EXPECT_EQ(advanced_transfer_block_idx, 2u);
}

TEST(BatchInputBuilderTest, LaterChunkUsesLogicalOffset) {
  const TransferKVInfo full_info = make_info({100, 101, 102, 103, 104});
  size_t advanced_transfer_block_idx = 0;

  const TransferKVInfo info =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_info,
          /*local_block_ids=*/{10, 11, 12, 13},
          /*next_transfer_block_idx=*/2,
          /*seq_len=*/64,
          /*block_size=*/16,
          &advanced_transfer_block_idx);

  expect_blocks(info, {12, 13}, {102, 103});
  EXPECT_EQ(advanced_transfer_block_idx, 4u);
}

TEST(BatchInputBuilderTest, PartialBoundaryRepeatsDirtyBlocks) {
  const TransferKVInfo full_info = make_info({100, 101, 102});
  size_t advanced_transfer_block_idx = 0;

  const TransferKVInfo info =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_info,
          /*local_block_ids=*/{10, 11, 12},
          /*next_transfer_block_idx=*/0,
          /*seq_len=*/33,
          /*block_size=*/16,
          &advanced_transfer_block_idx);

  expect_blocks(info, {10, 11, 12}, {100, 101, 102});
  EXPECT_EQ(advanced_transfer_block_idx, 2u);
}

TEST(BatchInputBuilderTest, SharedPrefixUsesLogicalMapping) {
  const TransferKVInfo full_info = make_info({100, 101, 102, 103, 104});
  size_t advanced_transfer_block_idx = 0;

  const TransferKVInfo info =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_info,
          /*local_block_ids=*/{10, 11, 12, 13, 14},
          /*next_transfer_block_idx=*/0,
          /*seq_len=*/80,
          /*block_size=*/16,
          &advanced_transfer_block_idx);

  expect_blocks(info, {10, 11, 12, 13, 14}, {100, 101, 102, 103, 104});
  EXPECT_EQ(advanced_transfer_block_idx, 5u);
}

TEST(BatchInputBuilderTest, SharedPrefixSlicesXTensorOffsets) {
  TransferKVInfo full_info = make_info({100, 101, 102, 103, 104});
  full_info.dst_xtensor_layer_offsets = {
      make_offsets({1000, 1001, 1002, 1003, 1004},
                   {2000, 2001, 2002, 2003, 2004}),
      make_offsets({3000, 3001, 3002, 3003, 3004},
                   {4000, 4001, 4002, 4003, 4004})};
  size_t advanced_transfer_block_idx = 0;

  const TransferKVInfo info =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_info,
          /*local_block_ids=*/{10, 11, 12, 13},
          /*next_transfer_block_idx=*/2,
          /*seq_len=*/64,
          /*block_size=*/16,
          &advanced_transfer_block_idx);

  expect_blocks(info, {12, 13}, {102, 103});
  EXPECT_EQ(advanced_transfer_block_idx, 4u);
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
  TransferKVInfo full_info = make_info({100, 101, 102});
  full_info.dst_xtensor_layer_offsets = {
      make_offsets({1000, 1001, 1002}, {2000, 2001, 2002}),
      make_offsets({3000, 3001, 3002}, {4000, 4001, 4002})};
  size_t advanced_transfer_block_idx = 0;

  const TransferKVInfo info =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_info,
          /*local_block_ids=*/{10, 11, 12},
          /*next_transfer_block_idx=*/0,
          /*seq_len=*/33,
          /*block_size=*/16,
          &advanced_transfer_block_idx);

  expect_blocks(info, {10, 11, 12}, {100, 101, 102});
  EXPECT_EQ(advanced_transfer_block_idx, 2u);
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
  const TransferKVInfo full_info = make_info({100, 101});
  size_t advanced_transfer_block_idx = 0;

  EXPECT_DEATH(
      {
        const TransferKVInfo info =
            BatchInputBuilderTestPeer::build_step_transfer_info(
                full_info,
                /*local_block_ids=*/{10, 11, 12},
                /*next_transfer_block_idx=*/0,
                /*seq_len=*/48,
                /*block_size=*/16,
                &advanced_transfer_block_idx);
        (void)info;
      },
      "remote");
}

TEST(BatchInputBuilderTest, DSV4FirstChunkSlicesFullRemoteAllocation) {
  const KVBlockTransferGroup full_group = make_dsv4_group({100, 101, 102, 103});
  size_t advanced_transfer_block_idx = 0;

  const KVBlockTransferGroup step_group =
      BatchInputBuilderTestPeer::build_group_step_transfer(
          full_group,
          /*local_block_ids=*/{10, 11},
          /*next_transfer_block_idx=*/0,
          /*seq_len=*/32,
          /*block_size=*/16,
          &advanced_transfer_block_idx);

  EXPECT_EQ(step_group.local_blocks_ids, (std::vector<uint64_t>{10, 11}));
  EXPECT_EQ(step_group.remote_blocks_ids, (std::vector<uint64_t>{100, 101}));
  EXPECT_EQ(advanced_transfer_block_idx, 2u);
}

TEST(BatchInputBuilderTest, DSV4LaterChunkSkipsExpiredSWABlocks) {
  KVBlockTransferGroup full_group = make_dsv4_group({100, 101, 102, 103});
  full_group.group_id = cache_group_id(BlockType::SWA);
  size_t advanced_transfer_block_idx = 2;

  const KVBlockTransferGroup step_group =
      BatchInputBuilderTestPeer::build_group_step_transfer(
          full_group,
          /*local_block_ids=*/{-1, -1, 12, 13},
          /*next_transfer_block_idx=*/2,
          /*seq_len=*/64,
          /*block_size=*/16,
          &advanced_transfer_block_idx);

  EXPECT_EQ(step_group.local_blocks_ids, (std::vector<uint64_t>{12, 13}));
  EXPECT_EQ(step_group.remote_blocks_ids, (std::vector<uint64_t>{102, 103}));
  EXPECT_EQ(advanced_transfer_block_idx, 4u);
}

TEST(BatchInputBuilderTest, DSV4PartialBlockIsRepeatedOnNextChunk) {
  const KVBlockTransferGroup full_group = make_dsv4_group({100, 101, 102});
  size_t advanced_transfer_block_idx = 0;

  const KVBlockTransferGroup step_group =
      BatchInputBuilderTestPeer::build_group_step_transfer(
          full_group,
          /*local_block_ids=*/{10, 11, 12},
          /*next_transfer_block_idx=*/0,
          /*seq_len=*/33,
          /*block_size=*/16,
          &advanced_transfer_block_idx);

  EXPECT_EQ(step_group.local_blocks_ids, (std::vector<uint64_t>{10, 11, 12}));
  EXPECT_EQ(step_group.remote_blocks_ids,
            (std::vector<uint64_t>{100, 101, 102}));
  EXPECT_EQ(advanced_transfer_block_idx, 2u);
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

TEST(BatchTest, DecodeForwardInputConsumesMtpBootstrap) {
  BlockManagerPool::Options options;
  options.num_blocks(8).block_size(4).enable_disagg_pd(true);
  options.max_seqs_per_batch(1024);
  BlockManagerPool manager(options, /*dp_size=*/1);

  Sequence sequence = make_basic_sequence({1, 2, 3});
  ASSERT_TRUE(manager.allocate(&sequence));
  ASSERT_GE(sequence.get_single_block_id(), 0);
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
  info.remote_blocks_ids = {100, 101, 102, 103};
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
  EXPECT_EQ(input.transfer_kv_infos[0].local_blocks_ids,
            (std::vector<uint64_t>{1, 2}));
  EXPECT_EQ(input.transfer_kv_infos[0].remote_blocks_ids,
            (std::vector<uint64_t>{100, 101}));
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
  info.remote_blocks_ids = {100, 101, 102};
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
  EXPECT_EQ(input.transfer_kv_infos[0].local_blocks_ids,
            (std::vector<uint64_t>{1, 2, 3}));
  EXPECT_EQ(input.transfer_kv_infos[0].remote_blocks_ids,
            (std::vector<uint64_t>{100, 101, 102}));
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
  info.remote_blocks_ids = {100, 101, 102, 103};
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
  EXPECT_EQ(input.transfer_kv_infos[0].local_blocks_ids,
            (std::vector<uint64_t>{1, 2}));
  EXPECT_EQ(input.transfer_kv_infos[0].remote_blocks_ids,
            (std::vector<uint64_t>{100, 101}));
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
  info.remote_blocks_ids = {100, 101};
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
  input.input_params.embedding.mtp_bootstrap_row_idxes = {0};
  input.input_params.embedding.mtp_bootstrap_embeddings =
      torch::tensor({{3.0f, 4.0f}});
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
  EXPECT_EQ(round_trip.transfer_kv_infos[0].local_blocks_ids,
            (std::vector<uint64_t>{1}));
  EXPECT_EQ(round_trip.transfer_kv_infos[0].remote_blocks_ids,
            (std::vector<uint64_t>{100}));
  EXPECT_EQ(round_trip.input_params.embedding.mtp_bootstrap_row_idxes,
            std::vector<int32_t>{0});
  ASSERT_TRUE(
      round_trip.input_params.embedding.mtp_bootstrap_embeddings.defined());
  EXPECT_TRUE(torch::equal(
      round_trip.input_params.embedding.mtp_bootstrap_embeddings.to(
          torch::kCPU),
      torch::tensor({{3.0f, 4.0f}})));
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

TEST(BatchTest, ForwardInputCpPartitionMatchesExpectedLayout) {
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
  seq_params.request_id = "req-cp";

  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);

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

  std::vector<Sequence*> sequences = {&seq};
  std::vector<uint32_t> budgets = {8};
  ModelArgs args;
  BatchInputBuilder forward_builder(sequences,
                                    budgets,
                                    {},
                                    {},
                                    nullptr,
                                    /*batch_id=*/1,
                                    &args,
                                    BatchForwardType::PREFILL);
  ForwardInput forward_input =
      forward_builder.build_forward_input(/*num_decoding_tokens=*/1,
                                          /*min_decoding_batch_size=*/0);

  ForwardInput cp_forward_input = forward_input;
  cp::cp_partition_inplace(cp_forward_input, /*cp_rank=*/0, /*cp_size=*/2);

  EXPECT_TRUE(
      equal(cp_forward_input.token_ids, std::vector<int32_t>({1, 2, 7, 8})));
  EXPECT_TRUE(
      equal(cp_forward_input.positions, std::vector<int32_t>({0, 1, 6, 7})));
  EXPECT_TRUE(equal(cp_forward_input.sampling_params.selected_token_idxes,
                    std::vector<int32_t>({3})));
  EXPECT_EQ(cp_forward_input.input_params.meta.q_max_seq_len, 4);
  EXPECT_EQ(cp_forward_input.input_params.meta.kv_max_seq_len, 4);

  const std::vector<int32_t>& q_seq_lens =
      cp_forward_input.input_params.attention.host.q_seq_lens;
  const std::vector<int32_t>& kv_seq_lens =
      cp_forward_input.input_params.attention.host.kv_seq_lens;
  EXPECT_TRUE((q_seq_lens == std::vector<int32_t>({4}) ||
               q_seq_lens == std::vector<int32_t>({0, 4})));
  EXPECT_TRUE((kv_seq_lens == std::vector<int32_t>({4}) ||
               kv_seq_lens == std::vector<int32_t>({0, 4})));
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

TEST(BatchTest, DecodeSingleBlockIdsStaySplitInTransportButShareSlotValue) {
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

  auto slot_block = manager.allocate(1);
  ASSERT_EQ(slot_block.size(), 1u);
  const int32_t expected_slot_id = slot_block[0].id();
  seq.add_blocks(BlockType::SINGLE, slot_block);

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
            expected_slot_id);
  EXPECT_EQ(forward_input.input_params.embedding.linear_state_ids[0],
            expected_slot_id);
}

TEST(BatchTest, LinearStateCheckpointSavesOnlyAtPrefillStepBoundary) {
  // Linear-state checkpoints are hashed per chunk-end boundary. This test's
  // save/restore boundaries (16, 20) are multiples of the KV block_size (4),
  // so set the chunk stride to block_size to keep them chunk-aligned and the
  // per-chunk hash chain identical to the per-block helper below.
  const int32_t prev_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(4);

  torch::Device device(Platform::type_torch(), 0);
  const uint32_t n_blocks = 24;
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

  Batch batch;
  batch.add(&aligned_seq, /*allowed_max_token=*/16);
  batch.add(&restore_seq, /*allowed_max_token=*/4);
  batch.add(&off_boundary_seq, /*allowed_max_token=*/15);
  batch.add(&decode_seq, /*allowed_max_token=*/1);

  ModelArgs args;
  args.layer_types({"linear_attention"});
  ForwardInput forward_input = batch.prepare_forward_input(
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

  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(
      prev_chunk_stride);
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
  KVBlockTransferGroup transfer_group;
  transfer_group.group_id = cache_group_id(BlockType::C128);
  transfer_group.local_blocks_ids = {11, 12};
  transfer_group.remote_blocks_ids = {101, 102, 103, 104};
  transfer_info.block_transfer_groups.emplace_back(std::move(transfer_group));
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
  ASSERT_EQ(from_shm.transfer_kv_infos[0].block_transfer_groups.size(), 1u);
  const auto& from_shm_group =
      from_shm.transfer_kv_infos[0].block_transfer_groups[0];
  EXPECT_EQ(from_shm_group.group_id, cache_group_id(BlockType::C128));
  EXPECT_EQ(from_shm_group.local_blocks_ids, (std::vector<uint64_t>{11, 12}));
  EXPECT_EQ(from_shm_group.remote_blocks_ids,
            (std::vector<uint64_t>{101, 102, 103, 104}));

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

// Smoke test: sweep batch sizes and, for each, report single-threaded vs
// multithreaded (8-worker pool) build_forward_input latency plus the split
// between the parallelized process_sequences() phase and the serial
// state_to_forward_input() phase. This diagnoses where time goes and at which
// batch size the multithreaded path starts to pay off. Not a strict latency
// gate; assertions only check the produced shape (inside the helpers).
TEST(BatchInputBuilderSmokeTest, BatchSizeSweepThreadingBreakdown) {
  const std::vector<int32_t> batch_sizes = {32, 128, 512, 2048, 5120};
  constexpr int32_t kIters = 10;
  ThreadPool thread_pool(/*num_threads=*/8);

  std::cout << "[ SMOKE    ] build_forward_input latency sweep (avg of "
            << kIters << " iters, " << thread_pool.size() << "-thread pool)\n"
            << "[ SMOKE    ] batch |  1-thread |  N-thread | speedup | "
               "process(1t) | process(Nt) | finalize\n";
  for (int32_t batch_size : batch_sizes) {
    const double st_ms =
        average_build_ms(batch_size, kIters, /*thread_pool=*/nullptr);
    const double mt_ms = average_build_ms(batch_size, kIters, &thread_pool);
    const PhaseTiming st_phases =
        average_phase_ms(batch_size, kIters, /*thread_pool=*/nullptr);
    const PhaseTiming mt_phases =
        average_phase_ms(batch_size, kIters, &thread_pool);

    char line[256];
    std::snprintf(line,
                  sizeof(line),
                  "[ SMOKE    ] %5d | %8.3f | %8.3f | %6.2fx | %10.3f | "
                  "%10.3f | %8.3f",
                  batch_size,
                  st_ms,
                  mt_ms,
                  mt_ms > 0.0 ? st_ms / mt_ms : 0.0,
                  st_phases.process_ms,
                  mt_phases.process_ms,
                  st_phases.finalize_ms);
    std::cout << line << std::endl;

    EXPECT_GT(st_ms, 0.0);
    EXPECT_GT(mt_ms, 0.0);
  }
}

// Smoke test: same sweep but with PREFILL sequences that carry a real
// per-sequence token workload. This is the regime the multithreaded builder
// targets: with meaningful work per sequence, process_sequences() distributes
// across workers and the N-thread path should beat single-threaded at larger
// batches. Contrast with the decode sweep above, where per-sequence work is a
// single token and threading cannot amortize its fixed dispatch+merge cost.
TEST(BatchInputBuilderSmokeTest, PrefillBatchSizeSweepThreadingSpeedup) {
  const std::vector<int32_t> batch_sizes = {32, 128, 512, 1024};
  constexpr int32_t kPromptLen = 512;
  constexpr int32_t kIters = 10;
  ThreadPool thread_pool(/*num_threads=*/8);

  std::cout << "[ SMOKE    ] prefill build_forward_input sweep (prompt_len="
            << kPromptLen << ", avg of " << kIters << " iters, "
            << thread_pool.size() << "-thread pool)\n"
            << "[ SMOKE    ] batch |  1-thread |  N-thread | speedup\n";
  for (int32_t batch_size : batch_sizes) {
    const double st_ms = average_prefill_build_ms(
        batch_size, kPromptLen, kIters, /*thread_pool=*/nullptr);
    const double mt_ms =
        average_prefill_build_ms(batch_size, kPromptLen, kIters, &thread_pool);

    char line[256];
    std::snprintf(line,
                  sizeof(line),
                  "[ SMOKE    ] %5d | %8.3f | %8.3f | %6.2fx",
                  batch_size,
                  st_ms,
                  mt_ms,
                  mt_ms > 0.0 ? st_ms / mt_ms : 0.0);
    std::cout << line << std::endl;

    EXPECT_GT(st_ms, 0.0);
    EXPECT_GT(mt_ms, 0.0);
  }
}

}  // namespace xllm
