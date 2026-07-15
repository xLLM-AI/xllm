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

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/batch/batch_input_builder.h"
#include "core/framework/block/block_manager_impl.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/request/stopping_checker.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/params_utils.h"

namespace xllm {

namespace {

void expect_linear_state_cache_op_eq(const LinearStateCacheOp& actual,
                                     const LinearStateCacheOp& expected) {
  EXPECT_EQ(actual.linear_state_id, expected.linear_state_id);
  EXPECT_EQ(actual.restore_requested, expected.restore_requested);
  EXPECT_EQ(actual.restore_src_slot_id, expected.restore_src_slot_id);
}

}  // namespace

template <typename T>
bool tensor_equals_vector(const torch::Tensor& tensor,
                          const std::vector<T>& values) {
  auto flat = tensor.flatten();
  if (flat.size(0) != values.size()) {
    return false;
  }
  for (int64_t i = 0; i < flat.size(0); ++i) {
    if (flat[i].item<T>() != values[static_cast<size_t>(i)]) {
      return false;
    }
  }
  return true;
}

TEST(BatchPackedInputTest, PackedProtoLazyUnpackPreservesLinearStateCacheOps) {
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
  seq_params.enable_schedule_overlap = true;

  torch::Tensor input_embedding;
  MMData mm_data;
  BlockManager::Options options;
  options.num_blocks(2).block_size(4);
  BlockManagerImpl manager(options);

  IncrementalDecoder decoder("", 1, false, false);
  Sequence seq(/*index=*/0,
               /*token_ids=*/{1, 2, 3, 4},
               input_embedding,
               mm_data,
               std::move(decoder),
               seq_params);

  seq.add_blocks(BlockType::KV, manager.allocate(1));

  std::vector<Sequence*> sequences = {&seq};
  std::vector<uint32_t> budgets = {4};
  BatchInputBuilder builder(sequences,
                            budgets,
                            {},
                            {},
                            nullptr,
                            /*batch_id=*/1,
                            nullptr,
                            BatchForwardType::DECODE);

  ForwardInput input =
      builder.build_forward_input(/*num_decoding_tokens=*/1,
                                  /*min_decoding_batch_size=*/0);
  LinearStateCacheOp restore_op;
  restore_op.linear_state_id = 7;
  restore_op.restore_requested = true;
  restore_op.restore_src_slot_id = 3;

  LinearStateCacheOp no_restore_op;
  no_restore_op.linear_state_id = 8;

  input.input_params.linear_state_cache_ops = {restore_op, no_restore_op};

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed_input));

  ForwardInput lazy_input;
  packed_proto_to_forward_input(
      packed_input, lazy_input, torch::Device(torch::kCPU), nullptr);
  EXPECT_TRUE(lazy_input.input_params.linear_state_cache_ops.empty());
  EXPECT_TRUE(lazy_input.input_host_buffer_has_layout);

  ForwardInput unpacked_input;
  ASSERT_TRUE(detail::unpack_from_input_host_buffer(lazy_input,
                                                    torch::Device(torch::kCPU),
                                                    torch::kFloat32,
                                                    unpacked_input,
                                                    false));
  ASSERT_EQ(unpacked_input.input_params.linear_state_cache_ops.size(), 2u);
  expect_linear_state_cache_op_eq(
      unpacked_input.input_params.linear_state_cache_ops[0], restore_op);
  expect_linear_state_cache_op_eq(
      unpacked_input.input_params.linear_state_cache_ops[1], no_restore_op);
}

TEST(BatchPackedInputTest, PackedProtoLazyUnpackRestoresSampleIdxes) {
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
  seq_params.enable_schedule_overlap = true;

  torch::Tensor input_embedding;
  MMData mm_data;
  BlockManager::Options options;
  options.num_blocks(2).block_size(4);
  BlockManagerImpl manager(options);

  IncrementalDecoder decoder("", 1, false, false);
  Sequence seq(/*index=*/0,
               /*token_ids=*/{1, 2, 3, 4},
               input_embedding,
               mm_data,
               std::move(decoder),
               seq_params);

  seq.add_blocks(BlockType::KV, manager.allocate(1));

  std::vector<Sequence*> sequences = {&seq};
  std::vector<uint32_t> budgets = {4};
  BatchInputBuilder builder(sequences,
                            budgets,
                            {},
                            {},
                            nullptr,
                            /*batch_id=*/1,
                            nullptr,
                            BatchForwardType::DECODE);

  ForwardInput input =
      builder.build_forward_input(/*num_decoding_tokens=*/1,
                                  /*min_decoding_batch_size=*/0);
  ASSERT_TRUE(input.sampling_params.sample_idxes.defined());

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed_input));

  ForwardInput lazy_input;
  packed_proto_to_forward_input(
      packed_input, lazy_input, torch::Device(torch::kCPU), nullptr);
  EXPECT_FALSE(lazy_input.sampling_params.sample_idxes.defined());
  EXPECT_TRUE(lazy_input.input_host_buffer_has_layout);

  ForwardInput unpacked_input;
  ASSERT_TRUE(detail::unpack_from_input_host_buffer(lazy_input,
                                                    torch::Device(torch::kCPU),
                                                    torch::kFloat32,
                                                    unpacked_input,
                                                    false));
  ASSERT_TRUE(unpacked_input.sampling_params.sample_idxes.defined());
  EXPECT_TRUE(tensor_equals_vector<int32_t>(
      unpacked_input.sampling_params.sample_idxes, {0}));
}

}  // namespace xllm
