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

#include "core/runtime/slot_buffer.h"

#include <gtest/gtest.h>
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>

#include <array>
#include <limits>
#include <mutex>
#include <numeric>

#include "core/framework/sampling/sampler.h"
#include "core/layers/common/attention_metadata.h"

namespace xllm {
namespace {

struct InputData {
  std::vector<int32_t> tokens;
  std::vector<int32_t> positions;
  std::vector<int32_t> slots;
  std::vector<int32_t> query;
  std::vector<int32_t> kv;
  std::vector<int32_t> cumulative;
  std::vector<int32_t> blocks;
  uint32_t width = 2;
};

InputData make_input(std::vector<int32_t> query, std::vector<int32_t> kv) {
  InputData input;
  input.query = std::move(query);
  input.kv = std::move(kv);
  input.cumulative.resize(input.query.size());
  std::partial_sum(
      input.query.begin(), input.query.end(), input.cumulative.begin());
  const int32_t tokens = input.cumulative.empty() ? 0 : input.cumulative.back();
  input.tokens.resize(tokens);
  input.positions.resize(tokens);
  input.slots.resize(tokens);
  input.blocks.resize(input.query.size() * input.width);
  std::iota(input.tokens.begin(), input.tokens.end(), 101);
  std::iota(input.positions.begin(), input.positions.end(), 201);
  std::iota(input.slots.begin(), input.slots.end(), 301);
  std::iota(input.blocks.begin(), input.blocks.end(), 401);
  return input;
}

ModelInputHostView view(const InputData& input) {
  return {input.tokens,
          input.positions,
          input.slots,
          input.query,
          input.kv,
          input.cumulative,
          input.blocks,
          input.width};
}

class SlotBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(SlotBuffer::create(capacity_, device_, binding_).ok());
    transfer_ = std::make_unique<Stream>(device_);
    consumer_ = std::make_unique<Stream>(device_);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  }

  void TearDown() override { EXPECT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS); }

  Status prepare_model(const ModelInputHostView& input,
                       const ModelInputBatch& batch,
                       const Stream& stream) {
    Status status = binding_->validate(input, batch, {}, 0, stream);
    if (status.ok()) {
      binding_->prepare(input, batch, {}, stream);
      ready_ = stream.record_event();
    }
    return status;
  }

  void expect_device_input(const InputData& input) {
    CHECK(consumer_->wait_event(ready_));
    auto guard = consumer_->set_stream_guard();
    const AttentionDeviceInput& device =
        binding_->model_params().attention.device;
    const std::array<torch::Tensor, 6> tensors = {binding_->tokens(),
                                                  binding_->positions(),
                                                  device.new_cache_slots,
                                                  device.q_seq_lens,
                                                  device.kv_seq_lens,
                                                  device.q_cu_seq_lens};
    const std::array<std::span<const int32_t>, 6> expected = {input.tokens,
                                                              input.positions,
                                                              input.slots,
                                                              input.query,
                                                              input.kv,
                                                              input.cumulative};
    for (uint32_t field = 0; field < tensors.size(); ++field) {
      torch::Tensor host = tensors[field].cpu();
      EXPECT_EQ(host.numel(), expected[field].size());
      const int32_t* data = host.data_ptr<int32_t>();
      for (uint32_t index = 0; index < expected[field].size(); ++index) {
        EXPECT_EQ(data[index], expected[field][index]);
      }
    }
    torch::Tensor blocks = device.block_tables.cpu();
    EXPECT_EQ(blocks.size(0), input.query.size());
    EXPECT_EQ(blocks.size(1), 5);
    const int32_t* values = blocks.data_ptr<int32_t>();
    for (uint32_t row = 0; row < input.query.size(); ++row) {
      for (uint32_t column = 0; column < input.width; ++column) {
        EXPECT_EQ(values[row * 5 + column],
                  input.blocks[row * input.width + column]);
      }
      for (uint32_t column = input.width; column < 5; ++column) {
        EXPECT_EQ(values[row * 5 + column], 0);
      }
    }
  }

  void expect_rejected(const ModelInputHostView& input,
                       const ModelInputBatch& batch) {
    const torch::Tensor before = binding_->tokens().cpu().clone();
    const uint64_t batch_id = binding_->model_params().meta.batch_id;
    const std::vector<int32_t> lengths =
        binding_->model_params().attention.host.kv_seq_lens;
    const void* tokens = binding_->tokens().data_ptr();
    EXPECT_FALSE(prepare_model(input, batch, *transfer_).ok());
    EXPECT_TRUE(torch::equal(before, binding_->tokens().cpu()));
    EXPECT_EQ(binding_->model_params().meta.batch_id, batch_id);
    EXPECT_EQ(binding_->model_params().attention.host.kv_seq_lens, lengths);
    EXPECT_EQ(binding_->tokens().data_ptr(), tokens);
  }

  const torch::Device device_{torch::kPrivateUse1, 0};
  SlotBufferCapacity capacity_{{16, 4, 5}, 16, 1024, 8};
  std::unique_ptr<SlotBuffer> binding_;
  StreamEventPtr ready_;
  std::unique_ptr<Stream> transfer_;
  std::unique_ptr<Stream> consumer_;
};

TEST_F(SlotBufferTest, PrefillUsesFixedInputsAndUncachedLengths) {
  InputData input = make_input({3, 2}, {3, 2});
  ASSERT_TRUE(prepare_model(view(input),
                            {BatchForwardType::PREFILL, 2, 41, true},
                            *transfer_)
                  .ok());
  expect_device_input(input);
  const ModelInputParams& params = binding_->model_params();
  EXPECT_EQ(params.attention.host.kv_cache_tokens_nums,
            (std::vector<int32_t>{0, 0}));
  EXPECT_EQ(params.attention.host.q_cu_seq_lens, (std::vector<int32_t>{3, 5}));
  EXPECT_EQ(params.meta.num_sequences, 2);
  EXPECT_EQ(params.meta.actual_num_sequences, 2);
  EXPECT_EQ(params.meta.batch_id, 41);
  EXPECT_TRUE(params.meta.is_graph_warmup);
  EXPECT_EQ(params.meta.q_max_seq_len, 3);
  EXPECT_EQ(params.meta.kv_max_seq_len, 3);
  EXPECT_EQ(params.attention.device.block_tables.stride(0), 5);
  EXPECT_EQ(params.attention.host.graph_q_seq_lens_data,
            params.attn_metadata->q_seq_lens_host.data_ptr<int32_t>());
  EXPECT_EQ(params.attention.host.graph_kv_seq_lens_data,
            params.attn_metadata->kv_seq_lens_host.data_ptr<int32_t>());
  EXPECT_FALSE(params.attention.device.kv_cache_tokens_nums.defined());
}

TEST_F(SlotBufferTest, MixedBatchKeepsActualAndPaddingRowsDistinct) {
  InputData input = make_input({2, 1, 0}, {5, 7, 0});
  ASSERT_TRUE(
      prepare_model(view(input), {BatchForwardType::MIXED, 2, 42}, *transfer_)
          .ok());
  expect_device_input(input);
  EXPECT_EQ(binding_->model_params().meta.num_sequences, 3);
  EXPECT_EQ(binding_->model_params().meta.actual_num_sequences, 2);
  EXPECT_EQ(binding_->model_params().meta.q_max_seq_len, 2);
  EXPECT_EQ(binding_->model_params().meta.kv_max_seq_len, 7);
  EXPECT_EQ(binding_->model_params().attention.host.kv_cache_tokens_nums,
            (std::vector<int32_t>{3, 6, 0}));
}

TEST_F(SlotBufferTest, VaryingSizesReuseAddressesAndHostCapacity) {
  InputData first = make_input({2, 2, 2, 2}, {7, 7, 7, 7});
  ASSERT_TRUE(prepare_model(view(first),
                            {BatchForwardType::CHUNKED_PREFILL, 4},
                            *transfer_)
                  .ok());
  expect_device_input(first);
  const void* token_address = binding_->tokens().data_ptr();
  const AttentionHostInput& host = binding_->model_params().attention.host;
  const int32_t* lengths = host.q_seq_lens.data();
  const int32_t* slots = host.new_cache_slots.data();
  const size_t lengths_capacity = host.q_seq_lens.capacity();
  const size_t slots_capacity = host.new_cache_slots.capacity();
  for (uint32_t iteration = 0; iteration < 16; ++iteration) {
    const uint32_t rows = iteration % 4 + 1;
    InputData input = make_input(std::vector<int32_t>(rows, 1),
                                 std::vector<int32_t>(rows, iteration + 2));
    input.tokens[0] += static_cast<int32_t>(iteration);
    ASSERT_TRUE(prepare_model(view(input),
                              {BatchForwardType::DECODE, rows, iteration},
                              *transfer_)
                    .ok());
    expect_device_input(input);
    EXPECT_EQ(host.q_seq_lens.data(), lengths);
    EXPECT_EQ(host.new_cache_slots.data(), slots);
    EXPECT_EQ(host.q_seq_lens.capacity(), lengths_capacity);
    EXPECT_EQ(host.new_cache_slots.capacity(), slots_capacity);
    EXPECT_EQ(binding_->tokens().data_ptr(), token_address);
    EXPECT_EQ(binding_->model_params().meta.q_max_seq_len, 1);
    EXPECT_EQ(binding_->model_params().meta.kv_max_seq_len, iteration + 2);
  }
}

TEST_F(SlotBufferTest, EmptyBatchClearsViewsAndMetadataWithoutReallocation) {
  InputData input = make_input({3, 2}, {8, 9});
  ASSERT_TRUE(prepare_model(view(input),
                            {BatchForwardType::CHUNKED_PREFILL, 2, 51, true},
                            *transfer_)
                  .ok());
  expect_device_input(input);
  const size_t capacity =
      binding_->model_params().attention.host.q_seq_lens.capacity();
  ASSERT_TRUE(
      prepare_model({}, {BatchForwardType::EMPTY, 0, 52}, *transfer_).ok());
  CHECK(consumer_->wait_event(ready_));
  ASSERT_EQ(consumer_->synchronize(), 0);
  EXPECT_EQ(binding_->tokens().numel(), 0);
  EXPECT_EQ(binding_->model_params().attention.device.block_tables.size(0), 0);
  EXPECT_TRUE(binding_->model_params().attention.host.q_seq_lens.empty());
  EXPECT_TRUE(
      binding_->model_params().attention.host.kv_cache_tokens_nums.empty());
  EXPECT_TRUE(binding_->model_params().attention.host.new_cache_slots.empty());
  EXPECT_EQ(binding_->model_params().attention.host.q_seq_lens.capacity(),
            capacity);
  EXPECT_EQ(binding_->model_params().meta.q_max_seq_len, 0);
  EXPECT_EQ(binding_->model_params().meta.kv_max_seq_len, 0);
  EXPECT_EQ(binding_->model_params().meta.actual_num_sequences, 0);
  EXPECT_EQ(binding_->model_params().meta.batch_id, 52);
  EXPECT_FALSE(binding_->model_params().meta.is_graph_warmup);
}

TEST_F(SlotBufferTest, InvalidBatchSemanticsPreservePreviousBinding) {
  InputData input = make_input({2, 1}, {5, 7});
  ASSERT_TRUE(
      prepare_model(view(input), {BatchForwardType::MIXED, 2, 63}, *transfer_)
          .ok());
  expect_device_input(input);
  for (const ModelInputBatch& batch :
       {ModelInputBatch{BatchForwardType(999), 2},
        ModelInputBatch{BatchForwardType::DECODE, 2},
        ModelInputBatch{BatchForwardType::PREFILL, 2},
        ModelInputBatch{BatchForwardType::MIXED, 3},
        ModelInputBatch{BatchForwardType::EMPTY, 0}}) {
    expect_rejected(view(input), batch);
  }
  InputData zero_query = make_input({2, 0}, {5, 7});
  expect_rejected(view(zero_query), {BatchForwardType::MIXED, 2});
  expect_rejected({}, {BatchForwardType::DECODE, 0});
}

TEST_F(SlotBufferTest, InvalidTransferInputPreservesPreviousBinding) {
  InputData input = make_input({1}, {4});
  ASSERT_TRUE(
      prepare_model(view(input), {BatchForwardType::DECODE, 1, 71}, *transfer_)
          .ok());
  expect_device_input(input);
  InputData wrong_shape = make_input({2}, {5});
  wrong_shape.positions.pop_back();
  expect_rejected(view(wrong_shape), {BatchForwardType::CHUNKED_PREFILL, 1});
  InputData oversized = make_input({17}, {17});
  expect_rejected(view(oversized), {BatchForwardType::PREFILL, 1});
}

TEST_F(SlotBufferTest, NextInputCanBorrowPreviousHostMetadata) {
  InputData first = make_input({2, 1}, {4, 3});
  ASSERT_TRUE(prepare_model(view(first),
                            {BatchForwardType::CHUNKED_PREFILL, 2},
                            *transfer_)
                  .ok());
  expect_device_input(first);
  InputData next = make_input({2, 2}, {2, 3});
  ModelInputHostView borrowed = view(next);
  borrowed.q_seq_lens =
      binding_->model_params().attention.host.kv_cache_tokens_nums;
  borrowed.kv_seq_lens = binding_->model_params().attention.host.q_cu_seq_lens;
  ASSERT_TRUE(prepare_model(
                  borrowed, {BatchForwardType::CHUNKED_PREFILL, 2}, *transfer_)
                  .ok());
  expect_device_input(next);
  EXPECT_EQ(binding_->model_params().attention.host.kv_seq_lens, next.kv);
  EXPECT_EQ(binding_->model_params().attention.host.kv_cache_tokens_nums,
            (std::vector<int32_t>{0, 1}));
}

TEST_F(SlotBufferTest, CallerSuppliedDummyRowHasNoActualSequence) {
  InputData input = make_input({1}, {1});
  ASSERT_TRUE(
      prepare_model(view(input), {BatchForwardType::DECODE, 0}, *transfer_)
          .ok());
  expect_device_input(input);
  EXPECT_EQ(binding_->model_params().meta.num_sequences, 1);
  EXPECT_EQ(binding_->model_params().meta.actual_num_sequences, 0);
  EXPECT_EQ(binding_->tokens().numel(), 1);
}

TEST_F(SlotBufferTest, PythonMetadataIsPrivateAndUsesFixedViews) {
  InputData first = make_input({3, 2}, {3, 2});
  ASSERT_TRUE(prepare_model(view(first),
                            {BatchForwardType::PREFILL, 2, 1, false},
                            *transfer_)
                  .ok());
  const auto metadata = binding_->model_params().attn_metadata;
  const void* token_address = binding_->tokens().data_ptr();
  const void* length_address = metadata->q_seq_lens.data_ptr();
  std::unique_ptr<SlotBuffer> other;
  ASSERT_TRUE(SlotBuffer::create(capacity_, device_, other).ok());
  const InputData second = make_input({1}, {4});
  const ModelInputBatch second_batch{BatchForwardType::DECODE, 1, 2, false};
  ASSERT_TRUE(
      other->validate(view(second), second_batch, {}, 0, *transfer_).ok());
  other->prepare(view(second), second_batch, {}, *transfer_);
  first.query.assign(first.query.size(), /*value=*/-99);
  EXPECT_EQ(metadata->q_seq_lens_vec, (std::vector<int32_t>{3, 2}));
  EXPECT_EQ(metadata->q_cu_seq_lens_host_vec, (std::vector<int64_t>{3, 5}));
  EXPECT_EQ(metadata->max_query_len, 3);
  EXPECT_TRUE(metadata->is_prefill);
  EXPECT_FALSE(metadata->block_table.defined());
  EXPECT_EQ(metadata->q_seq_lens.data_ptr(), length_address);
  EXPECT_EQ(binding_->tokens().data_ptr(), token_address);
  EXPECT_EQ(metadata->q_cu_seq_lens.data_ptr(),
            binding_->model_params().attention.device.q_cu_seq_lens.data_ptr());
  EXPECT_NE(metadata, other->model_params().attn_metadata);
  EXPECT_TRUE(other->model_params().attn_metadata->block_table.defined());
  ASSERT_EQ(transfer_->synchronize(), 0);
}

TEST_F(SlotBufferTest, MlaKeepsFinalBlockTableForEveryForwardType) {
  capacity_.enable_mla = true;
  ASSERT_TRUE(SlotBuffer::create(capacity_, device_, binding_).ok());
  ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  const std::array<BatchForwardType, 4> types = {
      BatchForwardType::PREFILL,
      BatchForwardType::CHUNKED_PREFILL,
      BatchForwardType::MIXED,
      BatchForwardType::DECODE};
  for (const BatchForwardType type : types) {
    SCOPED_TRACE(type.value());
    const bool decode = type.is_decode();
    const InputData input = make_input(
        decode ? std::vector<int32_t>{1, 1} : std::vector<int32_t>{3, 2},
        type.is_prefill() ? std::vector<int32_t>{3, 2}
                          : std::vector<int32_t>{6, 4});
    ASSERT_TRUE(prepare_model(view(input), {type, 2}, *transfer_).ok());
    expect_device_input(input);
    const auto metadata = binding_->model_params().attn_metadata;
    ASSERT_TRUE(metadata->block_table.defined());
    EXPECT_EQ(
        metadata->block_table.data_ptr(),
        binding_->model_params().attention.device.block_tables.data_ptr());
    EXPECT_EQ(
        metadata->q_cu_seq_lens.data_ptr(),
        binding_->model_params().attention.device.q_cu_seq_lens.data_ptr());
    EXPECT_EQ(metadata->kv_seq_lens.data_ptr(),
              binding_->model_params().attention.device.kv_seq_lens.data_ptr());
    EXPECT_EQ(
        metadata->slot_mapping.data_ptr(),
        binding_->model_params().attention.device.new_cache_slots.data_ptr());
    EXPECT_EQ(
        metadata->q_cu_seq_lens_host_vec,
        (decode ? std::vector<int64_t>{1, 2} : std::vector<int64_t>{3, 5}));
    EXPECT_EQ(metadata->kv_seq_lens_vec, input.kv);
  }
}

TEST_F(SlotBufferTest, CallerMayOverwriteModelDataAfterPrepare) {
  InputData input = make_input({2, 1}, {2, 1});
  const InputData expected = input;
  ASSERT_TRUE(
      prepare_model(view(input), {BatchForwardType::PREFILL, 2}, *transfer_)
          .ok());
  std::fill(input.tokens.begin(), input.tokens.end(), -99);
  std::fill(input.blocks.begin(), input.blocks.end(), -99);
  expect_device_input(expected);
}

TEST_F(SlotBufferTest, RejectsShapeCapacityAndStagingAliasesWithoutWrites) {
  InputData input = make_input({2, 1}, {4, 3});
  ASSERT_TRUE(
      prepare_model(view(input), {BatchForwardType::MIXED, 2}, *transfer_)
          .ok());
  expect_device_input(input);
  auto invalid = view(input);
  invalid.block_tables = std::span<const int32_t>(
      binding_->model_params().attention.host.block_tables.data_ptr<int32_t>(),
      input.blocks.size());
  expect_rejected(invalid, {BatchForwardType::MIXED, 2});
  invalid = view(input);
  invalid.kv_seq_lens = invalid.kv_seq_lens.first(1);
  expect_rejected(invalid, {BatchForwardType::MIXED, 2});
  invalid = view(input);
  invalid.new_cache_slots = {};
  expect_rejected(invalid, {BatchForwardType::MIXED, 2});
  invalid = view(input);
  invalid.block_table_width = 6;
  expect_rejected(invalid, {BatchForwardType::MIXED, 2});
  InputData bad = input;
  bad.cumulative.back() -= 1;
  expect_rejected(view(bad), {BatchForwardType::MIXED, 2});
  bad = input;
  bad.query[0] = -1;
  expect_rejected(view(bad), {BatchForwardType::MIXED, 2});
  bad = input;
  bad.kv[0] = 0;
  expect_rejected(view(bad), {BatchForwardType::MIXED, 2});
  bad = make_input({1, 1, 1, 1, 1}, {1, 1, 1, 1, 1});
  expect_rejected(view(bad), {BatchForwardType::PREFILL, 5});
  invalid = {};
  invalid.block_table_width = 1;
  expect_rejected(invalid, {BatchForwardType::EMPTY, 0});
}

TEST_F(SlotBufferTest, PreviousRowsReorderDuplicateAndRejectAtomically) {
  InputData input = make_input({2, 1, 1}, {2, 1, 1});
  input.tokens = {42, -2, -1, -2};
  const ModelInputBatch batch{BatchForwardType::MIXED, 3};
  ASSERT_TRUE(binding_->validate(view(input), batch, {}, 2, *transfer_).ok());
  binding_->prepare(view(input), batch, {}, *transfer_);
  ready_ = transfer_->record_event();
  ASSERT_TRUE(consumer_->wait_event(ready_));
  auto guard = consumer_->set_stream_guard();
  auto previous = torch::tensor({101, 202, 0, 0}, torch::kInt64).to(device_);
  binding_->patch_previous_tokens(previous);
  auto retained = binding_->tokens().cpu();
  EXPECT_TRUE(torch::equal(retained,
                           torch::tensor({42, 202, 101, 202}, torch::kInt32)));
  for (const std::vector<int32_t>& tokens :
       {std::vector<int32_t>{-1, 12, 13, 14},
        {42, -3, 13, 14},
        {42, std::numeric_limits<int32_t>::min(), 13, 14}}) {
    input.tokens = tokens;
    EXPECT_FALSE(
        binding_->validate(view(input), batch, {}, 2, *transfer_).ok());
    EXPECT_TRUE(torch::equal(binding_->tokens().cpu(), retained));
    EXPECT_TRUE(binding_->has_previous_tokens());
  }
  EXPECT_FALSE(binding_->validate(view(input), batch, {}, 5, *transfer_).ok());
  ASSERT_TRUE(
      binding_->validate({}, {BatchForwardType::EMPTY, 0}, {}, 0, *transfer_)
          .ok());
  binding_->prepare({}, {BatchForwardType::EMPTY, 0}, {}, *transfer_);
  EXPECT_FALSE(binding_->has_previous_tokens());
}

TEST_F(SlotBufferTest, InvalidSamplingDoesNotOverwriteModelOrPreviousMapping) {
  InputData input = make_input({1, 1}, {1, 1});
  ASSERT_TRUE(
      prepare_model(view(input), {BatchForwardType::DECODE, 2}, *transfer_)
          .ok());
  expect_device_input(input);
  const auto before = binding_->tokens().cpu();
  input.tokens = {-2, -1};
  SamplingParameters sampling;
  sampling.selected_token_idxes = torch::tensor({0, 1}, torch::kInt32);
  sampling.sample_idxes = torch::tensor({0, 1}, torch::kInt32);
  sampling.do_sample = torch::zeros({2}, torch::kInt32);
  EXPECT_FALSE(binding_
                   ->validate(view(input),
                              {BatchForwardType::DECODE, 2},
                              sampling,
                              2,
                              *transfer_)
                   .ok());
  EXPECT_FALSE(binding_->has_previous_tokens());
  EXPECT_FALSE(binding_->sampling_params().selected_token_idxes.defined());
  EXPECT_TRUE(torch::equal(binding_->tokens().cpu(), before));
}

TEST_F(SlotBufferTest, NarrowAndFullBlockTablesReuseStrideWithoutStalePages) {
  const uint64_t pinned = binding_->pinned_bytes();
  const uint64_t device = binding_->device_bytes();
  for (const uint32_t width : {5U, 1U, 3U, 5U}) {
    InputData input = make_input({1, 1}, {2, 2});
    input.width = width;
    input.blocks.resize(2 * width);
    std::iota(input.blocks.begin(), input.blocks.end(), 401);
    ASSERT_TRUE(
        prepare_model(view(input), {BatchForwardType::DECODE, 2}, *transfer_)
            .ok());
    expect_device_input(input);
    EXPECT_EQ(binding_->pinned_bytes(), pinned);
    EXPECT_EQ(binding_->device_bytes(), device);
  }
}

constexpr std::array<torch::Tensor SamplingParameters::*, 12> kInputs = {
    &SamplingParameters::selected_token_idxes,
    &SamplingParameters::frequency_penalties,
    &SamplingParameters::presence_penalties,
    &SamplingParameters::repetition_penalties,
    &SamplingParameters::temperatures,
    &SamplingParameters::top_p,
    &SamplingParameters::top_k,
    &SamplingParameters::unique_token_ids,
    &SamplingParameters::unique_token_counts,
    &SamplingParameters::unique_token_ids_lens,
    &SamplingParameters::sample_idxes,
    &SamplingParameters::do_sample};

SamplingParameters make_input(int32_t rows, int32_t width) {
  SamplingParameters input;
  input.selected_token_idxes = torch::arange(rows, torch::kInt32);
  input.sample_idxes = torch::arange(rows, torch::kInt32);
  input.do_sample =
      torch::arange(rows, torch::kInt32).remainder(/*other=*/2) == 0;
  input.frequency_penalties = torch::full({rows}, /*fill_value=*/0.15f);
  input.presence_penalties = torch::full({rows}, /*fill_value=*/0.2f);
  input.repetition_penalties = torch::full({rows}, /*fill_value=*/1.15f);
  input.temperatures = torch::full({rows}, /*fill_value=*/0.85f);
  input.top_p = torch::full({rows}, /*fill_value=*/0.9f);
  input.top_k = torch::full({rows}, /*fill_value=*/16, torch::kInt64);
  input.unique_token_ids =
      torch::arange(width, torch::kInt64).expand({rows, width}).clone();
  input.unique_token_counts =
      torch::full({rows, width}, /*fill_value=*/2, torch::kInt32);
  input.unique_token_ids_lens = torch::full({rows}, width, torch::kInt32);
  input.unique_token_ids_lens[0] = width - 1;
  input.unique_token_ids[0][width - 1] = 0;
  input.unique_token_counts[0][width - 1] = 0;
  input.all_random_sample = input.do_sample.all().item<bool>();
  input.all_greedy_sample = !input.do_sample.any().item<bool>();
  input.logprobs = true;
  input.max_top_logprobs = 5;
  return input;
}

torch::Tensor rng_state() {
  auto generator = at_npu::detail::getDefaultNPUGenerator(/*device_index=*/0);
  std::lock_guard<std::mutex> lock(generator.mutex());
  return generator.get_state();
}

void set_rng_state(const torch::Tensor& state) {
  auto generator = at_npu::detail::getDefaultNPUGenerator(/*device_index=*/0);
  std::lock_guard<std::mutex> lock(generator.mutex());
  generator.set_state(state);
}

void expect_equal(const torch::Tensor& expected, const torch::Tensor& actual) {
  ASSERT_EQ(expected.defined(), actual.defined());
  if (expected.defined()) {
    EXPECT_EQ(expected.scalar_type(), actual.scalar_type());
    EXPECT_EQ(expected.sizes(), actual.sizes());
    EXPECT_TRUE(torch::equal(expected.cpu(), actual.cpu()));
  }
}

class SlotSamplingInputTest
    : public ::testing::TestWithParam<torch::ScalarType> {
 protected:
  void SetUp() override {
    capacity_.parameter_dtype = GetParam();
    ASSERT_TRUE(SlotBuffer::create(capacity_, device_, binding_).ok());
    prepare_ = std::make_unique<Stream>(device_);
    launch_ = std::make_unique<Stream>(device_);
    ready_ = std::make_shared<StreamEvent>(device_.type());
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  }

  void TearDown() override { EXPECT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS); }

  Status prepare_sampling(const SamplingParameters& input,
                          uint32_t tokens,
                          const Stream& stream) {
    InputData model =
        tokens == 0
            ? InputData{}
            : make_input(std::vector<int32_t>{static_cast<int32_t>(tokens)},
                         std::vector<int32_t>{static_cast<int32_t>(tokens)});
    if (tokens == 0) {
      model.width = 0;
    }
    const ModelInputBatch batch{
        tokens == 0 ? BatchForwardType::EMPTY : BatchForwardType::PREFILL,
        tokens == 0 ? 0U : 1U};
    Status status = binding_->validate(view(model), batch, input, 0, stream);
    if (status.ok()) {
      binding_->prepare(view(model), batch, input, stream);
    }
    return status;
  }

  void handoff() {
    prepare_->record_event(*ready_);
    ASSERT_TRUE(launch_->wait_event(ready_));
  }

  void expect_inputs(const SamplingParameters& reference) {
    auto guard = launch_->set_stream_guard();
    for (const auto member : kInputs) {
      expect_equal(reference.*member, binding_->sampling_params().*member);
    }
    EXPECT_EQ(reference.all_greedy_sample,
              binding_->sampling_params().all_greedy_sample);
    EXPECT_EQ(reference.all_random_sample,
              binding_->sampling_params().all_random_sample);
    EXPECT_EQ(reference.logprobs, binding_->sampling_params().logprobs);
    EXPECT_EQ(reference.return_probs, binding_->sampling_params().return_probs);
    EXPECT_EQ(reference.max_top_logprobs,
              binding_->sampling_params().max_top_logprobs);
    ASSERT_EQ(aclrtSynchronizeStream(launch_->get_stream()->stream()),
              ACL_SUCCESS);
  }

  void expect_rejected(const SamplingParameters& input, uint32_t tokens = 8) {
    const SamplingParameters old = binding_->sampling_params();
    const torch::Tensor model_before = binding_->tokens().cpu();
    std::array<torch::Tensor, kInputs.size()> values;
    auto guard = launch_->set_stream_guard();
    for (uint32_t index = 0; index < kInputs.size(); ++index) {
      const torch::Tensor& tensor = old.*kInputs[index];
      values[index] = tensor.defined() ? tensor.cpu() : torch::Tensor();
    }
    EXPECT_FALSE(prepare_sampling(input, tokens, *prepare_).ok());
    EXPECT_TRUE(torch::equal(model_before, binding_->tokens().cpu()));
    for (uint32_t index = 0; index < kInputs.size(); ++index) {
      const torch::Tensor& tensor = binding_->sampling_params().*kInputs[index];
      expect_equal(values[index], tensor);
      if (tensor.defined()) {
        EXPECT_EQ(tensor.data_ptr(), (old.*kInputs[index]).data_ptr());
      }
    }
  }

  const torch::Device device_{torch::kPrivateUse1, 0};
  SlotBufferCapacity capacity_{{8, 8, 2}, 16, 128, 8};
  std::unique_ptr<SlotBuffer> binding_;
  std::unique_ptr<Stream> prepare_;
  std::unique_ptr<Stream> launch_;
  std::shared_ptr<StreamEvent> ready_;
};

TEST_P(SlotSamplingInputTest, CpuMaskExportSurvivesCallerAndSlotReuse) {
  const uint64_t pinned_bytes = binding_->pinned_bytes();
  const uint64_t device_bytes = binding_->device_bytes();
  auto input = make_input(/*rows=*/3, /*width=*/2);
  auto expected = input.do_sample.clone();
  ASSERT_TRUE(prepare_sampling(input, /*model_tokens=*/8, *prepare_).ok());
  input.do_sample.zero_();
  auto retained = binding_->copy_cpu_do_sample();
  EXPECT_TRUE(torch::equal(retained, expected));
  EXPECT_TRUE(retained.device().is_cpu());
  EXPECT_FALSE(retained.is_pinned());
  EXPECT_NE(retained.data_ptr(), input.do_sample.data_ptr());
  auto independent = binding_->copy_cpu_do_sample();
  independent.zero_();
  EXPECT_TRUE(torch::equal(binding_->copy_cpu_do_sample(), expected));
  ASSERT_EQ(prepare_->synchronize(), ACL_SUCCESS);

  auto invalid = make_input(/*rows=*/1, /*width=*/1);
  invalid.do_sample = torch::zeros({1}, torch::kInt32);
  EXPECT_FALSE(prepare_sampling(invalid, /*model_tokens=*/8, *prepare_).ok());
  EXPECT_TRUE(torch::equal(binding_->copy_cpu_do_sample(), expected));
  SamplingParameters empty;
  empty.do_sample = torch::empty({0}, torch::kBool);
  ASSERT_TRUE(prepare_sampling(empty, /*model_tokens=*/0, *prepare_).ok());
  auto empty_mask = binding_->copy_cpu_do_sample();
  ASSERT_TRUE(empty_mask.defined());
  EXPECT_EQ(empty_mask.numel(), 0);
  EXPECT_FALSE(empty_mask.is_pinned());
  ASSERT_TRUE(prepare_sampling({}, /*model_tokens=*/0, *prepare_).ok());
  EXPECT_FALSE(binding_->copy_cpu_do_sample().defined());
  EXPECT_EQ(binding_->pinned_bytes(), pinned_bytes);
  EXPECT_EQ(binding_->device_bytes(), device_bytes);
  binding_.reset();
  EXPECT_TRUE(torch::equal(retained, expected));
}

TEST_P(SlotSamplingInputTest, FixedAddressesSurviveShapeAndFeatureChanges) {
  std::array<const void*, kInputs.size()> addresses{};
  for (int32_t round = 0; round < 16; ++round) {
    SCOPED_TRACE(round);
    SamplingParameters input = make_input(1 + round % 4, 1 + round % 7);
    if (round == 15) {
      input.top_k.fill_(/*value=*/-7);
    }
    const SamplingParameters reference = input.to(device_, GetParam());
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    input.all_random_sample = !input.all_random_sample;
    input.all_greedy_sample = !input.all_greedy_sample;
    ASSERT_TRUE(prepare_sampling(input, /*model_tokens=*/8, *prepare_).ok());
    for (uint32_t index = 0; index < kInputs.size(); ++index) {
      const torch::Tensor& tensor = binding_->sampling_params().*kInputs[index];
      if (round == 0) {
        addresses[index] = tensor.data_ptr();
      }
      EXPECT_EQ(addresses[index], tensor.data_ptr());
      EXPECT_TRUE(tensor.is_contiguous());
      (input.*kInputs[index]).zero_();
    }
    handoff();
    expect_inputs(reference);

    SamplingParameters greedy;
    greedy.selected_token_idxes = torch::tensor({0}, torch::kInt32);
    greedy.sample_idxes = torch::tensor({0}, torch::kInt32);
    greedy.do_sample = torch::tensor({false}, torch::kBool);
    ASSERT_TRUE(prepare_sampling(greedy, /*model_tokens=*/8, *prepare_).ok());
    handoff();
    expect_inputs(greedy.to(device_, GetParam()));
  }
}

TEST_P(SlotSamplingInputTest, InvalidInputPreservesPriorBindingAndValues) {
  SamplingParameters valid = make_input(/*rows=*/4, /*width=*/6);
  ASSERT_TRUE(prepare_sampling(valid, /*model_tokens=*/8, *prepare_).ok());
  handoff();
  expect_inputs(valid.to(device_, GetParam()));

  auto invalid = valid;
  invalid.selected_token_idxes = torch::tensor({0, 1, -1, 3}, torch::kInt32);
  expect_rejected(invalid);
  invalid.selected_token_idxes = torch::tensor({0, 1, 8, 3}, torch::kInt32);
  expect_rejected(invalid);
  invalid = valid;
  invalid.sample_idxes = torch::tensor({0, 2, 1, 3}, torch::kInt32);
  expect_rejected(invalid);
  invalid.sample_idxes = torch::tensor({0, 1, 1, 3}, torch::kInt32);
  expect_rejected(invalid);
  invalid = valid;
  invalid.temperatures = torch::tensor({1.0f, 1.0f, -1.0f, 1.0f});
  expect_rejected(invalid);
  invalid.temperatures =
      torch::full({4}, std::numeric_limits<float>::quiet_NaN());
  expect_rejected(invalid);
  invalid = valid;
  invalid.top_p = torch::full({4}, /*fill_value=*/1.1f);
  expect_rejected(invalid);
  invalid = valid;
  invalid.unique_token_ids =
      torch::full({4, 6}, /*fill_value=*/128, torch::kInt64);
  expect_rejected(invalid);
  invalid = valid;
  invalid.unique_token_counts =
      torch::full({4, 5}, /*fill_value=*/1, torch::kInt32);
  expect_rejected(invalid);
  invalid = valid;
  invalid.unique_token_ids_lens =
      torch::full({4}, /*fill_value=*/7, torch::kInt32);
  expect_rejected(invalid);
  invalid = valid;
  invalid.presence_penalties = torch::Tensor();
  expect_rejected(invalid);
  invalid = valid;
  invalid.do_sample = torch::Tensor();
  expect_rejected(invalid);
  invalid = valid;
  invalid.top_k = torch::zeros({4}, torch::kInt32);
  expect_rejected(invalid);
  invalid = valid;
  invalid.frequency_penalties =
      torch::zeros({4, 2}).select(/*dim=*/1, /*index=*/0);
  expect_rejected(invalid);
  invalid = valid;
  invalid.max_top_logprobs = 9;
  expect_rejected(invalid);
  invalid = valid;
  invalid.filter_bitmask = torch::zeros({4, 4}, torch::kInt32);
  expect_rejected(invalid);
  invalid = valid;
  invalid.is_embeddings = true;
  expect_rejected(invalid);
  expect_rejected(valid.to(device_, GetParam()));
  expect_rejected(valid, /*tokens=*/0);
  expect_rejected(make_input(/*rows=*/9, /*width=*/3));
  expect_rejected(make_input(/*rows=*/4, /*width=*/17));
}

TEST_P(SlotSamplingInputTest, EmptyClearsBindingAndTransfersNothing) {
  auto input = make_input(/*rows=*/4, /*width=*/6);
  ASSERT_TRUE(prepare_sampling(input, /*model_tokens=*/8, *prepare_).ok());
  handoff();
  expect_inputs(input.to(device_, GetParam()));
  ASSERT_TRUE(prepare_sampling({}, /*model_tokens=*/0, *prepare_).ok());
  for (const auto member : kInputs) {
    EXPECT_FALSE((binding_->sampling_params().*member).defined());
  }
  EXPECT_FALSE(binding_->sampling_params().logprobs);
  EXPECT_FALSE(binding_->sampling_params().return_probs);
  EXPECT_EQ(binding_->sampling_params().max_top_logprobs, 0);
}

TEST_P(SlotSamplingInputTest,
       CapacityRejectsBeforeReplacingOwnerAndAccountsBytes) {
  const uint64_t float_bytes = GetParam() == torch::kFloat32 ? 4 : 2;
  const uint64_t sampling_bytes =
      8 * (16 + 5 * float_bytes) + 8 * 16 * 12 + 8 * 5;
  // Seven model regions occupy 256 bytes; mapping uses 2x8 int64 indices
  // plus int64/int32 gather temporaries on device. These feed KV budgeting.
  const uint64_t result_bytes = 8 * (1 + 8) * (sizeof(int64_t) + sizeof(float));
  EXPECT_EQ(binding_->pinned_bytes(),
            256 + sampling_bytes + result_bytes + 128);
  EXPECT_EQ(binding_->device_bytes(),
            256 + sampling_bytes + result_bytes + 128 + 96);
  const SlotBuffer* original = binding_.get();
  std::vector<SlotBufferCapacity> invalid(8, capacity_);
  invalid[0].model.max_tokens = 0;
  invalid[1].model.max_sequences = 0;
  invalid[2].model.max_blocks_per_sequence = 0;
  invalid[3].max_unique_tokens = 0;
  invalid[4].max_top_logprobs = 129;
  invalid[5].parameter_dtype = torch::kFloat64;
  invalid[6].model = {1, 2147483647, 1};
  invalid[6].max_unique_tokens = 2147483647;
  invalid[7].model = {1, 4294967295U, 4294967295U};
  for (const auto& capacity : invalid) {
    EXPECT_FALSE(SlotBuffer::create(capacity, device_, binding_).ok());
    EXPECT_EQ(binding_.get(), original);
  }
  EXPECT_FALSE(
      SlotBuffer::create(capacity_, torch::Device(torch::kCPU), binding_).ok());
  EXPECT_FALSE(SlotBuffer::create(
                   capacity_, torch::Device(torch::kPrivateUse1), binding_)
                   .ok());
  EXPECT_EQ(binding_.get(), original);
}

TEST_P(SlotSamplingInputTest,
       RealSamplerMatchesTokensProbabilitiesAndRngProgress) {
  const torch::Tensor original_rng = rng_state();
  for (int32_t mode = 0; mode < 7; ++mode) {
    SCOPED_TRACE(mode);
    SamplingParameters input = make_input(/*rows=*/4, /*width=*/6);
    if (mode == 0 || mode == 5) {
      input = SamplingParameters();
      input.selected_token_idxes = torch::arange(/*end=*/4, torch::kInt32);
      input.sample_idxes = torch::arange(/*end=*/4, torch::kInt32);
      input.do_sample = torch::zeros({4}, torch::kBool);
      input.return_probs = mode == 5;
    } else if (mode == 1) {
      input.do_sample.zero_();
    } else if (mode == 6) {
      input.do_sample.fill_(/*value=*/true);
      input.top_k = torch::Tensor();
      input.top_p.zero_();
    } else if (mode == 3) {
      input.sample_idxes = torch::tensor({1, 3}, torch::kInt32);
      input.do_sample = torch::tensor({false, true}, torch::kBool);
      input.top_k = torch::Tensor();
    } else {
      input.do_sample.fill_(/*value=*/true);
      if (mode == 4) {
        input.top_p = torch::Tensor();
      }
    }
    input.all_random_sample = input.do_sample.all().item<bool>();
    input.all_greedy_sample = !input.do_sample.any().item<bool>();
    const SamplingParameters legacy = input.to(device_, GetParam());
    const torch::Tensor initial_logits =
        (torch::arange(/*end=*/4 * 128, torch::kFloat32)
                 .remainder(/*other=*/73) *
             0.037f -
         1.4f)
            .view({4, 128})
            .to(device_, GetParam());
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    auto guard = launch_->set_stream_guard();
    torch::Tensor old_logits = initial_logits.clone();
    const torch::Tensor start_rng = rng_state();
    const SampleOutput expected = Sampler().forward(old_logits, legacy);
    ASSERT_EQ(aclrtSynchronizeStream(launch_->get_stream()->stream()),
              ACL_SUCCESS);
    const torch::Tensor end_rng = rng_state();
    set_rng_state(start_rng);
    ASSERT_TRUE(prepare_sampling(input, /*model_tokens=*/8, *prepare_).ok());
    EXPECT_TRUE(torch::equal(start_rng, rng_state()));
    handoff();
    torch::Tensor new_logits = initial_logits.clone();
    const SampleOutput actual =
        Sampler().forward(new_logits, binding_->sampling_params());
    ASSERT_EQ(aclrtSynchronizeStream(launch_->get_stream()->stream()),
              ACL_SUCCESS);
    EXPECT_TRUE(torch::equal(end_rng, rng_state()));
    expect_equal(old_logits, new_logits);
    expect_equal(expected.next_tokens, actual.next_tokens);
    expect_equal(expected.probs, actual.probs);
    expect_equal(expected.logprobs, actual.logprobs);
    expect_equal(expected.top_logprobs, actual.top_logprobs);
    expect_equal(expected.top_tokens, actual.top_tokens);
  }
  set_rng_state(original_rng);
}

INSTANTIATE_TEST_SUITE_P(ParameterTypes,
                         SlotSamplingInputTest,
                         ::testing::Values(torch::kFloat16,
                                           torch::kBFloat16,
                                           torch::kFloat32));

constexpr std::array<torch::Tensor TokenResultTensors::*, 4> kResults = {
    &TokenResultTensors::tokens,
    &TokenResultTensors::logprobs,
    &TokenResultTensors::top_tokens,
    &TokenResultTensors::top_logprobs};

void expect_result(const TokenResultTensors& values,
                   const TokenResultTensors& actual) {
  for (const auto member : kResults) {
    const torch::Tensor& expected_field = values.*member;
    const torch::Tensor& actual_field = actual.*member;
    ASSERT_EQ(expected_field.defined(), actual_field.defined());
    if (!actual_field.defined()) {
      continue;
    }
    EXPECT_TRUE(actual_field.device().is_cpu());
    EXPECT_FALSE(actual_field.is_pinned());
    EXPECT_TRUE(actual_field.is_contiguous());
    EXPECT_EQ(actual_field.sizes(), expected_field.sizes());
    EXPECT_EQ(actual_field.scalar_type(), expected_field.scalar_type());
    EXPECT_TRUE(torch::equal(actual_field, expected_field));
  }
}

class SlotBufferResultTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(SlotBuffer::create(capacity_, device_, storage_).ok());
    producer_ = std::make_unique<Stream>(device_);
    copy_ = std::make_unique<Stream>(device_);
    producer_ready_ = std::make_shared<StreamEvent>(device_.type());
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  }

  void TearDown() override {
    storage_.reset();
    EXPECT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  }

  Status prepare_result(uint32_t rows,
                        uint32_t top = 0,
                        bool logprobs = false) {
    InputData input = rows == 0 ? InputData{}
                                : make_input(std::vector<int32_t>(rows, 1),
                                             std::vector<int32_t>(rows, 1));
    if (rows == 0) {
      input.width = 0;
    }
    SamplingParameters sampling;
    if (rows != 0) {
      sampling.selected_token_idxes = torch::arange(rows, torch::kInt32);
      sampling.sample_idxes = sampling.selected_token_idxes;
      sampling.do_sample = torch::zeros({rows}, torch::kBool);
      sampling.logprobs = logprobs;
      sampling.max_top_logprobs = top;
    }
    const ModelInputBatch batch{
        rows == 0 ? BatchForwardType::EMPTY : BatchForwardType::DECODE, rows};
    Status status =
        storage_->validate(view(input), batch, sampling, 0, *producer_);
    if (status.ok()) {
      storage_->prepare(view(input), batch, sampling, *producer_);
    }
    return status;
  }

  TokenResultTensors produce(int32_t seed) {
    auto guard = producer_->set_stream_guard();
    TokenResultTensors expected;
    for (const auto member : kResults) {
      const torch::Tensor& target = storage_->device_result().*member;
      if (!target.defined()) {
        continue;
      }
      torch::Tensor values;
      if (target.scalar_type() == torch::kInt64) {
        values = (torch::arange(target.numel(), torch::kInt64) + seed)
                     .view(target.sizes());
      } else {
        values =
            (torch::arange(target.numel(), torch::kFloat32) * -0.125f - seed)
                .view(target.sizes());
      }
      target.copy_(values);
      expected.*member = std::move(values);
    }
    // A final Device write remains ordered by producer_ready, beyond the CPU
    // fixture transfers. The copy stream must observe this updated value.
    storage_->device_result().tokens.add_(/*other=*/17);
    expected.tokens.add_(/*other=*/17);
    record_producer();
    return expected;
  }

  void record_producer() { producer_->record_event(*producer_ready_); }

  void expect_producer_complete() {
    aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
    ASSERT_EQ(aclrtQueryEventStatus(producer_ready_->npu_event(), &status),
              ACL_SUCCESS);
    EXPECT_EQ(status, ACL_EVENT_RECORDED_STATUS_COMPLETE);
  }

  const torch::Device device_{torch::kPrivateUse1, 0};
  SlotBufferCapacity capacity_{{16, 4, 5}, 16, 1024, 5};
  std::unique_ptr<SlotBuffer> storage_;
  std::unique_ptr<Stream> producer_;
  std::unique_ptr<Stream> copy_;
  StreamEventPtr producer_ready_;
};

TEST_F(SlotBufferResultTest, SingleTokenResultsDetachCpuStorage) {
  ASSERT_TRUE(prepare_result(3, 3, true).ok());
  const TokenResultTensors expected = produce(/*seed=*/100);
  ASSERT_TRUE(storage_->copy_result_to_host(*copy_, producer_ready_).ok());
  const TokenResultTensors result = storage_->take_result();
  expect_producer_complete();
  storage_.reset();
  expect_result(expected, result);
}

TEST_F(SlotBufferResultTest, FixedAddressesAcrossShapeAndFeatureChanges) {
  ASSERT_TRUE(prepare_result(4, 5, true).ok());
  ASSERT_EQ(producer_->synchronize(), 0);
  std::array<const void*, kResults.size()> addresses{};
  for (uint32_t i = 0; i < kResults.size(); ++i) {
    addresses[i] = (storage_->device_result().*kResults[i]).data_ptr();
  }
  std::vector<TokenResultTensors> results;
  std::vector<TokenResultTensors> expected;
  results.reserve(/*new_cap=*/16);
  expected.reserve(/*new_cap=*/16);
  for (uint32_t round = 0; round < 16; ++round) {
    SCOPED_TRACE(round);
    const uint32_t rows = 1 + round % 4;
    const bool logprobs = round % 3 != 0;
    const uint32_t top = logprobs ? round % 5 : 0;
    ASSERT_TRUE(prepare_result(rows, top, logprobs).ok());
    for (uint32_t i = 0; i < kResults.size(); ++i) {
      const torch::Tensor& tensor = storage_->device_result().*kResults[i];
      if (tensor.defined()) {
        EXPECT_EQ(tensor.data_ptr(), addresses[i]);
        EXPECT_TRUE(tensor.is_contiguous());
      }
    }
    expected.emplace_back(produce(static_cast<int32_t>(100 * round)));
    ASSERT_TRUE(storage_->copy_result_to_host(*copy_, producer_ready_).ok());
    results.emplace_back(storage_->take_result());
    expect_result(expected.back(), results.back());
  }
  storage_.reset();
  for (uint32_t index = 0; index < results.size(); ++index) {
    expect_result(expected[index], results[index]);
  }
}

TEST_F(SlotBufferResultTest, PendingResultRejectsInputReuseWithoutWrites) {
  ASSERT_TRUE(prepare_result(2, 2, true).ok());
  const TokenResultTensors expected = produce(/*seed=*/23);
  ASSERT_TRUE(storage_->copy_result_to_host(*copy_, producer_ready_).ok());
  const void* tokens = storage_->device_result().tokens.data_ptr();
  EXPECT_EQ(prepare_result(1).code(), StatusCode::RESOURCE_EXHAUSTED);
  EXPECT_EQ(storage_->tokens().numel(), 2);
  EXPECT_EQ(storage_->device_result().tokens.data_ptr(), tokens);
  EXPECT_EQ(storage_->device_result().tokens.numel(), 2);
  expect_result(expected, storage_->take_result());
  ASSERT_TRUE(prepare_result(1).ok());
  const TokenResultTensors next = produce(/*seed=*/321);
  ASSERT_TRUE(storage_->copy_result_to_host(*copy_, producer_ready_).ok());
  expect_result(next, storage_->take_result());
}

TEST_F(SlotBufferResultTest, ZeroTopCapacityAndOptionalLogprobs) {
  capacity_.max_top_logprobs = 0;
  ASSERT_TRUE(SlotBuffer::create(capacity_, device_, storage_).ok());
  ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  ASSERT_TRUE(prepare_result(2, 0, true).ok());
  const TokenResultTensors expected = produce(/*seed=*/9);
  ASSERT_TRUE(storage_->copy_result_to_host(*copy_, producer_ready_).ok());
  expect_result(expected, storage_->take_result());
  ASSERT_TRUE(prepare_result(2, 0, false).ok());
  const TokenResultTensors greedy = produce(/*seed=*/11);
  ASSERT_TRUE(storage_->copy_result_to_host(*copy_, producer_ready_).ok());
  expect_result(greedy, storage_->take_result());
}

TEST_F(SlotBufferResultTest, EmptyResultStillRetiresProducerWork) {
  ASSERT_TRUE(prepare_result(0).ok());
  auto guard = producer_->set_stream_guard();
  torch::Tensor scratch =
      torch::ones({1024 * 1024}, torch::TensorOptions().device(device_));
  for (uint32_t i = 0; i < 32; ++i) {
    scratch.sin_();
  }
  record_producer();
  ASSERT_TRUE(storage_->copy_result_to_host(*copy_, producer_ready_).ok());
  const TokenResultTensors result = storage_->take_result();
  for (const auto member : kResults) {
    EXPECT_FALSE((result.*member).defined());
    EXPECT_FALSE((storage_->device_result().*member).defined());
  }
  expect_producer_complete();
}

TEST_F(SlotBufferResultTest, DiscardAndDestructionRetireUnclaimedCopies) {
  for (uint32_t round = 0; round < 4; ++round) {
    ASSERT_TRUE(prepare_result(4, 5, true).ok());
    produce(static_cast<int32_t>(round));
    auto guard = producer_->set_stream_guard();
    torch::Tensor scratch =
        torch::ones({1024 * 1024}, torch::TensorOptions().device(device_));
    for (uint32_t i = 0; i < 32; ++i) {
      scratch.sin_();
    }
    record_producer();
    ASSERT_TRUE(storage_->copy_result_to_host(*copy_, producer_ready_).ok());
    if (round == 3) {
      storage_.reset();
    } else {
      storage_->discard_result();
      storage_->discard_result();
    }
    expect_producer_complete();
  }
}

}  // namespace
}  // namespace xllm
