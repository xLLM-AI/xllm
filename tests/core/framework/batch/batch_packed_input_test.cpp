/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
#include <unistd.h>

#if defined(USE_CUDA)
#include <torch/cuda.h>
#endif

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/batch/forward_input_builder.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/model/dit_model_params.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "core/kv_cache/block/block_manager_impl.h"
#include "core/runtime/dit_forward_params.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/forward_shared_memory_manager.h"
#include "core/runtime/rec_forward_params.h"
#include "core/runtime/vlm_forward_params.h"
#include "worker.pb.h"

namespace xllm {

namespace {

static_assert(!std::is_same_v<decltype(LlmAttentionInput::host),
                              decltype(VlmAttentionInput::host)> &&
                  !std::is_same_v<decltype(LlmAttentionInput::host),
                                  decltype(RecAttentionInput::host)> &&
                  !std::is_same_v<decltype(VlmAttentionInput::host),
                                  decltype(RecAttentionInput::host)> &&
                  !std::is_same_v<decltype(LlmAttentionInput::device),
                                  decltype(VlmAttentionInput::device)> &&
                  !std::is_same_v<decltype(LlmAttentionInput::device),
                                  decltype(RecAttentionInput::device)> &&
                  !std::is_same_v<decltype(VlmAttentionInput::device),
                                  decltype(RecAttentionInput::device)>,
              "Each domain must declare its own attention storage types");
static_assert(
    !std::is_assignable_v<decltype(LlmAttentionInput::host)&,
                          const decltype(VlmAttentionInput::host)&> &&
        !std::is_assignable_v<decltype(VlmAttentionInput::host)&,
                              const decltype(LlmAttentionInput::host)&> &&
        !std::is_assignable_v<decltype(LlmAttentionInput::host)&,
                              const decltype(RecAttentionInput::host)&> &&
        !std::is_assignable_v<decltype(RecAttentionInput::host)&,
                              const decltype(LlmAttentionInput::host)&> &&
        !std::is_assignable_v<decltype(VlmAttentionInput::host)&,
                              const decltype(RecAttentionInput::host)&> &&
        !std::is_assignable_v<decltype(RecAttentionInput::host)&,
                              const decltype(VlmAttentionInput::host)&> &&
        !std::is_assignable_v<decltype(LlmAttentionInput::device)&,
                              const decltype(VlmAttentionInput::device)&> &&
        !std::is_assignable_v<decltype(VlmAttentionInput::device)&,
                              const decltype(LlmAttentionInput::device)&> &&
        !std::is_assignable_v<decltype(LlmAttentionInput::device)&,
                              const decltype(RecAttentionInput::device)&> &&
        !std::is_assignable_v<decltype(RecAttentionInput::device)&,
                              const decltype(LlmAttentionInput::device)&> &&
        !std::is_assignable_v<decltype(VlmAttentionInput::device)&,
                              const decltype(RecAttentionInput::device)&> &&
        !std::is_assignable_v<decltype(RecAttentionInput::device)&,
                              const decltype(VlmAttentionInput::device)&>,
    "Cross-domain attention copies must be explicit");

class ScopedContiguousInputBuffer final {
 public:
  explicit ScopedContiguousInputBuffer(bool enabled)
      : previous_enabled_(
            ExecutionConfig::get_instance().use_contiguous_input_buffer()) {
    ExecutionConfig::get_instance().use_contiguous_input_buffer(enabled);
  }

  ~ScopedContiguousInputBuffer() {
    ExecutionConfig::get_instance().use_contiguous_input_buffer(
        previous_enabled_);
  }

 private:
  bool previous_enabled_;
};

template <typename Input>
void expect_input_read_cancellation(const std::string& suffix) {
  ScopedContiguousInputBuffer contiguous_input_buffer(/*enabled=*/false);
  const std::string name =
      "shm_input_cancel_" + std::to_string(getpid()) + "_" + suffix;
  bool is_creator = false;
  ForwardSharedMemoryManager reader(
      name, /*size=*/1 << 20, is_creator, ForwardType::RAW_INPUT);
  auto read_input = [&reader]() {
    Input input;
    if constexpr (std::is_same_v<Input, DiTForwardInput>) {
      return reader.input_read(input);
    } else {
      return reader.input_read(input, torch::Device(torch::kCPU));
    }
  };
  auto result = std::async(std::launch::async, read_input);
  EXPECT_EQ(result.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  reader.stop_input_read();
  reader.stop_input_read();
  ASSERT_EQ(result.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  EXPECT_FALSE(result.get());
  EXPECT_FALSE(read_input());
}

TEST(ForwardSharedMemoryShutdownTest, CancelsIdleReadsForEveryInputDomain) {
  expect_input_read_cancellation<LlmForwardInput>("llm");
  expect_input_read_cancellation<VlmForwardInput>("vlm");
  expect_input_read_cancellation<RecForwardInput>("rec");
  expect_input_read_cancellation<DiTForwardInput>("dit");
}

TEST(ForwardSharedMemoryShutdownTest,
     CancellationDoesNotPublishOrConsumeInput) {
  ScopedContiguousInputBuffer contiguous_input_buffer(/*enabled=*/false);
  const std::string name = "shm_input_local_stop_" + std::to_string(getpid());
  bool is_creator = false;
  ForwardSharedMemoryManager writer(
      name, /*size=*/1 << 20, is_creator, ForwardType::RAW_INPUT);
  ForwardSharedMemoryManager stopped_reader(
      name, /*size=*/1 << 20, is_creator, ForwardType::RAW_INPUT);
  ForwardSharedMemoryManager active_reader(
      name, /*size=*/1 << 20, is_creator, ForwardType::RAW_INPUT);
  stopped_reader.stop_input_read();
  DiTForwardInput source;
  source.model_params.batch_size = 1;
  source.model_params.prompts = {
      "Keep the pending request for active readers."};
  ASSERT_TRUE(writer.input_write(source));
  DiTForwardInput cancelled;
  cancelled.model_params.batch_size = 7;
  EXPECT_FALSE(stopped_reader.input_read(cancelled));
  EXPECT_EQ(cancelled.model_params.batch_size, 7);
  DiTForwardInput received;
  ASSERT_TRUE(active_reader.input_read(received));
  EXPECT_EQ(received.model_params.batch_size, source.model_params.batch_size);
  EXPECT_EQ(received.model_params.prompts, source.model_params.prompts);
}

void expect_linear_state_cache_op_eq(const LinearStateCacheOp& actual,
                                     const LinearStateCacheOp& expected) {
  EXPECT_EQ(actual.linear_state_id, expected.linear_state_id);
  EXPECT_EQ(actual.reset_requested, expected.reset_requested);
  EXPECT_EQ(actual.restore_requested, expected.restore_requested);
  EXPECT_EQ(actual.restore_src_slot_id, expected.restore_src_slot_id);
}

void expect_dit_forward_input_eq(const DiTForwardInput& actual,
                                 const DiTForwardInput& expected) {
  EXPECT_EQ(actual.model_params.batch_size, expected.model_params.batch_size);
  EXPECT_EQ(actual.model_params.prompts, expected.model_params.prompts);
  EXPECT_EQ(actual.model_params.prompts_2, expected.model_params.prompts_2);
  EXPECT_EQ(actual.model_params.negative_prompts,
            expected.model_params.negative_prompts);
  EXPECT_EQ(actual.model_params.negative_prompts_2,
            expected.model_params.negative_prompts_2);
  EXPECT_EQ(actual.model_params.audio_prompt_text,
            expected.model_params.audio_prompt_text);
  EXPECT_EQ(actual.model_params.generation_params,
            expected.model_params.generation_params);

  ASSERT_EQ(actual.model_params.media_sources.size(),
            expected.model_params.media_sources.size());
  for (size_t index = 0; index < expected.model_params.media_sources.size();
       ++index) {
    const MediaNamedTensor& actual_source =
        actual.model_params.media_sources.at(index);
    const MediaNamedTensor& expected_source =
        expected.model_params.media_sources.at(index);
    EXPECT_EQ(actual_source.name, expected_source.name);
    EXPECT_EQ(actual_source.modality, expected_source.modality);
    EXPECT_TRUE(actual_source.tensor.device().is_cpu());
    EXPECT_EQ(actual_source.tensor.scalar_type(),
              expected_source.tensor.scalar_type());
    EXPECT_TRUE(torch::equal(actual_source.tensor, expected_source.tensor));
  }

  ASSERT_EQ(actual.model_params.tensor_sources.size(),
            expected.model_params.tensor_sources.size());
  for (size_t index = 0; index < expected.model_params.tensor_sources.size();
       ++index) {
    const NamedTensor& actual_source =
        actual.model_params.tensor_sources.entries()[index];
    const NamedTensor& expected_source =
        expected.model_params.tensor_sources.entries()[index];
    EXPECT_EQ(actual_source.name, expected_source.name);
    EXPECT_TRUE(actual_source.tensor.device().is_cpu());
    EXPECT_EQ(actual_source.tensor.scalar_type(),
              expected_source.tensor.scalar_type());
    EXPECT_TRUE(torch::equal(actual_source.tensor, expected_source.tensor));
  }
}

void overwrite_packed_uint64(std::string& payload,
                             size_t offset,
                             uint64_t value) {
  for (size_t index = 0; index < sizeof(value); ++index) {
    payload.at(offset + index) =
        static_cast<char>((value >> (index * 8)) & 0xff);
  }
}

uint64_t read_packed_uint64(const std::string& payload, size_t offset) {
  uint64_t value = 0;
  for (size_t index = 0; index < sizeof(value); ++index) {
    value |=
        static_cast<uint64_t>(static_cast<uint8_t>(payload.at(offset + index)))
        << (index * 8);
  }
  return value;
}

RecForwardInput make_rec_transport_input() {
  RecForwardInput input;
  input.token_ids = torch::tensor({11, 23}, torch::kInt32);
  input.positions = torch::tensor({0, 1}, torch::kInt32);
  input.input_params.meta.num_sequences = 1;
  input.input_params.meta.batch_id = 31;
  input.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  input.input_params.attention.host.q_seq_lens = {2};
  input.input_params.attention.host.q_cu_seq_lens = {0, 2};
  input.input_params.attention.host.kv_seq_lens = {2};
  input.input_params.attention.host.new_cache_slots = {3, 4};
  input.input_params.attention.host.block_tables =
      torch::tensor({{7, 9}}, torch::kInt32);
  input.input_params.features.mm_data.batch({MMData(
      MMType::EMBEDDING,
      MMDict{{"MULTI_MODAL_VALUES", torch::tensor({{1.5F, 2.5F}})},
             {"MULTI_MODAL_INDICES", torch::tensor({1}, torch::kInt64)}})});
  input.sampling_params.selected_token_idxes =
      torch::tensor({1}, torch::kInt32);
  input.sample_sequence_ids = {"rec#0"};
  input.sample_prior_output_rows = {-1};
  return input;
}

void expect_rec_transport_input(const RecForwardInput& input) {
  EXPECT_TRUE(
      torch::equal(input.token_ids, torch::tensor({11, 23}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(input.positions,
                           torch::tensor({0, 1}, input.positions.options())));
  EXPECT_EQ(input.input_params.meta.batch_id, 31);
  EXPECT_EQ(input.input_params.attention.host.kv_seq_lens,
            (std::vector<int32_t>{2}));
  EXPECT_TRUE(torch::equal(input.sampling_params.selected_token_idxes,
                           torch::tensor({1}, torch::kInt32)));
  EXPECT_EQ(input.sample_sequence_ids, (std::vector<std::string>{"rec#0"}));
  EXPECT_EQ(input.sample_prior_output_rows, (std::vector<int32_t>{-1}));
  const auto values = input.input_params.features.mm_data.get<torch::Tensor>(
      "MULTI_MODAL_VALUES");
  const auto indices = input.input_params.features.mm_data.get<torch::Tensor>(
      "MULTI_MODAL_INDICES");
  ASSERT_TRUE(values.has_value());
  ASSERT_TRUE(indices.has_value());
  EXPECT_TRUE(torch::equal(*values, torch::tensor({{1.5F, 2.5F}})));
  EXPECT_TRUE(torch::equal(*indices, torch::tensor({1}, torch::kInt64)));
  EXPECT_EQ(indices->scalar_type(), torch::kInt64);
  EXPECT_FALSE(input.has_step_meta());
  EXPECT_FALSE(input.input_params.has_onerec_params());
  EXPECT_FALSE(input.input_params.has_llmrec_params());
}

VlmForwardInput make_vlm_transport_input() {
  VlmForwardInput input;
  input.token_ids = torch::tensor({11, 23}, torch::kInt32);
  input.positions = torch::tensor({{0, 1}, {2, 3}, {4, 5}}, torch::kInt32);
  input.input_params.meta.num_sequences = 1;
  input.input_params.meta.batch_id = 47;
  input.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  input.input_params.attention.host.q_seq_lens = {2};
  input.input_params.attention.host.q_cu_seq_lens = {0, 2};
  input.input_params.attention.host.kv_seq_lens = {2};
  input.input_params.attention.host.new_cache_slots = {3, 4};
  input.input_params.attention.host.block_tables =
      torch::tensor({{7, 9}}, torch::kInt32);
  input.input_params.multimodal.mm_data.batch({MMData(
      MMType::IMAGE,
      MMDict{{"pixel_values", torch::tensor({{1.5F, 2.5F}})},
             {"image_grid_thw", torch::tensor({{1, 2, 2}}, torch::kInt64)}})});
  input.input_params.multimodal.deep_stacks = {torch::tensor({{3.5F, 4.5F}}),
                                               torch::tensor({{5.5F, 6.5F}})};
  input.sampling_params.selected_token_idxes =
      torch::tensor({1}, torch::kInt32);
  input.sample_sequence_ids = {"vlm#0"};
  input.sample_prior_output_rows = {-1};
  return input;
}

void expect_vlm_transport_input(const VlmForwardInput& input) {
  EXPECT_TRUE(
      torch::equal(input.token_ids, torch::tensor({11, 23}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      input.positions,
      torch::tensor({{0, 1}, {2, 3}, {4, 5}}, input.positions.options())));
  EXPECT_EQ(input.input_params.meta.batch_id, 47);
  EXPECT_EQ(input.input_params.attention.host.kv_seq_lens,
            (std::vector<int32_t>{2}));
  EXPECT_EQ(input.sample_sequence_ids, (std::vector<std::string>{"vlm#0"}));
  EXPECT_EQ(input.sample_prior_output_rows, (std::vector<int32_t>{-1}));
  const auto pixels =
      input.input_params.multimodal.mm_data.get<torch::Tensor>("pixel_values");
  const auto grid = input.input_params.multimodal.mm_data.get<torch::Tensor>(
      "image_grid_thw");
  ASSERT_TRUE(pixels.has_value());
  ASSERT_TRUE(grid.has_value());
  EXPECT_TRUE(torch::equal(*pixels, torch::tensor({{1.5F, 2.5F}})));
  EXPECT_TRUE(torch::equal(*grid, torch::tensor({{1, 2, 2}}, torch::kInt64)));
  ASSERT_EQ(input.input_params.multimodal.deep_stacks.size(), 2u);
  EXPECT_TRUE(torch::equal(input.input_params.multimodal.deep_stacks[0],
                           torch::tensor({{3.5F, 4.5F}})));
  EXPECT_TRUE(torch::equal(input.input_params.multimodal.deep_stacks[1],
                           torch::tensor({{5.5F, 6.5F}})));
}

template <typename Input>
class NoncontiguousSharedMemoryInputTest : public ::testing::Test {};

using TokenInputTypes =
    ::testing::Types<LlmForwardInput, RecForwardInput, VlmForwardInput>;
TYPED_TEST_SUITE(NoncontiguousSharedMemoryInputTest, TokenInputTypes);

TYPED_TEST(NoncontiguousSharedMemoryInputTest,
           RetainsTensorsAfterOverwriteAndUnmapping) {
  ScopedContiguousInputBuffer contiguous_input_buffer(/*enabled=*/false);
  std::vector<torch::Device> devices{torch::Device(torch::kCPU)};
  devices.reserve(2);
#if defined(USE_CUDA)
  if (torch::cuda::is_available()) {
    devices.emplace_back(torch::kCUDA, 0);
  }
#elif defined(USE_NPU)
  devices.emplace_back(torch::kPrivateUse1, 0);
#endif
  for (const auto& device : devices) {
    SCOPED_TRACE(device.str());
    for (const InputDeviceMaterializationPolicy policy :
         {InputDeviceMaterializationPolicy::MATERIALIZE_ON_READ,
          InputDeviceMaterializationPolicy::DEFER_TO_WORKER_PREPARE}) {
      SCOPED_TRACE(static_cast<int32_t>(policy));
      TypeParam source;
      source.token_ids = torch::tensor({11, 23}, torch::kInt32);
      source.positions = torch::tensor({0, 1}, torch::kInt32);
      source.input_params.meta.num_sequences = 1;
      source.input_params.attention.host.q_seq_lens = {2};
      source.input_params.attention.host.q_cu_seq_lens = {0, 2};
      source.input_params.attention.host.kv_seq_lens = {2};
      source.sampling_params.selected_token_idxes =
          torch::tensor({1}, torch::kInt32);
      TypeParam retained;
      {
        const std::string shm_name =
            ForwardSharedMemoryManager::create_unique_name(
                "batch_test_noncontiguous_input",
                /*dp_group=*/0,
                ForwardType::RAW_INPUT,
                /*rank=*/0);
        bool is_creator = false;
        ForwardSharedMemoryManager writer(
            shm_name, /*size=*/1 << 20, is_creator, ForwardType::RAW_INPUT);
        bool is_reader_creator = false;
        ForwardSharedMemoryManager reader(shm_name,
                                          /*size=*/1 << 20,
                                          is_reader_creator,
                                          ForwardType::RAW_INPUT);
        ASSERT_TRUE(writer.input_write(source));
        ASSERT_TRUE(reader.input_read(retained, device, policy));
        // Check ownership before unmapping so a regression fails safely.
        ASSERT_TRUE(retained.runtime.input_host_buffer.defined());
        EXPECT_FALSE(retained.runtime.device_input_buffer.defined());

        auto overwrite = source.clone();
        overwrite.token_ids = torch::tensor({91, 92}, torch::kInt32);
        overwrite.positions = torch::tensor({4, 5}, torch::kInt32);
        overwrite.sampling_params.selected_token_idxes =
            torch::tensor({0}, torch::kInt32);
        ASSERT_TRUE(writer.input_write(overwrite));
        ASSERT_TRUE(torch::equal(retained.token_ids, source.token_ids));
        ASSERT_TRUE(torch::equal(retained.positions, source.positions));
        ASSERT_TRUE(torch::equal(retained.sampling_params.selected_token_idxes,
                                 source.sampling_params.selected_token_idxes));
      }
      EXPECT_TRUE(torch::equal(retained.token_ids_host, source.token_ids));
      EXPECT_TRUE(torch::equal(retained.positions_host, source.positions));
      EXPECT_TRUE(torch::equal(retained.token_ids, source.token_ids));
      EXPECT_TRUE(torch::equal(retained.positions, source.positions));
      EXPECT_TRUE(torch::equal(retained.sampling_params.selected_token_idxes,
                               source.sampling_params.selected_token_idxes));
    }
  }
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

TEST(BatchPackedInputTest, PackedCopyKeepsStagingAliveAfterSourceRelease) {
  ScopedContiguousInputBuffer contiguous_input_buffer(/*enabled=*/false);
  LlmForwardInput lazy_input;
  {
    LlmForwardInput source;
    source.token_ids = torch::tensor({11, 23}, torch::kInt32);
    source.positions = torch::tensor({0, 1}, torch::kInt32);
    source.input_params.meta.num_sequences = 1;
    source.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
    source.input_params.attention.host.q_seq_lens = {2};
    source.input_params.attention.host.q_cu_seq_lens = {0, 2};
    source.input_params.attention.host.kv_seq_lens = {2};

    proto::PackedForwardInput packed_input;
    ASSERT_TRUE(source.to_packed_proto(&packed_input));
    ASSERT_TRUE(lazy_input.from_packed_proto(packed_input));
  }

  ASSERT_TRUE(lazy_input.runtime.input_host_buffer.defined());
  const void* staging_data = lazy_input.runtime.input_host_buffer.data_ptr();
  LlmForwardInput copied_input = lazy_input.clone();
  lazy_input = LlmForwardInput();

  LlmForwardInput materialized_input =
      copied_input.to(torch::Device(torch::kCPU), torch::kFloat32);
  copied_input = LlmForwardInput();

  EXPECT_EQ(materialized_input.runtime.input_host_buffer.data_ptr(),
            staging_data);
  EXPECT_FALSE(materialized_input.runtime.input_host_buffer_has_layout);
  EXPECT_TRUE(materialized_input.runtime.device_tensors_ready);
  EXPECT_TRUE(materialized_input.token_ids.device().is_cpu());
  EXPECT_TRUE(
      tensor_equals_vector<int32_t>(materialized_input.token_ids, {11, 23}));
  EXPECT_TRUE(
      tensor_equals_vector<int32_t>(materialized_input.positions, {0, 1}));
  EXPECT_EQ(materialized_input.input_params.attention.host.kv_seq_lens,
            std::vector<int32_t>({2}));
}

#if defined(USE_NPU)
namespace {

template <typename Input>
void expect_preparation_reuses_only_pinned_payload() {
  ScopedContiguousInputBuffer contiguous_input_buffer(/*enabled=*/true);
  const torch::Device device(torch::kPrivateUse1, 0);
  Input source;
  source.token_ids = torch::tensor({11, 23}, torch::kInt32);
  source.positions = torch::tensor({0, 1}, torch::kInt32);
  source.input_params.multi_block_tables = {
      torch::tensor({{3, 7}}, torch::kInt32)};
  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(source.to_packed_proto(&packed_input));
  Input lazy_input;
  ASSERT_TRUE(lazy_input.from_packed_proto(packed_input));
  ASSERT_TRUE(lazy_input.runtime.input_host_buffer.is_pinned());
  const void* pinned_data = lazy_input.runtime.input_host_buffer.data_ptr();
  Input prepared = lazy_input.to(device, torch::kFloat32);
  EXPECT_EQ(prepared.runtime.input_host_buffer.data_ptr(), pinned_data);
  lazy_input = Input();
  EXPECT_TRUE(torch::equal(prepared.token_ids.cpu(), source.token_ids));
  EXPECT_TRUE(torch::equal(prepared.positions_host, source.positions));

  Input unpinned_input;
  ASSERT_TRUE(unpinned_input.from_packed_proto(packed_input));
  torch::Tensor unpinned_buffer =
      torch::empty(unpinned_input.runtime.input_host_buffer.sizes(),
                   torch::TensorOptions().dtype(torch::kUInt8));
  unpinned_buffer.copy_(unpinned_input.runtime.input_host_buffer);
  ASSERT_FALSE(unpinned_buffer.is_pinned());
  unpinned_input.runtime.input_host_buffer = std::move(unpinned_buffer);
  const void* unpinned_data =
      unpinned_input.runtime.input_host_buffer.data_ptr();
  Input staged = unpinned_input.to(device, torch::kFloat32);
  EXPECT_NE(staged.runtime.input_host_buffer.data_ptr(), unpinned_data);
  EXPECT_TRUE(staged.runtime.input_host_buffer.is_pinned());
  unpinned_input = Input();
  EXPECT_TRUE(torch::equal(staged.token_ids.cpu(), source.token_ids));
  EXPECT_TRUE(torch::equal(staged.positions_host, source.positions));
  ASSERT_EQ(staged.input_params.multi_block_tables.size(), 1u);
  EXPECT_TRUE(torch::equal(staged.input_params.multi_block_tables[0],
                           source.input_params.multi_block_tables[0]));
}

}  // namespace

TEST(BatchPackedInputTest, LlmPreparationReusesOnlyPinnedPayload) {
  expect_preparation_reuses_only_pinned_payload<LlmForwardInput>();
}

TEST(BatchPackedInputTest, VlmPreparationReusesOnlyPinnedPayload) {
  expect_preparation_reuses_only_pinned_payload<VlmForwardInput>();
}

TEST(BatchPackedInputTest, MaterializedShmReadRebindsTaggedTensorArena) {
  ScopedContiguousInputBuffer contiguous_input_buffer(/*enabled=*/true);
  const torch::Device device(torch::kPrivateUse1, 0);
  LlmForwardInput source;
  source.token_ids = torch::tensor({11, 23}, torch::kInt32);
  source.positions = torch::tensor({0, 1}, torch::kInt32);
  source.input_params.meta.num_sequences = 1;
  source.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  source.input_params.attention.host.q_seq_lens = {2};
  source.input_params.attention.host.q_cu_seq_lens = {0, 2};
  source.input_params.attention.host.kv_seq_lens = {2};
  source.input_params.attention.host.new_cache_slots = {3, 4};
  source.input_params.attention.host.block_tables =
      torch::tensor({{7, 9}}, torch::kInt32);
  source.sampling_params.selected_token_idxes =
      torch::tensor({1}, torch::kInt32);
  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(source.to_packed_proto(&packed_input));
  ASSERT_GE(packed_input.payload().size(), 40u);
  EXPECT_EQ(static_cast<uint8_t>(packed_input.payload()[8]), 7u);
  EXPECT_EQ(static_cast<uint8_t>(packed_input.payload()[10]), 1u);
  const uint64_t arena_offset =
      read_packed_uint64(packed_input.payload(), /*offset=*/24);

  LlmForwardInput materialized_input;
  {
    const std::string shm_name = ForwardSharedMemoryManager::create_unique_name(
        "batch_test_materialized_tagged_input",
        /*dp_group=*/0,
        ForwardType::RAW_INPUT,
        /*rank=*/0);
    bool is_creator = false;
    ForwardSharedMemoryManager writer_manager(shm_name,
                                              /*size=*/1 << 20,
                                              is_creator,
                                              ForwardType::RAW_INPUT);
    bool is_reader_creator = false;
    ForwardSharedMemoryManager reader_manager(shm_name,
                                              /*size=*/1 << 20,
                                              is_reader_creator,
                                              ForwardType::RAW_INPUT);
    ASSERT_TRUE(writer_manager.input_write(source));
    reader_manager.input_read(
        materialized_input,
        device,
        InputDeviceMaterializationPolicy::MATERIALIZE_ON_READ);

    ASSERT_TRUE(materialized_input.runtime.input_host_buffer.defined());
    EXPECT_TRUE(materialized_input.runtime.input_host_buffer.is_pinned());
    ASSERT_TRUE(materialized_input.runtime.device_input_buffer.defined());
    EXPECT_EQ(materialized_input.runtime.device_input_buffer.device(), device);
    EXPECT_TRUE(materialized_input.runtime.device_tensors_ready);
    const uint8_t* retained_arena =
        materialized_input.runtime.input_host_buffer.data_ptr<uint8_t>() +
        arena_offset;
    EXPECT_EQ(materialized_input.token_ids_host.data_ptr(), retained_arena);
    EXPECT_EQ(materialized_input.token_ids.device(), device);
    EXPECT_EQ(materialized_input.positions.device(), device);

    LlmForwardInput overwrite_input = source.clone();
    overwrite_input.token_ids = torch::tensor({91, 92}, torch::kInt32);
    overwrite_input.positions = torch::tensor({4, 5}, torch::kInt32);
    ASSERT_TRUE(writer_manager.input_write(overwrite_input));
  }

  EXPECT_TRUE(tensor_equals_vector<int32_t>(materialized_input.token_ids_host,
                                            {11, 23}));
  EXPECT_TRUE(
      tensor_equals_vector<int32_t>(materialized_input.positions_host, {0, 1}));
  EXPECT_TRUE(tensor_equals_vector<int32_t>(materialized_input.token_ids.cpu(),
                                            {11, 23}));
  EXPECT_TRUE(tensor_equals_vector<int32_t>(materialized_input.positions.cpu(),
                                            {0, 1}));
  EXPECT_TRUE(tensor_equals_vector<int32_t>(
      materialized_input.input_params.attention.device.new_cache_slots.cpu(),
      {3, 4}));
  EXPECT_TRUE(torch::equal(
      materialized_input.input_params.attention.device.block_tables.cpu(),
      source.input_params.attention.host.block_tables));
  EXPECT_TRUE(tensor_equals_vector<int32_t>(
      materialized_input.sampling_params.selected_token_idxes.cpu(), {1}));
  EXPECT_EQ(materialized_input.input_params.attention.host.q_seq_lens,
            std::vector<int32_t>({2}));
  EXPECT_EQ(materialized_input.input_params.attention.host.q_cu_seq_lens,
            std::vector<int32_t>({0, 2}));
  EXPECT_EQ(materialized_input.input_params.attention.host.kv_seq_lens,
            std::vector<int32_t>({2}));
}
#endif

TEST(BatchPackedInputTest, CpuPreparationAndReadyCopyRetainExecutionSources) {
  ScopedContiguousInputBuffer contiguous_input_buffer(/*enabled=*/false);
  LlmForwardInput prepared_input;
  const void* retained_data = nullptr;
  const void* buffer_data = nullptr;
  {
    LlmForwardInput source;
    source.token_ids = torch::tensor({11, 23}, torch::kInt32);
    source.positions = torch::tensor({0, 1}, torch::kInt32);
    source.runtime.device_input_buffer = torch::tensor({7, 9}, torch::kUInt8);
    source.runtime.retained_device_tensors.reserve(1);
    source.runtime.retained_device_tensors.emplace_back(
        torch::tensor({3.0f, 5.0f}));
    source.runtime.kv_slot_layout = KvSlotLayout::NPU_CP_RECOVERED_PHYSICAL;
    buffer_data = source.runtime.device_input_buffer.data_ptr();
    retained_data = source.runtime.retained_device_tensors[0].data_ptr();

    prepared_input = source.to(torch::Device(torch::kCPU), torch::kFloat32);
  }

  LlmForwardInput ready_copy =
      prepared_input.to(torch::Device(torch::kCPU), torch::kFloat32);
  prepared_input = LlmForwardInput();

  ASSERT_EQ(ready_copy.runtime.retained_device_tensors.size(), 1u);
  EXPECT_EQ(ready_copy.runtime.device_input_buffer.data_ptr(), buffer_data);
  EXPECT_EQ(ready_copy.runtime.retained_device_tensors[0].data_ptr(),
            retained_data);
  EXPECT_TRUE(tensor_equals_vector<uint8_t>(
      ready_copy.runtime.device_input_buffer, {7, 9}));
  EXPECT_TRUE(tensor_equals_vector<float>(
      ready_copy.runtime.retained_device_tensors[0], {3.0f, 5.0f}));
  EXPECT_EQ(ready_copy.runtime.kv_slot_layout,
            KvSlotLayout::NPU_CP_RECOVERED_PHYSICAL);
  EXPECT_TRUE(ready_copy.runtime.device_tensors_ready);
  EXPECT_TRUE(
      tensor_equals_vector<int32_t>(ready_copy.host_token_ids(), {11, 23}));
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
  ForwardInputBuilder builder(sequences,
                              budgets,
                              {},
                              {},
                              nullptr,
                              /*batch_id=*/1,
                              nullptr,
                              BatchForwardType::DECODE);

  LlmForwardInput input =
      builder.build_forward_input(/*num_decoding_tokens=*/1,
                                  /*min_decoding_batch_size=*/0);
  LinearStateCacheOp restore_op;
  restore_op.linear_state_id = 7;
  restore_op.restore_requested = true;
  restore_op.restore_src_slot_id = 3;

  LinearStateCacheOp direct_read_op;
  direct_read_op.linear_state_id = 8;
  direct_read_op.restore_src_slot_id = 4;

  input.input_params.linear_state_cache_ops = {restore_op, direct_read_op};

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(input.to_packed_proto(&packed_input));

  LlmForwardInput lazy_input;
  ASSERT_TRUE(lazy_input.from_packed_proto(packed_input));
  EXPECT_TRUE(lazy_input.input_params.linear_state_cache_ops.empty());
  EXPECT_TRUE(lazy_input.runtime.input_host_buffer_has_layout);

  LlmForwardInput unpacked_input;
  unpacked_input.input_params.linear_state_cache_ops = {direct_read_op};
  ASSERT_TRUE(detail::unpack_from_input_host_buffer(lazy_input,
                                                    torch::Device(torch::kCPU),
                                                    torch::kFloat32,
                                                    unpacked_input,
                                                    false));
  ASSERT_EQ(unpacked_input.input_params.linear_state_cache_ops.size(), 2u);
  expect_linear_state_cache_op_eq(
      unpacked_input.input_params.linear_state_cache_ops[0], restore_op);
  expect_linear_state_cache_op_eq(
      unpacked_input.input_params.linear_state_cache_ops[1], direct_read_op);
}

namespace {

template <typename Input>
void expect_repeated_unpack_replaces_multi_block_tables() {
  ScopedContiguousInputBuffer contiguous_input_buffer(/*enabled=*/false);
  Input source;
  source.input_params.multi_block_tables = {
      torch::tensor({{3, 7}}, torch::kInt32),
      torch::tensor({{11, 13}}, torch::kInt32)};
  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(source.to_packed_proto(&packed_input));
  Input lazy_input;
  ASSERT_TRUE(lazy_input.from_packed_proto(packed_input));
  Input first_input;
  ASSERT_TRUE(detail::unpack_from_input_host_buffer(
      lazy_input,
      torch::Device(torch::kCPU),
      torch::kFloat32,
      first_input,
      /*materialize_device_buffer=*/false));
  ASSERT_EQ(first_input.input_params.multi_block_tables.size(), 2u);
  Input second_input;
  ASSERT_TRUE(detail::unpack_from_input_host_buffer(
      first_input,
      torch::Device(torch::kCPU),
      torch::kFloat32,
      second_input,
      /*materialize_device_buffer=*/false));
  ASSERT_EQ(second_input.input_params.multi_block_tables.size(), 2u);
  for (size_t index = 0; index < 2; ++index) {
    EXPECT_TRUE(
        torch::equal(second_input.input_params.multi_block_tables[index],
                     source.input_params.multi_block_tables[index]));
    EXPECT_NE(second_input.input_params.multi_block_tables[index].data_ptr(),
              first_input.input_params.multi_block_tables[index].data_ptr());
  }
  first_input.input_params.multi_block_tables[0].fill_(99);
  EXPECT_TRUE(torch::equal(second_input.input_params.multi_block_tables[0],
                           source.input_params.multi_block_tables[0]));
}

}  // namespace

TEST(BatchPackedInputTest, RepeatedLlmUnpackReplacesMultiBlockTables) {
  expect_repeated_unpack_replaces_multi_block_tables<LlmForwardInput>();
}

TEST(BatchPackedInputTest, RepeatedVlmUnpackReplacesMultiBlockTables) {
  expect_repeated_unpack_replaces_multi_block_tables<VlmForwardInput>();
}

namespace {

template <typename Input>
void expect_packed_sample_count_hint() {
  ScopedContiguousInputBuffer contiguous_input_buffer(/*enabled=*/false);
  Input source;
  source.sampling_params.selected_token_idxes =
      torch::tensor({0, 1}, torch::kInt32);
  EXPECT_FALSE(source.sampling_params.do_sample.defined());
  proto::PackedForwardInput packed;
  ASSERT_TRUE(source.to_packed_proto(&packed));
  ASSERT_TRUE(packed.has_sample_count());
  EXPECT_EQ(packed.sample_count(), 2);
  Input lazy;
  ASSERT_TRUE(lazy.from_packed_proto(packed));
  EXPECT_EQ(lazy.input_host_sample_count, 2);
  EXPECT_EQ(lazy.clone().input_host_sample_count, 2);
  EXPECT_FALSE(lazy.sampling_params.do_sample.defined());
  EXPECT_FALSE(lazy.sampling_params.selected_token_idxes.defined());

  const void* retained_data = lazy.runtime.input_host_buffer.data_ptr();
  packed.set_sample_count(-1);
  EXPECT_FALSE(lazy.from_packed_proto(packed));
  EXPECT_EQ(lazy.runtime.input_host_buffer.data_ptr(), retained_data);
  packed.set_sample_count(std::numeric_limits<int32_t>::max());
  EXPECT_FALSE(lazy.from_packed_proto(packed));
  packed.set_sample_count(static_cast<int32_t>(
      read_packed_uint64(packed.payload(), /*offset=*/32) / sizeof(int32_t) +
      1));
  EXPECT_FALSE(lazy.from_packed_proto(packed));

  const std::string retained_proto = packed.SerializeAsString();
  // Selected tokens define the wire count, including for greedy batches.
  // Expand a one-element view to exercise the int32 limit without allocating.
  const int64_t oversized_sample_count =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  source.sampling_params.selected_token_idxes =
      torch::zeros({1}, torch::kInt32).expand({oversized_sample_count});
  EXPECT_FALSE(source.to_packed_proto(&packed));
  EXPECT_EQ(packed.SerializeAsString(), retained_proto);

  packed.clear_sample_count();
  ASSERT_TRUE(lazy.from_packed_proto(packed));
  EXPECT_EQ(lazy.input_host_sample_count, kUnknownPackedSampleCount);
  Input unpacked;
  ASSERT_TRUE(detail::unpack_from_input_host_buffer(
      lazy,
      torch::Device(torch::kCPU),
      torch::kFloat32,
      unpacked,
      /*materialize_device_buffer=*/false));
  EXPECT_EQ(unpacked.input_host_sample_count, 2);

  Input empty;
  ASSERT_TRUE(empty.to_packed_proto(&packed));
  ASSERT_TRUE(lazy.from_packed_proto(packed));
  EXPECT_EQ(lazy.input_host_sample_count, 0);
  DiTForwardInput dit;
  ASSERT_TRUE(dit.to_packed_proto(&packed));
  EXPECT_FALSE(packed.has_sample_count());
}

template <typename Input>
void expect_packed_sample_count_mismatch_rejected() {
  ScopedContiguousInputBuffer contiguous_input_buffer(/*enabled=*/false);
  Input source;
  source.sampling_params.selected_token_idxes =
      torch::tensor({0, 1}, torch::kInt32);
  proto::PackedForwardInput packed;
  ASSERT_TRUE(source.to_packed_proto(&packed));
  packed.set_sample_count(1);
  Input lazy;
  ASSERT_TRUE(lazy.from_packed_proto(packed));
  EXPECT_DEATH(
      {
        Input unpacked;
        detail::unpack_from_input_host_buffer(
            lazy,
            torch::Device(torch::kCPU),
            torch::kFloat32,
            unpacked,
            /*materialize_device_buffer=*/false);
      },
      "Packed sample count does not match the sampling sequence count");
}

}  // namespace

TEST(BatchPackedInputTest, LlmPackedSampleCountAvoidsMaterialization) {
  expect_packed_sample_count_hint<LlmForwardInput>();
}

TEST(BatchPackedInputTest, VlmPackedSampleCountAvoidsMaterialization) {
  expect_packed_sample_count_hint<VlmForwardInput>();
}

TEST(BatchPackedInputDeathTest, LlmPackedSampleCountMustMatchPayload) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  expect_packed_sample_count_mismatch_rejected<LlmForwardInput>();
}

TEST(BatchPackedInputDeathTest, VlmPackedSampleCountMustMatchPayload) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  expect_packed_sample_count_mismatch_rejected<VlmForwardInput>();
}

TEST(BatchPackedInputTest, PackedProtoLazyUnpackRestoresDoSample) {
  RequestSamplingParam sampling_param;
  sampling_param.logprobs = true;
  RequestSamplingParam greedy_sampling_param = sampling_param;
  greedy_sampling_param.temperature = 0.0;

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
  options.num_blocks(4).block_size(4);
  BlockManagerImpl manager(options);

  IncrementalDecoder decoder("", 1, false, false);
  Sequence seq(/*index=*/0,
               /*token_ids=*/{1, 2, 3, 4},
               input_embedding,
               mm_data,
               std::move(decoder),
               seq_params);

  seq.add_blocks(BlockType::KV, manager.allocate(1));

  // do_sample is present only when random and greedy rows share a batch.
  SequenceParams greedy_seq_params = seq_params;
  greedy_seq_params.sampling_param = &greedy_sampling_param;
  IncrementalDecoder greedy_decoder("", 1, false, false);
  Sequence greedy_seq(/*index=*/1,
                      /*token_ids=*/{5, 6, 7, 8},
                      input_embedding,
                      mm_data,
                      std::move(greedy_decoder),
                      greedy_seq_params);
  greedy_seq.add_blocks(BlockType::KV, manager.allocate(1));

  std::vector<Sequence*> sequences = {&seq, &greedy_seq};
  std::vector<uint32_t> budgets = {4, 4};
  ForwardInputBuilder builder(sequences,
                              budgets,
                              {},
                              {},
                              nullptr,
                              /*batch_id=*/1,
                              nullptr,
                              BatchForwardType::DECODE);

  LlmForwardInput input =
      builder.build_forward_input(/*num_decoding_tokens=*/1,
                                  /*min_decoding_batch_size=*/0);
  ASSERT_TRUE(input.sampling_params.do_sample.defined());
  ASSERT_EQ(input.sampling_params.sampling_mode, SamplingMode::MIXED);
  input.sampling_params.min_p = torch::tensor({0.1F, 0.0F});
  input.sampling_params.logits_bias =
      torch::tensor({{0.0F, -INFINITY, 2.0F}, {0.0F, 0.0F, 0.0F}});
  input.sampling_params.seeds = torch::tensor({42, 43}, torch::kLong);
  input.sampling_params.seed_offsets = torch::tensor({7, 0}, torch::kLong);
  input.sampling_params.filter_bitmask =
      torch::tensor({{static_cast<int32_t>(0x5)}, {static_cast<int32_t>(0x7)}},
                    torch::TensorOptions().dtype(torch::kInt32));

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(input.to_packed_proto(&packed_input));

  LlmForwardInput lazy_input;
  ASSERT_TRUE(lazy_input.from_packed_proto(packed_input));
  EXPECT_FALSE(lazy_input.sampling_params.do_sample.defined());
  EXPECT_TRUE(lazy_input.runtime.input_host_buffer_has_layout);

  LlmForwardInput unpacked_input;
  ASSERT_TRUE(detail::unpack_from_input_host_buffer(lazy_input,
                                                    torch::Device(torch::kCPU),
                                                    torch::kFloat32,
                                                    unpacked_input,
                                                    false));
  ASSERT_TRUE(unpacked_input.sampling_params.do_sample.defined());
  EXPECT_EQ(unpacked_input.sampling_params.sampling_mode, SamplingMode::MIXED);
  EXPECT_TRUE(torch::equal(unpacked_input.sampling_params.do_sample,
                           torch::tensor({true, false}, torch::kBool)));
  EXPECT_EQ(unpacked_input.input_host_sample_count, 2);
  EXPECT_TRUE(torch::equal(unpacked_input.sampling_params.min_p,
                           input.sampling_params.min_p));
  EXPECT_TRUE(torch::equal(unpacked_input.sampling_params.logits_bias,
                           input.sampling_params.logits_bias));
  EXPECT_TRUE(torch::equal(unpacked_input.sampling_params.seeds,
                           input.sampling_params.seeds));
  EXPECT_TRUE(torch::equal(unpacked_input.sampling_params.seed_offsets,
                           input.sampling_params.seed_offsets));
  ASSERT_TRUE(unpacked_input.sampling_params.filter_bitmask.defined());
  EXPECT_TRUE(torch::equal(unpacked_input.sampling_params.filter_bitmask,
                           input.sampling_params.filter_bitmask));
}

TEST(BatchPackedInputTest, PackedProtoRejectsLegacyTokenLayout) {
  ScopedContiguousInputBuffer contiguous_input_buffer(/*enabled=*/false);
  LlmForwardInput input;
  input.token_ids = torch::tensor({11, 23}, torch::kInt32);
  input.positions = torch::tensor({0, 1}, torch::kInt32);
  input.input_params.meta.num_sequences = 1;
  input.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  input.input_params.attention.host.q_seq_lens = {2};
  input.input_params.attention.host.q_cu_seq_lens = {0, 2};
  input.input_params.attention.host.kv_seq_lens = {2};
  input.sample_sequence_ids = {"legacy#0"};
  input.sample_prior_output_rows = {-1};
  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(input.to_packed_proto(&packed_input));
  ASSERT_GE(packed_input.payload().size(), 40u);

  // Legacy peers start with layout sizes and have no domain or version.
  const uint64_t arena_offset =
      read_packed_uint64(packed_input.payload(), /*offset=*/24);
  ASSERT_GE(arena_offset, 40u);
  std::string legacy_payload = packed_input.payload().substr(16);
  overwrite_packed_uint64(legacy_payload, /*offset=*/8, arena_offset - 16);
  proto::PackedForwardInput legacy_input;
  legacy_input.set_payload(std::move(legacy_payload));

  LlmForwardInput lazy_input;
  EXPECT_FALSE(lazy_input.from_packed_proto(legacy_input));
  EXPECT_FALSE(lazy_input.runtime.input_host_buffer.defined());

  DiTForwardInput dit_input;
  EXPECT_FALSE(dit_input.from_packed_proto(legacy_input));
}

TEST(BatchPackedInputTest, NativeDomainsRejectUnsupportedPackedSchemas) {
  LlmForwardInput llm_source;
  VlmForwardInput vlm_source;
  RecForwardInput rec_source;
  DiTForwardInput dit_source;
  std::vector<proto::PackedForwardInput> packed_inputs(4);
  ASSERT_TRUE(llm_source.to_packed_proto(&packed_inputs[0]));
  ASSERT_TRUE(vlm_source.to_packed_proto(&packed_inputs[1]));
  ASSERT_TRUE(rec_source.to_packed_proto(&packed_inputs[2]));
  ASSERT_TRUE(dit_source.to_packed_proto(&packed_inputs[3]));

  for (const auto& packed_input : packed_inputs) {
    ASSERT_GE(packed_input.payload().size(), 40u);
    for (int32_t version : {0, 2, 4, 6, 127}) {
      SCOPED_TRACE(version);
      std::string payload = packed_input.payload();
      payload[8] = static_cast<char>(version);
      payload[9] = '\0';
      proto::PackedForwardInput unsupported_input;
      unsupported_input.set_payload(std::move(payload));
      LlmForwardInput llm_input;
      VlmForwardInput vlm_input;
      RecForwardInput rec_input;
      DiTForwardInput dit_input;
      EXPECT_FALSE(llm_input.from_packed_proto(unsupported_input));
      EXPECT_FALSE(vlm_input.from_packed_proto(unsupported_input));
      EXPECT_FALSE(rec_input.from_packed_proto(unsupported_input));
      EXPECT_FALSE(dit_input.from_packed_proto(unsupported_input));
      EXPECT_FALSE(llm_input.runtime.input_host_buffer.defined());
      EXPECT_FALSE(vlm_input.runtime.input_host_buffer.defined());
      EXPECT_FALSE(rec_input.runtime.input_host_buffer.defined());
      EXPECT_EQ(dit_input.model_params.tensor_sources.size(), 0u);
    }
  }
}

TEST(BatchPackedInputTest, PackedTokenDecoderRejectsNativeDiTDomain) {
  DiTForwardInput input;
  input.model_params.batch_size = 1;
  input.model_params.prompts = {"prompt"};
  input.model_params.tensor_sources.add("latent", torch::tensor({1.5f, 2.5f}));
  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(input.to_packed_proto(&packed_input));

  LlmForwardInput token_input;
  EXPECT_FALSE(token_input.from_packed_proto(packed_input));
  EXPECT_FALSE(token_input.runtime.input_host_buffer.defined());
}

TEST(BatchPackedInputTest,
     NativeRecPackedCopyPreservesPayloadAfterSourceRelease) {
  ScopedContiguousInputBuffer contiguous_input_buffer(/*enabled=*/false);
  RecForwardInput lazy_input;
  {
    auto input = make_rec_transport_input();
    proto::PackedForwardInput packed_input;
    ASSERT_TRUE(input.to_packed_proto(&packed_input));
    ASSERT_GE(packed_input.payload().size(), 40u);
    EXPECT_EQ(static_cast<uint8_t>(packed_input.payload()[10]), 3u);
    ASSERT_TRUE(lazy_input.from_packed_proto(packed_input));
    LlmForwardInput token_input;
    EXPECT_FALSE(token_input.from_packed_proto(packed_input));
    DiTForwardInput dit_input;
    EXPECT_FALSE(dit_input.from_packed_proto(packed_input));
  }
  ASSERT_TRUE(lazy_input.runtime.input_host_buffer_has_layout);
  RecForwardInput copied_input = lazy_input.clone();
  lazy_input = RecForwardInput();
  const auto input =
      copied_input.to(torch::Device(torch::kCPU), torch::kFloat32);
  copied_input = RecForwardInput();
  EXPECT_TRUE(input.runtime.device_tensors_ready);
  EXPECT_FALSE(input.runtime.input_host_buffer_has_layout);
  expect_rec_transport_input(input);
}

TEST(BatchPackedInputTest, NativeRecSharedMemoryRetainsPayloadAfterOverwrite) {
  ScopedContiguousInputBuffer contiguous_input_buffer(/*enabled=*/false);
  RecForwardInput lazy_input;
  {
    const std::string shm_name =
        ForwardSharedMemoryManager::create_unique_name("batch_test_native_rec",
                                                       /*dp_group=*/0,
                                                       ForwardType::RAW_INPUT,
                                                       /*rank=*/0);
    bool is_creator = false;
    ForwardSharedMemoryManager writer(
        shm_name, /*size=*/1 << 20, is_creator, ForwardType::RAW_INPUT);
    bool is_reader_creator = false;
    ForwardSharedMemoryManager reader(
        shm_name, /*size=*/1 << 20, is_reader_creator, ForwardType::RAW_INPUT);
    auto source = make_rec_transport_input();
    ASSERT_TRUE(writer.input_write(source));
    reader.input_read(
        lazy_input,
        torch::Device(torch::kCPU),
        InputDeviceMaterializationPolicy::DEFER_TO_WORKER_PREPARE);
    ASSERT_TRUE(lazy_input.runtime.input_host_buffer_has_layout);
    source.token_ids = torch::tensor({91, 92}, torch::kInt32);
    ASSERT_TRUE(writer.input_write(source));
  }
  const auto input = lazy_input.to(torch::Device(torch::kCPU), torch::kFloat32);
  expect_rec_transport_input(input);
}

TEST(BatchPackedInputTest, NativeRecDecoderRejectsTokenAndDiTPayloads) {
  LlmForwardInput token_input;
  token_input.token_ids = torch::tensor({11}, torch::kInt32);
  token_input.positions = torch::tensor({0}, torch::kInt32);
  proto::PackedForwardInput token_payload;
  ASSERT_TRUE(token_input.to_packed_proto(&token_payload));
  RecForwardInput input;
  EXPECT_FALSE(input.from_packed_proto(token_payload));
  EXPECT_FALSE(input.runtime.input_host_buffer.defined());
  DiTForwardInput dit_input;
  dit_input.model_params.batch_size = 1;
  dit_input.model_params.prompts = {"prompt"};
  proto::PackedForwardInput dit_payload;
  ASSERT_TRUE(dit_input.to_packed_proto(&dit_payload));
  EXPECT_FALSE(input.from_packed_proto(dit_payload));
  EXPECT_FALSE(input.runtime.input_host_buffer.defined());
}

TEST(BatchPackedInputTest, NativeRecTransportRejectsLocalDecodeStrategies) {
  proto::PackedForwardInput packed_input;
  auto input = make_rec_transport_input();
  input.input_params.mutable_onerec_params();
  EXPECT_FALSE(input.to_packed_proto(&packed_input));
  input.input_params.mutable_onerec_xattention_params();
  EXPECT_FALSE(input.to_packed_proto(&packed_input));
  input.input_params.mutable_llmrec_params();
  EXPECT_FALSE(input.to_packed_proto(&packed_input));
  input = make_rec_transport_input();
  input.step_decode = StepDecodeMeta{};
  EXPECT_FALSE(input.to_packed_proto(&packed_input));
  input = make_rec_transport_input();
  input.decoder_sampling_params.selected_token_idxes =
      torch::tensor({0}, torch::kInt32);
  EXPECT_FALSE(input.to_packed_proto(&packed_input));
}

TEST(BatchPackedInputTest, NativeVlmPackedCopyRetainsVisionAndMrope) {
  ScopedContiguousInputBuffer contiguous_input_buffer(/*enabled=*/false);
  VlmForwardInput lazy_input;
  {
    auto source = make_vlm_transport_input();
    proto::PackedForwardInput payload;
    ASSERT_TRUE(source.to_packed_proto(&payload));
    ASSERT_GE(payload.payload().size(), 40u);
    EXPECT_EQ(static_cast<uint8_t>(payload.payload()[10]), 2u);
    ASSERT_TRUE(lazy_input.from_packed_proto(payload));
  }
  ASSERT_TRUE(lazy_input.runtime.input_host_buffer_has_layout);
  auto copied_input = lazy_input.clone();
  lazy_input = VlmForwardInput();
  const auto prepared =
      copied_input.to(torch::Device(torch::kCPU), torch::kFloat32);
  copied_input = VlmForwardInput();
  EXPECT_TRUE(prepared.runtime.device_tensors_ready);
  expect_vlm_transport_input(prepared);
}

TEST(BatchPackedInputTest, NativeVlmSharedMemoryRetainsVisionAfterOverwrite) {
  ScopedContiguousInputBuffer contiguous_input_buffer(/*enabled=*/false);
  VlmForwardInput lazy_input;
  {
    const std::string shm_name =
        ForwardSharedMemoryManager::create_unique_name("batch_test_native_vlm",
                                                       /*dp_group=*/0,
                                                       ForwardType::RAW_INPUT,
                                                       /*rank=*/0);
    bool is_creator = false;
    ForwardSharedMemoryManager writer(
        shm_name, /*size=*/1 << 20, is_creator, ForwardType::RAW_INPUT);
    bool is_reader_creator = false;
    ForwardSharedMemoryManager reader(
        shm_name, /*size=*/1 << 20, is_reader_creator, ForwardType::RAW_INPUT);
    auto source = make_vlm_transport_input();
    ASSERT_TRUE(writer.input_write(source));
    reader.input_read(
        lazy_input,
        torch::Device(torch::kCPU),
        InputDeviceMaterializationPolicy::DEFER_TO_WORKER_PREPARE);
    source.token_ids = torch::tensor({91, 92}, torch::kInt32);
    source.input_params.multimodal.deep_stacks.clear();
    ASSERT_TRUE(writer.input_write(source));
  }
  const auto prepared =
      lazy_input.to(torch::Device(torch::kCPU), torch::kFloat32);
  expect_vlm_transport_input(prepared);
}

TEST(BatchPackedInputTest, NativeVlmDecodeKeepsDomainWithoutVisionData) {
  VlmForwardInput source;
  source.token_ids = torch::tensor({29}, torch::kInt32);
  source.positions = torch::tensor({9}, torch::kInt32);
  source.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  proto::PackedForwardInput payload;
  ASSERT_TRUE(source.to_packed_proto(&payload));
  VlmForwardInput lazy_input;
  ASSERT_TRUE(lazy_input.from_packed_proto(payload));
  const auto prepared =
      lazy_input.to(torch::Device(torch::kCPU), torch::kFloat32);
  EXPECT_TRUE(prepared.input_params.meta.batch_forward_type.is_decode());
  EXPECT_FALSE(prepared.input_params.multimodal.mm_data.valid());
  EXPECT_TRUE(torch::equal(prepared.token_ids.cpu(), source.token_ids));
  LlmForwardInput llm_input;
  RecForwardInput rec_input;
  DiTForwardInput dit_input;
  EXPECT_FALSE(llm_input.from_packed_proto(payload));
  EXPECT_FALSE(rec_input.from_packed_proto(payload));
  EXPECT_FALSE(dit_input.from_packed_proto(payload));
  payload.mutable_payload()->at(10) = 1;
  EXPECT_FALSE(lazy_input.from_packed_proto(payload));
}

TEST(BatchPackedInputTest, BorrowedExecutionViewUpdatesNativeMetadata) {
  static_assert(!std::is_copy_constructible_v<LlmForwardInput>);
  static_assert(!std::is_copy_constructible_v<VlmForwardInput>);
  static_assert(!std::is_copy_constructible_v<RecForwardInput>);
  static_assert(!std::is_copy_constructible_v<LlmModelParams>);
  static_assert(!std::is_copy_constructible_v<VlmModelParams>);
  static_assert(!std::is_copy_constructible_v<RecModelParams>);
  static_assert(!std::is_default_constructible_v<ModelParamsView>);
  static_assert(!std::is_constructible_v<ModelParamsView, RecModelParams&>);
  LlmModelParams owner;
  owner.attention.host.kv_seq_lens = {2};
  ModelParamsView view(owner);
  EXPECT_EQ(view.attention.host.kv_seq_lens.data(),
            owner.attention.host.kv_seq_lens.data());
  view.attention.host.kv_seq_lens[0] = 7;
  EXPECT_EQ(owner.attention.host.kv_seq_lens[0], 7);
  view.graph.input_tokens_override = torch::tensor({13}, torch::kInt32);
  EXPECT_TRUE(torch::equal(owner.graph.input_tokens_override,
                           torch::tensor({13}, torch::kInt32)));
  EXPECT_FALSE(view.has_multimodal());
}

TEST(BatchPackedInputTest,
     SnapshotViewRetainsDomainAndIndependentHostMetadata) {
  const auto snapshot_view = [] {
    VlmModelParams owner;
    owner.attention.host.kv_seq_lens = {2};
    owner.multimodal.deep_stacks = {torch::tensor({1.0F, 3.0F})};
    ModelParamsView borrowed(owner);
    auto retained = borrowed.clone().view();
    borrowed.attention.host.kv_seq_lens[0] = 9;
    return retained;
  }();
  EXPECT_EQ(snapshot_view.attention.host.kv_seq_lens,
            (std::vector<int32_t>{2}));
  EXPECT_TRUE(snapshot_view.has_multimodal());
  ASSERT_EQ(snapshot_view.multimodal().deep_stacks.size(), 1u);
  EXPECT_TRUE(torch::equal(snapshot_view.multimodal().deep_stacks[0],
                           torch::tensor({1.0F, 3.0F})));
}

TEST(BatchPackedInputTest, LlmSnapshotTransferMovesSpecExecutionTensors) {
  LlmModelParams owner;
  ModelParamsView params(owner);
  const torch::Tensor values = torch::tensor({1, 2}, torch::kInt32);
  params.embedding.mtp_shifted_token_ids = values;
  params.embedding.mtp_bootstrap_embeddings = values;
  params.embedding.extra_token_ids = {3};
  params.embedding.mtp_bootstrap_row_idxes = {4};
  params.graph.use_expanded_decode_for_spec_verify_attention = true;
  params.graph.expanded_kv_seq_lens = values;
  params.graph.expanded_block_tables = values;
  params.graph.expanded_paged_kv_indptr = values;
  params.graph.expanded_paged_kv_indices = values;
  params.graph.expanded_paged_kv_last_page_len = values;
  params.graph.expanded_tiling_data = values;
  params.graph.expanded_kv_seq_lens_vec = {5};
  params.graph.input_tokens_override = values;
  params.graph.spec_verify_draft_token_sources = {values};
  params.graph.spec_verify_source_addresses_stable = true;
  params.graph.spec_verify_static_graph_tasks_prepared = true;
  params.multi_block_tables = {values};
  params.mtp_shifted_token_ids = values;
  params.is_spec_verify = true;
  params.num_accepted_tokens = values;
  params.num_accepted_tokens_host = {6};
  params.mtp_topk_state = MtpTopkState::from_tensor(values);

  const torch::Device target_device("meta");
  ModelParamsView converted = params.clone().to(target_device).view();

  const std::vector<torch::Tensor> transferred_tensors = {
      converted.embedding.mtp_shifted_token_ids,
      converted.embedding.mtp_bootstrap_embeddings,
      converted.graph.expanded_kv_seq_lens,
      converted.graph.expanded_block_tables,
      converted.graph.expanded_paged_kv_indptr,
      converted.graph.expanded_paged_kv_indices,
      converted.graph.expanded_paged_kv_last_page_len,
      converted.graph.expanded_tiling_data,
      converted.graph.input_tokens_override,
      converted.graph.spec_verify_draft_token_sources.at(0),
      converted.mtp_shifted_token_ids,
      converted.num_accepted_tokens};
  for (const auto& tensor : transferred_tensors) {
    ASSERT_TRUE(tensor.defined());
    EXPECT_EQ(tensor.device(), target_device);
    EXPECT_EQ(tensor.sizes(), values.sizes());
    EXPECT_EQ(tensor.scalar_type(), values.scalar_type());
  }
  ASSERT_NE(converted.mtp_topk_state, nullptr);
  EXPECT_EQ(converted.mtp_topk_state->device(), target_device);
  EXPECT_FALSE(converted.has_multimodal());
  EXPECT_TRUE(converted.is_spec_verify);
  EXPECT_TRUE(converted.graph.use_expanded_decode_for_spec_verify_attention);
  EXPECT_TRUE(converted.graph.spec_verify_source_addresses_stable);
  EXPECT_TRUE(converted.graph.spec_verify_static_graph_tasks_prepared);
  EXPECT_EQ(converted.embedding.extra_token_ids, (std::vector<int32_t>{3}));
  EXPECT_EQ(converted.embedding.mtp_bootstrap_row_idxes,
            (std::vector<int32_t>{4}));
  EXPECT_EQ(converted.num_accepted_tokens_host, (std::vector<int64_t>{6}));
  ASSERT_EQ(converted.multi_block_tables.size(), 1u);
  EXPECT_TRUE(converted.multi_block_tables[0].device().is_cpu());
  EXPECT_TRUE(torch::equal(converted.multi_block_tables[0], values));
  converted.graph.expanded_kv_seq_lens_vec[0] = 9;
  EXPECT_EQ(params.graph.expanded_kv_seq_lens_vec, (std::vector<int32_t>{5}));
  EXPECT_TRUE(params.num_accepted_tokens.device().is_cpu());
  EXPECT_TRUE(params.mtp_topk_state->device().is_cpu());
}

TEST(BatchPackedInputTest, RecSnapshotTransfersNativeExecutionTensors) {
  RecModelParams source;
  const torch::Tensor values = torch::tensor({1, 2}, torch::kInt32);
  source.attention.host.kv_seq_lens = {2};
  source.attention.device.kv_seq_lens = values;
  source.embedding.input_embedding = values;
  source.embedding.linear_state_indices = values;
  source.embedding.request_ids = {"rec"};
  source.graph.attn_mask = values;
  source.graph.tiling_data = values;
  auto& strategy = source.mutable_llmrec_params();
  strategy.current_round_tensor = values;
  strategy.full_k_caches = {values};
  strategy.full_v_caches = {values};
  strategy.decode_positions_tensor_list = {values};

  const torch::Device target_device("meta");
  auto converted = source.clone().to(target_device);
  ASSERT_TRUE(converted.has_llmrec_params());
  EXPECT_FALSE(converted.has_onerec_params());
  const std::vector<torch::Tensor> tensors = {
      converted.attention.device.kv_seq_lens,
      converted.embedding.input_embedding,
      converted.embedding.linear_state_indices,
      converted.graph.attn_mask,
      converted.graph.tiling_data,
      converted.llmrec_params()->current_round_tensor,
      converted.llmrec_params()->full_k_caches.at(0),
      converted.llmrec_params()->full_v_caches.at(0),
      converted.llmrec_params()->decode_positions_tensor_list.at(0)};
  for (const auto& tensor : tensors) {
    ASSERT_TRUE(tensor.defined());
    EXPECT_EQ(tensor.device(), target_device);
    EXPECT_EQ(tensor.sizes(), values.sizes());
    EXPECT_EQ(tensor.scalar_type(), values.scalar_type());
  }
  converted.attention.host.kv_seq_lens[0] = 9;
  EXPECT_EQ(source.attention.host.kv_seq_lens, (std::vector<int32_t>{2}));
  EXPECT_EQ(converted.embedding.request_ids, (std::vector<std::string>{"rec"}));
  EXPECT_TRUE(source.llmrec_params()->full_k_caches[0].device().is_cpu());
}

TEST(BatchPackedInputTest, VlmDraftConversionExcludesTargetOnlyState) {
  auto target = make_vlm_transport_input();
  target.input_params.embedding.linear_state_ids = {17};
  target.input_params.embedding.linear_state_indices =
      torch::tensor({17}, torch::kInt32);
  target.input_params.linear_state_cache_ops = {{17, true, false, -1}};
  auto draft = make_llm_draft_input(target);
  EXPECT_TRUE(torch::equal(draft.token_ids, target.token_ids));
  EXPECT_TRUE(torch::equal(draft.positions, target.positions));
  EXPECT_TRUE(draft.input_params.embedding.linear_state_ids.empty());
  EXPECT_FALSE(draft.input_params.embedding.linear_state_indices.defined());
  EXPECT_TRUE(draft.input_params.linear_state_cache_ops.empty());
  draft.input_params.attention.host.kv_seq_lens[0] = 9;
  EXPECT_EQ(target.input_params.attention.host.kv_seq_lens[0], 2);
  EXPECT_EQ(target.input_params.embedding.linear_state_ids,
            (std::vector<int32_t>{17}));
  expect_vlm_transport_input(target);
}

TEST(BatchPackedInputTest, NativeVlmReadyCopyRetainsRuntimeLifetime) {
  ScopedContiguousInputBuffer contiguous_input_buffer(/*enabled=*/false);
  VlmForwardInput prepared;
  {
    auto source = make_vlm_transport_input();
    source.runtime.retained_device_tensors = {
        torch::tensor({7, 9}, torch::kInt32)};
    source.runtime.kv_slot_layout = KvSlotLayout::NPU_CP_RECOVERED_PHYSICAL;
    prepared = source.to(torch::Device(torch::kCPU), torch::kFloat32);
  }
  const auto ready_copy =
      prepared.to(torch::Device(torch::kCPU), torch::kFloat32);
  prepared = VlmForwardInput();
  EXPECT_EQ(ready_copy.runtime.kv_slot_layout,
            KvSlotLayout::NPU_CP_RECOVERED_PHYSICAL);
  ASSERT_EQ(ready_copy.runtime.retained_device_tensors.size(), 1u);
  EXPECT_TRUE(torch::equal(ready_copy.runtime.retained_device_tensors[0],
                           torch::tensor({7, 9}, torch::kInt32)));
  expect_vlm_transport_input(ready_copy);
}

TEST(BatchPackedInputTest,
     NativeDiTPackedProtoPreservesInputsAndGenerationParams) {
  DiTForwardInput dit_input;
  dit_input.model_params.batch_size = 2;
  dit_input.model_params.prompts = {"first prompt", "second prompt"};
  dit_input.model_params.prompts_2 = {"first detail", "second detail"};
  dit_input.model_params.negative_prompts = {"first exclusion",
                                             "second exclusion"};
  dit_input.model_params.negative_prompts_2 = {"first negative detail",
                                               "second negative detail"};
  dit_input.model_params.audio_prompt_text = "spoken prompt transcript";
  dit_input.model_params.media_sources.add(
      "unknown", "image", torch::tensor({1, 2}, torch::dtype(torch::kUInt8)));
  dit_input.model_params.media_sources.add(
      "unknown", "image", torch::tensor({3, 4}, torch::dtype(torch::kUInt8)));
  dit_input.model_params.media_sources.add(
      "mask_image",
      "image",
      torch::tensor({5, 6}, torch::dtype(torch::kUInt8)));
  dit_input.model_params.media_sources.add(
      "prompt_audio", "audio", torch::tensor({0.25f, 0.5f}));
  dit_input.model_params.tensor_sources.add("prompt_embed",
                                            torch::tensor({1.5f, 2.5f}));
  dit_input.model_params.tensor_sources.add("latent",
                                            torch::tensor({3.5f, 4.5f}));
  DiTGenerationParams& params = dit_input.model_params.generation_params;
  params.width = 640;
  params.height = 480;
  params.num_inference_steps = 31;
  params.true_cfg_scale = 2.5f;
  params.guidance_scale = 6.5f;
  params.num_images_per_prompt = 2;
  params.num_videos_per_prompt = 3;
  params.seed = 1234567;
  params.seed_is_set = true;
  params.max_sequence_length = 768;
  params.strength = 0.75f;
  params.enable_cfg_renorm = false;
  params.cfg_renorm_min = 0.25f;
  params.audio_duration_frames = 321;
  params.audio_steps = 24;
  params.audio_guidance_method = "apg";
  params.audio_sampling_rate = 48000;
  params.num_frames = 49;
  params.video_fps = 12.5;
  params.guidance_scale_2 = 4.25f;
  params.seconds = 7;
  params.boundary_ratio = 0.8f;
  params.flow_shift = 1.25f;
  params.max_new_tokens = 96;
  params.diffusion_steps = 12;
  params.temperature = 0.6f;
  params.top_k = 32;
  params.top_p = 0.85f;
  params.repetition_penalty = 1.2f;

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(dit_input.to_packed_proto(&packed_input));
  ASSERT_GE(packed_input.payload().size(), 40u);
  EXPECT_EQ(static_cast<uint8_t>(packed_input.payload()[10]), 4u);

  DiTForwardInput unpacked_input;
  ASSERT_TRUE(unpacked_input.from_packed_proto(packed_input));
  expect_dit_forward_input_eq(unpacked_input, dit_input);

  proto::DiTForwardInput proto_input;
  ASSERT_TRUE(dit_input.to_proto(&proto_input));
  DiTForwardInput proto_input_round_trip;
  ASSERT_TRUE(proto_input_round_trip.from_proto(proto_input));
  expect_dit_forward_input_eq(proto_input_round_trip, dit_input);

  proto::DiTGenerationParams proto_params;
  ASSERT_TRUE(params.to_proto(&proto_params));
  DiTGenerationParams proto_round_trip;
  ASSERT_TRUE(proto_round_trip.from_proto(proto_params));
  EXPECT_EQ(proto_round_trip, params);
}

TEST(BatchPackedInputTest, NativeDiTPackedProtoOwnsSourcesAfterPayloadRelease) {
  DiTForwardInput unpacked_input;
  {
    DiTForwardInput source;
    source.model_params.batch_size = 1;
    source.model_params.prompts = {"retained prompt"};
    source.model_params.media_sources.add(
        "image", "image", torch::tensor({1, 2}, torch::dtype(torch::kUInt8)));
    source.model_params.media_sources.add(
        "prompt_audio", "audio", torch::tensor({0.25f, 0.5f}));
    source.model_params.tensor_sources.add("latent",
                                           torch::tensor({1.5f, 2.5f}));
    source.model_params.tensor_sources.add(
        "empty", torch::empty({0, 2}, torch::dtype(torch::kFloat32)));
    proto::PackedForwardInput packed_input;
    ASSERT_TRUE(source.to_packed_proto(&packed_input));
    ASSERT_TRUE(unpacked_input.from_packed_proto(packed_input));
    packed_input.mutable_payload()->assign(packed_input.payload().size(), '\0');
  }

  EXPECT_EQ(unpacked_input.model_params.prompts,
            std::vector<std::string>({"retained prompt"}));
  ASSERT_EQ(unpacked_input.model_params.media_sources.size(), 2u);
  EXPECT_TRUE(
      torch::equal(unpacked_input.model_params.media_sources.at(0).tensor,
                   torch::tensor({1, 2}, torch::dtype(torch::kUInt8))));
  EXPECT_TRUE(
      torch::equal(unpacked_input.model_params.media_sources.at(1).tensor,
                   torch::tensor({0.25f, 0.5f})));
  ASSERT_EQ(unpacked_input.model_params.tensor_sources.size(), 2u);
  EXPECT_TRUE(torch::equal(
      unpacked_input.model_params.tensor_sources.entries()[0].tensor,
      torch::tensor({1.5f, 2.5f})));
  EXPECT_TRUE(
      unpacked_input.model_params.tensor_sources.entries()[1].tensor.defined());
  EXPECT_EQ(unpacked_input.model_params.tensor_sources.entries()[1]
                .tensor.sizes()
                .vec(),
            std::vector<int64_t>({0, 2}));

  const DiTModelParams prepared_input =
      unpacked_input.model_params.to(torch::Device(torch::kCPU));
  EXPECT_EQ(prepared_input.media_sources.at(0).tensor.scalar_type(),
            torch::kUInt8);
  EXPECT_EQ(prepared_input.media_sources.at(1).tensor.scalar_type(),
            torch::kFloat32);
  EXPECT_EQ(prepared_input.tensor_sources.entries()[0].tensor.scalar_type(),
            torch::kBFloat16);
  EXPECT_EQ(prepared_input.tensor_sources.entries()[1].tensor.scalar_type(),
            torch::kBFloat16);
  EXPECT_TRUE(torch::equal(prepared_input.tensor_sources.entries()[0].tensor,
                           torch::tensor({1.5f, 2.5f}, torch::kBFloat16)));
  EXPECT_TRUE(torch::equal(prepared_input.media_sources.at(1).tensor,
                           torch::tensor({0.25f, 0.5f})));
}

TEST(BatchPackedInputTest,
     DiTModelParamsConversionRetainsInputsAfterSourceRelease) {
  const TensorParameters image_parameters = {{"frame_count", int64_t{2}}};
  const TensorParameters latent_parameters = {{"layout", std::string("BCHW")}};
  DiTModelParams prepared;
  {
    DiTModelParams source;
    source.batch_size = 1;
    source.prompts = {"prompt"};
    source.prompts_2 = {"detail"};
    source.negative_prompts = {"exclusion"};
    source.negative_prompts_2 = {"negative detail"};
    source.audio_prompt_text = "spoken transcript";
    source.generation_params.width = 640;
    source.generation_params.seed = 0;
    source.generation_params.seed_is_set = true;
    source.media_sources.add("image",
                             "image",
                             torch::tensor({1, 2}, torch::kUInt8),
                             image_parameters);
    source.media_sources.add(
        "prompt_audio", "audio", torch::tensor({0.25f, 0.5f}));
    source.tensor_sources.add(
        "latent", torch::tensor({1.5f, 2.5f}), latent_parameters);

    prepared = source.to(torch::Device(torch::kCPU), torch::kBFloat16);
    ASSERT_EQ(prepared.tensor_sources.size(), 1u);
    ASSERT_EQ(prepared.media_sources.size(), 2u);
    EXPECT_EQ(source.tensor_sources.entries()[0].tensor.scalar_type(),
              torch::kFloat32);
    EXPECT_NE(prepared.tensor_sources.entries()[0].tensor.data_ptr(),
              source.tensor_sources.entries()[0].tensor.data_ptr());
    EXPECT_EQ(prepared.media_sources.at(0).tensor.data_ptr(),
              source.media_sources.at(0).tensor.data_ptr());
    source.prompts[0] = "changed prompt";
    source.generation_params.width = 128;
  }

  EXPECT_EQ(prepared.batch_size, 1);
  EXPECT_EQ(prepared.prompts, std::vector<std::string>{"prompt"});
  EXPECT_EQ(prepared.prompts_2, std::vector<std::string>{"detail"});
  EXPECT_EQ(prepared.negative_prompts, std::vector<std::string>{"exclusion"});
  EXPECT_EQ(prepared.negative_prompts_2,
            std::vector<std::string>{"negative detail"});
  EXPECT_EQ(prepared.audio_prompt_text, "spoken transcript");
  EXPECT_EQ(prepared.generation_params.width, 640);
  EXPECT_EQ(prepared.generation_params.seed, 0);
  EXPECT_TRUE(prepared.generation_params.seed_is_set);
  ASSERT_EQ(prepared.media_sources.size(), 2u);
  EXPECT_EQ(prepared.media_sources.at(0).name, "image");
  EXPECT_EQ(prepared.media_sources.at(0).modality, "image");
  EXPECT_EQ(prepared.media_sources.at(0).parameters, image_parameters);
  EXPECT_TRUE(torch::equal(prepared.media_sources.at(0).tensor,
                           torch::tensor({1, 2}, torch::kUInt8)));
  EXPECT_TRUE(torch::equal(prepared.media_sources.at(1).tensor,
                           torch::tensor({0.25f, 0.5f})));
  ASSERT_EQ(prepared.tensor_sources.size(), 1u);
  EXPECT_EQ(prepared.tensor_sources.entries()[0].name, "latent");
  EXPECT_EQ(prepared.tensor_sources.entries()[0].parameters, latent_parameters);
  EXPECT_TRUE(torch::equal(prepared.tensor_sources.entries()[0].tensor,
                           torch::tensor({1.5f, 2.5f}, torch::kBFloat16)));
}

TEST(BatchPackedInputTest, NativeDiTPackedProtoRoundTripsEmptyInput) {
  DiTForwardInput input;
  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(input.to_packed_proto(&packed_input));

  DiTForwardInput unpacked_input;
  ASSERT_TRUE(unpacked_input.from_packed_proto(packed_input));
  expect_dit_forward_input_eq(unpacked_input, input);
}

TEST(BatchPackedInputTest, NativeDiTPackedProtoRejectsMalformedPayloads) {
  DiTForwardInput input;
  input.model_params.batch_size = 1;
  input.model_params.prompts = {"prompt"};
  input.model_params.tensor_sources.add("latent", torch::tensor({1.5f, 2.5f}));
  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(input.to_packed_proto(&packed_input));
  DiTForwardInput baseline;
  ASSERT_TRUE(baseline.from_packed_proto(packed_input));
  ASSERT_GT(packed_input.payload().size(), 40u);
  const uint64_t descriptor_bytes =
      read_packed_uint64(packed_input.payload(), /*offset=*/16);
  const uint64_t arena_offset =
      read_packed_uint64(packed_input.payload(), /*offset=*/24);
  const uint64_t arena_bytes =
      read_packed_uint64(packed_input.payload(), /*offset=*/32);

  std::vector<std::pair<std::string, std::string>> malformed_payloads;
  malformed_payloads.reserve(13);
  malformed_payloads.emplace_back("empty", "");
  malformed_payloads.emplace_back("truncated header",
                                  packed_input.payload().substr(0, 39));
  malformed_payloads.emplace_back("unknown version", packed_input.payload());
  malformed_payloads.back().second[8] = 127;
  malformed_payloads.emplace_back("token domain", packed_input.payload());
  malformed_payloads.back().second[10] = 1;
  malformed_payloads.emplace_back("reserved VLM domain",
                                  packed_input.payload());
  malformed_payloads.back().second[10] = 2;
  malformed_payloads.emplace_back("reserved Rec domain",
                                  packed_input.payload());
  malformed_payloads.back().second[10] = 3;
  malformed_payloads.emplace_back("unknown domain", packed_input.payload());
  malformed_payloads.back().second[10] = 127;
  malformed_payloads.emplace_back("overlapping arena", packed_input.payload());
  overwrite_packed_uint64(malformed_payloads.back().second,
                          /*offset=*/24,
                          /*value=*/0);
  malformed_payloads.emplace_back("arena beyond payload",
                                  packed_input.payload());
  overwrite_packed_uint64(
      malformed_payloads.back().second,
      /*offset=*/24,
      static_cast<uint64_t>(packed_input.payload().size()) + 16);
  malformed_payloads.emplace_back("overflowing descriptor",
                                  packed_input.payload());
  overwrite_packed_uint64(malformed_payloads.back().second,
                          /*offset=*/16,
                          std::numeric_limits<uint64_t>::max());
  malformed_payloads.emplace_back(
      "truncated arena",
      packed_input.payload().substr(0, packed_input.payload().size() - 1));
  malformed_payloads.emplace_back("unconsumed descriptor byte",
                                  packed_input.payload());
  malformed_payloads.back().second.insert(
      static_cast<size_t>(40 + descriptor_bytes), /*count=*/16, /*ch=*/'\0');
  overwrite_packed_uint64(malformed_payloads.back().second,
                          /*offset=*/16,
                          descriptor_bytes + 1);
  overwrite_packed_uint64(malformed_payloads.back().second,
                          /*offset=*/24,
                          arena_offset + 16);
  malformed_payloads.emplace_back("unconsumed arena byte",
                                  packed_input.payload());
  malformed_payloads.back().second += '\0';
  overwrite_packed_uint64(malformed_payloads.back().second,
                          /*offset=*/32,
                          arena_bytes + 1);

  for (const auto& [name, payload] : malformed_payloads) {
    SCOPED_TRACE(name);
    proto::PackedForwardInput malformed_input;
    malformed_input.set_payload(payload);
    DiTForwardInput unpacked_input;
    EXPECT_FALSE(unpacked_input.from_packed_proto(malformed_input));
  }
}

TEST(BatchPackedInputTest, PackedProtoLazyToPreservesJsonMetadata) {
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);

  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.enable_schedule_overlap = true;

  MMData mm_data;
  BlockManager::Options options;
  options.num_blocks(2).block_size(4);
  BlockManagerImpl manager(options);

  IncrementalDecoder decoder("", 1, false, false);
  Sequence sequence(/*index=*/0,
                    /*token_ids=*/{1, 2, 3, 4},
                    torch::Tensor(),
                    mm_data,
                    std::move(decoder),
                    seq_params);
  sequence.add_blocks(BlockType::KV, manager.allocate(1));

  std::vector<Sequence*> sequences = {&sequence};
  std::vector<uint32_t> allowed_max_tokens = {4};
  ForwardInputBuilder builder(sequences,
                              allowed_max_tokens,
                              {},
                              {},
                              nullptr,
                              /*batch_id=*/2,
                              nullptr,
                              BatchForwardType::DECODE);
  LlmForwardInput input = builder.build_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0);

  JsonObjectGrammar grammar({"{", "}", "stop"}, /*stop_token_ids=*/{2});
  JsonObjectGrammarState state = grammar.initial_state();
  ASSERT_TRUE(state.accept_token(/*open_object=*/0));
  input.json_object_state_snapshots = {state.snapshot()};
  input.sample_sequence_ids = {"req-json#0"};
  input.sample_prior_output_rows = {-1};

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(input.to_packed_proto(&packed_input));

  LlmForwardInput lazy_input;
  ASSERT_TRUE(lazy_input.from_packed_proto(packed_input));
  EXPECT_TRUE(lazy_input.runtime.input_host_buffer_has_layout);
  EXPECT_TRUE(lazy_input.json_object_state_snapshots.empty());

  const LlmForwardInput materialized_input =
      lazy_input.to(torch::Device(torch::kCPU), torch::kFloat32);
  ASSERT_EQ(materialized_input.json_object_state_snapshots.size(), 1u);
  EXPECT_EQ(materialized_input.json_object_state_snapshots[0].token_ids,
            std::vector<int32_t>({0}));
  EXPECT_FALSE(
      materialized_input.json_object_state_snapshots[0].reasoning_enabled);
  EXPECT_EQ(materialized_input.sample_sequence_ids,
            std::vector<std::string>({"req-json#0"}));
  EXPECT_EQ(materialized_input.sample_prior_output_rows,
            std::vector<int32_t>({-1}));
}

}  // namespace xllm
