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

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "core/common/types.h"
#include "core/framework/config/execution_config.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/forward_shared_memory_transport.h"
#include "core/runtime/vlm_forward_params.h"
#include "worker.pb.h"

namespace xllm {
namespace {

using LayerOffsets = std::vector<VirtualKVCacheLayerGlobalOffsets>;

LayerOffsets make_sparse_role_offsets() {
  constexpr uint64_t kLargeOffset = uint64_t{1} << 34;
  return {{{0, kLargeOffset, 128}, {64, kLargeOffset + 64, 256}, {}},
          {{512, 768, 1024}, {576, 832, 1088}, {640, kLargeOffset + 128, 1152}},
          {{1280, 1536, 1792}, {1344, 1600, 1856}, {}}};
}

void expect_role_offsets(const LayerOffsets& actual,
                         const LayerOffsets& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t layer_id = 0; layer_id < expected.size(); ++layer_id) {
    SCOPED_TRACE(layer_id);
    EXPECT_EQ(actual[layer_id].k_offsets, expected[layer_id].k_offsets);
    EXPECT_EQ(actual[layer_id].v_offsets, expected[layer_id].v_offsets);
    EXPECT_EQ(actual[layer_id].index_offsets, expected[layer_id].index_offsets);
  }
}

template <typename Input>
Input make_input(const LayerOffsets& offsets) {
  Input input;
  input.token_ids = torch::tensor({11, 23}, torch::kInt32);
  input.positions = torch::tensor({0, 1}, torch::kInt32);
  input.input_params.meta.num_sequences = 1;
  input.input_params.attention.host.q_seq_lens = {2};
  input.input_params.attention.host.q_cu_seq_lens = {0, 2};
  input.input_params.attention.host.kv_seq_lens = {2};
  TransferKVInfo transfer;
  transfer.request_id = "sparse-index";
  transfer.rank_local_mapping = true;
  transfer.dst_virtual_kv_cache_layer_global_offsets = offsets;
  transfer.mappings = {{0, {3, 0, 7}, {8, 1, 9}, 0}};
  input.transfer_kv_infos.emplace_back(std::move(transfer));
  return input;
}

template <typename Input>
class ForwardSharedMemoryInputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    previous_contiguous_ =
        ExecutionConfig::get_instance().use_contiguous_input_buffer();
    ExecutionConfig::get_instance().use_contiguous_input_buffer(false);
  }

  void TearDown() override {
    ExecutionConfig::get_instance().use_contiguous_input_buffer(
        previous_contiguous_);
  }

 private:
  bool previous_contiguous_ = false;
};

using TokenInputTypes = ::testing::Types<LlmForwardInput, VlmForwardInput>;

TYPED_TEST_SUITE(ForwardSharedMemoryInputTest, TokenInputTypes);

TYPED_TEST(ForwardSharedMemoryInputTest, RejectsPreviousPackedInputSchema) {
  const TypeParam source = make_input<TypeParam>(make_sparse_role_offsets());
  proto::PackedForwardInput packed;
  ASSERT_TRUE(source.to_packed_proto(&packed));
  constexpr uint16_t kPreviousVersion = 10;
  ASSERT_GE(packed.payload().size(),
            sizeof(uint64_t) + sizeof(kPreviousVersion));
  std::memcpy(packed.mutable_payload()->data() + sizeof(uint64_t),
              &kPreviousVersion,
              sizeof(kPreviousVersion));
  TypeParam decoded;
  EXPECT_FALSE(decoded.from_packed_proto(packed));
}

TYPED_TEST(ForwardSharedMemoryInputTest,
           PackedInputPreservesExecutionMetadata) {
  TypeParam source = make_input<TypeParam>(make_sparse_role_offsets());
  source.input_params.meta.execution_batch = {1, 2, {2}, {17}, {0, 2}, {1}};
  proto::PackedForwardInput packed;
  ASSERT_TRUE(source.to_packed_proto(&packed));
  TypeParam lazy;
  ASSERT_TRUE(lazy.from_packed_proto(packed));
  const TypeParam decoded =
      lazy.to(torch::Device(torch::kCPU), torch::kFloat32);
  const auto& batch = decoded.input_params.meta.execution_batch;
  EXPECT_EQ(batch.num_reqs, 1);
  EXPECT_EQ(batch.num_tokens, 2);
  EXPECT_EQ(batch.num_scheduled_tokens, (std::vector<int32_t>{2}));
  EXPECT_EQ(batch.num_computed_tokens, (std::vector<int32_t>{17}));
  EXPECT_EQ(batch.query_start_loc, (std::vector<int32_t>{0, 2}));
  EXPECT_EQ(batch.is_prefilling, (std::vector<uint8_t>{1}));
}

TYPED_TEST(ForwardSharedMemoryInputTest,
           PackedProtoPreservesSparseIndexOffsetsAfterSourceRelease) {
  const LayerOffsets expected = make_sparse_role_offsets();
  TypeParam lazy_input;
  {
    TypeParam source = make_input<TypeParam>(expected);
    proto::PackedForwardInput packed_input;
    ASSERT_TRUE(source.to_packed_proto(&packed_input));
    ASSERT_TRUE(lazy_input.from_packed_proto(packed_input));
    source.transfer_kv_infos.clear();
    packed_input.mutable_payload()->assign(packed_input.payload().size(), '\0');
  }

  TypeParam decoded =
      lazy_input.to(torch::Device(torch::kCPU), torch::kFloat32);
  ASSERT_EQ(decoded.transfer_kv_infos.size(), 1);
  const auto& transfer = decoded.transfer_kv_infos.front();
  EXPECT_EQ(transfer.request_id, "sparse-index");
  EXPECT_TRUE(transfer.rank_local_mapping);
  expect_role_offsets(transfer.dst_virtual_kv_cache_layer_global_offsets,
                      expected);
  ASSERT_EQ(transfer.mappings.size(), 1);
  EXPECT_EQ(transfer.mappings.front().remote_ids,
            (std::vector<uint64_t>{8, 1, 9}));
}

TYPED_TEST(ForwardSharedMemoryInputTest,
           SharedMemoryReplacesSparseIndexOffsetsOnFollowingInput) {
  std::string name = "forward_input_offsets_" + std::to_string(getpid()) + "_" +
                     ::testing::UnitTest::GetInstance()
                         ->current_test_info()
                         ->test_suite_name();
  std::replace(name.begin(), name.end(), '/', '_');
  bool is_creator = false;
  constexpr size_t kSegmentBytes = 1 << 20;
  ForwardSharedMemoryTransport writer(
      name, kSegmentBytes, is_creator, ForwardType::RAW_INPUT);
  ASSERT_TRUE(is_creator);
  ForwardSharedMemoryTransport reader(
      name, kSegmentBytes, is_creator, ForwardType::RAW_INPUT);
  ASSERT_FALSE(is_creator);
  const LayerOffsets expected = make_sparse_role_offsets();
  TypeParam source = make_input<TypeParam>(expected);
  ASSERT_TRUE(
      ForwardSharedMemoryTransport::can_write_input(source, kSegmentBytes));
  ASSERT_TRUE(writer.input_write(source));
  source.transfer_kv_infos.clear();

  TypeParam decoded;
  ASSERT_TRUE(reader.input_read(
      decoded,
      torch::Device(torch::kCPU),
      InputDeviceMaterializationPolicy::DEFER_TO_WORKER_PREPARE));
  ASSERT_EQ(decoded.transfer_kv_infos.size(), 1);
  expect_role_offsets(decoded.transfer_kv_infos.front()
                          .dst_virtual_kv_cache_layer_global_offsets,
                      expected);

  const LayerOffsets shorter = {{{4096}, {8192}, {}}};
  source = make_input<TypeParam>(shorter);
  ASSERT_TRUE(writer.input_write(source));
  ASSERT_TRUE(reader.input_read(
      decoded,
      torch::Device(torch::kCPU),
      InputDeviceMaterializationPolicy::DEFER_TO_WORKER_PREPARE));
  ASSERT_EQ(decoded.transfer_kv_infos.size(), 1);
  expect_role_offsets(decoded.transfer_kv_infos.front()
                          .dst_virtual_kv_cache_layer_global_offsets,
                      shorter);
}

TYPED_TEST(ForwardSharedMemoryInputTest,
           PackedInputPreservesNoncontiguousCpuTensorValues) {
  TypeParam source = make_input<TypeParam>({});
  source.token_ids = torch::tensor({11, 0, 23, 0}, torch::kInt32)
                         .slice(/*dim=*/0, /*start=*/0, /*end=*/4, /*step=*/2);
  source.positions = torch::tensor({0, 0, 1, 0}, torch::kInt32)
                         .slice(/*dim=*/0, /*start=*/0, /*end=*/4, /*step=*/2);
  ASSERT_FALSE(source.token_ids.is_contiguous());
  ASSERT_FALSE(source.positions.is_contiguous());
  proto::PackedForwardInput packed;
  ASSERT_TRUE(source.to_packed_proto(&packed));
  source.token_ids.zero_();
  source.positions.zero_();

  TypeParam lazy;
  ASSERT_TRUE(lazy.from_packed_proto(packed));
  const TypeParam decoded =
      lazy.to(torch::Device(torch::kCPU), torch::kFloat32);
  EXPECT_TRUE(
      torch::equal(decoded.token_ids, torch::tensor({11, 23}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(decoded.positions, torch::tensor({0, 1}, torch::kInt32)));
}

}  // namespace
}  // namespace xllm
