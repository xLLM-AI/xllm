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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/framework/multimodal/mm_visitor.h"
#include "core/runtime/forward_shared_memory_transport.h"
#include "core/runtime/vlm_forward_params.h"

namespace xllm {
class ForwardSharedMemoryTransportTestPeer final {
 public:
  static void observe_input_wait(ForwardSharedMemoryTransport& transport,
                                 std::function<void()> observer) {
    transport.input_wait_observer_ = std::move(observer);
  }
};

namespace {

class VlmHostMetadataTransportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    previous_contiguous_ =
        ExecutionConfig::get_instance().use_contiguous_input_buffer();
    ExecutionConfig::get_instance().use_contiguous_input_buffer(true);
  }

  void TearDown() override {
    ExecutionConfig::get_instance().use_contiguous_input_buffer(
        previous_contiguous_);
  }

 private:
  bool previous_contiguous_ = false;
};

VlmForwardInput make_vision_input(int64_t frames, float pixel_value = 1.0F) {
  VlmForwardInput input;
  input.token_ids = torch::tensor({11, 23}, torch::kInt32);
  input.positions = torch::tensor({0, 1}, torch::kInt32);
  input.input_params.meta.num_sequences = 1;
  input.input_params.attention.host.q_seq_lens = {2};
  input.input_params.attention.host.q_cu_seq_lens = {0, 2};
  input.input_params.attention.host.kv_seq_lens = {2};
  MMItemVec items;
  items.reserve(2);
  items.emplace_back(
      MMType::IMAGE,
      MMDict{{"image_grid_thw", torch::tensor({{1, 4, 4}}, torch::kInt32)},
             {"pixel_values", torch::full({16, 8}, pixel_value)}});
  items.emplace_back(
      MMType::VIDEO,
      MMDict{
          {"video_grid_thw",
           torch::tensor({{frames, int64_t{4}, int64_t{4}}}, torch::kInt64)},
          {"pixel_values_videos", torch::full({frames * 16, 8}, pixel_value)}});
  std::vector<MMData> requests;
  requests.reserve(1);
  requests.emplace_back(MMType::IMAGE | MMType::VIDEO, std::move(items));
  input.input_params.multimodal.mm_data.batch(std::move(requests));
  return input;
}

void expect_host_grids(VlmForwardInput& input, int64_t frames) {
  auto& data = input.input_params.multimodal.mm_data;
  EncoderInputGatherVisitor gather;
  ASSERT_TRUE(data.foreach (gather));
  ASSERT_TRUE(gather.finish(data));
  const torch::Tensor image_grid =
      data.get<torch::Tensor>("image_grid_thw").value();
  const torch::Tensor video_grid =
      data.get<torch::Tensor>("video_grid_thw").value();
  EXPECT_TRUE(image_grid.device().is_cpu());
  EXPECT_TRUE(video_grid.device().is_cpu());
  EXPECT_TRUE(
      torch::equal(image_grid, torch::tensor({{1, 4, 4}}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      video_grid,
      torch::tensor({{frames, int64_t{4}, int64_t{4}}}, torch::kInt64)));
  EXPECT_EQ(data.get<torch::Tensor>("pixel_values")->device().type(),
            torch::kPrivateUse1);
  EXPECT_EQ(data.get<torch::Tensor>("pixel_values_videos")->device().type(),
            torch::kPrivateUse1);
}

TEST_F(VlmHostMetadataTransportTest, PackedInputPreservesHostGridPolicy) {
  proto::PackedForwardInput packed;
  ASSERT_TRUE(make_vision_input(/*frames=*/2).to_packed_proto(&packed));
  VlmForwardInput lazy;
  ASSERT_TRUE(lazy.from_packed_proto(packed));
  ASSERT_FALSE(lazy.input_params.multimodal.mm_data.valid());
  const torch::Device device(torch::kPrivateUse1, 0);
  auto prepared =
      lazy.to(device, torch::kFloat32, /*keep_vision_grid_on_cpu=*/true);
  ASSERT_TRUE(prepared.runtime.device_tensors_ready);
  expect_host_grids(prepared, /*frames=*/2);
  auto reused =
      prepared.to(device, torch::kFloat32, /*keep_vision_grid_on_cpu=*/true);
  expect_host_grids(reused, /*frames=*/2);

  auto legacy = lazy.to(device, torch::kFloat32);
  EXPECT_EQ(legacy.input_params.multimodal.mm_data
                .get<torch::Tensor>("image_grid_thw")
                ->device()
                .type(),
            torch::kPrivateUse1);
}

TEST_F(VlmHostMetadataTransportTest,
       SharedMemoryReuseDoesNotOverwriteHostGrids) {
  const std::string name = "opt03_host_grid_" + std::to_string(getpid());
  constexpr size_t kSegmentBytes = 1 << 20;
  bool is_creator = false;
  ForwardSharedMemoryTransport writer(
      name, kSegmentBytes, is_creator, ForwardType::RAW_INPUT);
  ASSERT_TRUE(is_creator);
  ForwardSharedMemoryTransport reader(
      name, kSegmentBytes, is_creator, ForwardType::RAW_INPUT);
  const torch::Device device(torch::kPrivateUse1, 0);
  VlmForwardInput first;
  std::promise<void> entered_wait;
  auto entered = entered_wait.get_future();
  std::promise<void> resume_wait;
  auto resume = resume_wait.get_future();
  ForwardSharedMemoryTransportTestPeer::observe_input_wait(reader, [&]() {
    entered_wait.set_value();
    resume.wait();
  });
  auto pending_read = std::async(std::launch::async, [&]() {
    return reader.input_read(
        first, device, InputDeviceMaterializationPolicy::MATERIALIZE_ON_READ);
  });
  const std::future_status wait_status =
      entered.wait_for(std::chrono::seconds(5));
  // Publish only after the reader has entered wait_for_input. A policy cached
  // before the wait will deterministically retain the old false value.
  reader.set_keep_vision_grid_on_cpu(/*enabled=*/true);
  const bool write_ok = writer.input_write(make_vision_input(/*frames=*/2));
  if (!write_ok || wait_status != std::future_status::ready) {
    reader.stop_input_read();
  }
  resume_wait.set_value();
  const bool read_ok = pending_read.get();
  ASSERT_EQ(wait_status, std::future_status::ready);
  ASSERT_TRUE(write_ok);
  ASSERT_TRUE(read_ok);
  ForwardSharedMemoryTransportTestPeer::observe_input_wait(reader, {});
  ASSERT_TRUE(writer.input_write(make_vision_input(/*frames=*/3)));
  VlmForwardInput second;
  ASSERT_TRUE(reader.input_read(
      second, device, InputDeviceMaterializationPolicy::MATERIALIZE_ON_READ));
  expect_host_grids(first, /*frames=*/2);
  expect_host_grids(second, /*frames=*/3);
}

TEST_F(VlmHostMetadataTransportTest,
       DeferredPrepareOwnsPayloadAcrossReuseAndUnmapping) {
  const torch::Device device(torch::kPrivateUse1, 0);
  VlmForwardInput first;
  VlmForwardInput second;
  {
    const std::string name = "opt03_deferred_grid_" + std::to_string(getpid());
    constexpr size_t kSegmentBytes = 1 << 20;
    bool is_creator = false;
    ForwardSharedMemoryTransport writer(
        name, kSegmentBytes, is_creator, ForwardType::RAW_INPUT);
    ASSERT_TRUE(is_creator);
    ForwardSharedMemoryTransport reader(
        name, kSegmentBytes, is_creator, ForwardType::RAW_INPUT);
    reader.set_keep_vision_grid_on_cpu(/*enabled=*/true);
    ASSERT_TRUE(writer.input_write(make_vision_input(/*frames=*/2)));
    ASSERT_TRUE(reader.input_read(
        first,
        device,
        InputDeviceMaterializationPolicy::DEFER_TO_WORKER_PREPARE));
    ASSERT_FALSE(first.runtime.device_tensors_ready);
    ASSERT_TRUE(first.runtime.input_host_buffer_has_layout);
    // Change pixels as well as the grid so stale/aliased payloads are visible.
    auto next = make_vision_input(/*frames=*/3, /*pixel_value=*/7.0F);
    ASSERT_TRUE(writer.input_write(next));
    ASSERT_TRUE(reader.input_read(
        second,
        device,
        InputDeviceMaterializationPolicy::DEFER_TO_WORKER_PREPARE));
  }
  // Both SHM mappings are gone before worker-side unpacking/materialization.
  auto prepared_first = first.to(device,
                                 torch::kFloat32,
                                 /*keep_vision_grid_on_cpu=*/true);
  auto prepared_second = second.to(device,
                                   torch::kFloat32,
                                   /*keep_vision_grid_on_cpu=*/true);
  expect_host_grids(prepared_first, /*frames=*/2);
  expect_host_grids(prepared_second, /*frames=*/3);
  for (const std::string key : {"pixel_values", "pixel_values_videos"}) {
    const auto old_pixels =
        prepared_first.input_params.multimodal.mm_data.get<torch::Tensor>(key)
            .value()
            .cpu();
    const auto new_pixels =
        prepared_second.input_params.multimodal.mm_data.get<torch::Tensor>(key)
            .value()
            .cpu();
    EXPECT_TRUE(torch::equal(old_pixels, torch::ones_like(old_pixels)));
    EXPECT_TRUE(torch::equal(new_pixels, torch::full_like(new_pixels, 7)));
  }
}

}  // namespace
}  // namespace xllm
