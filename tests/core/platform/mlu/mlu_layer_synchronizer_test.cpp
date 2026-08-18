/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/platform/mlu/mlu_layer_synchronizer.h"

#include <cnrt.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>

#include "core/framework/model/model_input_params.h"
#include "core/platform/device.h"
#include "core/platform/layer_synchronizer.h"
#include "core/platform/platform.h"

namespace xllm::mlu {
namespace {

struct QueueGateState {
  std::atomic<bool> gate_open{false};
  std::atomic<bool> wait_thread_started{false};
  std::atomic<bool> copy_callback_finished{false};
  std::atomic<bool> compute_callback_saw_copy{false};
};

bool wait_for_wait_thread(QueueGateState* state) {
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!state->wait_thread_started.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  return state->wait_thread_started.load(std::memory_order_acquire);
}

void wait_for_queue_gate(void* user_data) {
  QueueGateState* state = static_cast<QueueGateState*>(user_data);
  while (!state->gate_open.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  state->copy_callback_finished.store(true, std::memory_order_release);
}

void observe_copy_completion(void* user_data) {
  QueueGateState* state = static_cast<QueueGateState*>(user_data);
  state->compute_callback_saw_copy.store(
      state->copy_callback_finished.load(std::memory_order_acquire),
      std::memory_order_release);
}

class MLULayerSynchronizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() < 1) {
      GTEST_SKIP() << "MLU device is required for layer synchronizer tests.";
    }
    device_ = std::make_unique<Device>(/*device_index=*/0);
    device_->set_device();
  }

  std::unique_ptr<Device> device_;
};

TEST_F(MLULayerSynchronizerTest, RecordsAndWaitsOnProvidedCopyStream) {
  std::shared_ptr<LayerSynchronizer> synchronizer =
      create_layer_synchronizer(/*num_layers=*/2);
  ASSERT_NE(synchronizer, nullptr);
  ASSERT_EQ(synchronizer->size(), 2U);
  std::unique_ptr<Stream> copy_stream = device_->get_stream_from_pool();
  torch::Tensor tensor = torch::zeros(
      {16},
      torch::TensorOptions().dtype(torch::kInt32).device(device_->unwrap()));
  {
    const c10::StreamGuard guard = copy_stream->set_stream_guard();
    tensor.fill_(37);
  }

  ASSERT_TRUE(
      synchronizer->record_stream(/*layer_index=*/0, copy_stream.get()));
  ASSERT_TRUE(synchronizer->synchronize_layer(/*layer_index=*/0));
  EXPECT_TRUE(torch::all(tensor.cpu() == 37).item<bool>());
}

TEST_F(MLULayerSynchronizerTest,
       EnqueuesCopyDependencyWithoutBlockingCallingThread) {
  std::shared_ptr<LayerSynchronizer> synchronizer =
      create_layer_synchronizer(/*num_layers=*/1);
  ASSERT_NE(synchronizer, nullptr);
  std::unique_ptr<Stream> copy_stream = device_->get_stream_from_pool();
  std::unique_ptr<Stream> compute_stream = device_->get_stream_from_pool();
  QueueGateState state;

  ASSERT_EQ(
      cnrtInvokeHostFunc(
          copy_stream->get_stream()->stream(), wait_for_queue_gate, &state),
      cnrtSuccess);
  const bool event_recorded =
      synchronizer->record_stream(/*layer_index=*/0, copy_stream.get());
  if (!event_recorded) {
    state.gate_open.store(true, std::memory_order_release);
    EXPECT_EQ(copy_stream->synchronize(), 0);
  }
  ASSERT_TRUE(event_recorded);

  std::future<bool> wait_result = std::async(
      std::launch::async, [&synchronizer, &compute_stream, &state]() {
        const c10::StreamGuard guard = compute_stream->set_stream_guard();
        state.wait_thread_started.store(true, std::memory_order_release);
        if (!synchronizer->synchronize_layer(/*layer_index=*/0)) {
          return false;
        }
        return cnrtInvokeHostFunc(compute_stream->get_stream()->stream(),
                                  observe_copy_completion,
                                  &state) == cnrtSuccess;
      });

  const bool wait_thread_started = wait_for_wait_thread(&state);
  if (!wait_thread_started) {
    state.gate_open.store(true, std::memory_order_release);
    EXPECT_EQ(copy_stream->synchronize(), 0);
  }
  ASSERT_TRUE(wait_thread_started);
  const std::future_status blocked_status =
      wait_result.wait_for(std::chrono::milliseconds(100));
  state.gate_open.store(true, std::memory_order_release);

  ASSERT_EQ(wait_result.wait_for(std::chrono::seconds(2)),
            std::future_status::ready);
  ASSERT_TRUE(wait_result.get());
  ASSERT_EQ(compute_stream->synchronize(), 0);
  EXPECT_EQ(blocked_status, std::future_status::ready);
  EXPECT_TRUE(state.copy_callback_finished.load(std::memory_order_acquire));
  EXPECT_TRUE(state.compute_callback_saw_copy.load(std::memory_order_acquire));
}

TEST_F(MLULayerSynchronizerTest,
       DirectSynchronizerBlocksUntilCopyEventCompletes) {
  MLULayerSynchronizerImpl synchronizer(/*num_layers=*/1);
  std::unique_ptr<Stream> copy_stream = device_->get_stream_from_pool();
  QueueGateState state;

  ASSERT_EQ(
      cnrtInvokeHostFunc(
          copy_stream->get_stream()->stream(), wait_for_queue_gate, &state),
      cnrtSuccess);
  const bool event_recorded =
      synchronizer.record_stream(/*layer_index=*/0, copy_stream.get());
  if (!event_recorded) {
    state.gate_open.store(true, std::memory_order_release);
    EXPECT_EQ(copy_stream->synchronize(), 0);
  }
  ASSERT_TRUE(event_recorded);

  std::future<bool> wait_result =
      std::async(std::launch::async, [&synchronizer, &state]() {
        state.wait_thread_started.store(true, std::memory_order_release);
        return synchronizer.synchronize_layer(/*layer_index=*/0);
      });
  const bool wait_thread_started = wait_for_wait_thread(&state);
  if (!wait_thread_started) {
    state.gate_open.store(true, std::memory_order_release);
    EXPECT_EQ(copy_stream->synchronize(), 0);
  }
  ASSERT_TRUE(wait_thread_started);
  const std::future_status blocked_status =
      wait_result.wait_for(std::chrono::milliseconds(100));
  state.gate_open.store(true, std::memory_order_release);

  ASSERT_EQ(wait_result.wait_for(std::chrono::seconds(2)),
            std::future_status::ready);
  EXPECT_TRUE(wait_result.get());
  ASSERT_EQ(copy_stream->synchronize(), 0);
  EXPECT_EQ(blocked_status, std::future_status::timeout);
  EXPECT_TRUE(state.copy_callback_finished.load(std::memory_order_acquire));
}

TEST_F(MLULayerSynchronizerTest, AbortUnblocksAnUnrecordedRange) {
  std::shared_ptr<LayerSynchronizer> synchronizer =
      create_layer_synchronizer(/*num_layers=*/2);
  ASSERT_NE(synchronizer, nullptr);
  std::future<bool> wait_result =
      std::async(std::launch::async, [synchronizer]() {
        return synchronizer->synchronize_layer(/*layer_index=*/1);
      });
  EXPECT_EQ(wait_result.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);

  synchronizer->abort();

  ASSERT_EQ(wait_result.wait_for(std::chrono::seconds(2)),
            std::future_status::ready);
  EXPECT_FALSE(wait_result.get());
}

TEST_F(MLULayerSynchronizerTest, RecordFailureWaitsForOwnerAbort) {
  std::shared_ptr<LayerSynchronizer> synchronizer =
      create_layer_synchronizer(/*num_layers=*/2);
  ASSERT_NE(synchronizer, nullptr);

  std::future<bool> wait_result =
      std::async(std::launch::async, [synchronizer]() {
        return synchronizer->synchronize_layer(/*layer_index=*/1);
      });

  EXPECT_FALSE(
      synchronizer->record_stream(/*layer_index=*/0, /*stream=*/nullptr));
  EXPECT_EQ(wait_result.wait_for(std::chrono::milliseconds(100)),
            std::future_status::timeout);

  synchronizer->abort();

  ASSERT_EQ(wait_result.wait_for(std::chrono::seconds(2)),
            std::future_status::ready);
  EXPECT_FALSE(wait_result.get());
}

TEST_F(MLULayerSynchronizerTest, ModelInputWaitsAtRangeBoundaries) {
  std::shared_ptr<LayerSynchronizer> synchronizer =
      create_layer_synchronizer(/*num_layers=*/2);
  ASSERT_NE(synchronizer, nullptr);
  std::unique_ptr<Stream> copy_stream = device_->get_stream_from_pool();
  ASSERT_TRUE(
      synchronizer->record_stream(/*layer_index=*/0, copy_stream.get()));
  ModelInputParams params;
  params.parallel.layer_wise_load_synchronizer = synchronizer;
  params.parallel.layers_per_bacth_copy = 2;

  EXPECT_TRUE(params.synchronize_layer(/*layer_idx=*/0));
  EXPECT_TRUE(params.synchronize_layer(/*layer_idx=*/1));
  synchronizer->abort();
  EXPECT_FALSE(params.synchronize_layer(/*layer_idx=*/2));
}

}  // namespace
}  // namespace xllm::mlu
