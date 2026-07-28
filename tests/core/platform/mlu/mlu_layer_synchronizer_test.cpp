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

#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <memory>

#include "core/framework/model/model_input_params.h"
#include "core/platform/device.h"
#include "core/platform/layer_synchronizer.h"
#include "core/platform/platform.h"

namespace xllm::mlu {
namespace {

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

TEST_F(MLULayerSynchronizerTest, RecordFailureAbortsPendingWaits) {
  std::shared_ptr<LayerSynchronizer> synchronizer =
      create_layer_synchronizer(/*num_layers=*/2);
  ASSERT_NE(synchronizer, nullptr);

  EXPECT_FALSE(
      synchronizer->record_stream(/*layer_index=*/0, /*stream=*/nullptr));
  EXPECT_FALSE(synchronizer->synchronize_layer(/*layer_index=*/1));
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
