/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/kv_cache_transfer/host_transfer/basic_transfer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "framework/kv_cache_transfer/host_transfer/layout.h"
#include "platform/device.h"
#include "platform/layer_synchronizer.h"
#include "platform/platform.h"

namespace xllm {
namespace {

TEST(BasicHostKVTransferTest, RoundTripUsesConfiguredLayerEventGroups) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "An accelerator device is required for Host KV transfer.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const torch::TensorOptions host_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  const torch::TensorOptions device_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap());
  torch::Tensor host_key = torch::zeros({2, 4, 2}, host_options);

  HostKVGroupLayout group;
  group.group_id = 5;
  group.host_roles.emplace(KVCacheTensorRole::KEY, host_key);
  group.layers.reserve(4);
  std::vector<torch::Tensor> device_layers;
  device_layers.reserve(4);
  for (int64_t layer_index = 0; layer_index < 4; ++layer_index) {
    torch::Tensor blocks = torch::zeros({2, 2}, device_options);
    blocks[0].fill_(10.0 + static_cast<double>(layer_index));
    device_layers.emplace_back(blocks);
    group.layers.emplace_back(
        HostKVLayerLayout{layer_index,
                          layer_index,
                          {{KVCacheTensorRole::KEY, std::move(blocks)}}});
  }
  HostKVLayout layout(/*num_layers=*/4, {std::move(group)}, device.unwrap());
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  BasicHostKVTransfer transfer(std::move(layout),
                               device,
                               *compute_stream,
                               /*layer_copy_batches=*/2);

  const HostKVRequest offload_request{{HostKVMapping{5, 0, 0}}};
  ASSERT_TRUE(transfer.offload(offload_request));
  const HostKVRequest load_request{{HostKVMapping{5, 0, 1}}};
  HostKVLoadHandle handle = transfer.prepare_load();
  ASSERT_NE(handle.synchronizer, nullptr);
  EXPECT_EQ(handle.synchronizer->size(), 2U);
  EXPECT_EQ(handle.layers_per_event, 2U);
  ASSERT_TRUE(transfer.load(load_request, handle));
  ASSERT_TRUE(handle.synchronizer->synchronize_layer(/*layer_index=*/0));
  ASSERT_TRUE(handle.synchronizer->synchronize_layer(/*layer_index=*/1));

  for (const torch::Tensor& blocks : device_layers) {
    EXPECT_TRUE(torch::equal(blocks[0], blocks[1]));
  }
  transfer.drain();
  transfer.drain();
}

}  // namespace
}  // namespace xllm
