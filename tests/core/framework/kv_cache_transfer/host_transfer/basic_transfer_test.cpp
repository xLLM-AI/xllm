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

#include "framework/kv_cache_transfer/host_transfer/compact_transfer.h"
#include "framework/kv_cache_transfer/host_transfer/layout.h"
#include "framework/kv_cache_transfer/host_transfer/transfer.h"
#include "platform/device.h"
#include "platform/layer_synchronizer.h"
#include "platform/platform.h"

namespace xllm {
namespace {

HostKVLayout make_layout(const Device& device, int64_t num_layers) {
  const torch::TensorOptions host_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  const torch::TensorOptions device_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap());
  HostKVGroupLayout group;
  group.group_id = 9;
  group.host_roles.emplace(KVCacheTensorRole::KEY,
                           torch::zeros({1, num_layers, 1}, host_options));
  group.layers.reserve(num_layers);
  for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    group.layers.emplace_back(HostKVLayerLayout{
        layer_id,
        layer_id,
        {{KVCacheTensorRole::KEY, torch::zeros({1, 1}, device_options)}}});
  }
  return HostKVLayout(num_layers, {std::move(group)}, device.unwrap());
}

TEST(BasicHostKVTransferTest, KeepsLayerBatchingEdgeSemantics) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "An accelerator device is required for Host KV transfer.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  std::unique_ptr<Stream> compute_stream = device.current_stream();

  BasicHostKVTransfer single_window(make_layout(device, /*num_layers=*/4),
                                    device,
                                    *compute_stream,
                                    /*layer_copy_batches=*/0);
  HostKVLoadHandle single_handle = single_window.prepare_load();
  EXPECT_EQ(single_handle.synchronizer->size(), 1U);
  EXPECT_EQ(single_handle.layers_per_event, 4U);
  single_window.drain();

  BasicHostKVTransfer per_layer(make_layout(device, /*num_layers=*/4),
                                device,
                                *compute_stream,
                                /*layer_copy_batches=*/8);
  HostKVLoadHandle per_layer_handle = per_layer.prepare_load();
  EXPECT_EQ(per_layer_handle.synchronizer->size(), 4U);
  EXPECT_EQ(per_layer_handle.layers_per_event, 1U);
  per_layer.drain();
}

TEST(HostKVTransferFactoryTest, SelectsConfiguredStrategy) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "An accelerator device is required for Host KV transfer.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  std::unique_ptr<Stream> compute_stream = device.current_stream();

  HostKVTransferConfig config;
  std::unique_ptr<HostKVTransfer> automatic = create_host_kv_transfer(
      make_layout(device, /*num_layers=*/1), device, *compute_stream, config);
  if (Platform::supports_compact_host_kv_transfer()) {
    EXPECT_NE(dynamic_cast<CompactHostKVTransfer*>(automatic.get()), nullptr);
  } else {
    EXPECT_NE(dynamic_cast<BasicHostKVTransfer*>(automatic.get()), nullptr);
  }
  automatic->drain();

  config.mode = HostKVTransferMode::BASIC;
  std::unique_ptr<HostKVTransfer> basic = create_host_kv_transfer(
      make_layout(device, /*num_layers=*/1), device, *compute_stream, config);
  EXPECT_NE(dynamic_cast<BasicHostKVTransfer*>(basic.get()), nullptr);
  basic->drain();
}

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
