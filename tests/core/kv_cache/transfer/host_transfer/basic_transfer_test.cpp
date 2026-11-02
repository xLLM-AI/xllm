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

#include "core/kv_cache/transfer/host_transfer/basic_transfer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/kv_cache/transfer/host_transfer/compact_transfer.h"
#include "core/kv_cache/transfer/host_transfer/layout.h"
#include "core/kv_cache/transfer/host_transfer/transfer.h"
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

  BasicHostKVTransfer composite(make_layout(device, /*num_layers=*/4),
                                device,
                                *compute_stream,
                                /*layer_copy_batches=*/4);
  HostKVLoadHandle composite_handle = composite.prepare_load(/*draft=*/true);
  EXPECT_EQ(composite_handle.synchronizer->size(), 5U);
  composite.drain();
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

TEST(BasicHostKVTransferTest, AbortedReloadCannotBeQueuedOnConsumerStream) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "An accelerator device is required for Host KV transfer.";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  auto synchronizer = create_layer_synchronizer(/*num_layers=*/2);
  ASSERT_NE(synchronizer, nullptr);
  synchronizer->abort();
  auto consumer = device.get_stream_from_pool();
  auto guard = consumer->set_stream_guard();
  EXPECT_FALSE(synchronizer->wait_all_on_current_stream());
}

TEST(BasicHostKVTransferTest, RoundTripUsesConfiguredLayerEventGroups) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "An accelerator device is required for Host KV transfer.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  std::unique_ptr<Stream> compute_stream = device.get_stream_from_pool();
  auto compute_guard = compute_stream->set_stream_guard();
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
  BasicHostKVTransfer transfer(std::move(layout),
                               device,
                               *compute_stream,
                               /*layer_copy_batches=*/2);

  const HostKVRequest offload_request{{HostKVMapping{5, 0, 0}}};
  ASSERT_TRUE(transfer.offload(offload_request));
  for (int64_t layer_index = 0; layer_index < 4; ++layer_index) {
    const torch::Tensor expected =
        torch::full({2}, 10.0 + static_cast<double>(layer_index), host_options);
    EXPECT_TRUE(torch::equal(host_key[0][layer_index], expected));
  }
  const HostKVRequest load_request{{HostKVMapping{5, 0, 1}}};
  HostKVLoadHandle handle = transfer.prepare_load();
  handle.producer_ready = compute_stream->record_event();
  ASSERT_NE(handle.producer_ready, nullptr);
  ASSERT_NE(handle.synchronizer, nullptr);
  EXPECT_EQ(handle.synchronizer->size(), 2U);
  EXPECT_EQ(handle.layers_per_event, 2U);
  ASSERT_TRUE(transfer.load(load_request, handle));
  {
    auto consumer = device.get_stream_from_pool();
    auto consumer_guard = consumer->set_stream_guard();
    for (int64_t layer_index = 0; layer_index < 4; ++layer_index) {
      if (layer_index % handle.layers_per_event == 0) {
        ASSERT_TRUE(handle.synchronizer->wait_layer_on_current_stream(
            layer_index / handle.layers_per_event));
      }
      const torch::Tensor copied = device_layers[layer_index][1] * 2.0;
      const torch::Tensor expected = torch::full(
          {2}, 2.0 * (10.0 + static_cast<double>(layer_index)), host_options);
      EXPECT_TRUE(torch::equal(copied.cpu(), expected));
    }
  }
  transfer.drain();
  transfer.drain();
}

TEST(BasicHostKVTransferTest, RestoresNarrowedBlockViewsAcrossRoles) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "An accelerator device is required for Host KV transfer.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  std::unique_ptr<Stream> compute_stream = device.get_stream_from_pool();
  auto compute_guard = compute_stream->set_stream_guard();
  const torch::TensorOptions host_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  const torch::TensorOptions device_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap());

  torch::Tensor conv_storage = torch::zeros({2, 5, 3}, device_options);
  torch::Tensor ssm_storage = torch::zeros({8, 2}, device_options);
  torch::Tensor tail_storage = torch::zeros({2, 2, 4, 3}, device_options);
  conv_storage[0].fill_(11.0);
  conv_storage[1].fill_(22.0);
  ssm_storage.narrow(0, 0, 4).fill_(33.0);
  ssm_storage.narrow(0, 4, 4).fill_(44.0);
  tail_storage[0].fill_(55.0);
  tail_storage[1].fill_(66.0);

  torch::Tensor conv_view = conv_storage.narrow(1, 0, 3);
  torch::Tensor ssm_view = ssm_storage.view({2, 4, 2}).narrow(1, 0, 1);

  torch::Tensor conv_host = torch::zeros({2, 1, 3, 3}, host_options);
  torch::Tensor ssm_host = torch::zeros({2, 1, 1, 2}, host_options);
  torch::Tensor tail_host = torch::zeros({2, 1, 2, 4, 3}, host_options);
  HostKVGroupLayout group;
  group.group_id = 7;
  group.host_roles.emplace(KVCacheTensorRole::CONV, conv_host);
  group.host_roles.emplace(KVCacheTensorRole::SSM, ssm_host);
  group.host_roles.emplace(KVCacheTensorRole::KPOOL_TAIL, tail_host);
  group.layers.emplace_back(
      HostKVLayerLayout{0,
                        0,
                        {{KVCacheTensorRole::CONV, conv_view},
                         {KVCacheTensorRole::SSM, ssm_view},
                         {KVCacheTensorRole::KPOOL_TAIL, tail_storage}}});
  HostKVLayout layout(/*num_layers=*/1, {std::move(group)}, device.unwrap());
  BasicHostKVTransfer transfer(std::move(layout),
                               device,
                               *compute_stream,
                               /*layer_copy_batches=*/1);

  const HostKVRequest offload_request{{HostKVMapping{7, 0, 1}}};
  ASSERT_TRUE(transfer.offload(offload_request));
  EXPECT_TRUE(
      torch::equal(conv_host[0][0], torch::full({3, 3}, 22.0, host_options)));
  EXPECT_TRUE(
      torch::equal(ssm_host[0][0], torch::full({1, 2}, 44.0, host_options)));
  EXPECT_TRUE(torch::equal(tail_host[0][0],
                           torch::full({2, 4, 3}, 66.0, host_options)));

  const double kSentinel = -1.0;
  conv_storage.fill_(kSentinel);
  ssm_storage.fill_(kSentinel);
  tail_storage.fill_(kSentinel);
  const HostKVRequest load_request{{HostKVMapping{7, 0, 1}}};
  HostKVLoadHandle handle = transfer.prepare_load();
  handle.producer_ready = compute_stream->record_event();
  ASSERT_NE(handle.producer_ready, nullptr);
  ASSERT_NE(handle.synchronizer, nullptr);
  ASSERT_TRUE(transfer.load(load_request, handle));
  {
    auto consumer = device.get_stream_from_pool();
    auto consumer_guard = consumer->set_stream_guard();
    ASSERT_TRUE(handle.synchronizer->wait_layer_on_current_stream(0));
    EXPECT_TRUE(torch::equal(conv_storage[1].narrow(0, 0, 3).cpu(),
                             torch::full({3, 3}, 22.0, host_options)));
    EXPECT_TRUE(torch::equal(ssm_storage.narrow(0, 4, 1).cpu(),
                             torch::full({1, 2}, 44.0, host_options)));
    EXPECT_TRUE(torch::equal(tail_storage[1].cpu(),
                             torch::full({2, 4, 3}, 66.0, host_options)));
    EXPECT_TRUE(torch::equal(conv_storage[1].narrow(0, 3, 2).cpu(),
                             torch::full({2, 3}, kSentinel, host_options)));
    EXPECT_TRUE(torch::equal(ssm_storage.narrow(0, 5, 3).cpu(),
                             torch::full({3, 2}, kSentinel, host_options)));
    EXPECT_TRUE(torch::equal(conv_storage[0].cpu(),
                             torch::full({5, 3}, kSentinel, host_options)));
    EXPECT_TRUE(torch::equal(ssm_storage.narrow(0, 0, 4).cpu(),
                             torch::full({4, 2}, kSentinel, host_options)));
    EXPECT_TRUE(torch::equal(tail_storage[0].cpu(),
                             torch::full({2, 4, 3}, kSentinel, host_options)));
  }
  transfer.drain();
}

}  // namespace
}  // namespace xllm
