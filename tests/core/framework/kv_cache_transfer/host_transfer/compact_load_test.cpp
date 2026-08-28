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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "framework/kv_cache_transfer/host_transfer/compact_transfer.h"
#include "framework/kv_cache_transfer/host_transfer/layout.h"
#include "platform/batch_memcpy.h"
#include "platform/device.h"
#include "platform/layer_synchronizer.h"
#include "platform/platform.h"

namespace xllm {
namespace {

TEST(CompactLoadTest, UsesApprovedDefaultStagingBudget) {
  EXPECT_EQ(CompactTransferConfig{}.load_target_bytes, 64ULL * 1024 * 1024);
}

using CompactLoadLayerLayout = HostKVLayerLayout;
using CompactLoadGroupLayout = HostKVGroupLayout;
using CompactLoadLayout = HostKVLayoutInput;
using CompactLoadMapping = HostKVMapping;
using CompactLoadRequest = HostKVRequest;

class RecordingH2DBatchMemcpy final : public BatchMemcpy {
 public:
  explicit RecordingH2DBatchMemcpy(const Device& device, bool fail_h2d = false)
      : delegate_(create_batch_memcpy(device)), fail_h2d_(fail_h2d) {
    CHECK(delegate_ != nullptr);
  }

  void init(int32_t device_id) override { delegate_->init(device_id); }

  bool submit_h2d(const std::vector<torch::Tensor>& src_tensors,
                  const std::vector<torch::Tensor>& dst_tensors,
                  Stream* stream) override {
    descriptor_bytes_.emplace_back();
    descriptor_bytes_.back().reserve(src_tensors.size());
    for (const torch::Tensor& source : src_tensors) {
      descriptor_bytes_.back().emplace_back(source.nbytes());
    }
    if (fail_h2d_) {
      return false;
    }
    return delegate_->submit_h2d(src_tensors, dst_tensors, stream);
  }

  bool copy_d2h(const std::vector<torch::Tensor>& src_tensors,
                const std::vector<torch::Tensor>& dst_tensors,
                Stream* stream) override {
    return delegate_->copy_d2h(src_tensors, dst_tensors, stream);
  }

  const std::vector<std::vector<size_t>>& descriptor_bytes() const {
    return descriptor_bytes_;
  }

 private:
  std::unique_ptr<BatchMemcpy> delegate_;
  bool fail_h2d_ = false;
  std::vector<std::vector<size_t>> descriptor_bytes_;
};

class BorrowedBatchMemcpy final : public BatchMemcpy {
 public:
  explicit BorrowedBatchMemcpy(BatchMemcpy& delegate) : delegate_(delegate) {}

  void init(int32_t device_id) override { delegate_.init(device_id); }

  bool submit_h2d(const std::vector<torch::Tensor>& src_tensors,
                  const std::vector<torch::Tensor>& dst_tensors,
                  Stream* stream) override {
    return delegate_.submit_h2d(src_tensors, dst_tensors, stream);
  }

  bool submit_d2h(const std::vector<torch::Tensor>& src_tensors,
                  const std::vector<torch::Tensor>& dst_tensors,
                  Stream* stream) override {
    return delegate_.submit_d2h(src_tensors, dst_tensors, stream);
  }

  bool copy_d2h(const std::vector<torch::Tensor>& src_tensors,
                const std::vector<torch::Tensor>& dst_tensors,
                Stream* stream) override {
    return delegate_.copy_d2h(src_tensors, dst_tensors, stream);
  }

 private:
  BatchMemcpy& delegate_;
};

class CompactLoadHarness final {
 public:
  CompactLoadHarness(const HostKVLayoutInput& layout,
                     const Device& device,
                     uint32_t layer_copy_batches,
                     BatchMemcpy& batch_memcpy,
                     size_t target_bytes = 16ULL * 1024 * 1024) {
    compute_stream_ = device.current_stream();
    CompactTransferConfig config;
    config.load_target_bytes = target_bytes;
    config.offload_target_bytes = 16ULL * 1024 * 1024;
    transfer_ = std::make_unique<CompactHostKVTransfer>(
        HostKVLayout(layout, device.unwrap()),
        device,
        *compute_stream_,
        layer_copy_batches,
        std::make_unique<BorrowedBatchMemcpy>(batch_memcpy),
        config);
  }

  bool execute(const HostKVRequest& request,
               const std::shared_ptr<LayerSynchronizer>& synchronizer) {
    HostKVLoadHandle handle = transfer_->prepare_load();
    handle.synchronizer = synchronizer;
    return transfer_->load(request, handle);
  }

  void drain() { transfer_->drain(); }

 private:
  std::unique_ptr<Stream> compute_stream_;
  std::unique_ptr<CompactHostKVTransfer> transfer_;
};

class RejectingSynchronizer final : public LayerSynchronizer {
 public:
  bool synchronize_layer(int64_t /*layer_index*/) override { return false; }
  bool record_stream(int64_t /*layer_index*/, Stream* /*stream*/) override {
    return false;
  }
  void abort() override { aborted_ = true; }
  uint32_t size() const override { return 1; }
  bool aborted() const { return aborted_; }

 private:
  bool aborted_ = false;
};

torch::Tensor make_host_blocks(int64_t block_count,
                               int64_t layer_count,
                               double offset) {
  torch::Tensor blocks = torch::empty(
      {block_count, layer_count, 2},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  for (int64_t block_id = 0; block_id < block_count; ++block_id) {
    for (int64_t layer_id = 0; layer_id < layer_count; ++layer_id) {
      blocks[block_id][layer_id].fill_(offset + block_id * 10 + layer_id);
    }
  }
  return blocks;
}

CompactLoadLayout make_layout(const torch::Tensor& host_key,
                              const torch::Tensor& device_key_layer_zero,
                              const torch::Tensor& device_key_layer_one) {
  CompactLoadGroupLayout group;
  group.group_id = 17;
  group.host_roles.emplace(KVCacheTensorRole::KEY, host_key);
  group.layers = {
      CompactLoadLayerLayout{
          0, 0, {{KVCacheTensorRole::KEY, device_key_layer_zero}}},
      CompactLoadLayerLayout{
          1, 1, {{KVCacheTensorRole::KEY, device_key_layer_one}}},
  };

  CompactLoadLayout layout;
  layout.num_layers = 2;
  layout.groups = {std::move(group)};
  return layout;
}

CompactLoadRequest make_request() {
  CompactLoadRequest request;
  request.mappings = {
      CompactLoadMapping{17, 3, 2},
      CompactLoadMapping{17, 0, 4},
      CompactLoadMapping{17, 4, 0},
      CompactLoadMapping{17, 1, 3},
      CompactLoadMapping{17, 2, 1},
  };
  return request;
}

CompactLoadLayout make_single_role_layout(int64_t num_layers,
                                          int64_t group_layer_slot,
                                          const torch::Tensor& host_tensor,
                                          const torch::Tensor& device_tensor) {
  CompactLoadGroupLayout group;
  group.group_id = 61;
  group.host_roles.emplace(KVCacheTensorRole::KEY, host_tensor);
  group.layers = {
      CompactLoadLayerLayout{num_layers - 1,
                             group_layer_slot,
                             {{KVCacheTensorRole::KEY, device_tensor}}},
  };

  CompactLoadLayout layout;
  layout.num_layers = num_layers;
  layout.groups = {std::move(group)};
  return layout;
}

TEST(HostKVLayoutTest, SortsGroupsAndFiltersActiveRoleLayers) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for host KV layout.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();

  torch::Tensor host_key =
      make_host_blocks(/*block_count=*/2, /*layer_count=*/2, /*offset=*/10.0);
  torch::Tensor host_value =
      make_host_blocks(/*block_count=*/2, /*layer_count=*/2, /*offset=*/20.0);
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap());
  torch::Tensor key_layer_one = torch::zeros({2, 2}, options);
  torch::Tensor value_layer_zero = torch::zeros({2, 2}, options);

  HostKVGroupLayout group;
  group.group_id = 17;
  group.host_roles.emplace(KVCacheTensorRole::KEY, host_key);
  group.host_roles.emplace(KVCacheTensorRole::VALUE, host_value);
  group.layers = {
      HostKVLayerLayout{1, 1, {{KVCacheTensorRole::KEY, key_layer_one}}},
      HostKVLayerLayout{0, 0, {{KVCacheTensorRole::VALUE, value_layer_zero}}},
  };

  HostKVLayout layout(/*num_layers=*/2, {std::move(group)}, device.unwrap());
  const std::vector<KVCacheTensorRole::Value>& roles = layout.active_roles(17);
  ASSERT_EQ(roles.size(), 2);
  EXPECT_EQ(roles[0], KVCacheTensorRole::KEY);
  EXPECT_EQ(roles[1], KVCacheTensorRole::VALUE);

  const std::vector<const HostKVLayerLayout*> key_layers =
      layout.active_layers(17,
                           KVCacheTensorRole::KEY,
                           /*begin_layer=*/0,
                           /*end_layer=*/2);
  ASSERT_EQ(key_layers.size(), 1);
  EXPECT_EQ(key_layers[0]->absolute_layer_id, 1);
  EXPECT_EQ(key_layers[0]->group_layer_slot, 1);

  const std::vector<const torch::Tensor*> value_tensors =
      layout.active_tensors(17,
                            KVCacheTensorRole::VALUE,
                            /*begin_layer=*/0,
                            /*end_layer=*/1);
  ASSERT_EQ(value_tensors.size(), 1);
  EXPECT_TRUE(torch::equal(*value_tensors[0], value_layer_zero));
}

TEST(CompactLoadTest, RestoresLayerWindowFromDirectHostSlices) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for compact H2D.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();

  constexpr int64_t kBlockCount = 5;
  torch::Tensor host_key =
      make_host_blocks(kBlockCount, /*layer_count=*/2, /*offset=*/100.0);
  torch::Tensor device_key_layer_zero = torch::full(
      {kBlockCount, 2},
      -1.0,
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap()));
  torch::Tensor device_key_layer_one = torch::full(
      {kBlockCount, 2},
      -1.0,
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap()));

  RecordingH2DBatchMemcpy batch_memcpy(device);
  CompactLoadHarness compact_load(
      make_layout(host_key, device_key_layer_zero, device_key_layer_one),
      device,
      /*layer_copy_batches=*/1,
      batch_memcpy,
      /*target_bytes=*/32);
  std::shared_ptr<LayerSynchronizer> synchronizer =
      create_layer_synchronizer(/*num_layers=*/1);
  ASSERT_NE(synchronizer, nullptr);

  const bool submitted = compact_load.execute(make_request(), synchronizer);
  ASSERT_TRUE(submitted);
  ASSERT_TRUE(synchronizer->synchronize_layer(/*layer_index=*/0));
  ASSERT_EQ(batch_memcpy.descriptor_bytes().size(), 3);
  EXPECT_EQ(batch_memcpy.descriptor_bytes()[0],
            (std::vector<size_t>{16U, 16U, 16U}));
  EXPECT_EQ(batch_memcpy.descriptor_bytes()[1],
            (std::vector<size_t>{16U, 16U, 16U}));
  EXPECT_EQ(batch_memcpy.descriptor_bytes()[2], (std::vector<size_t>{16U, 8U}));

  const std::vector<int64_t> destination_ids = {2, 4, 0, 3, 1};
  const std::vector<int64_t> source_ids = {3, 0, 4, 1, 2};
  for (size_t mapping_index = 0; mapping_index < destination_ids.size();
       ++mapping_index) {
    const int64_t destination_id = destination_ids[mapping_index];
    const int64_t source_id = source_ids[mapping_index];
    EXPECT_TRUE(torch::equal(device_key_layer_zero[destination_id],
                             host_key[source_id][0].to(device.unwrap())));
  }

  for (size_t mapping_index = 0; mapping_index < destination_ids.size();
       ++mapping_index) {
    const int64_t destination_id = destination_ids[mapping_index];
    const int64_t source_id = source_ids[mapping_index];
    EXPECT_TRUE(torch::equal(device_key_layer_one[destination_id],
                             host_key[source_id][1].to(device.unwrap())));
  }
  compact_load.drain();
  compact_load.drain();
}

TEST(CompactLoadTest, UsesGlmLayerWindowsForMainKvDescriptors) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for compact H2D.";
  }

  constexpr int64_t kLayerCount = 78;
  constexpr int64_t kBlockElements = 73728;
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::Tensor host_key = torch::zeros(
      {1, kLayerCount, kBlockElements},
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU));
  const torch::TensorOptions device_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());

  CompactLoadGroupLayout group;
  group.group_id = 71;
  group.host_roles.emplace(KVCacheTensorRole::KEY, host_key);
  group.layers.reserve(kLayerCount);
  for (int64_t layer_id = 0; layer_id < kLayerCount; ++layer_id) {
    group.layers.emplace_back(CompactLoadLayerLayout{
        layer_id,
        layer_id,
        {{KVCacheTensorRole::KEY,
          torch::zeros({1, kBlockElements}, device_options)}}});
  }
  CompactLoadLayout layout;
  layout.num_layers = kLayerCount;
  layout.groups = {std::move(group)};

  auto batch_memcpy = std::make_unique<RecordingH2DBatchMemcpy>(device);
  RecordingH2DBatchMemcpy* recording = batch_memcpy.get();
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  CompactHostKVTransfer transfer(HostKVLayout(layout, device.unwrap()),
                                 device,
                                 *compute_stream,
                                 /*layer_copy_batches=*/4,
                                 std::move(batch_memcpy));
  HostKVLoadHandle handle = transfer.prepare_load();
  ASSERT_EQ(handle.synchronizer->size(), 5U);
  ASSERT_EQ(handle.layers_per_event, 19U);
  ASSERT_TRUE(transfer.load(CompactLoadRequest{{CompactLoadMapping{71, 0, 0}}},
                            handle));
  ASSERT_TRUE(handle.synchronizer->synchronize_layer(/*layer_index=*/4));

  ASSERT_EQ(recording->descriptor_bytes().size(), 5);
  for (size_t range_index = 0; range_index < 4; ++range_index) {
    EXPECT_EQ(recording->descriptor_bytes()[range_index][0], 2801664U);
    EXPECT_EQ(recording->descriptor_bytes()[range_index][1], 8U);
  }
  EXPECT_EQ(recording->descriptor_bytes()[4],
            (std::vector<size_t>{294912U, 8U}));
  transfer.drain();
}

TEST(CompactLoadTest, AbortsLoadWhenSubmissionFails) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for compact H2D.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::Tensor host_key =
      make_host_blocks(/*block_count=*/1, /*layer_count=*/1, /*offset=*/10.0);
  torch::Tensor device_key = torch::zeros(
      {1, 2},
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap()));
  auto batch_memcpy =
      std::make_unique<RecordingH2DBatchMemcpy>(device, /*fail_h2d=*/true);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  CompactHostKVTransfer transfer(
      HostKVLayout(make_single_role_layout(/*num_layers=*/1,
                                           /*group_layer_slot=*/0,
                                           host_key,
                                           device_key),
                   device.unwrap()),
      device,
      *compute_stream,
      /*layer_copy_batches=*/1,
      std::move(batch_memcpy));
  HostKVLoadHandle handle = transfer.prepare_load();

  EXPECT_FALSE(transfer.load(CompactLoadRequest{{CompactLoadMapping{61, 0, 0}}},
                             handle));
  EXPECT_FALSE(handle.synchronizer->synchronize_layer(/*layer_index=*/0));
  transfer.drain();
}

TEST(CompactLoadTest, DrainsAndAbortsWhenReadyEventFails) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for compact H2D.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::Tensor host_key =
      make_host_blocks(/*block_count=*/1, /*layer_count=*/1, /*offset=*/10.0);
  torch::Tensor device_key = torch::zeros(
      {1, 2},
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap()));
  auto batch_memcpy = std::make_unique<RecordingH2DBatchMemcpy>(device);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  CompactHostKVTransfer transfer(
      HostKVLayout(make_single_role_layout(/*num_layers=*/1,
                                           /*group_layer_slot=*/0,
                                           host_key,
                                           device_key),
                   device.unwrap()),
      device,
      *compute_stream,
      /*layer_copy_batches=*/1,
      std::move(batch_memcpy));
  auto synchronizer = std::make_shared<RejectingSynchronizer>();

  EXPECT_FALSE(transfer.load(CompactLoadRequest{{CompactLoadMapping{61, 0, 0}}},
                             {synchronizer, /*layers_per_event=*/1}));
  EXPECT_TRUE(synchronizer->aborted());
  transfer.drain();
}

TEST(CompactLoadTest, RestoresOnlyRolesActiveAtEachLayer) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for compact H2D.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();

  constexpr int64_t kBlockCount = 3;
  torch::Tensor host_key =
      make_host_blocks(kBlockCount, /*layer_count=*/2, /*offset=*/10.0);
  torch::Tensor host_value =
      make_host_blocks(kBlockCount, /*layer_count=*/2, /*offset=*/100.0);
  torch::Tensor device_key = torch::full(
      {kBlockCount, 2},
      -1.0,
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap()));
  torch::Tensor device_value = torch::full(
      {kBlockCount, 2},
      -2.0,
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap()));

  CompactLoadGroupLayout group;
  group.group_id = 23;
  group.host_roles.emplace(KVCacheTensorRole::KEY, host_key);
  group.host_roles.emplace(KVCacheTensorRole::VALUE, host_value);
  group.layers = {
      CompactLoadLayerLayout{0, 0, {{KVCacheTensorRole::KEY, device_key}}},
      CompactLoadLayerLayout{1, 1, {{KVCacheTensorRole::VALUE, device_value}}},
  };
  CompactLoadLayout layout;
  layout.num_layers = 2;
  layout.groups = {std::move(group)};

  RecordingH2DBatchMemcpy batch_memcpy(device);
  CompactLoadHarness compact_load(layout,
                                  device,
                                  /*layer_copy_batches=*/2,
                                  batch_memcpy,
                                  /*target_bytes=*/16);
  std::shared_ptr<LayerSynchronizer> synchronizer =
      create_layer_synchronizer(/*num_layers=*/2);
  ASSERT_NE(synchronizer, nullptr);
  CompactLoadRequest request;
  request.mappings = {
      CompactLoadMapping{23, 2, 1},
      CompactLoadMapping{23, 0, 2},
  };

  ASSERT_TRUE(compact_load.execute(request, synchronizer));
  ASSERT_TRUE(synchronizer->synchronize_layer(/*layer_index=*/0));
  ASSERT_TRUE(synchronizer->synchronize_layer(/*layer_index=*/1));
  ASSERT_EQ(batch_memcpy.descriptor_bytes().size(), 2);
  EXPECT_EQ(batch_memcpy.descriptor_bytes()[0],
            (std::vector<size_t>{8U, 8U, 16U}));
  EXPECT_EQ(batch_memcpy.descriptor_bytes()[1],
            (std::vector<size_t>{8U, 8U, 16U}));

  EXPECT_TRUE(torch::equal(device_key[1], host_key[2][0].to(device.unwrap())));
  EXPECT_TRUE(torch::equal(device_key[2], host_key[0][0].to(device.unwrap())));
  EXPECT_TRUE(
      torch::equal(device_value[1], host_value[2][1].to(device.unwrap())));
  EXPECT_TRUE(
      torch::equal(device_value[2], host_value[0][1].to(device.unwrap())));
  EXPECT_TRUE(torch::equal(device_key[0], torch::full_like(device_key[0], -1)));
  EXPECT_TRUE(
      torch::equal(device_value[0], torch::full_like(device_value[0], -2)));
}

TEST(CompactLoadTest, RestoresSplitMlaAndIndexerRolesThroughOneLayout) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for compact H2D.";
  }

  constexpr int64_t kBlockCount = 3;
  constexpr int32_t kGroupId = 53;
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();

  torch::Tensor host_key =
      make_host_blocks(kBlockCount, /*layer_count=*/3, /*offset=*/10.0);
  torch::Tensor host_value =
      make_host_blocks(kBlockCount, /*layer_count=*/3, /*offset=*/100.0);
  torch::Tensor host_index = torch::empty(
      {kBlockCount, 3, 1},
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  torch::Tensor host_index_scale = torch::empty(
      {kBlockCount, 3, 1},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  for (int64_t block_id = 0; block_id < kBlockCount; ++block_id) {
    for (int64_t layer_id = 0; layer_id < 3; ++layer_id) {
      host_index[block_id][layer_id].fill_(
          static_cast<int32_t>(1000 + block_id * 10 + layer_id));
      host_index_scale[block_id][layer_id].fill_(
          0.25F + static_cast<float>(block_id * 10 + layer_id));
    }
  }

  const torch::TensorOptions float_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap());
  torch::Tensor split_key = torch::full({kBlockCount, 2}, -1.0, float_options);
  torch::Tensor split_value =
      torch::full({kBlockCount, 2}, -2.0, float_options);
  torch::Tensor mla_key_layer_one =
      torch::full({kBlockCount, 2}, -3.0, float_options);
  torch::Tensor mla_key_layer_two =
      torch::full({kBlockCount, 2}, -6.0, float_options);
  torch::Tensor index = torch::full(
      {kBlockCount, 1},
      -4,
      torch::TensorOptions().dtype(torch::kInt32).device(device.unwrap()));
  torch::Tensor index_scale =
      torch::full({kBlockCount, 1}, -5.0, float_options);

  CompactLoadGroupLayout group;
  group.group_id = kGroupId;
  group.host_roles.emplace(KVCacheTensorRole::KEY, host_key);
  group.host_roles.emplace(KVCacheTensorRole::VALUE, host_value);
  group.host_roles.emplace(KVCacheTensorRole::INDEX, host_index);
  group.host_roles.emplace(KVCacheTensorRole::INDEX_SCALE, host_index_scale);
  group.layers = {
      CompactLoadLayerLayout{0,
                             0,
                             {{KVCacheTensorRole::KEY, split_key},
                              {KVCacheTensorRole::VALUE, split_value}}},
      CompactLoadLayerLayout{
          1, 1, {{KVCacheTensorRole::KEY, mla_key_layer_one}}},
      CompactLoadLayerLayout{2,
                             2,
                             {{KVCacheTensorRole::KEY, mla_key_layer_two},
                              {KVCacheTensorRole::INDEX, index},
                              {KVCacheTensorRole::INDEX_SCALE, index_scale}}},
  };
  CompactLoadLayout layout;
  layout.num_layers = 3;
  layout.groups = {std::move(group)};

  RecordingH2DBatchMemcpy batch_memcpy(device);
  CompactLoadHarness compact_load(layout,
                                  device,
                                  /*layer_copy_batches=*/1,
                                  batch_memcpy,
                                  /*target_bytes=*/8);
  std::shared_ptr<LayerSynchronizer> synchronizer =
      create_layer_synchronizer(/*num_layers=*/1);
  ASSERT_NE(synchronizer, nullptr);
  CompactLoadRequest request;
  request.mappings = {
      CompactLoadMapping{kGroupId, 2, 0},
      CompactLoadMapping{kGroupId, 0, 2},
  };

  ASSERT_TRUE(compact_load.execute(request, synchronizer));
  ASSERT_TRUE(synchronizer->synchronize_layer(/*layer_index=*/0));
  ASSERT_EQ(batch_memcpy.descriptor_bytes().size(), 2);
  EXPECT_EQ(batch_memcpy.descriptor_bytes()[0],
            (std::vector<size_t>{24U, 24U, 12U, 12U, 8U}));

  EXPECT_TRUE(torch::equal(split_key[0], host_key[2][0].to(device.unwrap())));
  EXPECT_TRUE(torch::equal(split_key[2], host_key[0][0].to(device.unwrap())));
  EXPECT_TRUE(
      torch::equal(split_value[0], host_value[2][0].to(device.unwrap())));
  EXPECT_TRUE(
      torch::equal(split_value[2], host_value[0][0].to(device.unwrap())));
  EXPECT_TRUE(
      torch::equal(mla_key_layer_one[0], host_key[2][1].to(device.unwrap())));
  EXPECT_TRUE(
      torch::equal(mla_key_layer_one[2], host_key[0][1].to(device.unwrap())));
  EXPECT_TRUE(
      torch::equal(mla_key_layer_two[0], host_key[2][2].to(device.unwrap())));
  EXPECT_TRUE(
      torch::equal(mla_key_layer_two[2], host_key[0][2].to(device.unwrap())));
  EXPECT_TRUE(torch::equal(index[0], host_index[2][2].to(device.unwrap())));
  EXPECT_TRUE(torch::equal(index[2], host_index[0][2].to(device.unwrap())));
  EXPECT_TRUE(
      torch::equal(index_scale[0], host_index_scale[2][2].to(device.unwrap())));
  EXPECT_TRUE(
      torch::equal(index_scale[2], host_index_scale[0][2].to(device.unwrap())));
  compact_load.drain();
}

TEST(HostKVLayoutTest, RejectsInvalidRoleLayouts) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for compact H2D.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();

  torch::Tensor host_float =
      make_host_blocks(/*block_count=*/2, /*layer_count=*/1, /*offset=*/0.0);
  torch::Tensor device_float = torch::zeros(
      {2, 2},
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap()));
  torch::Tensor device_int = torch::zeros(
      {2, 2},
      torch::TensorOptions().dtype(torch::kInt32).device(device.unwrap()));
  torch::Tensor host_wide = torch::zeros(
      {2, 1, 3},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

  EXPECT_DEATH(
      {
        HostKVLayout(make_single_role_layout(/*num_layers=*/1,
                                             /*group_layer_slot=*/0,
                                             host_float,
                                             device_int),
                     device.unwrap());
      },
      "host and device role dtypes must match");
  EXPECT_DEATH(
      {
        HostKVLayout(make_single_role_layout(/*num_layers=*/1,
                                             /*group_layer_slot=*/0,
                                             host_wide,
                                             device_float),
                     device.unwrap());
      },
      "host and device block shapes must match");
  EXPECT_DEATH(([&] {
                 CompactLoadGroupLayout group;
                 group.group_id = 61;
                 group.host_roles.emplace(KVCacheTensorRole::VALUE, host_float);
                 group.layers = {CompactLoadLayerLayout{
                     0, 0, {{KVCacheTensorRole::KEY, device_float}}}};
                 CompactLoadLayout layout;
                 layout.num_layers = 1;
                 layout.groups = {std::move(group)};
                 HostKVLayout(layout, device.unwrap());
               }()),
               "active device role is missing its host tensor");
  EXPECT_DEATH(
      {
        HostKVLayout(make_single_role_layout(/*num_layers=*/2,
                                             /*group_layer_slot=*/1,
                                             host_float,
                                             device_float),
                     device.unwrap());
      },
      "group layer slot is out of range");
}

TEST(CompactLoadTest, RejectsNoncontiguousWindowBeforeSubmission) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for compact H2D.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::Tensor host_key =
      make_host_blocks(/*block_count=*/1, /*layer_count=*/3, /*offset=*/0.0);
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap());
  CompactLoadGroupLayout group;
  group.group_id = 81;
  group.host_roles.emplace(KVCacheTensorRole::KEY, host_key);
  group.layers = {
      CompactLoadLayerLayout{
          0, 0, {{KVCacheTensorRole::KEY, torch::zeros({1, 2}, options)}}},
      CompactLoadLayerLayout{
          1, 2, {{KVCacheTensorRole::KEY, torch::zeros({1, 2}, options)}}},
  };
  CompactLoadLayout layout;
  layout.num_layers = 2;
  layout.groups = {std::move(group)};
  RecordingH2DBatchMemcpy batch_memcpy(device);

  EXPECT_DEATH(
      {
        CompactLoadHarness compact_load(layout,
                                        device,
                                        /*layer_copy_batches=*/1,
                                        batch_memcpy);
      },
      "contiguous group-layer slots");
}

TEST(CompactLoadTest, RestoresPartialGroupsWithIndependentRoleNamespaces) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for compact H2D.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();

  constexpr int64_t kBlockCount = 3;
  torch::Tensor host_group_a =
      make_host_blocks(kBlockCount, /*layer_count=*/1, /*offset=*/10.0);
  torch::Tensor host_group_b = torch::empty(
      {kBlockCount, 1, 4},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  for (int64_t block_id = 0; block_id < kBlockCount; ++block_id) {
    host_group_b[block_id][0].fill_(100.0 + block_id);
  }
  torch::Tensor device_group_a = torch::full(
      {kBlockCount, 2},
      -1.0,
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap()));
  torch::Tensor device_group_b = torch::full(
      {kBlockCount, 4},
      -2.0,
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap()));

  CompactLoadGroupLayout group_a;
  group_a.group_id = 31;
  group_a.host_roles.emplace(KVCacheTensorRole::KEY, host_group_a);
  group_a.layers = {
      CompactLoadLayerLayout{0, 0, {{KVCacheTensorRole::KEY, device_group_a}}},
  };
  CompactLoadGroupLayout group_b;
  group_b.group_id = 47;
  group_b.host_roles.emplace(KVCacheTensorRole::KEY, host_group_b);
  group_b.layers = {
      CompactLoadLayerLayout{0, 0, {{KVCacheTensorRole::KEY, device_group_b}}},
  };
  CompactLoadLayout layout;
  layout.num_layers = 1;
  layout.groups = {std::move(group_a), std::move(group_b)};
  RecordingH2DBatchMemcpy batch_memcpy(device);
  CompactLoadHarness compact_load(layout,
                                  device,
                                  /*layer_copy_batches=*/1,
                                  batch_memcpy,
                                  /*target_bytes=*/16);

  std::shared_ptr<LayerSynchronizer> synchronizer =
      create_layer_synchronizer(/*num_layers=*/1);
  ASSERT_NE(synchronizer, nullptr);
  CompactLoadRequest partial_request;
  partial_request.mappings = {
      CompactLoadMapping{47, 2, 1},
      CompactLoadMapping{47, 0, 2},
  };

  ASSERT_TRUE(compact_load.execute(partial_request, synchronizer));
  ASSERT_TRUE(synchronizer->synchronize_layer(/*layer_index=*/0));
  EXPECT_TRUE(
      torch::equal(device_group_b[1], host_group_b[2][0].to(device.unwrap())));
  EXPECT_TRUE(
      torch::equal(device_group_b[2], host_group_b[0][0].to(device.unwrap())));
  EXPECT_TRUE(
      torch::equal(device_group_a, torch::full_like(device_group_a, -1)));

  synchronizer = create_layer_synchronizer(/*num_layers=*/1);
  ASSERT_NE(synchronizer, nullptr);
  CompactLoadRequest full_request;
  full_request.mappings = {
      CompactLoadMapping{31, 1, 0},
      CompactLoadMapping{31, 0, 2},
      CompactLoadMapping{47, 1, 0},
  };

  ASSERT_TRUE(compact_load.execute(full_request, synchronizer));
  ASSERT_TRUE(synchronizer->synchronize_layer(/*layer_index=*/0));
  EXPECT_TRUE(
      torch::equal(device_group_a[0], host_group_a[1][0].to(device.unwrap())));
  EXPECT_TRUE(
      torch::equal(device_group_a[2], host_group_a[0][0].to(device.unwrap())));
  EXPECT_TRUE(
      torch::equal(device_group_b[0], host_group_b[1][0].to(device.unwrap())));
  compact_load.drain();
}

}  // namespace
}  // namespace xllm
