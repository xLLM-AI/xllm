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

#include "framework/kv_cache_transfer/host_transfer/transfer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "framework/kv_cache_transfer/host_transfer/layout.h"
#include "platform/layer_synchronizer.h"

namespace xllm {
namespace {

class TestSynchronizer final : public LayerSynchronizer {
 public:
  explicit TestSynchronizer(uint32_t size) : size_(size) {}

  bool synchronize_layer(int64_t /*layer_index*/) override { return true; }
  bool record_stream(int64_t /*layer_index*/, Stream* /*stream*/) override {
    return true;
  }
  void abort() override { aborted_ = true; }
  uint32_t size() const override { return size_; }
  bool aborted() const { return aborted_; }

 private:
  uint32_t size_ = 0;
  bool aborted_ = false;
};

HostKVLayout make_layout() {
  std::vector<HostKVGroupLayout> groups;
  groups.reserve(2);
  for (int32_t group_id : {3, 7}) {
    HostKVGroupLayout group;
    group.group_id = group_id;
    group.host_roles.emplace(
        KVCacheTensorRole::KEY,
        torch::zeros({2, 1, 2}, torch::TensorOptions().dtype(torch::kFloat32)));
    group.layers.emplace_back(HostKVLayerLayout{
        0,
        0,
        {{KVCacheTensorRole::KEY,
          torch::zeros({2, 2},
                       torch::TensorOptions().dtype(torch::kFloat32))}}});
    groups.emplace_back(std::move(group));
  }
  return HostKVLayout(
      /*num_layers=*/1, std::move(groups), torch::Device(torch::kCPU));
}

class TestHostKVTransfer final : public HostKVTransfer {
 public:
  TestHostKVTransfer() : HostKVTransfer(make_layout()) {}

  HostKVLoadHandle prepare_load() override {
    return {std::make_shared<TestSynchronizer>(/*size=*/1),
            /*layers_per_event=*/1};
  }

  void drain() override { drained_ = true; }

  uint32_t load_calls() const { return load_calls_; }
  uint32_t offload_calls() const { return offload_calls_; }
  bool drained() const { return drained_; }
  void fail_load() { load_success_ = false; }

 protected:
  uint32_t load_event_count() const override { return 1; }
  uint32_t layers_per_event() const override { return 1; }

  bool load_impl(const HostKVRequest& /*request*/,
                 const HostKVLoadHandle& /*handle*/) override {
    ++load_calls_;
    return load_success_;
  }

  bool offload_impl(const HostKVRequest& /*request*/) override {
    ++offload_calls_;
    return true;
  }

 private:
  uint32_t load_calls_ = 0;
  uint32_t offload_calls_ = 0;
  bool drained_ = false;
  bool load_success_ = true;
};

TEST(HostKVTransferTest, RejectsInvalidRequestsBeforeSubmission) {
  TestHostKVTransfer transfer;
  HostKVLoadHandle handle = transfer.prepare_load();

  EXPECT_FALSE(transfer.load(HostKVRequest{}, handle));
  EXPECT_FALSE(transfer.offload(HostKVRequest{}));
  EXPECT_FALSE(transfer.load(HostKVRequest{{HostKVMapping{99, 0, 0}}}, handle));
  EXPECT_FALSE(transfer.offload(HostKVRequest{{HostKVMapping{3, 2, 0}}}));
  EXPECT_FALSE(transfer.load(HostKVRequest{{HostKVMapping{3, 0, 2}}}, handle));
  EXPECT_EQ(transfer.load_calls(), 0U);
  EXPECT_EQ(transfer.offload_calls(), 0U);
}

TEST(HostKVTransferTest, EnforcesDirectionScopedDestinations) {
  TestHostKVTransfer transfer;
  HostKVLoadHandle handle = transfer.prepare_load();

  EXPECT_FALSE(transfer.load(
      HostKVRequest{{HostKVMapping{3, 0, 1}, HostKVMapping{3, 1, 1}}}, handle));
  EXPECT_FALSE(transfer.offload(
      HostKVRequest{{HostKVMapping{3, 1, 0}, HostKVMapping{3, 1, 1}}}));

  const HostKVRequest distinct_groups{
      {HostKVMapping{3, 0, 1}, HostKVMapping{7, 0, 1}}};
  EXPECT_TRUE(transfer.load(distinct_groups, handle));
  EXPECT_TRUE(transfer.offload(distinct_groups));
  EXPECT_EQ(transfer.load_calls(), 1U);
  EXPECT_EQ(transfer.offload_calls(), 1U);
}

TEST(HostKVTransferTest, RejectsInvalidLoadHandleAndAbortsIt) {
  TestHostKVTransfer transfer;
  const HostKVRequest request{{HostKVMapping{3, 0, 1}}};

  HostKVLoadHandle empty_handle;
  EXPECT_FALSE(transfer.load(request, empty_handle));

  auto wrong_size = std::make_shared<TestSynchronizer>(/*size=*/2);
  EXPECT_FALSE(transfer.load(request, {wrong_size, /*layers_per_event=*/1}));
  EXPECT_TRUE(wrong_size->aborted());

  auto wrong_granularity = std::make_shared<TestSynchronizer>(/*size=*/1);
  EXPECT_FALSE(
      transfer.load(request, {wrong_granularity, /*layers_per_event=*/2}));
  EXPECT_TRUE(wrong_granularity->aborted());
  EXPECT_EQ(transfer.load_calls(), 0U);
}

TEST(HostKVTransferTest, AbortsHandleWhenStrategySubmissionFails) {
  TestHostKVTransfer transfer;
  transfer.fail_load();
  auto synchronizer = std::make_shared<TestSynchronizer>(/*size=*/1);

  EXPECT_FALSE(transfer.load(HostKVRequest{{HostKVMapping{3, 0, 1}}},
                             {synchronizer, /*layers_per_event=*/1}));
  EXPECT_TRUE(synchronizer->aborted());
}

}  // namespace
}  // namespace xllm
