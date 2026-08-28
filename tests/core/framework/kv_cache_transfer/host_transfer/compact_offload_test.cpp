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

#include <cnrt.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/kv_cache_transfer/host_transfer/compact_transfer.h"
#include "framework/kv_cache_transfer/host_transfer/layout.h"
#include "platform/batch_memcpy.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "platform/stream.h"

namespace xllm {
namespace {

class SynchronousD2HBatchMemcpy final : public BatchMemcpy {
 public:
  void init(int32_t /*device_id*/) override {}

  bool submit_h2d(const std::vector<torch::Tensor>& /*src_tensors*/,
                  const std::vector<torch::Tensor>& /*dst_tensors*/,
                  Stream* /*stream*/) override {
    ADD_FAILURE() << "Compact D2H must not submit H2D copies.";
    return false;
  }

  bool submit_d2h(const std::vector<torch::Tensor>& src_tensors,
                  const std::vector<torch::Tensor>& dst_tensors,
                  Stream* stream) override {
    CHECK(stream != nullptr);
    CHECK_EQ(stream->synchronize(), 0);
    ++submit_count_;
    descriptor_counts_.emplace_back(src_tensors.size());
    descriptor_count_ += src_tensors.size();
    if (src_tensors.size() != dst_tensors.size()) {
      return false;
    }
    for (size_t index = 0; index < src_tensors.size(); ++index) {
      dst_tensors[index].copy_(src_tensors[index].to(torch::kCPU));
    }
    return true;
  }

  bool copy_d2h(const std::vector<torch::Tensor>& /*src_tensors*/,
                const std::vector<torch::Tensor>& /*dst_tensors*/,
                Stream* /*stream*/) override {
    ADD_FAILURE() << "Compact D2H must use submit_d2h.";
    return false;
  }

  size_t submit_count() const { return submit_count_; }
  size_t descriptor_count() const { return descriptor_count_; }
  const std::vector<size_t>& descriptor_counts() const {
    return descriptor_counts_;
  }

 private:
  size_t submit_count_ = 0;
  size_t descriptor_count_ = 0;
  std::vector<size_t> descriptor_counts_;
};

class CountingD2HBatchMemcpy final : public BatchMemcpy {
 public:
  void init(int32_t /*device_id*/) override {}

  bool submit_h2d(const std::vector<torch::Tensor>& /*src_tensors*/,
                  const std::vector<torch::Tensor>& /*dst_tensors*/,
                  Stream* /*stream*/) override {
    ADD_FAILURE() << "Compact D2H must not submit H2D copies.";
    return false;
  }

  bool submit_d2h(const std::vector<torch::Tensor>& src_tensors,
                  const std::vector<torch::Tensor>& dst_tensors,
                  Stream* /*stream*/) override {
    EXPECT_EQ(src_tensors.size(), dst_tensors.size());
    ++submit_count_;
    descriptor_count_ += src_tensors.size();
    return true;
  }

  bool copy_d2h(const std::vector<torch::Tensor>& /*src_tensors*/,
                const std::vector<torch::Tensor>& /*dst_tensors*/,
                Stream* /*stream*/) override {
    ADD_FAILURE() << "Compact D2H must use submit_d2h.";
    return false;
  }

  size_t submit_count() const { return submit_count_; }
  size_t descriptor_count() const { return descriptor_count_; }

 private:
  size_t submit_count_ = 0;
  size_t descriptor_count_ = 0;
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

class CompactOffloadHarness final {
 public:
  CompactOffloadHarness(HostKVLayout layout,
                        const Device& device,
                        BatchMemcpy& batch_memcpy,
                        size_t target_bytes = 64ULL * 1024 * 1024) {
    compute_stream_ = device.current_stream();
    const uint32_t layer_copy_batches =
        static_cast<uint32_t>(layout.num_layers());
    CompactTransferConfig config;
    config.load_target_bytes = 8;
    config.offload_target_bytes = target_bytes;
    transfer_ = std::make_unique<CompactHostKVTransfer>(
        std::move(layout),
        device,
        *compute_stream_,
        layer_copy_batches,
        std::make_unique<BorrowedBatchMemcpy>(batch_memcpy),
        config);
  }

  bool execute(const HostKVRequest& request, const Stream& /*compute_stream*/) {
    return transfer_->offload(request);
  }

  void drain() { transfer_->drain(); }

 private:
  std::unique_ptr<Stream> compute_stream_;
  std::unique_ptr<CompactHostKVTransfer> transfer_;
};

void wait_for_queue_gate(void* user_data) {
  std::atomic<bool>* gate_open = static_cast<std::atomic<bool>*>(user_data);
  while (!gate_open->load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

class GatedD2HBatchMemcpy final : public BatchMemcpy {
 public:
  GatedD2HBatchMemcpy(const Device& device, std::atomic<bool>* gate_open)
      : delegate_(create_batch_memcpy(device)), gate_open_(gate_open) {
    CHECK(delegate_ != nullptr);
    CHECK(gate_open_ != nullptr);
  }

  void init(int32_t /*device_id*/) override {}

  bool submit_h2d(const std::vector<torch::Tensor>& /*src_tensors*/,
                  const std::vector<torch::Tensor>& /*dst_tensors*/,
                  Stream* /*stream*/) override {
    ADD_FAILURE() << "Compact D2H must not submit H2D copies.";
    return false;
  }

  bool submit_d2h(const std::vector<torch::Tensor>& src_tensors,
                  const std::vector<torch::Tensor>& dst_tensors,
                  Stream* stream) override {
    CHECK(stream != nullptr);
    if (!gate_submitted_) {
      if (cnrtInvokeHostFunc(stream->get_stream()->stream(),
                             wait_for_queue_gate,
                             gate_open_) != cnrtSuccess) {
        return false;
      }
      gate_submitted_.store(true, std::memory_order_release);
    }
    ++submit_count_;
    return delegate_->submit_d2h(src_tensors, dst_tensors, stream);
  }

  bool copy_d2h(const std::vector<torch::Tensor>& /*src_tensors*/,
                const std::vector<torch::Tensor>& /*dst_tensors*/,
                Stream* /*stream*/) override {
    ADD_FAILURE() << "Compact D2H must use submit_d2h.";
    return false;
  }

  bool gate_submitted() const {
    return gate_submitted_.load(std::memory_order_acquire);
  }
  size_t submit_count() const { return submit_count_; }

 private:
  std::unique_ptr<BatchMemcpy> delegate_;
  std::atomic<bool>* gate_open_ = nullptr;
  std::atomic<bool> gate_submitted_{false};
  size_t submit_count_ = 0;
};

class WarningLogSink final : public google::LogSink {
 public:
  void send(google::LogSeverity severity,
            const char* /*full_filename*/,
            const char* /*base_filename*/,
            int /*line*/,
            const google::LogMessageTime& /*logmsgtime*/,
            const char* message,
            size_t message_len) override {
    if (severity != google::GLOG_WARNING) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    messages_.append(message, message_len);
  }

  bool contains(const std::string& message) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return messages_.find(message) != std::string::npos;
  }

 private:
  mutable std::mutex mutex_;
  std::string messages_;
};

class ScopedLogSink final {
 public:
  explicit ScopedLogSink(google::LogSink* sink) : sink_(sink) {
    CHECK(sink_ != nullptr);
    google::AddLogSink(sink_);
  }

  ~ScopedLogSink() { google::RemoveLogSink(sink_); }

  ScopedLogSink(const ScopedLogSink&) = delete;
  ScopedLogSink& operator=(const ScopedLogSink&) = delete;

 private:
  google::LogSink* sink_ = nullptr;
};

bool wait_for_gate_submission(const GatedD2HBatchMemcpy& batch_memcpy) {
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!batch_memcpy.gate_submitted() &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  return batch_memcpy.gate_submitted();
}

torch::Tensor make_host_blocks(int64_t block_count, int64_t layer_count) {
  return torch::full(
      {block_count, layer_count, 2},
      -1.0,
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
}

torch::Tensor make_device_blocks(const Device& device,
                                 int64_t block_count,
                                 double offset) {
  torch::Tensor blocks = torch::empty(
      {block_count, 2},
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap()));
  for (int64_t block_id = 0; block_id < block_count; ++block_id) {
    blocks[block_id].fill_(offset + static_cast<double>(block_id));
  }
  return blocks;
}

HostKVLayoutInput make_layout(const torch::Tensor& host_key,
                              const torch::Tensor& host_value,
                              const torch::Tensor& key_layer_zero,
                              const torch::Tensor& key_layer_two,
                              const torch::Tensor& value_layer_two) {
  HostKVGroupLayout group;
  group.group_id = 17;
  group.host_roles.emplace(KVCacheTensorRole::KEY, host_key);
  group.host_roles.emplace(KVCacheTensorRole::VALUE, host_value);
  group.layers = {
      HostKVLayerLayout{0, 0, {{KVCacheTensorRole::KEY, key_layer_zero}}},
      HostKVLayerLayout{2,
                        1,
                        {{KVCacheTensorRole::KEY, key_layer_two},
                         {KVCacheTensorRole::VALUE, value_layer_two}}},
  };
  HostKVLayoutInput layout;
  layout.num_layers = 3;
  layout.groups = {std::move(group)};
  return layout;
}

TEST(CompactOffloadTest, CopiesPaddedRolesAcrossTiles) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for compact D2H.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();

  constexpr int64_t kBlockCount = 4;
  torch::Tensor host_key = make_host_blocks(kBlockCount, /*layer_count=*/2);
  torch::Tensor host_value = make_host_blocks(kBlockCount, /*layer_count=*/2);
  torch::Tensor key_layer_zero =
      make_device_blocks(device, kBlockCount, /*offset=*/10.0);
  torch::Tensor key_layer_two =
      make_device_blocks(device, kBlockCount, /*offset=*/20.0);
  torch::Tensor value_layer_two =
      make_device_blocks(device, kBlockCount, /*offset=*/30.0);
  HostKVLayout layout(
      make_layout(
          host_key, host_value, key_layer_zero, key_layer_two, value_layer_two),
      device.unwrap());
  SynchronousD2HBatchMemcpy batch_memcpy;
  CompactOffloadHarness compact_offload(layout,
                                        device,
                                        batch_memcpy,
                                        /*target_bytes=*/96);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  ASSERT_NE(compute_stream, nullptr);

  HostKVRequest request;
  request.mappings = {
      HostKVMapping{17, 3, 2},
      HostKVMapping{17, 0, 3},
      HostKVMapping{17, 1, 3},
      HostKVMapping{17, 2, 0},
  };
  ASSERT_TRUE(compact_offload.execute(request, *compute_stream));

  EXPECT_EQ(batch_memcpy.submit_count(), 2);
  EXPECT_EQ(batch_memcpy.descriptor_count(), 8);
  EXPECT_EQ(batch_memcpy.descriptor_counts(), (std::vector<size_t>{6, 2}));
  const std::vector<int64_t> host_blocks = {2, 3, 0, 1};
  const std::vector<int64_t> device_blocks = {0, 2, 3, 3};
  for (size_t index = 0; index < host_blocks.size(); ++index) {
    const int64_t host_block = host_blocks[index];
    const int64_t device_block = device_blocks[index];
    EXPECT_TRUE(torch::equal(host_key[host_block][0],
                             key_layer_zero[device_block].to(torch::kCPU)));
    EXPECT_TRUE(torch::equal(host_key[host_block][1],
                             key_layer_two[device_block].to(torch::kCPU)));
    EXPECT_TRUE(torch::equal(host_value[host_block][1],
                             value_layer_two[device_block].to(torch::kCPU)));
    EXPECT_TRUE(torch::equal(host_value[host_block][0],
                             torch::zeros_like(host_value[host_block][0])));
  }
  compact_offload.drain();
  compact_offload.drain();
}

TEST(CompactOffloadTest, RejectsInvalidRequestsWithoutSubmittingOrScattering) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for compact D2H.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();

  torch::Tensor host_key =
      make_host_blocks(/*block_count=*/3, /*layer_count=*/2);
  torch::Tensor host_value =
      make_host_blocks(/*block_count=*/3, /*layer_count=*/2);
  torch::Tensor key_layer_zero =
      make_device_blocks(device, /*block_count=*/3, /*offset=*/10.0);
  torch::Tensor key_layer_two =
      make_device_blocks(device, /*block_count=*/3, /*offset=*/20.0);
  torch::Tensor value_layer_two =
      make_device_blocks(device, /*block_count=*/3, /*offset=*/30.0);
  HostKVLayout layout(
      make_layout(
          host_key, host_value, key_layer_zero, key_layer_two, value_layer_two),
      device.unwrap());
  SynchronousD2HBatchMemcpy batch_memcpy;
  CompactOffloadHarness compact_offload(layout,
                                        device,
                                        batch_memcpy,
                                        /*target_bytes=*/8);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  ASSERT_NE(compute_stream, nullptr);

  const std::vector<HostKVRequest> invalid_requests = {
      HostKVRequest{},
      HostKVRequest{{HostKVMapping{99, 0, 0}}},
      HostKVRequest{{HostKVMapping{17, -1, 0}}},
      HostKVRequest{{HostKVMapping{17, 0, -1}}},
      HostKVRequest{{HostKVMapping{17, 3, 0}}},
      HostKVRequest{{HostKVMapping{17, 0, 3}}},
      HostKVRequest{{HostKVMapping{17, 0, 0}, HostKVMapping{17, 0, 1}}},
  };
  for (const HostKVRequest& request : invalid_requests) {
    EXPECT_FALSE(compact_offload.execute(request, *compute_stream));
  }
  EXPECT_EQ(batch_memcpy.submit_count(), 0);
  EXPECT_TRUE(torch::equal(host_key, torch::full_like(host_key, -1.0)));
  EXPECT_TRUE(torch::equal(host_value, torch::full_like(host_value, -1.0)));
  compact_offload.drain();
}

TEST(CompactOffloadTest, CopiesMultipleGroupsWithPartialLayerRoles) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for compact D2H.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();

  torch::Tensor host_key =
      make_host_blocks(/*block_count=*/4, /*layer_count=*/2);
  torch::Tensor host_value =
      make_host_blocks(/*block_count=*/4, /*layer_count=*/2);
  torch::Tensor key_layer_zero =
      make_device_blocks(device, /*block_count=*/4, /*offset=*/10.0);
  torch::Tensor key_layer_two =
      make_device_blocks(device, /*block_count=*/4, /*offset=*/20.0);
  torch::Tensor value_layer_two =
      make_device_blocks(device, /*block_count=*/4, /*offset=*/30.0);
  torch::Tensor second_host_key =
      make_host_blocks(/*block_count=*/3, /*layer_count=*/1);
  torch::Tensor second_key_layer =
      make_device_blocks(device, /*block_count=*/3, /*offset=*/40.0);
  HostKVLayoutInput layout_input = make_layout(
      host_key, host_value, key_layer_zero, key_layer_two, value_layer_two);
  HostKVGroupLayout second_group;
  second_group.group_id = 31;
  second_group.host_roles.emplace(KVCacheTensorRole::KEY, second_host_key);
  second_group.layers = {
      HostKVLayerLayout{1, 0, {{KVCacheTensorRole::KEY, second_key_layer}}}};
  layout_input.groups.emplace_back(std::move(second_group));
  HostKVLayout layout(layout_input, device.unwrap());
  SynchronousD2HBatchMemcpy batch_memcpy;
  CompactOffloadHarness compact_offload(layout,
                                        device,
                                        batch_memcpy,
                                        /*target_bytes=*/8);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  ASSERT_NE(compute_stream, nullptr);

  HostKVRequest request;
  request.mappings = {
      HostKVMapping{31, 1, 2},
      HostKVMapping{17, 3, 1},
      HostKVMapping{17, 0, 2},
  };
  ASSERT_TRUE(compact_offload.execute(request, *compute_stream));

  EXPECT_EQ(batch_memcpy.submit_count(), 3);
  EXPECT_EQ(batch_memcpy.descriptor_count(), 5);
  EXPECT_TRUE(torch::equal(host_key[3][0], key_layer_zero[1].to(torch::kCPU)));
  EXPECT_TRUE(torch::equal(host_key[3][1], key_layer_two[1].to(torch::kCPU)));
  EXPECT_TRUE(
      torch::equal(host_value[3][1], value_layer_two[1].to(torch::kCPU)));
  EXPECT_TRUE(torch::equal(host_key[0][0], key_layer_zero[2].to(torch::kCPU)));
  EXPECT_TRUE(torch::equal(host_key[0][1], key_layer_two[2].to(torch::kCPU)));
  EXPECT_TRUE(
      torch::equal(host_value[0][1], value_layer_two[2].to(torch::kCPU)));
  EXPECT_TRUE(
      torch::equal(second_host_key[1][0], second_key_layer[2].to(torch::kCPU)));
  EXPECT_TRUE(
      torch::equal(host_value[3][0], torch::zeros_like(host_value[3][0])));
  compact_offload.drain();
}

TEST(CompactOffloadTest, ExpandsBothSlotsForOversizedMapping) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for compact D2H.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();

  torch::Tensor host_key =
      make_host_blocks(/*block_count=*/1, /*layer_count=*/2);
  torch::Tensor host_value =
      make_host_blocks(/*block_count=*/1, /*layer_count=*/2);
  torch::Tensor key_layer_zero =
      make_device_blocks(device, /*block_count=*/1, /*offset=*/10.0);
  torch::Tensor key_layer_two =
      make_device_blocks(device, /*block_count=*/1, /*offset=*/20.0);
  torch::Tensor value_layer_two =
      make_device_blocks(device, /*block_count=*/1, /*offset=*/30.0);
  HostKVLayout layout(
      make_layout(
          host_key, host_value, key_layer_zero, key_layer_two, value_layer_two),
      device.unwrap());
  SynchronousD2HBatchMemcpy batch_memcpy;
  WarningLogSink warning_sink;
  {
    ScopedLogSink log_sink(&warning_sink);
    CompactOffloadHarness compact_offload(layout,
                                          device,
                                          batch_memcpy,
                                          /*target_bytes=*/8);
  }
  EXPECT_TRUE(
      warning_sink.contains("budget_bytes=8, slot_bytes=512, "
                            "total_slot_bytes=1024, "
                            "total_extra_hbm_bytes=1008"));
}

TEST(CompactOffloadTest, FinalFlushWaitsForQueuedD2HBeforeReusingSlots) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for compact D2H.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::Tensor key_layer_zero =
      make_device_blocks(device, /*block_count=*/3, /*offset=*/10.0);
  torch::Tensor key_layer_two =
      make_device_blocks(device, /*block_count=*/3, /*offset=*/20.0);
  torch::Tensor value_layer_two =
      make_device_blocks(device, /*block_count=*/3, /*offset=*/30.0);
  torch::Tensor host_key;
  HostPageAlignedRegion host_key_region;
  create_host_page_aligned_tensor(
      {3, 2, 2}, torch::kFloat32, &host_key, &host_key_region);
  host_key.fill_(-1.0);
  torch::Tensor host_value;
  HostPageAlignedRegion host_value_region;
  create_host_page_aligned_tensor(
      {3, 2, 2}, torch::kFloat32, &host_value, &host_value_region);
  host_value.fill_(-1.0);
  HostKVLayout layout(
      make_layout(
          host_key, host_value, key_layer_zero, key_layer_two, value_layer_two),
      device.unwrap());
  std::atomic<bool> gate_open{false};
  GatedD2HBatchMemcpy batch_memcpy(device, &gate_open);
  CompactOffloadHarness compact_offload(layout,
                                        device,
                                        batch_memcpy,
                                        /*target_bytes=*/24);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  ASSERT_NE(compute_stream, nullptr);

  const HostKVRequest request{{HostKVMapping{17, 2, 1},
                               HostKVMapping{17, 0, 2},
                               HostKVMapping{17, 1, 0}}};
  std::future<bool> result =
      std::async(std::launch::async,
                 [&compact_offload, &compute_stream, &device, &request]() {
                   device.set_device();
                   device.init_device_context();
                   return compact_offload.execute(request, *compute_stream);
                 });

  if (!wait_for_gate_submission(batch_memcpy)) {
    gate_open.store(true, std::memory_order_release);
    ADD_FAILURE() << "Compact D2H did not submit the queued D2H copy.";
    return;
  }
  EXPECT_EQ(result.wait_for(std::chrono::milliseconds(100)),
            std::future_status::timeout);
  gate_open.store(true, std::memory_order_release);

  ASSERT_EQ(result.wait_for(std::chrono::seconds(2)),
            std::future_status::ready);
  EXPECT_TRUE(result.get());
  EXPECT_EQ(batch_memcpy.submit_count(), 3U);
  for (const HostKVMapping& mapping : request.mappings) {
    EXPECT_TRUE(
        torch::equal(host_key[mapping.host_block_id][0],
                     key_layer_zero[mapping.device_block_id].to(torch::kCPU)));
    EXPECT_TRUE(
        torch::equal(host_key[mapping.host_block_id][1],
                     key_layer_two[mapping.device_block_id].to(torch::kCPU)));
    EXPECT_TRUE(
        torch::equal(host_value[mapping.host_block_id][1],
                     value_layer_two[mapping.device_block_id].to(torch::kCPU)));
    EXPECT_TRUE(
        torch::equal(host_value[mapping.host_block_id][0],
                     torch::zeros_like(host_value[mapping.host_block_id][0])));
  }
}

TEST(CompactOffloadTest, ClearsSharedSlotsWhenGroupsChange) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for compact D2H.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::Tensor host_a = make_host_blocks(/*block_count=*/2, /*layer_count=*/3);
  torch::Tensor host_b = make_host_blocks(/*block_count=*/2, /*layer_count=*/3);
  torch::Tensor layer_a_zero =
      make_device_blocks(device, /*block_count=*/2, /*offset=*/10.0);
  torch::Tensor layer_a_two =
      make_device_blocks(device, /*block_count=*/2, /*offset=*/20.0);
  torch::Tensor layer_b_one =
      make_device_blocks(device, /*block_count=*/2, /*offset=*/30.0);
  HostKVGroupLayout group_a;
  group_a.group_id = 1;
  group_a.host_roles.emplace(KVCacheTensorRole::KEY, host_a);
  group_a.layers = {
      HostKVLayerLayout{0, 0, {{KVCacheTensorRole::KEY, layer_a_zero}}},
      HostKVLayerLayout{2, 2, {{KVCacheTensorRole::KEY, layer_a_two}}},
  };
  HostKVGroupLayout group_b;
  group_b.group_id = 2;
  group_b.host_roles.emplace(KVCacheTensorRole::KEY, host_b);
  group_b.layers = {
      HostKVLayerLayout{1, 1, {{KVCacheTensorRole::KEY, layer_b_one}}},
  };
  HostKVLayout layout(
      /*num_layers=*/3,
      {std::move(group_a), std::move(group_b)},
      device.unwrap());
  SynchronousD2HBatchMemcpy batch_memcpy;
  CompactOffloadHarness compact_offload(layout,
                                        device,
                                        batch_memcpy,
                                        /*target_bytes=*/24);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  ASSERT_NE(compute_stream, nullptr);
  const HostKVRequest request_a{
      {HostKVMapping{1, 0, 0}, HostKVMapping{1, 1, 1}}};
  const HostKVRequest request_b{
      {HostKVMapping{2, 0, 1}, HostKVMapping{2, 1, 0}}};

  ASSERT_TRUE(compact_offload.execute(request_a, *compute_stream));
  ASSERT_TRUE(compact_offload.execute(request_b, *compute_stream));
  host_b.fill_(-1.0);
  ASSERT_TRUE(compact_offload.execute(request_b, *compute_stream));

  EXPECT_EQ(batch_memcpy.submit_count(), 6U);
  EXPECT_EQ(batch_memcpy.descriptor_count(), 6U);
  const std::vector<int64_t> device_blocks = {1, 0};
  for (int64_t host_block = 0; host_block < 2; ++host_block) {
    EXPECT_TRUE(torch::equal(host_b[host_block][0],
                             torch::zeros_like(host_b[host_block][0])));
    EXPECT_TRUE(
        torch::equal(host_b[host_block][1],
                     layer_b_one[device_blocks[host_block]].to(torch::kCPU)));
    EXPECT_TRUE(torch::equal(host_b[host_block][2],
                             torch::zeros_like(host_b[host_block][2])));
  }
}

TEST(CompactOffloadTest, UsesOneDescriptorPerMappingAndActiveRole) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for compact D2H.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  constexpr int64_t kMappingCount = 512;
  torch::Tensor host_key = make_host_blocks(kMappingCount, /*layer_count=*/2);
  torch::Tensor host_value = make_host_blocks(kMappingCount, /*layer_count=*/2);
  torch::Tensor key_layer_zero =
      make_device_blocks(device, kMappingCount, /*offset=*/10.0);
  torch::Tensor key_layer_two =
      make_device_blocks(device, kMappingCount, /*offset=*/20.0);
  torch::Tensor value_layer_two =
      make_device_blocks(device, kMappingCount, /*offset=*/30.0);
  HostKVLayout layout(
      make_layout(
          host_key, host_value, key_layer_zero, key_layer_two, value_layer_two),
      device.unwrap());
  CountingD2HBatchMemcpy batch_memcpy;
  CompactOffloadHarness compact_offload(layout, device, batch_memcpy);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  ASSERT_NE(compute_stream, nullptr);
  HostKVRequest request;
  request.mappings.reserve(kMappingCount);
  for (int64_t block_id = 0; block_id < kMappingCount; ++block_id) {
    request.mappings.emplace_back(HostKVMapping{17, block_id, block_id});
  }

  ASSERT_TRUE(compact_offload.execute(request, *compute_stream));

  EXPECT_EQ(batch_memcpy.submit_count(), 1U);
  EXPECT_EQ(batch_memcpy.descriptor_count(), 1024U);
}

}  // namespace
}  // namespace xllm
