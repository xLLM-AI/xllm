/* Copyright 2025-2026 The xLLM Authors.

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

#include "framework/kv_cache_transfer/mooncake_transfer_engine.h"

#include <brpc/controller.h>
#include <gtest/gtest.h>

#if defined(USE_NPU)
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache_transfer/kv_cache_transfer.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "util/net.h"
#include "worker.pb.h"

#define private public
#define protected public
#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"
#undef private
#undef protected

namespace xllm {

namespace {

constexpr size_t kScaleBlockBytes = 96 * sizeof(float);

TransferKVInfo make_info(int32_t dst_dp_size,
                         int32_t dst_tp_size,
                         int32_t dst_dp_rank) {
  TransferKVInfo info;
  info.request_id = "req";
  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.local_ids = {11, 12};
  mapping.remote_ids = {21, 22};
  info.mappings.emplace_back(std::move(mapping));
  info.dp_rank = dst_dp_rank;
  info.remote_instance_info.dp_size = dst_dp_size;

  int32_t dst_world_size = dst_dp_size * dst_tp_size;
  for (int32_t i = 0; i < dst_world_size; ++i) {
    info.remote_instance_info.cluster_ids.emplace_back(
        static_cast<uint64_t>(100 + i));
    info.remote_instance_info.addrs.emplace_back("addr_" + std::to_string(i));
  }

  return info;
}

ParallelArgs make_args(int32_t rank, int32_t world_size, int32_t dp_size) {
  return ParallelArgs(rank, world_size, dp_size, nullptr);
}

void expect_same_mappings(const std::vector<KVTransferMapping>& lhs,
                          const std::vector<KVTransferMapping>& rhs) {
  ASSERT_EQ(lhs.size(), rhs.size());
  for (size_t index = 0; index < lhs.size(); ++index) {
    EXPECT_EQ(lhs[index].group_id, rhs[index].group_id);
    EXPECT_EQ(lhs[index].local_ids, rhs[index].local_ids);
    EXPECT_EQ(lhs[index].remote_ids, rhs[index].remote_ids);
  }
}

void expect_same_merge(
    const std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo>& lhs,
    const std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo>& rhs) {
  ASSERT_EQ(lhs.size(), rhs.size());
  for (const auto& [key, lhs_info] : lhs) {
    auto it = rhs.find(key);
    ASSERT_NE(it, rhs.end());
    const KVCacheTransfer::KVCacheInfo& rhs_info = it->second;
    EXPECT_EQ(lhs_info.dst_cluster_id, rhs_info.dst_cluster_id);
    EXPECT_EQ(lhs_info.dst_addr, rhs_info.dst_addr);
    expect_same_mappings(lhs_info.mappings, rhs_info.mappings);
  }
}

class RecordingMooncakeTransferEngine final : public MooncakeTransferEngine {
 public:
  struct MoveCall {
    std::string remote_addr;
    std::vector<BufferTransferMapping> mappings;
    MoveOpcode opcode;
  };

  RecordingMooncakeTransferEngine(uint16_t listen_port,
                                  const torch::Device& device)
      : MooncakeTransferEngine(listen_port, device) {}

  bool register_memory(std::vector<void*> addrs,
                       std::vector<size_t> lens,
                       std::vector<uint64_t> buf_bytes) override {
    registered_addrs.emplace_back(std::move(addrs));
    registered_lens.emplace_back(std::move(lens));
    registered_block_bytes.emplace_back(std::move(buf_bytes));
    return true;
  }

  bool move_memory_groups(const std::string& remote_addr,
                          const std::vector<BufferTransferMapping>& mappings,
                          MoveOpcode opcode) override {
    move_calls.emplace_back(MoveCall{remote_addr, mappings, opcode});
    return move_result;
  }

  bool move_result = true;
  std::vector<std::vector<void*>> registered_addrs;
  std::vector<std::vector<size_t>> registered_lens;
  std::vector<std::vector<uint64_t>> registered_block_bytes;
  std::vector<MoveCall> move_calls;
};

#if defined(USE_NPU)
constexpr int32_t kValidatePushCommand = 1;
constexpr int32_t kPreparePullCommand = 2;
constexpr int32_t kStopChildCommand = 3;
constexpr char kPeerCommandFdEnv[] = "XLLM_MOONCAKE_TEST_PEER_COMMAND_FD";
constexpr char kPeerStatusFdEnv[] = "XLLM_MOONCAKE_TEST_PEER_STATUS_FD";
constexpr char kPeerListenPortEnv[] = "XLLM_MOONCAKE_TEST_PEER_LISTEN_PORT";
constexpr char kPeerDeviceIndexEnv[] = "XLLM_MOONCAKE_TEST_PEER_DEVICE_INDEX";
constexpr char kControllerProcessEnv[] =
    "XLLM_MOONCAKE_TEST_CONTROLLER_PROCESS";

bool write_all(int fd, const void* data, size_t size) {
  const char* cursor = static_cast<const char*>(data);
  while (size > 0) {
    const ssize_t written = write(fd, cursor, size);
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written <= 0) {
      return false;
    }
    cursor += written;
    size -= static_cast<size_t>(written);
  }
  return true;
}

bool read_all(int fd, void* data, size_t size) {
  char* cursor = static_cast<char*>(data);
  while (size > 0) {
    const ssize_t received = read(fd, cursor, size);
    if (received < 0 && errno == EINTR) {
      continue;
    }
    if (received <= 0) {
      return false;
    }
    cursor += received;
    size -= static_cast<size_t>(received);
  }
  return true;
}

bool write_endpoint(int fd,
                    uint64_t cluster_id,
                    uint16_t listen_port,
                    const std::string& addr) {
  const uint32_t addr_size = static_cast<uint32_t>(addr.size());
  return write_all(fd, &cluster_id, sizeof(cluster_id)) &&
         write_all(fd, &listen_port, sizeof(listen_port)) &&
         write_all(fd, &addr_size, sizeof(addr_size)) &&
         write_all(fd, addr.data(), addr_size);
}

bool read_endpoint(int fd,
                   uint64_t* cluster_id,
                   uint16_t* listen_port,
                   std::string* addr) {
  uint32_t addr_size = 0;
  if (!read_all(fd, cluster_id, sizeof(*cluster_id)) ||
      !read_all(fd, listen_port, sizeof(*listen_port)) ||
      !read_all(fd, &addr_size, sizeof(addr_size)) || addr_size == 0 ||
      addr_size > 1024) {
    return false;
  }
  addr->resize(addr_size);
  return read_all(fd, addr->data(), addr_size);
}

class ChildProcessGuard final {
 public:
  explicit ChildProcessGuard(pid_t pid) : pid_(pid) {}
  ~ChildProcessGuard() {
    if (pid_ > 0) {
      kill(pid_, SIGKILL);
      while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {
      }
    }
  }

  void release() { pid_ = -1; }

 private:
  pid_t pid_;
};

class ScopedSigpipeIgnore final {
 public:
  ScopedSigpipeIgnore() : previous_handler_(signal(SIGPIPE, SIG_IGN)) {}
  ~ScopedSigpipeIgnore() {
    if (previous_handler_ != SIG_ERR) {
      signal(SIGPIPE, previous_handler_);
    }
  }

 private:
  using SignalHandler = void (*)(int);
  SignalHandler previous_handler_;
};

class ScopedEnvironmentVariable final {
 public:
  explicit ScopedEnvironmentVariable(const char* name) : name_(name) {
    const char* value = std::getenv(name);
    if (value != nullptr) {
      original_value_ = value;
    }
  }

  ~ScopedEnvironmentVariable() {
    if (original_value_.has_value()) {
      setenv(name_.c_str(), original_value_->c_str(), /*overwrite=*/1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  bool set(const char* value) {
    return setenv(name_.c_str(), value, /*overwrite=*/1) == 0;
  }

 private:
  std::string name_;
  std::optional<std::string> original_value_;
};

struct NpuMixedTransferCaches {
  torch::Tensor backing;
  torch::Tensor conv;
  torch::Tensor ssm;
  torch::Tensor key;
  torch::Tensor value;
  torch::Tensor index;
  torch::Tensor index_scale;
  std::vector<KVCache> caches;
};

NpuMixedTransferCaches make_npu_mixed_transfer_caches(
    const torch::Device& device) {
  NpuMixedTransferCaches tensors;
  tensors.backing = torch::zeros({6, 2, 1024, 512},
                                 torch::dtype(torch::kBFloat16).device(device));
  tensors.conv = tensors.backing.index({0});
  tensors.ssm = tensors.backing.index({1});
  tensors.key = tensors.backing.index({2});
  tensors.value = tensors.backing.index({3});
  tensors.index = tensors.backing.index({4});
  tensors.index_scale = tensors.backing.index({5});
  tensors.caches.emplace_back(
      LinearAttentionKVCacheTensors{tensors.conv, tensors.ssm});
  tensors.caches.emplace_back(
      IndexedKVCacheTensors{KVCacheTensors{tensors.key, tensors.value},
                            tensors.index,
                            tensors.index_scale});
  return tensors;
}

void fill_mixed_transfer_block(NpuMixedTransferCaches* tensors,
                               int64_t block_id,
                               bool pull_pattern) {
  const double offset = pull_pattern ? 4.0 : 0.0;
  tensors->conv.index({block_id}).fill_(1.25 + offset);
  tensors->ssm.index({block_id}).fill_(-2.5 - offset);
  tensors->key.index({block_id}).fill_(3.5 + offset);
  tensors->value.index({block_id}).fill_(-4.5 - offset);
  tensors->index.index({block_id}).fill_(pull_pattern ? 17 : 42);
  tensors->index_scale.index({block_id}).fill_(pull_pattern ? 0.25 : 0.125);
}

void zero_mixed_transfer_block(NpuMixedTransferCaches* tensors,
                               int64_t block_id) {
  tensors->conv.index({block_id}).zero_();
  tensors->ssm.index({block_id}).zero_();
  tensors->key.index({block_id}).zero_();
  tensors->value.index({block_id}).zero_();
  tensors->index.index({block_id}).zero_();
  tensors->index_scale.index({block_id}).zero_();
}

bool tensor_block_has_value(const torch::Tensor& tensor,
                            int64_t block_id,
                            double value) {
  const torch::Tensor block = tensor.index({block_id}).cpu();
  return torch::equal(block, torch::full_like(block, value));
}

bool mixed_transfer_block_matches(const NpuMixedTransferCaches& tensors,
                                  int64_t block_id,
                                  bool pull_pattern) {
  const double offset = pull_pattern ? 4.0 : 0.0;
  return tensor_block_has_value(tensors.conv, block_id, 1.25 + offset) &&
         tensor_block_has_value(tensors.ssm, block_id, -2.5 - offset) &&
         tensor_block_has_value(tensors.key, block_id, 3.5 + offset) &&
         tensor_block_has_value(tensors.value, block_id, -4.5 - offset) &&
         tensor_block_has_value(
             tensors.index, block_id, pull_pattern ? 17 : 42) &&
         tensor_block_has_value(
             tensors.index_scale, block_id, pull_pattern ? 0.25 : 0.125);
}

int run_npu_round_trip_peer(int command_fd,
                            int status_fd,
                            uint16_t listen_port,
                            int32_t device_index) {
  Device remote_device(device_index);
  remote_device.set_device();
  remote_device.init_device_context();
  const torch::Device remote_torch_device = remote_device.unwrap();
  MooncakeKVCacheTransferDefault remote_transfer(device_index,
                                                 listen_port,
                                                 remote_torch_device,
                                                 /*model_type=*/"test");
  remote_transfer.initialize(device_index);
  NpuMixedTransferCaches remote_caches =
      make_npu_mixed_transfer_caches(remote_torch_device);
  remote_transfer.register_kv_cache(
      remote_caches.caches, KVCacheShape(), torch::kBFloat16);

  const auto& layers = remote_transfer.main_layout_.layers;
  const bool layout_matches =
      layers.size() == 2 && layers[0].size() == 2 && layers[1].size() == 4 &&
      layers[0][0].role == KVCacheTensorRole::CONV &&
      layers[0][1].role == KVCacheTensorRole::SSM &&
      layers[0][0].group_id == cache_group_id(BlockType::LINEAR) &&
      layers[0][1].group_id == cache_group_id(BlockType::LINEAR) &&
      layers[1][0].role == KVCacheTensorRole::KEY &&
      layers[1][1].role == KVCacheTensorRole::VALUE &&
      layers[1][2].role == KVCacheTensorRole::INDEX &&
      layers[1][3].role == KVCacheTensorRole::INDEX_SCALE &&
      layers[1][0].group_id == cache_group_id(BlockType::KV) &&
      layers[1][1].group_id == cache_group_id(BlockType::KV) &&
      layers[1][2].group_id == cache_group_id(BlockType::KV) &&
      layers[1][3].group_id == cache_group_id(BlockType::KV);
  if (!layout_matches) {
    return 10;
  }

  uint64_t remote_cluster_id = 0;
  std::string remote_addr;
  remote_transfer.get_cache_info(remote_cluster_id, remote_addr);
  if (remote_addr.empty() ||
      !write_endpoint(status_fd, remote_cluster_id, listen_port, remote_addr)) {
    return 11;
  }

  while (true) {
    int32_t command = 0;
    if (!read_all(command_fd, &command, sizeof(command))) {
      return 12;
    }

    uint8_t success = 0;
    if (command == kValidatePushCommand) {
      remote_device.set_device();
      success = remote_device.synchronize_default_stream() == 0 &&
                        mixed_transfer_block_matches(remote_caches,
                                                     /*block_id=*/1,
                                                     /*pull_pattern=*/false)
                    ? 1
                    : 0;
    } else if (command == kPreparePullCommand) {
      remote_device.set_device();
      fill_mixed_transfer_block(
          &remote_caches, /*block_id=*/1, /*pull_pattern=*/true);
      success = remote_device.synchronize_default_stream() == 0 ? 1 : 0;
    } else if (command == kStopChildCommand) {
      close(command_fd);
      close(status_fd);
      // The peer is an exec-isolated test process. The transfer and remote
      // session have already been verified and closed by the parent before
      // this command. Bypass third-party process-global teardown, which can
      // terminate on a still-joinable TransferEngine thread.
      _exit(0);
    } else {
      return 13;
    }

    if (!write_all(status_fd, &success, sizeof(success))) {
      return 14;
    }
  }
}
#endif

#if defined(USE_MLU)
KVCacheShape make_indexer_int8_transfer_shape() {
  proto::KVCacheShape proto_shape;
  for (int64_t dim : std::vector<int64_t>{2, 1, 1, 16}) {
    proto_shape.add_key_cache_shape(dim);
    proto_shape.add_value_cache_shape(dim);
  }
  for (int64_t dim : std::vector<int64_t>{2, 96, 1, 8}) {
    proto_shape.add_index_cache_shape(dim);
  }
  for (int64_t dim : std::vector<int64_t>{2, 96, 1}) {
    proto_shape.add_index_cache_scale_shape(dim);
  }
  return KVCacheShape::from_proto(proto_shape);
}

std::vector<KVCache> make_mixed_transfer_caches(const torch::Device& device) {
  std::shared_ptr<KVCacheTensorAllocator> allocator =
      default_kv_tensor_allocator();
  auto make_full_cache = [&allocator, &device]() {
    torch::Tensor key = allocator->allocate(
        KVCacheTensorRole::KEY, {2, 1, 1, 16}, torch::kBFloat16, device);
    torch::Tensor index = allocator->allocate(
        KVCacheTensorRole::INDEX, {2, 96, 1, 8}, torch::kChar, device);
    torch::Tensor index_scale = allocator->allocate(
        KVCacheTensorRole::INDEX_SCALE, {2, 96, 1}, torch::kFloat32, device);
    return KVCache(IndexedKVCacheTensors{
        KVCacheTensors{key, torch::Tensor()}, index, index_scale});
  };
  auto make_shared_cache = [&allocator, &device]() {
    torch::Tensor key = allocator->allocate(
        KVCacheTensorRole::KEY, {2, 1, 1, 16}, torch::kChar, device);
    torch::Tensor key_scale = allocator->allocate(
        KVCacheTensorRole::KEY_SCALE, {2, 1, 1}, torch::kFloat32, device);
    return KVCache(QuantizedKVCacheTensors{
        KVCacheTensors{key, torch::Tensor()}, key_scale, torch::Tensor()});
  };

  std::vector<KVCache> caches;
  caches.reserve(4);
  caches.emplace_back(make_full_cache());
  caches.emplace_back(make_shared_cache());
  caches.emplace_back(make_full_cache());
  caches.emplace_back(make_shared_cache());
  return caches;
}
#endif

}  // namespace

TEST(MooncakeTransferEngineServiceTest, OpenSessionRejectsMissingAddr) {
  MooncakeTransferEngineService service;
  proto::SessionInfo request;
  proto::Status response;
  brpc::Controller cntl;

  service.OpenSession(&cntl, &request, &response, nullptr);

  EXPECT_FALSE(response.ok());
}

TEST(MooncakeTransferEngineServiceTest, CloseSessionRejectsMissingAddr) {
  MooncakeTransferEngineService service;
  proto::SessionInfo request;
  proto::Status response;
  brpc::Controller cntl;

  service.CloseSession(&cntl, &request, &response, nullptr);

  EXPECT_FALSE(response.ok());
}

TEST(MooncakeTransferEngineServiceTest, CloseSessionWithoutHandleReturnsTrue) {
  MooncakeTransferEngineService service;
  proto::SessionInfo request;
  request.set_addr("127.0.0.1:5001");
  proto::Status response;
  brpc::Controller cntl;

  service.CloseSession(&cntl, &request, &response, nullptr);

  EXPECT_TRUE(response.ok());
}

TEST(MooncakeKVCacheTransferDefaultTest,
     PullUsesGroupSpecificMappingsForKvAndLinearBuffers) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  caches.emplace_back(LinearAttentionKVCacheTensors{torch::zeros({1, 3}),
                                                    torch::zeros({1, 5})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  KVTransferMapping kv_mapping;
  kv_mapping.group_id = cache_group_id(BlockType::KV);
  kv_mapping.local_ids = {1, 2};
  kv_mapping.remote_ids = {11, 12};
  KVTransferMapping linear_mapping;
  linear_mapping.group_id = cache_group_id(BlockType::LINEAR);
  linear_mapping.local_ids = {0};
  linear_mapping.remote_ids = {7};

  ASSERT_TRUE(transfer.pull_kv_blocks(
      /*src_cluster_id=*/1, "remote", {kv_mapping, linear_mapping}));
  ASSERT_EQ(engine_observer->move_calls.size(), 1U);
  const RecordingMooncakeTransferEngine::MoveCall& call =
      engine_observer->move_calls[0];
  EXPECT_EQ(call.remote_addr, "remote");
  EXPECT_EQ(call.opcode, MooncakeTransferEngine::MoveOpcode::READ);
  ASSERT_EQ(call.mappings.size(), 4U);
  EXPECT_EQ(call.mappings[0].buf_id, 0);
  EXPECT_EQ(call.mappings[1].buf_id, 1);
  EXPECT_EQ(call.mappings[0].local_ids, kv_mapping.local_ids);
  EXPECT_EQ(call.mappings[1].remote_ids, kv_mapping.remote_ids);
  EXPECT_EQ(call.mappings[2].buf_id, 2);
  EXPECT_EQ(call.mappings[3].buf_id, 3);
  EXPECT_EQ(call.mappings[2].local_ids, linear_mapping.local_ids);
  EXPECT_EQ(call.mappings[3].remote_ids, linear_mapping.remote_ids);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     GroupedPullUsesSwaAndCompressedMappings) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"deepseek_v4",
                                          std::move(engine));
  DeepSeekV4KVCacheTensors tensors;
  tensors.key_cache = torch::zeros({4, 2});
  tensors.swa_cache = torch::zeros({4, 3});
  tensors.compressed_block_type = BlockType::C4;
  std::vector<KVCache> caches;
  caches.emplace_back(tensors);
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  KVTransferMapping swa_mapping;
  swa_mapping.group_id = cache_group_id(BlockType::SWA);
  swa_mapping.local_ids = {1};
  swa_mapping.remote_ids = {11};
  KVTransferMapping c4_mapping;
  c4_mapping.group_id = cache_group_id(BlockType::C4);
  c4_mapping.local_ids = {2};
  c4_mapping.remote_ids = {12};

  ASSERT_TRUE(transfer.pull_kv_blocks(
      /*src_cluster_id=*/1, "remote", {c4_mapping, swa_mapping}));
  ASSERT_EQ(engine_observer->move_calls.size(), 1U);
  const std::vector<MooncakeTransferEngine::BufferTransferMapping>& mappings =
      engine_observer->move_calls[0].mappings;
  ASSERT_EQ(mappings.size(), 2U);
  EXPECT_EQ(mappings[0].buf_id, 0);
  EXPECT_EQ(mappings[0].local_ids, swa_mapping.local_ids);
  EXPECT_EQ(mappings[1].buf_id, 1);
  EXPECT_EQ(mappings[1].remote_ids, c4_mapping.remote_ids);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     PullCoversMainAndSpecLayoutsInOneRead) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> main_caches;
  main_caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  std::vector<KVCache> spec_caches;
  spec_caches.emplace_back(LinearAttentionKVCacheTensors{torch::zeros({1, 3}),
                                                         torch::zeros({1, 5})});
  transfer.register_kv_cache(main_caches, KVCacheShape(), torch::kFloat32);
  transfer.register_kv_cache_spec(spec_caches, KVCacheShape(), torch::kFloat32);

  KVTransferMapping kv_mapping;
  kv_mapping.group_id = cache_group_id(BlockType::KV);
  kv_mapping.local_ids = {1};
  kv_mapping.remote_ids = {11};
  KVTransferMapping linear_mapping;
  linear_mapping.group_id = cache_group_id(BlockType::LINEAR);
  linear_mapping.local_ids = {0};
  linear_mapping.remote_ids = {7};

  ASSERT_TRUE(transfer.pull_kv_blocks(
      /*src_cluster_id=*/1, "remote", {kv_mapping, linear_mapping}));
  ASSERT_EQ(engine_observer->move_calls.size(), 1U);
  const std::vector<MooncakeTransferEngine::BufferTransferMapping>& mappings =
      engine_observer->move_calls[0].mappings;
  ASSERT_EQ(mappings.size(), 4U);
  EXPECT_EQ(mappings[0].buf_id, 0);
  EXPECT_EQ(mappings[1].buf_id, 1);
  EXPECT_EQ(mappings[2].buf_id, 2);
  EXPECT_EQ(mappings[3].buf_id, 3);
}

TEST(MooncakeKVCacheTransferDefaultTest, PullRejectsInvalidMappings) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  KVTransferMapping mismatched;
  mismatched.group_id = cache_group_id(BlockType::KV);
  mismatched.local_ids = {1, 2};
  mismatched.remote_ids = {11};
  EXPECT_FALSE(
      transfer.pull_kv_blocks(/*src_cluster_id=*/1, "remote", {mismatched}));

  KVTransferMapping duplicate;
  duplicate.group_id = cache_group_id(BlockType::KV);
  duplicate.local_ids = {1};
  duplicate.remote_ids = {11};
  EXPECT_FALSE(transfer.pull_kv_blocks(
      /*src_cluster_id=*/1, "remote", {duplicate, duplicate}));

  KVTransferMapping wrong_group;
  wrong_group.group_id = cache_group_id(BlockType::LINEAR);
  wrong_group.local_ids = {0};
  wrong_group.remote_ids = {7};
  EXPECT_FALSE(
      transfer.pull_kv_blocks(/*src_cluster_id=*/1, "remote", {wrong_group}));
  EXPECT_TRUE(engine_observer->move_calls.empty());
}

#if defined(USE_NPU)
TEST(MooncakeKVCacheTransferDefaultTest,
     PushRejectsDuplicateMappingsBeforeMerge) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/1,
                                  /*dst_dp_rank=*/0);
  info.mappings.emplace_back(info.mappings[0]);
  const ParallelArgs parallel_args = make_args(/*rank=*/0,
                                               /*world_size=*/1,
                                               /*dp_size=*/1);
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer;

  folly::SemiFuture<bool> future = transfer.push_kv_blocks_async(
      {info}, parallel_args, synchronizer, /*is_spec_draft=*/false);
  EXPECT_FALSE(std::move(future).get());
  EXPECT_TRUE(engine_observer->move_calls.empty());
}

TEST(MooncakeKVCacheTransferDefaultTest,
     KvSplitFilterAcceptsCompleteAndPartialFinalCoverage) {
  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/1,
                                  /*dst_dp_rank=*/0);
  info.mappings[0].remote_ids = {21, 22, 23};

  std::vector<TransferKVInfo> rank_zero_infos = filter_kv_split_infos(
      /*kv_split_rank=*/0, /*kv_split_size=*/2, {info});
  ASSERT_EQ(rank_zero_infos.size(), 1U);
  ASSERT_EQ(rank_zero_infos[0].mappings.size(), 1U);
  EXPECT_EQ(rank_zero_infos[0].mappings[0].local_ids,
            (std::vector<uint64_t>{11, 12}));
  EXPECT_EQ(rank_zero_infos[0].mappings[0].remote_ids,
            (std::vector<uint64_t>{21, 23}));

  std::vector<TransferKVInfo> rank_one_infos = filter_kv_split_infos(
      /*kv_split_rank=*/1, /*kv_split_size=*/2, {info});
  ASSERT_EQ(rank_one_infos.size(), 1U);
  ASSERT_EQ(rank_one_infos[0].mappings.size(), 1U);
  EXPECT_EQ(rank_one_infos[0].mappings[0].local_ids,
            (std::vector<uint64_t>{11}));
  EXPECT_EQ(rank_one_infos[0].mappings[0].remote_ids,
            (std::vector<uint64_t>{22}));
}

TEST(MooncakeKVCacheTransferDefaultTest,
     PushRejectsIncompleteKvSplitCoverageBeforeFilter) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/1,
                                  /*dst_dp_rank=*/0);
  info.mappings[0].remote_ids = {21, 22};
  ParallelArgs parallel_args = make_args(/*rank=*/0,
                                         /*world_size=*/2,
                                         /*dp_size=*/1);
  parallel_args.kv_split_size(2);
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer;

  folly::SemiFuture<bool> future = transfer.push_kv_blocks_async(
      {info}, parallel_args, synchronizer, /*is_spec_draft=*/false);
  EXPECT_FALSE(std::move(future).get());
  EXPECT_TRUE(engine_observer->move_calls.empty());
}

TEST(MooncakeKVCacheTransferDefaultTest,
     PushRejectsExcessKvSplitCoverageBeforeFilter) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/1,
                                  /*dst_dp_rank=*/0);
  info.mappings[0].remote_ids = {21, 22, 23, 24, 25};
  ParallelArgs parallel_args = make_args(/*rank=*/0,
                                         /*world_size=*/2,
                                         /*dp_size=*/1);
  parallel_args.kv_split_size(2);
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer;

  folly::SemiFuture<bool> future = transfer.push_kv_blocks_async(
      {info}, parallel_args, synchronizer, /*is_spec_draft=*/false);
  EXPECT_FALSE(std::move(future).get());
  EXPECT_TRUE(engine_observer->move_calls.empty());
}

TEST(MooncakeKVCacheTransferDefaultTest,
     DISABLED_NpuLinearIndexerScaleRoundTripPeerProcess) {
  const char* command_fd_env = std::getenv(kPeerCommandFdEnv);
  const char* status_fd_env = std::getenv(kPeerStatusFdEnv);
  const char* listen_port_env = std::getenv(kPeerListenPortEnv);
  const char* device_index_env = std::getenv(kPeerDeviceIndexEnv);
  ASSERT_NE(command_fd_env, nullptr);
  ASSERT_NE(status_fd_env, nullptr);
  ASSERT_NE(listen_port_env, nullptr);
  ASSERT_NE(device_index_env, nullptr);

  const int command_fd = std::atoi(command_fd_env);
  const int status_fd = std::atoi(status_fd_env);
  const int listen_port = std::atoi(listen_port_env);
  const int device_index = std::atoi(device_index_env);
  ASSERT_GE(command_fd, 0);
  ASSERT_GE(status_fd, 0);
  ASSERT_GT(listen_port, 0);
  ASSERT_LE(listen_port, UINT16_MAX);
  ASSERT_GE(device_index, 0);
  EXPECT_EQ(run_npu_round_trip_peer(command_fd,
                                    status_fd,
                                    static_cast<uint16_t>(listen_port),
                                    device_index),
            0);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     NpuLinearIndexerScalePushAndPullRoundTrip) {
  if (std::getenv(kControllerProcessEnv) == nullptr) {
    ScopedEnvironmentVariable controller_process(kControllerProcessEnv);
    ASSERT_TRUE(controller_process.set("1"));

    const pid_t controller_pid = fork();
    ASSERT_GE(controller_pid, 0);
    if (controller_pid == 0) {
      execl("/proc/self/exe",
            "mooncake_transfer_engine_test",
            "--gtest_filter=MooncakeKVCacheTransferDefaultTest."
            "NpuLinearIndexerScalePushAndPullRoundTrip",
            "--gtest_color=no",
            static_cast<char*>(nullptr));
      _exit(127);
    }

    ChildProcessGuard controller_guard(controller_pid);
    int controller_status = 0;
    pid_t waited_pid = -1;
    do {
      waited_pid = waitpid(controller_pid, &controller_status, 0);
    } while (waited_pid < 0 && errno == EINTR);
    if (waited_pid == controller_pid) {
      controller_guard.release();
    }
    ASSERT_EQ(waited_pid, controller_pid);
    ASSERT_TRUE(WIFEXITED(controller_status));
    EXPECT_EQ(WEXITSTATUS(controller_status), 0);
    return;
  }

  const int32_t device_count = Platform::device_count();
  if (device_count < 2) {
    GTEST_SKIP() << "Two NPU devices are required for Mooncake memory "
                    "transfer.";
  }
  const int32_t remote_device_index = device_count > 4 ? 4 : 1;

  const int32_t local_listen_port = net::get_local_free_port();
  int32_t remote_listen_port = net::get_local_free_port();
  while (remote_listen_port == local_listen_port) {
    remote_listen_port = net::get_local_free_port();
  }
  ASSERT_GT(local_listen_port, 0);
  ASSERT_GT(remote_listen_port, 0);

  int parent_to_child[2];
  int child_to_parent[2];
  ASSERT_EQ(pipe(parent_to_child), 0);
  ASSERT_EQ(pipe(child_to_parent), 0);
  ScopedSigpipeIgnore sigpipe_guard;
  ScopedEnvironmentVariable hccl_base_port("HCCL_IF_BASE_PORT");
  ASSERT_TRUE(hccl_base_port.set("35439"));

  ASSERT_EQ(setenv(kPeerCommandFdEnv,
                   std::to_string(parent_to_child[0]).c_str(),
                   /*overwrite=*/1),
            0);
  ASSERT_EQ(setenv(kPeerStatusFdEnv,
                   std::to_string(child_to_parent[1]).c_str(),
                   /*overwrite=*/1),
            0);
  ASSERT_EQ(setenv(kPeerListenPortEnv,
                   std::to_string(remote_listen_port).c_str(),
                   /*overwrite=*/1),
            0);
  ASSERT_EQ(setenv(kPeerDeviceIndexEnv,
                   std::to_string(remote_device_index).c_str(),
                   /*overwrite=*/1),
            0);

  const pid_t child_pid = fork();
  ASSERT_GE(child_pid, 0);
  if (child_pid == 0) {
    close(parent_to_child[1]);
    close(child_to_parent[0]);
    execl("/proc/self/exe",
          "mooncake_transfer_engine_test",
          "--gtest_filter=MooncakeKVCacheTransferDefaultTest."
          "DISABLED_NpuLinearIndexerScaleRoundTripPeerProcess",
          "--gtest_also_run_disabled_tests",
          "--gtest_color=no",
          static_cast<char*>(nullptr));
    _exit(127);
  }

  ASSERT_TRUE(hccl_base_port.set("34439"));

  unsetenv(kPeerCommandFdEnv);
  unsetenv(kPeerStatusFdEnv);
  unsetenv(kPeerListenPortEnv);
  unsetenv(kPeerDeviceIndexEnv);

  close(parent_to_child[0]);
  close(child_to_parent[1]);
  ChildProcessGuard child_guard(child_pid);

  Device local_device(/*device_id=*/0);
  local_device.set_device();
  local_device.init_device_context();
  const torch::Device local_torch_device = local_device.unwrap();
  MooncakeKVCacheTransferDefault local_transfer(
      /*device_id=*/0,
      static_cast<uint16_t>(local_listen_port),
      local_torch_device,
      /*model_type=*/"test");
  local_transfer.initialize(/*device_id=*/0);
  NpuMixedTransferCaches local_caches =
      make_npu_mixed_transfer_caches(local_torch_device);
  local_transfer.register_kv_cache(
      local_caches.caches, KVCacheShape(), torch::kBFloat16);

  ASSERT_EQ(local_transfer.main_layout_.layers.size(), 2U);
  ASSERT_EQ(local_transfer.main_layout_.layers[0].size(), 2U);
  ASSERT_EQ(local_transfer.main_layout_.layers[1].size(), 4U);
  EXPECT_EQ(local_transfer.main_layout_.layers[0][0].role,
            KVCacheTensorRole::CONV);
  EXPECT_EQ(local_transfer.main_layout_.layers[0][1].role,
            KVCacheTensorRole::SSM);
  EXPECT_EQ(local_transfer.main_layout_.layers[1][0].role,
            KVCacheTensorRole::KEY);
  EXPECT_EQ(local_transfer.main_layout_.layers[1][1].role,
            KVCacheTensorRole::VALUE);
  EXPECT_EQ(local_transfer.main_layout_.layers[1][2].role,
            KVCacheTensorRole::INDEX);
  EXPECT_EQ(local_transfer.main_layout_.layers[1][3].role,
            KVCacheTensorRole::INDEX_SCALE);
  for (const auto& buffer : local_transfer.main_layout_.layers[0]) {
    EXPECT_EQ(buffer.group_id, cache_group_id(BlockType::LINEAR));
  }
  for (const auto& buffer : local_transfer.main_layout_.layers[1]) {
    EXPECT_EQ(buffer.group_id, cache_group_id(BlockType::KV));
  }

  uint64_t local_cluster_id = 0;
  std::string local_addr;
  local_transfer.get_cache_info(local_cluster_id, local_addr);
  ASSERT_FALSE(local_addr.empty());

  uint64_t remote_cluster_id = 0;
  uint16_t received_remote_port = 0;
  std::string remote_addr;
  ASSERT_TRUE(read_endpoint(child_to_parent[0],
                            &remote_cluster_id,
                            &received_remote_port,
                            &remote_addr));
  ASSERT_EQ(received_remote_port, static_cast<uint16_t>(remote_listen_port));
  ASSERT_FALSE(remote_addr.empty());
  ASSERT_TRUE(local_transfer.link_cluster(
      remote_cluster_id, remote_addr, received_remote_port));

  KVTransferMapping linear_mapping;
  linear_mapping.group_id = cache_group_id(BlockType::LINEAR);
  linear_mapping.local_ids = {0};
  linear_mapping.remote_ids = {1};
  KVTransferMapping kv_mapping;
  kv_mapping.group_id = cache_group_id(BlockType::KV);
  kv_mapping.local_ids = {0};
  kv_mapping.remote_ids = {1};

  local_device.set_device();
  fill_mixed_transfer_block(
      &local_caches, /*block_id=*/0, /*pull_pattern=*/false);
  ASSERT_EQ(local_device.synchronize_default_stream(), 0);

  KVCacheTransfer::KVCacheInfo info;
  info.dst_cluster_id = remote_cluster_id;
  info.dst_addr = remote_addr;
  info.mappings = {linear_mapping, kv_mapping};
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged_infos;
  merged_infos.emplace(remote_addr, std::move(info));
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer =
      std::make_shared<KVPushSynchronizerImpl>(/*num_layers=*/2);
  ASSERT_TRUE(synchronizer->record_event(/*layer_index=*/0,
                                         /*device_index=*/0));
  ASSERT_TRUE(synchronizer->record_event(/*layer_index=*/1,
                                         /*device_index=*/0));
  ASSERT_TRUE(local_transfer.push_kv_blocks(merged_infos,
                                            synchronizer,
                                            /*is_spec_draft=*/false,
                                            /*kv_split_rank=*/0,
                                            /*kv_split_size=*/1));

  int32_t command = kValidatePushCommand;
  ASSERT_TRUE(write_all(parent_to_child[1], &command, sizeof(command)));
  uint8_t child_success = 0;
  ASSERT_TRUE(
      read_all(child_to_parent[0], &child_success, sizeof(child_success)));
  ASSERT_EQ(child_success, 1);

  command = kPreparePullCommand;
  ASSERT_TRUE(write_all(parent_to_child[1], &command, sizeof(command)));
  ASSERT_TRUE(
      read_all(child_to_parent[0], &child_success, sizeof(child_success)));
  ASSERT_EQ(child_success, 1);

  local_device.set_device();
  zero_mixed_transfer_block(&local_caches, /*block_id=*/0);
  ASSERT_EQ(local_device.synchronize_default_stream(), 0);
  ASSERT_TRUE(local_transfer.pull_kv_blocks(
      remote_cluster_id, remote_addr, {linear_mapping, kv_mapping}));
  ASSERT_EQ(local_device.synchronize_default_stream(), 0);
  EXPECT_TRUE(mixed_transfer_block_matches(
      local_caches, /*block_id=*/0, /*pull_pattern=*/true));

  ASSERT_TRUE(local_transfer.unlink_cluster(remote_cluster_id,
                                            remote_addr,
                                            received_remote_port,
                                            /*force_flag=*/true));
  command = kStopChildCommand;
  ASSERT_TRUE(write_all(parent_to_child[1], &command, sizeof(command)));
  close(parent_to_child[1]);
  close(child_to_parent[0]);

  int child_status = 0;
  pid_t waited_pid = -1;
  do {
    waited_pid = waitpid(child_pid, &child_status, 0);
  } while (waited_pid < 0 && errno == EINTR);
  if (waited_pid == child_pid) {
    child_guard.release();
  }
  ASSERT_EQ(waited_pid, child_pid);
  ASSERT_TRUE(WIFEXITED(child_status));
  EXPECT_EQ(WEXITSTATUS(child_status), 0);

  const int32_t exit_status = ::testing::Test::HasFailure() ? 1 : 0;
  _exit(exit_status);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     PushPropagatesSynchronizeLayerFailure) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "NPU device is required for synchronizer failure test.";
  }
  Device device(/*device_id=*/0);
  device.set_device();
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.local_ids = {1};
  mapping.remote_ids = {2};
  KVCacheTransfer::KVCacheInfo info;
  info.dst_cluster_id = 1;
  info.dst_addr = "remote";
  info.mappings.emplace_back(std::move(mapping));
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged_infos;
  merged_infos.emplace("remote", std::move(info));
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer =
      std::make_shared<KVPushSynchronizerImpl>(/*num_layers=*/1);
  synchronizer->abort();

  EXPECT_FALSE(transfer.push_kv_blocks(merged_infos,
                                       synchronizer,
                                       /*is_spec_draft=*/false,
                                       /*kv_split_rank=*/0,
                                       /*kv_split_size=*/1));
  EXPECT_TRUE(engine_observer->move_calls.empty());
}

#endif

#if defined(USE_MLU)
TEST(MooncakeKVCacheTransferDefaultTest, OwnerRankMergesSingleDst) {
  MooncakeKVCacheTransferDefault transfer(
      0, 0, torch::Device(torch::kCPU), "test");
  transfer.has_v_cache_ = false;

  const TransferKVInfo info = make_info(1, 3, 0);
  const ParallelArgs parallel_args = make_args(2, 8, 1);
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged_kv_infos;

  transfer.merge_kv_blocks(merged_kv_infos, {info}, parallel_args);

  ASSERT_EQ(merged_kv_infos.size(), 1U);
  const KVCacheTransfer::KVCacheInfo& kv_info = merged_kv_infos.begin()->second;
  EXPECT_EQ(kv_info.dst_cluster_id, 102U);
  EXPECT_EQ(kv_info.dst_addr, "addr_2");
  expect_same_mappings(kv_info.mappings, info.mappings);
}

TEST(MooncakeKVCacheTransferDefaultTest, MluCpKeepsCompleteKvBlockMapping) {
  MooncakeKVCacheTransferDefault transfer(
      0, 0, torch::Device(torch::kCPU), "test");
  transfer.has_v_cache_ = false;

  const TransferKVInfo info = make_info(1, 4, 0);
  ParallelArgs parallel_args(
      2, 4, 1, 4, /*process_group=*/nullptr, /*ep_size=*/1);
  parallel_args.kv_split_size(1);
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged_kv_infos;

  transfer.merge_kv_blocks(merged_kv_infos, {info}, parallel_args);

  ASSERT_EQ(merged_kv_infos.size(), 1U);
  const KVCacheTransfer::KVCacheInfo& kv_info = merged_kv_infos.begin()->second;
  EXPECT_EQ(kv_info.dst_cluster_id, 102U);
  expect_same_mappings(kv_info.mappings, info.mappings);
}

TEST(MooncakeKVCacheTransferDefaultTest, WrappedOwnerRankKeepsMerge) {
  MooncakeKVCacheTransferDefault transfer(
      0, 0, torch::Device(torch::kCPU), "test");
  transfer.has_v_cache_ = false;

  const TransferKVInfo info = make_info(2, 3, 1);
  const ParallelArgs parallel_args = make_args(5, 8, 1);
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged_kv_infos;

  transfer.merge_kv_blocks(merged_kv_infos, {info}, parallel_args);

  ASSERT_EQ(merged_kv_infos.size(), 1U);
  const KVCacheTransfer::KVCacheInfo& kv_info = merged_kv_infos.begin()->second;
  EXPECT_EQ(kv_info.dst_cluster_id, 105U);
  EXPECT_EQ(kv_info.dst_addr, "addr_5");
  expect_same_mappings(kv_info.mappings, info.mappings);
}

TEST(MooncakeKVCacheTransferDefaultTest, HasVCacheUsesBaseMerge) {
  MooncakeKVCacheTransferDefault transfer(
      0, 0, torch::Device(torch::kCPU), "test");
  transfer.has_v_cache_ = true;

  const TransferKVInfo info = make_info(2, 3, 1);
  const ParallelArgs parallel_args = make_args(5, 8, 1);
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged_kv_infos;
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> base_kv_infos;

  transfer.merge_kv_blocks(merged_kv_infos, {info}, parallel_args);
  transfer.KVCacheTransfer::merge_kv_blocks(
      base_kv_infos, {info}, parallel_args);

  expect_same_merge(merged_kv_infos, base_kv_infos);
}

TEST(MooncakeKVCacheTransferDefaultTest, SmallSrcTpUsesBaseMerge) {
  MooncakeKVCacheTransferDefault transfer(
      0, 0, torch::Device(torch::kCPU), "test");
  transfer.has_v_cache_ = false;

  const TransferKVInfo info = make_info(1, 4, 0);
  const ParallelArgs parallel_args = make_args(1, 2, 1);
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged_kv_infos;
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> base_kv_infos;

  transfer.merge_kv_blocks(merged_kv_infos, {info}, parallel_args);
  transfer.KVCacheTransfer::merge_kv_blocks(
      base_kv_infos, {info}, parallel_args);

  expect_same_merge(merged_kv_infos, base_kv_infos);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     AddBufUsesLogicalLengthWithoutChangingBlockBytes) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for Mooncake registration tests.";
  }
  Device device(/*device_id=*/0);
  device.set_device();
  const torch::Device torch_device = device.unwrap();
  MooncakeKVCacheTransferDefault transfer(
      /*device_id=*/0,
      /*listen_port=*/0,
      torch_device,
      /*model_type=*/"test");
  const torch::Tensor tensor = torch::zeros(
      {2, 96, 1}, torch::dtype(torch::kFloat32).device(torch_device));
  std::vector<void*> addrs;
  std::vector<size_t> lens;
  std::vector<uint64_t> block_bytes;

  transfer.add_buf(tensor, addrs, lens, block_bytes);

  ASSERT_EQ(addrs.size(), 1U);
  EXPECT_EQ(addrs[0], tensor.data_ptr());
  EXPECT_EQ(lens[0], tensor.nbytes());
  EXPECT_EQ(block_bytes[0], kScaleBlockBytes);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     RegistersMixedLayersFromProtocolRolesInStableOrder) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for Mooncake registration tests.";
  }
  Device device(/*device_id=*/0);
  device.set_device();
  const torch::Device torch_device = device.unwrap();
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch_device);
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch_device,
                                          /*model_type=*/"glm_moe_dsa",
                                          std::move(engine));
  std::vector<KVCache> caches = make_mixed_transfer_caches(torch_device);
  const KVCacheShape shape = make_indexer_int8_transfer_shape();

  transfer.register_kv_cache(caches, shape, torch::kBFloat16);

  ASSERT_EQ(engine_observer->registered_addrs.size(), 1U);
  ASSERT_EQ(engine_observer->registered_addrs[0].size(), 8U);
  const std::vector<void*> expected_addrs = {
      caches[0].get_k_cache().data_ptr(),
      caches[0].get_index_cache().data_ptr(),
      caches[0].get_indexer_cache_scale()->data_ptr(),
      caches[1].get_k_cache().data_ptr(),
      caches[2].get_k_cache().data_ptr(),
      caches[2].get_index_cache().data_ptr(),
      caches[2].get_indexer_cache_scale()->data_ptr(),
      caches[3].get_k_cache().data_ptr()};
  EXPECT_EQ(engine_observer->registered_addrs[0], expected_addrs);
  EXPECT_EQ(engine_observer->registered_lens[0][2],
            caches[0].get_indexer_cache_scale()->nbytes());
  EXPECT_EQ(engine_observer->registered_block_bytes[0][2],
            caches[0].get_indexer_cache_scale()->nbytes() / 2);

  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.local_ids = {1};
  mapping.remote_ids = {0};
  EXPECT_TRUE(transfer.pull_kv_blocks(
      /*src_cluster_id=*/1, /*src_addr=*/"remote", {mapping}));
  ASSERT_EQ(engine_observer->move_calls.size(), 1U);
  ASSERT_EQ(engine_observer->move_calls[0].mappings.size(), 8U);
  for (size_t index = 0; index < engine_observer->move_calls[0].mappings.size();
       ++index) {
    EXPECT_EQ(engine_observer->move_calls[0].mappings[index].buf_id,
              static_cast<int64_t>(index));
  }
}

TEST(MooncakeKVCacheTransferDefaultTest,
     SpecRegistrationStartsAfterActualMainBufferCount) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for Mooncake registration tests.";
  }
  Device device(/*device_id=*/0);
  device.set_device();
  const torch::Device torch_device = device.unwrap();
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch_device);
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch_device,
                                          /*model_type=*/"glm_moe_dsa",
                                          std::move(engine));
  std::vector<KVCache> main_caches = make_mixed_transfer_caches(torch_device);
  std::vector<KVCache> draft_source = make_mixed_transfer_caches(torch_device);
  std::vector<KVCache> draft_caches;
  draft_caches.reserve(2);
  draft_caches.emplace_back(std::move(draft_source[1]));
  draft_caches.emplace_back(std::move(draft_source[0]));
  const KVCacheShape shape = make_indexer_int8_transfer_shape();

  transfer.register_kv_cache(main_caches, shape, torch::kBFloat16);
  transfer.register_kv_cache_spec(draft_caches, shape, torch::kBFloat16);

  ASSERT_EQ(engine_observer->registered_addrs.size(), 2U);
  EXPECT_EQ(engine_observer->registered_addrs[0].size(), 8U);
  EXPECT_EQ(engine_observer->registered_addrs[1].size(), 4U);
  ASSERT_EQ(transfer.spec_layout_.layers.size(), 2U);
  ASSERT_EQ(transfer.spec_layout_.layers[0].size(), 1U);
  ASSERT_EQ(transfer.spec_layout_.layers[1].size(), 3U);
  EXPECT_EQ(transfer.spec_layout_.layers[0][0].buf_id, 8);
  EXPECT_EQ(transfer.spec_layout_.layers[1][0].buf_id, 9);
  EXPECT_EQ(transfer.spec_layout_.layers[1][1].buf_id, 10);
  EXPECT_EQ(transfer.spec_layout_.layers[1][2].buf_id, 11);
}

TEST(MooncakeKVCacheTransferDefaultTest, AddBufRejectsNonContiguousTensor) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  MooncakeKVCacheTransferDefault transfer(
      /*device_id=*/0,
      /*listen_port=*/0,
      torch::Device(torch::kCPU),
      /*model_type=*/"test");
  torch::Tensor tensor = torch::zeros({2, 96, 2}, torch::kFloat32)
                             .transpose(/*dim0=*/1, /*dim1=*/2);
  std::vector<void*> addrs;
  std::vector<size_t> lens;
  std::vector<uint64_t> block_bytes;

  EXPECT_DEATH(transfer.add_buf(tensor, addrs, lens, block_bytes),
               "contiguous");
}

TEST(MooncakeKVCacheTransferDefaultTest,
     IndexScaleRegistersAndRoundTripsWithKvBlocks) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for Mooncake memory transfer.";
  }

  Device device(/*device_id=*/0);
  device.set_device();
  const torch::Device torch_device = device.unwrap();
  const int32_t listen_port = net::get_local_free_port();
  ASSERT_GT(listen_port, 0);

  MooncakeKVCacheTransferDefault transfer(
      /*device_id=*/0,
      static_cast<uint16_t>(listen_port),
      torch_device,
      /*model_type=*/"deepseek_v32");
  transfer.initialize(/*device_id=*/0);

  const KVCacheShape shape = make_indexer_int8_transfer_shape();
  KVCacheCreateOptions options;
  options.device(torch_device)
      .dtype(torch::kBFloat16)
      .num_layers(1)
      .model_type("deepseek_v32")
      .enable_lighting_indexer(true)
      .enable_indexer_cache_quant(true);
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, shape, options);
  ASSERT_EQ(caches.size(), 1U);

  KVCache& cache = caches[0];
  torch::Tensor key_cache = cache.get_k_cache();
  torch::Tensor value_cache = cache.get_v_cache();
  torch::Tensor index_cache = cache.get_index_cache();
  std::optional<torch::Tensor> index_scale = cache.get_indexer_cache_scale();
  ASSERT_TRUE(index_scale.has_value());
  ASSERT_EQ(index_cache.scalar_type(), torch::kChar);
  ASSERT_EQ(index_scale->scalar_type(), torch::kFloat32);
  EXPECT_EQ(index_scale->nbytes(), 2 * kScaleBlockBytes);
  EXPECT_EQ(index_scale->storage().nbytes(), index_scale->nbytes());

  key_cache.index({0}).fill_(1.25);
  key_cache.index({1}).zero_();
  value_cache.index({0}).fill_(-2.5);
  value_cache.index({1}).zero_();
  index_cache.index({0}).fill_(42);
  index_cache.index({1}).zero_();
  index_scale->index({0}).fill_(0.125);
  index_scale->index({1}).zero_();
  device.synchronize_default_stream();

  transfer.register_kv_cache(caches, shape, torch::kBFloat16);

  ASSERT_EQ(transfer.main_layout_.layers.size(), 1U);
  ASSERT_EQ(transfer.main_layout_.layers[0].size(), 4U);
  for (size_t index = 0; index < transfer.main_layout_.layers[0].size();
       ++index) {
    EXPECT_EQ(transfer.main_layout_.layers[0][index].buf_id,
              static_cast<int64_t>(index));
  }

  uint64_t cluster_id = 0;
  std::string addr;
  transfer.get_cache_info(cluster_id, addr);
  ASSERT_FALSE(addr.empty());
  ASSERT_TRUE(transfer.link_cluster(
      /*cluster_id=*/0, addr, static_cast<uint16_t>(listen_port)));
  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.local_ids = {1};
  mapping.remote_ids = {0};
  ASSERT_TRUE(transfer.pull_kv_blocks(
      /*src_cluster_id=*/0, addr, {mapping}));
  device.synchronize_default_stream();

  EXPECT_TRUE(torch::equal(key_cache.index({1}), key_cache.index({0})));
  EXPECT_TRUE(torch::equal(value_cache.index({1}), value_cache.index({0})));
  EXPECT_TRUE(torch::equal(index_cache.index({1}), index_cache.index({0})));
  EXPECT_TRUE(torch::equal(index_scale->index({1}), index_scale->index({0})));

  EXPECT_TRUE(transfer.unlink_cluster(
      /*cluster_id=*/0,
      addr,
      static_cast<uint16_t>(listen_port),
      /*force_flag=*/true));
}
#endif

}  // namespace xllm
