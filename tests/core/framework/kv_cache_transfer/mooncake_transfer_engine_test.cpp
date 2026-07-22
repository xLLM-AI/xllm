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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache_transfer/kv_cache_transfer.h"
#include "platform/device.h"
#if defined(USE_MLU)
#include "platform/mlu/mlu_tensor_alloc.h"
#endif
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
constexpr size_t kAlignedRegisterBytes = 2097408;

TransferKVInfo make_info(int32_t dst_dp_size,
                         int32_t dst_tp_size,
                         int32_t dst_dp_rank) {
  TransferKVInfo info;
  info.request_id = "req";
  info.local_blocks_ids = {11, 12};
  info.remote_blocks_ids = {21, 22};
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
    EXPECT_EQ(lhs_info.src_blocks, rhs_info.src_blocks);
    EXPECT_EQ(lhs_info.dst_blocks, rhs_info.dst_blocks);
  }
}

#if defined(USE_MLU)
class RecordingMooncakeTransferEngine final : public MooncakeTransferEngine {
 public:
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

  bool pull_memory_blocks(const std::string& /*remote_addr*/,
                          const std::vector<uint64_t>& /*src_blocks*/,
                          const std::vector<uint64_t>& /*dst_blocks*/,
                          const std::vector<int64_t>& buf_ids) override {
    pulled_buf_ids = buf_ids;
    return true;
  }

  std::vector<std::vector<void*>> registered_addrs;
  std::vector<std::vector<size_t>> registered_lens;
  std::vector<std::vector<uint64_t>> registered_block_bytes;
  std::vector<int64_t> pulled_buf_ids;
};

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
      mlu_mooncake_tensor_allocator();
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
  EXPECT_EQ(kv_info.src_blocks, info.local_blocks_ids);
  EXPECT_EQ(kv_info.dst_blocks, info.remote_blocks_ids);
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
  EXPECT_EQ(kv_info.src_blocks, info.local_blocks_ids);
  EXPECT_EQ(kv_info.dst_blocks, info.remote_blocks_ids);
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
  EXPECT_EQ(kv_info.src_blocks, info.local_blocks_ids);
  EXPECT_EQ(kv_info.dst_blocks, info.remote_blocks_ids);
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

TEST(MooncakeKVCacheTransferDefaultTest, SpecDraftBufIdsUseSpecOffset) {
  MooncakeKVCacheTransferDefault transfer(
      0, 0, torch::Device(torch::kCPU), "test");
  transfer.has_v_cache_ = true;
  transfer.main_layout_.num_layers = 40;
  transfer.main_layout_.buf_cnt = 2;
  transfer.main_layout_.offset = 0;
  transfer.main_layout_.registered = true;
  transfer.spec_layout_.num_layers = 1;
  transfer.spec_layout_.buf_cnt = 2;
  transfer.spec_layout_.offset = 80;
  transfer.spec_layout_.registered = true;

  EXPECT_EQ(transfer.get_buf_ids({0}, false), (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(transfer.get_buf_ids({0}, true), (std::vector<int64_t>{80, 81}));
}

TEST(MooncakeKVCacheTransferDefaultTest,
     VariableLayerBufIdsUseRegistrationOffsets) {
  MooncakeKVCacheTransferDefault transfer(
      0, 0, torch::Device(torch::kCPU), "test");
  transfer.main_layout_.num_layers = 4;
  transfer.main_layout_.offset = 10;
  transfer.main_layout_.layer_offsets = {0, 4, 6, 8, 12};
  transfer.main_layout_.total_buf_cnt = 12;
  transfer.main_layout_.registered = true;

  EXPECT_EQ(transfer.get_buf_ids({1, 3}, false),
            (std::vector<int64_t>{14, 15, 18, 19, 20, 21}));
}

TEST(MooncakeKVCacheTransferDefaultTest,
     AddBufUsesRdmaRegisterableLengthWithoutChangingBlockBytes) {
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
  const torch::Tensor tensor = mlu::alloc_rdma_registerable_zero_tensor(
      {2, 96, 1}, torch::kFloat32, torch_device);
  std::vector<void*> addrs;
  std::vector<size_t> lens;
  std::vector<uint64_t> block_bytes;

  transfer.add_buf(tensor,
                   addrs,
                   lens,
                   block_bytes,
                   MooncakeKVCacheTransferDefault::RegisterLengthPolicy::
                       RDMA_REGISTERABLE_BYTES);

  ASSERT_EQ(addrs.size(), 1U);
  EXPECT_EQ(addrs[0], tensor.data_ptr());
  EXPECT_EQ(lens[0], kAlignedRegisterBytes);
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
            caches[0].get_indexer_cache_scale()->storage().nbytes());
  EXPECT_EQ(engine_observer->registered_block_bytes[0][2],
            caches[0].get_indexer_cache_scale()->nbytes() / 2);

  EXPECT_TRUE(transfer.pull_kv_blocks(/*src_cluster_id=*/1,
                                      /*src_addr=*/"remote",
                                      /*src_blocks=*/{0},
                                      /*dst_blocks=*/{1},
                                      /*src_linear_state_ids=*/{},
                                      /*dst_linear_state_ids=*/{}));
  EXPECT_EQ(engine_observer->pulled_buf_ids,
            (std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}));
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
  EXPECT_EQ(transfer.get_buf_ids({0, 1}, /*is_spec_draft=*/true),
            (std::vector<int64_t>{8, 9, 10, 11}));
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

TEST(MooncakeKVCacheTransferDefaultTest, AddBufRejectsRdmaLengthBeyondStorage) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for Mooncake registration tests.";
  }
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  Device device(/*device_id=*/0);
  device.set_device();
  const torch::Device torch_device = device.unwrap();
  MooncakeKVCacheTransferDefault transfer(
      /*device_id=*/0,
      /*listen_port=*/0,
      torch_device,
      /*model_type=*/"test");
  const torch::Tensor tensor =
      mlu::alloc_zero_tensor({2, 96, 1}, torch::kFloat32, torch_device);
  std::vector<void*> addrs;
  std::vector<size_t> lens;
  std::vector<uint64_t> block_bytes;

  EXPECT_DEATH(
      transfer.add_buf(tensor,
                       addrs,
                       lens,
                       block_bytes,
                       MooncakeKVCacheTransferDefault::RegisterLengthPolicy::
                           RDMA_REGISTERABLE_BYTES),
      "exceeds tensor storage capacity");
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
      .enable_indexer_cache_quant(true)
      .tensor_allocator(mlu_mooncake_tensor_allocator());
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
  EXPECT_EQ(index_scale->storage().nbytes(), kAlignedRegisterBytes);

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

  EXPECT_EQ(transfer.main_layout_.buf_cnt, 4);
  EXPECT_EQ(transfer.get_buf_ids({0}, /*is_spec_draft=*/false),
            (std::vector<int64_t>{0, 1, 2, 3}));

  uint64_t cluster_id = 0;
  std::string addr;
  transfer.get_cache_info(cluster_id, addr);
  ASSERT_FALSE(addr.empty());
  ASSERT_TRUE(transfer.link_cluster(
      /*cluster_id=*/0, addr, static_cast<uint16_t>(listen_port)));
  ASSERT_TRUE(transfer.pull_kv_blocks(/*src_cluster_id=*/0,
                                      addr,
                                      /*src_blocks=*/{0},
                                      /*dst_blocks=*/{1},
                                      /*src_linear_state_ids=*/{},
                                      /*dst_linear_state_ids=*/{}));
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
