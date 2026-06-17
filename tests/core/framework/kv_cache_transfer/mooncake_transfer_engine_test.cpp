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

#include "framework/kv_cache_transfer/mooncake_transfer_engine.h"

#include <brpc/controller.h>
#include <gtest/gtest.h>

#include <unordered_map>
#include <vector>

#include "framework/kv_cache_transfer/kv_cache_transfer.h"

#define private public
#define protected public
#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"
#undef private
#undef protected

namespace xllm {

namespace {

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
#endif

}  // namespace xllm
