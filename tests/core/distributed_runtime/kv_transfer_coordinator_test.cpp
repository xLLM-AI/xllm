/* Copyright 2026 The xLLM Authors.

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

#include "core/distributed_runtime/kv_transfer_coordinator.h"

#include <folly/ExceptionWrapper.h>
#include <folly/futures/Future.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/distributed_runtime/distributed_worker_manager.h"

namespace xllm {
namespace {

class RecordingTransferWorkerClient final : public WorkerClient {
 public:
  bool pull_kv_blocks(uint64_t src_cluster_id,
                      const std::string& src_addr,
                      const std::vector<KVTransferMapping>& mappings) override {
    ++pull_calls_;
    src_cluster_id_ = src_cluster_id;
    src_addr_ = src_addr;
    mappings_ = mappings;
    return pull_result_;
  }

  folly::SemiFuture<uint32_t> transfer_kv_blocks(
      const std::vector<BlockTransferInfo>& block_transfer_info) override {
    ++transfer_calls_;
    transfer_info_ = block_transfer_info;
    if (transfer_query_) {
      return transfer_query_();
    }
    return folly::makeSemiFuture(transfer_result_);
  }

  folly::SemiFuture<uint32_t> transfer_kv_blocks(
      uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info) override {
    ++batch_transfer_calls_;
    batch_id_ = batch_id;
    transfer_info_ = block_transfer_info;
    return folly::makeSemiFuture(
        static_cast<uint32_t>(block_transfer_info.size()));
  }

  void prefetch_from_storage(
      const std::shared_ptr<const StoragePrefetchRequest>& request,
      std::shared_ptr<PrefetchResult> result,
      size_t worker_index) override {
    ++prefetch_calls_;
    prefetch_request_ = request;
    prefetch_result_ = std::move(result);
    prefetch_worker_index_ = worker_index;
  }

  int32_t pull_calls_ = 0;
  bool pull_result_ = true;
  uint64_t src_cluster_id_ = 0;
  std::string src_addr_;
  std::vector<KVTransferMapping> mappings_;
  int32_t transfer_calls_ = 0;
  uint32_t transfer_result_ = 0;
  std::function<folly::SemiFuture<uint32_t>()> transfer_query_;
  int32_t batch_transfer_calls_ = 0;
  uint64_t batch_id_ = 0;
  std::vector<BlockTransferInfo> transfer_info_;
  int32_t prefetch_calls_ = 0;
  std::shared_ptr<const StoragePrefetchRequest> prefetch_request_;
  std::shared_ptr<PrefetchResult> prefetch_result_;
  size_t prefetch_worker_index_ = 0;
};

std::vector<std::shared_ptr<RecordingTransferWorkerClient>> make_clients(
    size_t worker_count) {
  std::vector<std::shared_ptr<RecordingTransferWorkerClient>> clients;
  clients.reserve(worker_count);
  for (size_t rank = 0; rank < worker_count; ++rank) {
    auto client = std::make_shared<RecordingTransferWorkerClient>();
    client->transfer_result_ = static_cast<uint32_t>(rank + 1);
    clients.emplace_back(std::move(client));
  }
  return clients;
}

std::vector<BlockTransferInfo> make_transfer_info(TransferType type) {
  const uint8_t hash_key[XXH3_128BITS_HASH_VALUE_LEN] = {};
  return {BlockTransferInfo(/*src_id=*/11, /*dst_id=*/13, hash_key, type)};
}

std::shared_ptr<const StoragePrefetchRequest> make_prefetch_request() {
  auto request = std::make_shared<StoragePrefetchRequest>();
  const uint8_t hash_key[XXH3_128BITS_HASH_VALUE_LEN] = {};
  request->units.reserve(4);
  request->transfer_infos.reserve(4);
  for (int32_t offset = 0; offset < 4; ++offset) {
    request->transfer_infos.emplace_back(/*src_id=*/offset + 1,
                                         /*dst_id=*/offset + 7,
                                         hash_key,
                                         TransferType::G2H);
    PrefetchUnit unit;
    unit.gated_end = static_cast<uint32_t>(offset + 1);
    unit.end = unit.gated_end;
    request->units.emplace_back(unit);
  }
  return request;
}

size_t contiguous_gated_prefix(const PrefetchSummary& summary) {
  size_t prefix = 0;
  while (prefix < summary.gated_hits.size() && summary.gated_hits[prefix]) {
    ++prefix;
  }
  return prefix;
}

void expect_transfer_info(const std::vector<BlockTransferInfo>& actual,
                          const std::vector<BlockTransferInfo>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(actual[index].src_block_id, expected[index].src_block_id);
    EXPECT_EQ(actual[index].dst_block_id, expected[index].dst_block_id);
    EXPECT_EQ(actual[index].block_type, expected[index].block_type);
    EXPECT_EQ(actual[index].transfer_type, expected[index].transfer_type);
    EXPECT_TRUE(std::equal(std::begin(actual[index].hash_key),
                           std::end(actual[index].hash_key),
                           std::begin(expected[index].hash_key)));
  }
}

}  // namespace

class KVTransferCoordinatorTest : public ::testing::Test {
 protected:
  KVTransferCoordinator make_coordinator(
      const std::vector<std::shared_ptr<RecordingTransferWorkerClient>>&
          clients,
      int32_t dp_size = 1,
      uint32_t prefetch_timeout_ms = 0) {
    std::vector<std::shared_ptr<WorkerClient>> worker_clients;
    worker_clients.reserve(clients.size());
    for (const auto& client : clients) {
      worker_clients.emplace_back(client);
    }
    auto manager = std::shared_ptr<DistributedWorkerManager>(
        new DistributedWorkerManager(std::move(worker_clients)));
    return KVTransferCoordinator(
        KVTransferCoordinator::Options{
            .dp_size = dp_size,
            .prefetch_timeout_ms = prefetch_timeout_ms,
            .prefetch_batch_size = 2},
        std::move(manager));
  }
};

TEST_F(KVTransferCoordinatorTest, PullWaitsForAllWorkersAfterFailure) {
  auto clients = make_clients(/*worker_count=*/8);
  clients[4]->pull_result_ = false;
  auto coordinator = make_coordinator(clients, /*dp_size=*/2);
  const std::vector<uint64_t> ids = {101, 102, 103, 104, 105, 106, 107, 108};
  const std::vector<std::string> addrs = {"worker-0",
                                          "worker-1",
                                          "worker-2",
                                          "worker-3",
                                          "worker-4",
                                          "worker-5",
                                          "worker-6",
                                          "worker-7"};
  const std::vector<KVTransferMapping> mappings = {KVTransferMapping{
      .group_id = 7, .local_ids = {11, 13}, .remote_ids = {17, 19}}};

  EXPECT_FALSE(coordinator.pull_kv_blocks(/*src_dp_size=*/2,
                                          /*src_dp_rank=*/0,
                                          ids,
                                          addrs,
                                          /*dst_dp_rank=*/1,
                                          mappings));

  for (size_t rank = 0; rank < 4; ++rank) {
    EXPECT_EQ(clients[rank]->pull_calls_, 0);
    const auto& target = clients[rank + 4];
    EXPECT_EQ(target->pull_calls_, 1);
    EXPECT_EQ(target->src_cluster_id_, ids[rank]);
    EXPECT_EQ(target->src_addr_, addrs[rank]);
    ASSERT_EQ(target->mappings_.size(), 1u);
    EXPECT_EQ(target->mappings_[0].group_id, mappings[0].group_id);
    EXPECT_EQ(target->mappings_[0].local_ids, mappings[0].local_ids);
    EXPECT_EQ(target->mappings_[0].remote_ids, mappings[0].remote_ids);
  }

  clients[4]->pull_result_ = true;
  EXPECT_TRUE(coordinator.pull_kv_blocks(/*src_dp_size=*/2,
                                         /*src_dp_rank=*/0,
                                         ids,
                                         addrs,
                                         /*dst_dp_rank=*/1,
                                         mappings));
}

TEST_F(KVTransferCoordinatorTest, PullRejectsInvalidInputsBeforeDispatch) {
  auto clients = make_clients(/*worker_count=*/4);
  auto coordinator = make_coordinator(clients, /*dp_size=*/2);
  const std::vector<uint64_t> ids = {101, 102, 103, 104};
  const std::vector<std::string> addrs = {"one", "two", "three", "four"};
  const auto expect_rejected = [&coordinator](
                                   int32_t src_dp_size,
                                   int32_t src_dp_rank,
                                   const std::vector<uint64_t>& source_ids,
                                   const std::vector<std::string>& source_addrs,
                                   int32_t dst_dp_rank) {
    EXPECT_FALSE(coordinator.pull_kv_blocks(src_dp_size,
                                            src_dp_rank,
                                            source_ids,
                                            source_addrs,
                                            dst_dp_rank,
                                            /*mappings=*/{}));
  };
  expect_rejected(/*src_dp_size=*/2,
                  /*src_dp_rank=*/0,
                  ids,
                  {},
                  /*dst_dp_rank=*/0);
  expect_rejected(/*src_dp_size=*/0,
                  /*src_dp_rank=*/0,
                  ids,
                  addrs,
                  /*dst_dp_rank=*/0);
  expect_rejected(/*src_dp_size=*/3,
                  /*src_dp_rank=*/0,
                  ids,
                  addrs,
                  /*dst_dp_rank=*/0);
  expect_rejected(/*src_dp_size=*/2,
                  /*src_dp_rank=*/-1,
                  ids,
                  addrs,
                  /*dst_dp_rank=*/0);
  expect_rejected(/*src_dp_size=*/2,
                  /*src_dp_rank=*/2,
                  ids,
                  addrs,
                  /*dst_dp_rank=*/0);
  expect_rejected(/*src_dp_size=*/2,
                  /*src_dp_rank=*/0,
                  ids,
                  addrs,
                  /*dst_dp_rank=*/2);
  expect_rejected(/*src_dp_size=*/1,
                  /*src_dp_rank=*/0,
                  ids,
                  addrs,
                  /*dst_dp_rank=*/0);
  expect_rejected(/*src_dp_size=*/2,
                  /*src_dp_rank=*/0,
                  {},
                  {},
                  /*dst_dp_rank=*/0);
  for (const auto& client : clients) {
    EXPECT_EQ(client->pull_calls_, 0);
  }

  auto empty_coordinator = make_coordinator(/*clients=*/{});
  EXPECT_FALSE(empty_coordinator.pull_kv_blocks(/*src_dp_size=*/2,
                                                /*src_dp_rank=*/0,
                                                ids,
                                                addrs,
                                                /*dst_dp_rank=*/0,
                                                /*mappings=*/{}));
}

TEST_F(KVTransferCoordinatorTest, TransferReturnsFullDPGroupInOrder) {
  // DP=2, CP=2, attention TP=2: all four workers in DP 1 must participate.
  auto clients = make_clients(/*worker_count=*/8);
  folly::Promise<uint32_t> first_result;
  clients[4]->transfer_query_ = [&first_result] {
    return first_result.getSemiFuture();
  };
  auto coordinator = make_coordinator(clients, /*dp_size=*/2);
  const auto infos = make_transfer_info(TransferType::D2H2G);
  auto futures = coordinator.transfer_kv_blocks(/*dp_rank=*/1, infos);
  ASSERT_EQ(futures.size(), 4u);

  for (size_t rank = 0; rank < 4; ++rank) {
    EXPECT_EQ(clients[rank]->transfer_calls_, 0);
    EXPECT_EQ(clients[rank + 4]->transfer_calls_, 1);
    expect_transfer_info(clients[rank + 4]->transfer_info_, infos);
  }
  // A later rank can complete before rank 0, while the returned order stays
  // stable and the coordinator leaves waiting to its consumer.
  EXPECT_EQ(std::move(futures[3]).get(), 8u);
  first_result.setValue(41);
  EXPECT_EQ(std::move(futures[0]).get(), 41u);
  EXPECT_EQ(std::move(futures[1]).get(), 6u);
  EXPECT_EQ(std::move(futures[2]).get(), 7u);
}

TEST_F(KVTransferCoordinatorTest, TransferKeepsFailuresAndDispatchesAll) {
  auto clients = make_clients(/*worker_count=*/3);
  clients[1]->transfer_query_ = [] {
    folly::Promise<uint32_t> result;
    auto future = result.getSemiFuture();
    result.setException(folly::make_exception_wrapper<std::runtime_error>(
        "KV transfer failed"));
    return future;
  };
  auto coordinator = make_coordinator(clients);
  auto futures = coordinator.transfer_kv_blocks(
      /*dp_rank=*/0, make_transfer_info(TransferType::D2H2G));
  ASSERT_EQ(futures.size(), 3u);
  EXPECT_EQ(std::move(futures[0]).get(), 1u);
  EXPECT_THROW(std::move(futures[1]).get(), std::runtime_error);
  EXPECT_EQ(std::move(futures[2]).get(), 3u);
  for (const auto& client : clients) {
    EXPECT_EQ(client->transfer_calls_, 1);
  }
}

TEST_F(KVTransferCoordinatorTest, BatchTransferUsesFullDPGroup) {
  auto clients = make_clients(/*worker_count=*/8);
  auto coordinator = make_coordinator(clients, /*dp_size=*/2);
  const auto infos = make_transfer_info(TransferType::H2D);
  coordinator.transfer_kv_blocks(/*dp_rank=*/1, /*batch_id=*/73, infos);
  for (size_t rank = 0; rank < 4; ++rank) {
    EXPECT_EQ(clients[rank]->batch_transfer_calls_, 0);
    const auto& target = clients[rank + 4];
    EXPECT_EQ(target->batch_transfer_calls_, 1);
    EXPECT_EQ(target->batch_id_, 73u);
    expect_transfer_info(target->transfer_info_, infos);
  }
}

TEST_F(KVTransferCoordinatorTest, PrefetchCompletesAllCPWorkersOnce) {
  auto clients = make_clients(/*worker_count=*/8);
  auto coordinator = make_coordinator(clients, /*dp_size=*/2);
  const auto request = make_prefetch_request();
  int32_t done_calls = 0;
  size_t common_hit_units = 0;
  bool workers_quiescent = false;
  coordinator.prefetch_from_storage(
      /*dp_rank=*/1,
      request,
      [] { return false; },
      [&](PrefetchSummary summary, bool quiescent) {
        ++done_calls;
        common_hit_units = contiguous_gated_prefix(summary);
        workers_quiescent = quiescent;
      });
  const auto result = clients[4]->prefetch_result_;
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->worker_count(), 4u);
  EXPECT_EQ(result->stream_idle_timeout_ms(), -1);
  for (size_t rank = 0; rank < 4; ++rank) {
    EXPECT_EQ(clients[rank]->prefetch_calls_, 0);
    const auto& target = clients[rank + 4];
    EXPECT_EQ(target->prefetch_calls_, 1);
    EXPECT_EQ(target->prefetch_request_, request);
    EXPECT_EQ(target->prefetch_result_, result);
    EXPECT_EQ(target->prefetch_worker_index_, rank);
    EXPECT_EQ(result->record_batch_result(
                  rank, /*gated_hits=*/{1, 1}, /*non_gated_hits=*/{1, 1}),
              PrefetchControl::CONTINUE);
  }
  for (size_t rank = 0; rank < 3; ++rank) {
    EXPECT_EQ(result->record_batch_result(
                  rank, /*gated_hits=*/{1, 1}, /*non_gated_hits=*/{1, 1}),
              PrefetchControl::STOP);
    result->mark_worker_ended(rank,
                              /*worker_ok=*/true,
                              /*worker_quiescent=*/true);
    EXPECT_EQ(done_calls, 0);
  }
  EXPECT_EQ(result->record_batch_result(/*worker_index=*/3,
                                        /*gated_hits=*/{1, 0},
                                        /*non_gated_hits=*/{1, 0}),
            PrefetchControl::STOP);
  result->mark_worker_ended(/*worker_index=*/3,
                            /*worker_ok=*/true,
                            /*worker_quiescent=*/true);
  EXPECT_EQ(done_calls, 1);
  EXPECT_EQ(common_hit_units, 3u);
  EXPECT_TRUE(workers_quiescent);
  result->mark_worker_ended(/*worker_index=*/3,
                            /*worker_ok=*/true,
                            /*worker_quiescent=*/true);
  EXPECT_EQ(done_calls, 1);
}

TEST_F(KVTransferCoordinatorTest, PrefetchPreservesTimeoutAndStop) {
  auto clients = make_clients(/*worker_count=*/2);
  auto coordinator = make_coordinator(clients,
                                      /*dp_size=*/1,
                                      /*prefetch_timeout_ms=*/60000);
  bool stop_requested = false;
  int32_t done_calls = 0;
  coordinator.prefetch_from_storage(
      /*dp_rank=*/0,
      make_prefetch_request(),
      [&stop_requested] { return stop_requested; },
      [&done_calls](PrefetchSummary summary, bool workers_quiescent) {
        ++done_calls;
        EXPECT_EQ(contiguous_gated_prefix(summary), 2u);
        EXPECT_TRUE(workers_quiescent);
      });
  const auto result = clients[0]->prefetch_result_;
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->stream_idle_timeout_ms(), 60000);
  stop_requested = true;
  for (size_t rank = 0; rank < clients.size(); ++rank) {
    EXPECT_EQ(result->record_batch_result(
                  rank, /*gated_hits=*/{1, 1}, /*non_gated_hits=*/{1, 1}),
              PrefetchControl::STOP);
    result->mark_worker_ended(rank,
                              /*worker_ok=*/true,
                              /*worker_quiescent=*/true);
  }
  EXPECT_EQ(done_calls, 1);
}

TEST_F(KVTransferCoordinatorTest, PrefetchReportsUncertainCompletion) {
  auto clients = make_clients(/*worker_count=*/2);
  auto coordinator = make_coordinator(clients);
  int32_t done_calls = 0;
  coordinator.prefetch_from_storage(
      /*dp_rank=*/0,
      make_prefetch_request(),
      [] { return false; },
      [&done_calls](PrefetchSummary summary, bool workers_quiescent) {
        ++done_calls;
        EXPECT_EQ(contiguous_gated_prefix(summary), 0u);
        EXPECT_FALSE(workers_quiescent);
      });
  const auto result = clients[0]->prefetch_result_;
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->record_batch_result(/*worker_index=*/0,
                                        /*gated_hits=*/{1, 0},
                                        /*non_gated_hits=*/{1, 0}),
            PrefetchControl::STOP);
  result->mark_worker_ended(/*worker_index=*/0,
                            /*worker_ok=*/true,
                            /*worker_quiescent=*/true);
  EXPECT_EQ(done_calls, 0);
  result->mark_worker_ended(/*worker_index=*/1,
                            /*worker_ok=*/false,
                            /*worker_quiescent=*/false);
  EXPECT_EQ(done_calls, 1);
}

}  // namespace xllm
