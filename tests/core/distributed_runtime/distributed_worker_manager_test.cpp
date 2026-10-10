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

#include "core/distributed_runtime/distributed_worker_manager.h"

#include <folly/ExceptionWrapper.h>
#include <folly/futures/Future.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace xllm {
namespace {

class RecordingWorkerClient final : public WorkerClient {
 public:
  folly::SemiFuture<int64_t> get_active_activation_memory_async() override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++activation_memory_calls_;
    if (activation_memory_query_) {
      return activation_memory_query_();
    }
    return folly::makeSemiFuture(activation_memory_);
  }

  bool set_speculative_validate_time_predictor(
      const SpeculativeProfileRegistry::ValidateTimePredictor& predictor)
      override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++predictor_calls_;
    predictor_ = predictor;
    return predictor_result_;
  }

  void get_cache_info(uint64_t& cluster_id,
                      std::string& addr,
                      uint16_t& port) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++cache_info_calls_;
    cluster_id = cache_cluster_id_;
    addr = cache_addr_;
    port = cache_port_;
  }

  bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                    const std::vector<std::string>& addrs,
                    const std::vector<uint16_t>& ports) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++link_cluster_calls_;
    cluster_ids_ = cluster_ids;
    addrs_ = addrs;
    ports_ = ports;
    return link_cluster_result_;
  }

  bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                      const std::vector<std::string>& addrs,
                      const std::vector<uint16_t>& ports) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++unlink_cluster_calls_;
    cluster_ids_ = cluster_ids;
    addrs_ = addrs;
    ports_ = ports;
    return unlink_cluster_result_;
  }

  bool link_p2p(const std::string& remote_addr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++link_p2p_calls_;
    remote_addr_ = remote_addr;
    return link_p2p_result_;
  }

  bool unlink_p2p(const std::string& remote_addr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++unlink_p2p_calls_;
    remote_addr_ = remote_addr;
    return unlink_p2p_result_;
  }

  int32_t activation_memory_calls_ = 0;
  int64_t activation_memory_ = 0;
  std::function<folly::SemiFuture<int64_t>()> activation_memory_query_;
  int32_t predictor_calls_ = 0;
  SpeculativeProfileRegistry::ValidateTimePredictor predictor_;
  bool predictor_result_ = true;
  int32_t cache_info_calls_ = 0;
  uint64_t cache_cluster_id_ = 0;
  std::string cache_addr_;
  uint16_t cache_port_ = 0;
  bool link_cluster_result_ = true;
  bool unlink_cluster_result_ = true;
  bool link_p2p_result_ = true;
  bool unlink_p2p_result_ = true;
  int32_t link_cluster_calls_ = 0;
  int32_t unlink_cluster_calls_ = 0;
  int32_t link_p2p_calls_ = 0;
  int32_t unlink_p2p_calls_ = 0;
  std::vector<uint64_t> cluster_ids_;
  std::vector<std::string> addrs_;
  std::vector<uint16_t> ports_;
  std::string remote_addr_;

 private:
  std::mutex mutex_;
};

}  // namespace

class DistributedWorkerManagerTest : public ::testing::Test {
 protected:
  std::unique_ptr<DistributedWorkerManager> make_manager(
      std::vector<std::shared_ptr<RecordingWorkerClient>> clients) {
    std::vector<std::shared_ptr<WorkerClient>> worker_clients;
    worker_clients.reserve(clients.size());
    for (auto& client : clients) {
      worker_clients.emplace_back(std::move(client));
    }
    return std::unique_ptr<DistributedWorkerManager>(
        new DistributedWorkerManager(std::move(worker_clients)));
  }

  static bool has_link_threadpool(const DistributedWorkerManager& manager) {
    return manager.link_threadpool_ != nullptr;
  }

  static std::vector<uint64_t> cluster_ids() { return {11, 22}; }
  static std::vector<std::string> addrs() {
    return {"127.0.0.1:1001", "127.0.0.1:1002"};
  }
  static std::vector<uint16_t> ports() { return {1001, 1002}; }
};

TEST_F(DistributedWorkerManagerTest, AppendsCacheEndpointsInWorkerRankOrder) {
  auto first = std::make_shared<RecordingWorkerClient>();
  first->cache_cluster_id_ = 22;
  first->cache_addr_ = "worker-rank-0";
  first->cache_port_ = 1002;
  auto second = std::make_shared<RecordingWorkerClient>();
  second->cache_cluster_id_ = 11;
  second->cache_addr_ = "worker-rank-1";
  second->cache_port_ = 1001;
  auto manager = make_manager({first, second});
  std::vector<uint64_t> ids = {99};
  std::vector<std::string> addresses = {"existing"};
  std::vector<uint16_t> source_ports = {999};

  manager->get_cache_info(ids, addresses, source_ports);

  EXPECT_EQ(ids, (std::vector<uint64_t>{99, 22, 11}));
  EXPECT_EQ(
      addresses,
      (std::vector<std::string>{"existing", "worker-rank-0", "worker-rank-1"}));
  EXPECT_EQ(source_ports, (std::vector<uint16_t>{999, 1002, 1001}));
  EXPECT_EQ(first->cache_info_calls_, 1);
  EXPECT_EQ(second->cache_info_calls_, 1);
  EXPECT_FALSE(has_link_threadpool(*manager));

  manager->get_cache_info(ids, addresses, source_ports);

  EXPECT_EQ(ids, (std::vector<uint64_t>{99, 22, 11, 22, 11}));
  EXPECT_EQ(addresses,
            (std::vector<std::string>{"existing",
                                      "worker-rank-0",
                                      "worker-rank-1",
                                      "worker-rank-0",
                                      "worker-rank-1"}));
  EXPECT_EQ(source_ports, (std::vector<uint16_t>{999, 1002, 1001, 1002, 1001}));
}

TEST_F(DistributedWorkerManagerTest, PreservesCacheEndpointsWithoutWorkers) {
  auto manager = make_manager({});
  std::vector<uint64_t> ids = {99};
  std::vector<std::string> addresses = {"existing"};
  std::vector<uint16_t> source_ports = {999};

  manager->get_cache_info(ids, addresses, source_ports);

  EXPECT_EQ(ids, (std::vector<uint64_t>{99}));
  EXPECT_EQ(addresses, (std::vector<std::string>{"existing"}));
  EXPECT_EQ(source_ports, (std::vector<uint16_t>{999}));
}

TEST_F(DistributedWorkerManagerTest, BroadcastsPredictorToAllWorkers) {
  auto first = std::make_shared<RecordingWorkerClient>();
  auto second = std::make_shared<RecordingWorkerClient>();
  auto manager = make_manager({first, second});
  const SpeculativeProfileRegistry::ValidateTimePredictor predictor{
      .intercept_ms = 1.5, .query_token_ms = 2.5, .query_prefix_ms = 3.5};

  EXPECT_TRUE(manager->set_speculative_validate_time_predictor(predictor));

  for (const auto& client : {first, second}) {
    EXPECT_EQ(client->predictor_calls_, 1);
    EXPECT_EQ(client->predictor_.intercept_ms, predictor.intercept_ms);
    EXPECT_EQ(client->predictor_.query_token_ms, predictor.query_token_ms);
    EXPECT_EQ(client->predictor_.query_prefix_ms, predictor.query_prefix_ms);
  }
}

TEST_F(DistributedWorkerManagerTest, ReportsPredictorBroadcastFailures) {
  auto first = std::make_shared<RecordingWorkerClient>();
  auto second = std::make_shared<RecordingWorkerClient>();
  second->predictor_result_ = false;
  auto manager = make_manager({first, second});
  const SpeculativeProfileRegistry::ValidateTimePredictor predictor{
      .intercept_ms = 4.5, .query_token_ms = 5.5, .query_prefix_ms = 6.5};

  EXPECT_FALSE(manager->set_speculative_validate_time_predictor(predictor));
  EXPECT_EQ(first->predictor_calls_, 1);
  EXPECT_EQ(second->predictor_calls_, 1);
}

TEST_F(DistributedWorkerManagerTest, QueriesAllWorkersBeforeWaiting) {
  folly::Promise<int64_t> first_result;
  auto first = std::make_shared<RecordingWorkerClient>();
  first->activation_memory_query_ = [&first_result] {
    return first_result.getSemiFuture();
  };
  std::promise<void> second_dispatched;
  auto second = std::make_shared<RecordingWorkerClient>();
  second->activation_memory_query_ = [&second_dispatched] {
    second_dispatched.set_value();
    return folly::makeSemiFuture<int64_t>(2048);
  };
  auto manager = make_manager({first, second});
  auto dispatched = second_dispatched.get_future();

  auto query = std::async(std::launch::async, [&manager] {
    return manager->get_active_activation_memory();
  });
  const std::future_status dispatch_status =
      dispatched.wait_for(std::chrono::seconds(5));
  // Release the first worker even on timeout so the query cannot outlive the
  // test.
  first_result.setValue(4096);

  EXPECT_EQ(dispatch_status, std::future_status::ready);
  EXPECT_EQ(query.get(), (std::vector<int64_t>{4096, 2048}));
  EXPECT_EQ(first->activation_memory_calls_, 1);
  EXPECT_EQ(second->activation_memory_calls_, 1);
  EXPECT_FALSE(has_link_threadpool(*manager));
}

TEST_F(DistributedWorkerManagerTest, PropagatesActivationMemoryQueryFailure) {
  auto first = std::make_shared<RecordingWorkerClient>();
  first->activation_memory_query_ = [] {
    folly::Promise<int64_t> result;
    auto future = result.getSemiFuture();
    result.setException(folly::make_exception_wrapper<std::runtime_error>(
        "worker activation memory query failed"));
    return future;
  };
  auto second = std::make_shared<RecordingWorkerClient>();
  second->activation_memory_ = 2048;
  auto manager = make_manager({first, second});

  EXPECT_THROW(manager->get_active_activation_memory(), std::runtime_error);

  EXPECT_EQ(first->activation_memory_calls_, 1);
  EXPECT_EQ(second->activation_memory_calls_, 1);
}

TEST_F(DistributedWorkerManagerTest, ReturnsNoActivationMemoryWithoutWorkers) {
  auto manager = make_manager({});

  EXPECT_TRUE(manager->get_active_activation_memory().empty());
}

TEST_F(DistributedWorkerManagerTest, LinksAndUnlinksAllWorkers) {
  auto first = std::make_shared<RecordingWorkerClient>();
  auto second = std::make_shared<RecordingWorkerClient>();
  auto manager = make_manager({first, second});
  EXPECT_FALSE(has_link_threadpool(*manager));

  EXPECT_TRUE(manager->link_cluster(cluster_ids(),
                                    addrs(),
                                    ports(),
                                    /*src_dp_size=*/1));
  EXPECT_TRUE(has_link_threadpool(*manager));
  EXPECT_TRUE(manager->unlink_cluster(cluster_ids(),
                                      addrs(),
                                      ports(),
                                      /*src_dp_size=*/1));
  EXPECT_TRUE(manager->link_p2p({"p2p-0", "p2p-1"}));
  EXPECT_TRUE(manager->unlink_p2p({"p2p-0", "p2p-1"}));

  for (const auto& client : {first, second}) {
    EXPECT_EQ(client->link_cluster_calls_, 1);
    EXPECT_EQ(client->unlink_cluster_calls_, 1);
    EXPECT_EQ(client->cluster_ids_, cluster_ids());
    EXPECT_EQ(client->addrs_, addrs());
    EXPECT_EQ(client->ports_, ports());
    EXPECT_EQ(client->link_p2p_calls_, 1);
    EXPECT_EQ(client->unlink_p2p_calls_, 1);
  }
  EXPECT_EQ(first->remote_addr_, "p2p-0");
  EXPECT_EQ(second->remote_addr_, "p2p-1");
}

TEST_F(DistributedWorkerManagerTest, ReportsWorkerFailures) {
  auto first = std::make_shared<RecordingWorkerClient>();
  auto second = std::make_shared<RecordingWorkerClient>();
  second->link_cluster_result_ = false;
  second->unlink_cluster_result_ = false;
  second->link_p2p_result_ = false;
  second->unlink_p2p_result_ = false;
  auto manager = make_manager({first, second});

  EXPECT_FALSE(manager->link_cluster(cluster_ids(),
                                     addrs(),
                                     ports(),
                                     /*src_dp_size=*/1));
  EXPECT_FALSE(manager->unlink_cluster(cluster_ids(),
                                       addrs(),
                                       ports(),
                                       /*src_dp_size=*/1));
  EXPECT_FALSE(manager->link_p2p({"p2p-0", "p2p-1"}));
  EXPECT_FALSE(manager->unlink_p2p({"p2p-0", "p2p-1"}));
  EXPECT_EQ(first->link_cluster_calls_, 1);
  EXPECT_EQ(second->link_cluster_calls_, 1);
  EXPECT_EQ(first->link_p2p_calls_, 1);
  EXPECT_EQ(second->link_p2p_calls_, 1);
  EXPECT_EQ(first->unlink_cluster_calls_, 1);
  EXPECT_EQ(second->unlink_cluster_calls_, 1);
  EXPECT_EQ(first->unlink_p2p_calls_, 1);
  EXPECT_EQ(second->unlink_p2p_calls_, 1);
}

TEST_F(DistributedWorkerManagerTest, RejectsInvalidTopologyWithoutDispatch) {
  auto client = std::make_shared<RecordingWorkerClient>();
  auto manager = make_manager({client});
  EXPECT_FALSE(has_link_threadpool(*manager));

  const auto expect_rejected = [&manager](
                                   const std::vector<uint64_t>& ids,
                                   const std::vector<std::string>& addresses,
                                   const std::vector<uint16_t>& source_ports,
                                   int32_t dp_size,
                                   int32_t kv_split_size) {
    EXPECT_FALSE(manager->link_cluster(
        ids, addresses, source_ports, dp_size, kv_split_size));
    EXPECT_FALSE(manager->unlink_cluster(
        ids, addresses, source_ports, dp_size, kv_split_size));
  };
  expect_rejected({}, {}, {}, /*dp_size=*/1, /*kv_split_size=*/1);
  expect_rejected(cluster_ids(),
                  addrs(),
                  ports(),
                  /*dp_size=*/0,
                  /*kv_split_size=*/1);
  expect_rejected(cluster_ids(),
                  addrs(),
                  ports(),
                  /*dp_size=*/1,
                  /*kv_split_size=*/0);
  expect_rejected(cluster_ids(),
                  addrs(),
                  ports(),
                  /*dp_size=*/3,
                  /*kv_split_size=*/1);
  expect_rejected(cluster_ids(),
                  addrs(),
                  ports(),
                  /*dp_size=*/1,
                  /*kv_split_size=*/3);
  expect_rejected(cluster_ids(),
                  {"one"},
                  ports(),
                  /*dp_size=*/1,
                  /*kv_split_size=*/1);
  expect_rejected(cluster_ids(),
                  addrs(),
                  {1001},
                  /*dp_size=*/1,
                  /*kv_split_size=*/1);
  EXPECT_FALSE(manager->link_p2p({}));
  EXPECT_FALSE(manager->unlink_p2p({"one", "two"}));
  EXPECT_FALSE(has_link_threadpool(*manager));
  EXPECT_EQ(client->link_cluster_calls_, 0);
  EXPECT_EQ(client->unlink_cluster_calls_, 0);
  EXPECT_EQ(client->link_p2p_calls_, 0);
  EXPECT_EQ(client->unlink_p2p_calls_, 0);
}

TEST_F(DistributedWorkerManagerTest, RejectsOperationsWithoutWorkers) {
  auto manager = make_manager({});

  EXPECT_FALSE(manager->link_cluster(cluster_ids(),
                                     addrs(),
                                     ports(),
                                     /*src_dp_size=*/1));
  EXPECT_FALSE(manager->unlink_cluster(cluster_ids(),
                                       addrs(),
                                       ports(),
                                       /*src_dp_size=*/1));
  EXPECT_FALSE(manager->link_p2p({}));
  EXPECT_FALSE(manager->unlink_p2p({}));
}

}  // namespace xllm
