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

#include "core/distributed_runtime/worker_memory_client.h"

#include <brpc/closure_guard.h>
#include <brpc/server.h>
#include <bthread/bthread.h>
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/framework/config/service_config.h"

namespace xllm {
namespace {

using LayerOffsets = std::vector<VirtualKVCacheLayerGlobalOffsets>;

enum class ResponseMode {
  NORMAL,
  TRANSPORT_FAILURE,
  EMPTY_RESPONSE,
  INVALID_K_OFFSETS,
  INVALID_V_OFFSETS,
  INVALID_INDEX_OFFSETS,
};

enum class MemoryOperation {
  INIT_POOL,
  PREPARE_KV,
  COMMIT_KV,
  ABORT_KV,
  PREPARE_WEIGHT,
  COMMIT_WEIGHT,
  ABORT_WEIGHT,
};

struct MemoryInvocation {
  MemoryOperation operation;
  uint64_t transaction_id;
};

class OffsetTestService final : public proto::ModelMemoryDist {
 public:
  explicit OffsetTestService(ResponseMode mode, bool withhold_mutations = false)
      : mode_(mode), withhold_mutations_(withhold_mutations) {
    invocations_.reserve(6);
    pending_replies_.reserve(6);
  }

  void Hello(google::protobuf::RpcController* /*controller*/,
             const proto::Status* /*request*/,
             proto::Status* response,
             google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    std::lock_guard<std::mutex> lock(mutex_);
    ++hello_calls_;
    response->set_ok(true);
  }

  void GetMemoryInfo(google::protobuf::RpcController* controller,
                     const proto::Status* /*request*/,
                     proto::MemoryInfoResponse* response,
                     google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    std::lock_guard<std::mutex> lock(mutex_);
    ++memory_info_calls_;
    if (mode_ == ResponseMode::TRANSPORT_FAILURE) {
      controller->SetFailed("injected memory info transport failure");
      return;
    }
    response->set_available_memory(4096);
    response->set_total_memory(8192);
    response->set_pool_initialized(true);
    response->set_pool_allocatable_pages(6);
    response->set_pool_capacity_pages(32);
    response->set_pool_active_pages(4);
    response->set_growth_memory_reserve_bytes(1024);
  }

  void InitPhysicalPagePool(google::protobuf::RpcController* /*controller*/,
                            const proto::InitPhysicalPagePoolRequest* request,
                            proto::Status* response,
                            google::protobuf::Closure* done) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      initialization_request_ = *request;
    }
    complete_mutation(
        MemoryOperation::INIT_POOL, /*transaction_id=*/0, response, done);
  }

  void PrepareKvMapping(google::protobuf::RpcController* /*controller*/,
                        const proto::PrepareKvMappingRequest* request,
                        proto::Status* response,
                        google::protobuf::Closure* done) override {
    complete_mutation(
        MemoryOperation::PREPARE_KV, request->transaction_id(), response, done);
  }

  void CommitKvMapping(google::protobuf::RpcController* /*controller*/,
                       const proto::KvMappingTransactionRequest* request,
                       proto::Status* response,
                       google::protobuf::Closure* done) override {
    complete_mutation(
        MemoryOperation::COMMIT_KV, request->transaction_id(), response, done);
  }

  void AbortKvMapping(google::protobuf::RpcController* /*controller*/,
                      const proto::KvMappingTransactionRequest* request,
                      proto::Status* response,
                      google::protobuf::Closure* done) override {
    complete_mutation(
        MemoryOperation::ABORT_KV, request->transaction_id(), response, done);
  }

  void PrepareWeightPages(google::protobuf::RpcController* /*controller*/,
                          const proto::PrepareWeightPagesRequest* request,
                          proto::Status* response,
                          google::protobuf::Closure* done) override {
    complete_mutation(MemoryOperation::PREPARE_WEIGHT,
                      request->transaction_id(),
                      response,
                      done);
  }

  void CommitWeightPages(google::protobuf::RpcController* /*controller*/,
                         const proto::KvMappingTransactionRequest* request,
                         proto::Status* response,
                         google::protobuf::Closure* done) override {
    complete_mutation(MemoryOperation::COMMIT_WEIGHT,
                      request->transaction_id(),
                      response,
                      done);
  }

  void AbortWeightPages(google::protobuf::RpcController* /*controller*/,
                        const proto::KvMappingTransactionRequest* request,
                        proto::Status* response,
                        google::protobuf::Closure* done) override {
    complete_mutation(MemoryOperation::ABORT_WEIGHT,
                      request->transaction_id(),
                      response,
                      done);
  }

  void GetVirtualKVCacheOffsets(
      google::protobuf::RpcController* controller,
      const proto::GetVirtualKVCacheOffsetsRequest* request,
      proto::GetVirtualKVCacheOffsetsResponse* response,
      google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      request_ = *request;
    }
    if (mode_ == ResponseMode::TRANSPORT_FAILURE) {
      controller->SetFailed("injected virtual-memory offset lookup failure");
      return;
    }
    if (mode_ == ResponseMode::EMPTY_RESPONSE) {
      return;
    }

    const LayerOffsets offsets = {{{300, 0, 700}, {301, 1, 701}, {302, 2, 702}},
                                  {{1300, 1000, 1700}, {1301, 1001, 1701}, {}}};
    for (const auto& offsets_for_layer : offsets) {
      auto* layer = response->add_layer_offsets();
      for (uint64_t offset : offsets_for_layer.k_offsets) {
        layer->add_k_offsets(offset);
      }
      for (uint64_t offset : offsets_for_layer.v_offsets) {
        layer->add_v_offsets(offset);
      }
      for (uint64_t offset : offsets_for_layer.index_offsets) {
        layer->add_index_offsets(offset);
      }
    }
    switch (mode_) {
      case ResponseMode::INVALID_K_OFFSETS:
        response->mutable_layer_offsets(0)->mutable_k_offsets()->RemoveLast();
        break;
      case ResponseMode::INVALID_V_OFFSETS:
        response->mutable_layer_offsets(0)->mutable_v_offsets()->RemoveLast();
        break;
      case ResponseMode::INVALID_INDEX_OFFSETS:
        response->mutable_layer_offsets(0)
            ->mutable_index_offsets()
            ->RemoveLast();
        break;
      default:
        break;
    }
  }

  proto::GetVirtualKVCacheOffsetsRequest request() {
    std::lock_guard<std::mutex> lock(mutex_);
    return request_;
  }

  proto::InitPhysicalPagePoolRequest initialization_request() {
    std::lock_guard<std::mutex> lock(mutex_);
    return initialization_request_;
  }

  int32_t hello_calls() {
    std::lock_guard<std::mutex> lock(mutex_);
    return hello_calls_;
  }

  int32_t memory_info_calls() {
    std::lock_guard<std::mutex> lock(mutex_);
    return memory_info_calls_;
  }

  std::vector<MemoryInvocation> invocations() {
    std::lock_guard<std::mutex> lock(mutex_);
    return invocations_;
  }

  size_t pending_replies() {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_replies_.size();
  }

  void release_replies() {
    std::vector<google::protobuf::Closure*> pending;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Also release calls that reach the server after test cleanup begins.
      withhold_mutations_ = false;
      pending.swap(pending_replies_);
    }
    for (google::protobuf::Closure* done : pending) {
      done->Run();
    }
  }

 private:
  void complete_mutation(MemoryOperation operation,
                         uint64_t transaction_id,
                         proto::Status* response,
                         google::protobuf::Closure* done) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      invocations_.emplace_back(MemoryInvocation{operation, transaction_id});
      response->set_ok(true);
      if (withhold_mutations_) {
        pending_replies_.emplace_back(done);
        return;
      }
    }
    done->Run();
  }

  const ResponseMode mode_;
  std::mutex mutex_;
  proto::GetVirtualKVCacheOffsetsRequest request_;
  proto::InitPhysicalPagePoolRequest initialization_request_;
  int32_t hello_calls_ = 0;
  int32_t memory_info_calls_ = 0;
  bool withhold_mutations_;
  std::vector<MemoryInvocation> invocations_;
  std::vector<google::protobuf::Closure*> pending_replies_;
};

class OffsetTestServer final {
 public:
  explicit OffsetTestServer(ResponseMode mode = ResponseMode::NORMAL,
                            bool withhold_mutations = false)
      : service_(mode, withhold_mutations) {}

  ~OffsetTestServer() {
    service_.release_replies();
    server_.Stop(/*timeout_ms=*/0);
    server_.Join();
  }

  int32_t start() {
    const int32_t result =
        server_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE);
    if (result != 0) {
      return result;
    }
    return server_.Start("127.0.0.1:0", nullptr);
  }

  std::string address() const {
    return "127.0.0.1:" + std::to_string(server_.listen_address().port);
  }

  OffsetTestService& service() { return service_; }

 private:
  OffsetTestService service_;
  brpc::Server server_;
};

// Declare after the client so withheld server replies are released before the
// client's thread pools join, including assertion failures against older code.
class PendingReplyGuard final {
 public:
  explicit PendingReplyGuard(OffsetTestService& service) : service_(service) {}
  ~PendingReplyGuard() { service_.release_replies(); }

 private:
  OffsetTestService& service_;
};

template <typename Predicate>
bool wait_until(Predicate ready) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!ready() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return ready();
}

struct TransactionCase {
  MemoryOperation operation;
  folly::SemiFuture<bool> (*invoke)(WorkerMemoryClient&, uint64_t);
};

const std::array<TransactionCase, 6> kTransactionCases = {{
    {MemoryOperation::PREPARE_KV,
     [](WorkerMemoryClient& client, uint64_t id) {
       return client.prepare_kv_mapping_async(id, "model", {0, 4096});
     }},
    {MemoryOperation::COMMIT_KV,
     [](WorkerMemoryClient& client, uint64_t id) {
       return client.commit_kv_mapping_async(id);
     }},
    {MemoryOperation::ABORT_KV,
     [](WorkerMemoryClient& client, uint64_t id) {
       return client.abort_kv_mapping_async(id);
     }},
    {MemoryOperation::PREPARE_WEIGHT,
     [](WorkerMemoryClient& client, uint64_t id) {
       return client.prepare_weight_pages_async(id, "model", /*num_pages=*/2);
     }},
    {MemoryOperation::COMMIT_WEIGHT,
     [](WorkerMemoryClient& client, uint64_t id) {
       return client.commit_weight_pages_async(id);
     }},
    {MemoryOperation::ABORT_WEIGHT,
     [](WorkerMemoryClient& client, uint64_t id) {
       return client.abort_weight_pages_async(id);
     }},
}};

class WorkerMemoryClientTimeoutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto& config = ServiceConfig::get_instance();
    previous_timeout_ms_ = config.rpc_channel_timeout_ms();
    config.rpc_channel_timeout_ms(200);
  }

  void TearDown() override {
    ServiceConfig::get_instance().rpc_channel_timeout_ms(previous_timeout_ms_);
  }

 private:
  int32_t previous_timeout_ms_ = -1;
};

struct OffsetLookupState {
  WorkerMemoryClient* client;
  const std::vector<int32_t>* block_ids;
  LayerOffsets offsets;
};

TEST(WorkerMemoryClientTest, PreservesBlockOrderAndLayerOffsets) {
  OffsetTestServer server;
  ASSERT_EQ(server.start(), 0);
  WorkerMemoryClient client(
      /*global_rank=*/1, server.address(), torch::Device(torch::kCPU));
  ASSERT_EQ(server.service().hello_calls(), 1);
  const std::vector<int32_t> block_ids = {3, 0, 7};

  OffsetLookupState lookup{&client, &block_ids, {}};
  bthread_t lookup_thread;
  ASSERT_EQ(bthread_start_background(
                &lookup_thread,
                nullptr,
                [](void* arg) -> void* {
                  auto* state = static_cast<OffsetLookupState*>(arg);
                  state->offsets = state->client->get_virtual_kv_cache_offsets(
                      "forked-model", *state->block_ids);
                  return nullptr;
                },
                &lookup),
            0);
  ASSERT_EQ(bthread_join(lookup_thread, nullptr), 0);

  const LayerOffsets expected = {{{300, 0, 700}, {301, 1, 701}, {302, 2, 702}},
                                 {{1300, 1000, 1700}, {1301, 1001, 1701}, {}}};
  ASSERT_EQ(lookup.offsets.size(), expected.size());
  for (size_t layer_id = 0; layer_id < expected.size(); ++layer_id) {
    EXPECT_EQ(lookup.offsets[layer_id].k_offsets, expected[layer_id].k_offsets);
    EXPECT_EQ(lookup.offsets[layer_id].v_offsets, expected[layer_id].v_offsets);
    EXPECT_EQ(lookup.offsets[layer_id].index_offsets,
              expected[layer_id].index_offsets);
  }
  const auto request = server.service().request();
  EXPECT_EQ(request.model_id(), "forked-model");
  EXPECT_EQ(std::vector<int32_t>(request.block_ids().begin(),
                                 request.block_ids().end()),
            block_ids);
  EXPECT_EQ(block_ids, (std::vector<int32_t>{3, 0, 7}));
}

TEST(WorkerMemoryClientTest, TransportFailureReturnsEmptyOffsets) {
  OffsetTestServer server(ResponseMode::TRANSPORT_FAILURE);
  ASSERT_EQ(server.start(), 0);
  WorkerMemoryClient client(
      /*global_rank=*/1, server.address(), torch::Device(torch::kCPU));

  EXPECT_TRUE(
      client.get_virtual_kv_cache_offsets("missing-model", {0}).empty());
}

TEST(WorkerMemoryClientTest, RejectsPartialRoleOffsetResponses) {
  for (ResponseMode mode : {ResponseMode::INVALID_K_OFFSETS,
                            ResponseMode::INVALID_V_OFFSETS,
                            ResponseMode::INVALID_INDEX_OFFSETS}) {
    SCOPED_TRACE(static_cast<int32_t>(mode));
    OffsetTestServer server(mode);
    ASSERT_EQ(server.start(), 0);
    WorkerMemoryClient client(
        /*global_rank=*/1, server.address(), torch::Device(torch::kCPU));
    EXPECT_TRUE(
        client.get_virtual_kv_cache_offsets("model", {3, 0, 7}).empty());
  }
}

TEST(WorkerMemoryClientTest, EmptyWorkerResponseRemainsEmpty) {
  OffsetTestServer server(ResponseMode::EMPTY_RESPONSE);
  ASSERT_EQ(server.start(), 0);
  WorkerMemoryClient client(
      /*global_rank=*/1, server.address(), torch::Device(torch::kCPU));

  EXPECT_TRUE(
      client.get_virtual_kv_cache_offsets("missing-model", {0}).empty());
}

TEST(WorkerMemoryClientTest, CapacityQueryReportsTransportFailure) {
  OffsetTestServer server(ResponseMode::TRANSPORT_FAILURE);
  ASSERT_EQ(server.start(), 0);
  WorkerMemoryClient client(
      /*global_rank=*/1, server.address(), torch::Device(torch::kCPU));

  auto capacity = client.get_memory_info_async(/*timeout_ms=*/1000);
  EXPECT_THROW(std::move(capacity).get(), std::runtime_error);
}

TEST_F(WorkerMemoryClientTimeoutTest,
       CapacityProbeBypassesBlockedMutationQueue) {
  OffsetTestServer server(ResponseMode::NORMAL, /*withhold_mutations=*/true);
  ASSERT_EQ(server.start(), 0);
  WorkerMemoryClient client(
      /*global_rank=*/1, server.address(), torch::Device(torch::kCPU));
  PendingReplyGuard release_guard(server.service());
  auto initialization = client.init_physical_page_pool_async(/*num_pages=*/8);
  ASSERT_TRUE(
      wait_until([&] { return server.service().pending_replies() > 0; }));

  // A held mutation owns the legacy RPC queue. The probe must finish through
  // its own queue while initialization remains blocked at the server.
  auto capacity = client.get_memory_info_async(/*timeout_ms=*/1000);
  EXPECT_TRUE(wait_until([&] { return capacity.isReady(); }));
  EXPECT_FALSE(initialization.isReady());
  EXPECT_EQ(server.service().memory_info_calls(), 1);
  const auto invocations = server.service().invocations();
  ASSERT_EQ(invocations.size(), 1u);
  EXPECT_EQ(invocations.front().operation, MemoryOperation::INIT_POOL);

  // Release before any blocking get, so the old shared-queue implementation
  // fails the readiness check without hanging this test or client destruction.
  server.service().release_replies();
  ASSERT_TRUE(wait_until([&] { return capacity.isReady(); }));
  const MemoryInfo info = std::move(capacity).get();
  EXPECT_EQ(info.available_memory, 4096);
  EXPECT_EQ(info.total_memory, 8192);
  EXPECT_TRUE(info.pool_initialized);
  EXPECT_EQ(info.pool_allocatable_pages, 6u);
  EXPECT_EQ(info.pool_capacity_pages, 32u);
  EXPECT_EQ(info.pool_active_pages, 4u);
  EXPECT_EQ(info.growth_memory_reserve_bytes, 1024u);
  ASSERT_TRUE(wait_until([&] { return initialization.isReady(); }));
  EXPECT_TRUE(std::move(initialization).get());
}

TEST(WorkerMemoryClientTest,
     PreservesArenaAndGrowthLimitsDuringInitialization) {
  OffsetTestServer server;
  ASSERT_EQ(server.start(), 0);
  WorkerMemoryClient client(
      /*global_rank=*/1, server.address(), torch::Device(torch::kCPU));

  EXPECT_TRUE(client
                  .init_physical_page_pool_async(
                      /*max_pages=*/32,
                      /*initial_pages=*/4,
                      /*lazy=*/true,
                      /*growth_memory_reserve_bytes=*/4096)
                  .get());
  const auto request = server.service().initialization_request();
  EXPECT_EQ(request.num_pages(), 32);
  ASSERT_TRUE(request.has_initial_pages());
  EXPECT_EQ(request.initial_pages(), 4);
  EXPECT_TRUE(request.lazy());
  EXPECT_EQ(request.growth_memory_reserve_bytes(), 4096u);
}

TEST(WorkerMemoryClientTest, PreservesEmptyInitialArenaDuringInitialization) {
  OffsetTestServer server;
  ASSERT_EQ(server.start(), 0);
  WorkerMemoryClient client(
      /*global_rank=*/1, server.address(), torch::Device(torch::kCPU));

  EXPECT_TRUE(client
                  .init_physical_page_pool_async(
                      /*max_pages=*/32,
                      /*initial_pages=*/0,
                      /*lazy=*/true,
                      /*growth_memory_reserve_bytes=*/0)
                  .get());
  const auto request = server.service().initialization_request();
  ASSERT_TRUE(request.has_initial_pages());
  EXPECT_EQ(request.initial_pages(), 0);
  EXPECT_EQ(request.growth_memory_reserve_bytes(), 0u);
}

TEST_F(WorkerMemoryClientTimeoutTest,
       MappingTransactionsPreserveSuccessfulReplies) {
  ServiceConfig::get_instance().rpc_channel_timeout_ms(1000);
  OffsetTestServer server;
  ASSERT_EQ(server.start(), 0);
  WorkerMemoryClient client(
      /*global_rank=*/1, server.address(), torch::Device(torch::kCPU));
  PendingReplyGuard release_guard(server.service());
  uint64_t transaction_id = 100;
  for (const TransactionCase& test_case : kTransactionCases) {
    SCOPED_TRACE(static_cast<int32_t>(test_case.operation));
    auto result = test_case.invoke(client, transaction_id);
    ASSERT_TRUE(wait_until([&] { return result.isReady(); }));
    EXPECT_TRUE(std::move(result).get());
    const auto invocations = server.service().invocations();
    ASSERT_FALSE(invocations.empty());
    EXPECT_EQ(invocations.back().operation, test_case.operation);
    EXPECT_EQ(invocations.back().transaction_id, transaction_id);
    ++transaction_id;
  }
  EXPECT_EQ(server.service().invocations().size(), kTransactionCases.size());
}

TEST_F(WorkerMemoryClientTimeoutTest,
       MappingTransactionsFailWhenRepliesAreWithheld) {
  for (const TransactionCase& test_case : kTransactionCases) {
    SCOPED_TRACE(static_cast<int32_t>(test_case.operation));
    OffsetTestServer server(ResponseMode::NORMAL, /*withhold_mutations=*/true);
    ASSERT_EQ(server.start(), 0);
    WorkerMemoryClient client(
        /*global_rank=*/1, server.address(), torch::Device(torch::kCPU));
    PendingReplyGuard release_guard(server.service());
    auto result = test_case.invoke(client, /*transaction_id=*/42);

    // The server accepts the mutation but never acknowledges it. The client
    // must return false using its configured 200 ms transaction deadline.
    const bool completed = wait_until([&] { return result.isReady(); });
    EXPECT_TRUE(completed);
    const auto invocations = server.service().invocations();
    ASSERT_FALSE(invocations.empty());
    EXPECT_EQ(invocations.front().operation, test_case.operation);
    EXPECT_EQ(invocations.front().transaction_id, 42u);
    EXPECT_GT(server.service().pending_replies(), 0u);

    // Drain held closures even on pre-fix code, where no RPC deadline exists.
    server.service().release_replies();
    ASSERT_TRUE(wait_until([&] { return result.isReady(); }));
    EXPECT_FALSE(std::move(result).get());
  }
}

}  // namespace
}  // namespace xllm
