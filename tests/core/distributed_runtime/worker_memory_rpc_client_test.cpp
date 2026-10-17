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

#include "core/distributed_runtime/worker_memory_rpc_client.h"

#include <brpc/closure_guard.h>
#include <brpc/server.h>
#include <bthread/bthread.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

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

class OffsetTestService final : public proto::ModelMemoryDist {
 public:
  explicit OffsetTestService(ResponseMode mode) : mode_(mode) {}

  void Hello(google::protobuf::RpcController* /*controller*/,
             const proto::Status* /*request*/,
             proto::Status* response,
             google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    std::lock_guard<std::mutex> lock(mutex_);
    ++hello_calls_;
    response->set_ok(true);
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

  int32_t hello_calls() {
    std::lock_guard<std::mutex> lock(mutex_);
    return hello_calls_;
  }

 private:
  const ResponseMode mode_;
  std::mutex mutex_;
  proto::GetVirtualKVCacheOffsetsRequest request_;
  int32_t hello_calls_ = 0;
};

class OffsetTestServer final {
 public:
  explicit OffsetTestServer(ResponseMode mode = ResponseMode::NORMAL)
      : service_(mode) {}

  ~OffsetTestServer() {
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

struct OffsetLookupState {
  WorkerMemoryRpcClient* client;
  const std::vector<int32_t>* block_ids;
  LayerOffsets offsets;
};

TEST(WorkerMemoryRpcClientTest, PreservesBlockOrderAndLayerOffsets) {
  OffsetTestServer server;
  ASSERT_EQ(server.start(), 0);
  WorkerMemoryRpcClient client(
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

TEST(WorkerMemoryRpcClientTest, TransportFailureReturnsEmptyOffsets) {
  OffsetTestServer server(ResponseMode::TRANSPORT_FAILURE);
  ASSERT_EQ(server.start(), 0);
  WorkerMemoryRpcClient client(
      /*global_rank=*/1, server.address(), torch::Device(torch::kCPU));

  EXPECT_TRUE(
      client.get_virtual_kv_cache_offsets("missing-model", {0}).empty());
}

TEST(WorkerMemoryRpcClientTest, RejectsPartialRoleOffsetResponses) {
  for (ResponseMode mode : {ResponseMode::INVALID_K_OFFSETS,
                            ResponseMode::INVALID_V_OFFSETS,
                            ResponseMode::INVALID_INDEX_OFFSETS}) {
    SCOPED_TRACE(static_cast<int32_t>(mode));
    OffsetTestServer server(mode);
    ASSERT_EQ(server.start(), 0);
    WorkerMemoryRpcClient client(
        /*global_rank=*/1, server.address(), torch::Device(torch::kCPU));
    EXPECT_TRUE(
        client.get_virtual_kv_cache_offsets("model", {3, 0, 7}).empty());
  }
}

TEST(WorkerMemoryRpcClientTest, EmptyWorkerResponseRemainsEmpty) {
  OffsetTestServer server(ResponseMode::EMPTY_RESPONSE);
  ASSERT_EQ(server.start(), 0);
  WorkerMemoryRpcClient client(
      /*global_rank=*/1, server.address(), torch::Device(torch::kCPU));

  EXPECT_TRUE(
      client.get_virtual_kv_cache_offsets("missing-model", {0}).empty());
}

}  // namespace
}  // namespace xllm
