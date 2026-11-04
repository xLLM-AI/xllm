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

#include "core/distributed_runtime/comm_channel.h"

#include <brpc/closure_guard.h>
#include <brpc/server.h>
#include <butil/endpoint.h>
#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <mutex>

#include "core/distributed_runtime/remote_worker.h"
#include "runtime/rec_forward_params.h"
#include "runtime/vlm_forward_params.h"

namespace xllm {
namespace {

class ModelRoutingService final : public proto::DistributeWorker {
 public:
  void Hello(google::protobuf::RpcController* /*controller*/,
             const proto::Status* /*request*/,
             proto::Status* response,
             google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    response->set_ok(true);
  }

  void TransferBlocks(google::protobuf::RpcController* /*controller*/,
                      const proto::BlockTransferInfos* request,
                      proto::TransferStatus* response,
                      google::protobuf::Closure* done) override {
    std::lock_guard<std::mutex> lock(restore_mutex_);
    restore_batch_id_ = request->batch_id();
    registration_response_ = response;
    registration_done_ = done;
    registration_count_ = static_cast<uint32_t>(request->transfer_infos_size());
    registration_requested_.set_value();
  }

  void WaitForKVBlocksTransfer(
      google::protobuf::RpcController* controller,
      const proto::WaitForKVBlocksTransferRequest* request,
      proto::Status* response,
      google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    std::lock_guard<std::mutex> lock(restore_mutex_);
    if (request->batch_id() != restore_batch_id_ ||
        registration_done_ != nullptr) {
      controller->SetFailed("Restore has not been registered");
      return;
    }
    completion_requested_.set_value();
    if (completion_allowed_) {
      response->set_ok(true);
      return;
    }
    completion_response_ = response;
    completion_done_ = guard.release();
  }

  std::future<void> registration_requested() {
    return registration_requested_.get_future();
  }

  std::future<void> completion_requested() {
    return completion_requested_.get_future();
  }

  void allow_restore_registration() {
    std::lock_guard<std::mutex> lock(restore_mutex_);
    if (registration_done_ == nullptr) {
      return;
    }
    registration_response_->set_success_cnt(registration_count_);
    brpc::ClosureGuard guard(registration_done_);
    registration_done_ = nullptr;
  }

  void allow_restore_completion() {
    std::lock_guard<std::mutex> lock(restore_mutex_);
    completion_allowed_ = true;
    if (completion_done_ == nullptr) {
      return;
    }
    completion_response_->set_ok(true);
    brpc::ClosureGuard guard(completion_done_);
    completion_done_ = nullptr;
  }

  void Sleep(google::protobuf::RpcController* /*controller*/,
             const proto::SleepRequest* request,
             proto::Status* response,
             google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    response->set_ok(request->master_status() == proto::DEEP_SLEEP &&
                     request->transition_id() == 41);
  }

  void Wakeup(google::protobuf::RpcController* /*controller*/,
              const proto::WakeupRequest* request,
              proto::Status* response,
              google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    response->set_ok(
        request->master_status() == proto::DEEP_SLEEP &&
        request->transition_id() == 41 && request->remote_addrs_size() == 1 &&
        request->remote_addrs(0) == "source:1234" &&
        request->src_weight_segments_size() == 1 &&
        request->src_weight_segments(0).segments_size() == 1 &&
        request->src_weight_segments(0).segments(0).offset() == 128 &&
        request->src_weight_segments(0).segments(0).size() == 256);
  }

  void ExecuteLLMModel(google::protobuf::RpcController* controller,
                       const proto::LLMForwardRequest* request,
                       proto::LLMForwardResponse* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    if (!request->has_packed_input() ||
        request->packed_input().payload().size() < 11 ||
        static_cast<uint8_t>(request->packed_input().payload()[10]) != 1) {
      controller->SetFailed("Unexpected packed input domain");
      return;
    }
    response->add_outputs()->add_tokens()->set_id(1);
  }

  void ExecuteVLMModel(google::protobuf::RpcController* controller,
                       const proto::VLMForwardRequest* request,
                       proto::VLMForwardResponse* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    if (!request->has_packed_input() ||
        request->packed_input().payload().size() < 11 ||
        static_cast<uint8_t>(request->packed_input().payload()[10]) != 2) {
      controller->SetFailed("Unexpected packed input domain");
      return;
    }
    response->add_outputs()->add_tokens()->set_id(2);
  }

  void ExecuteRECModel(google::protobuf::RpcController* controller,
                       const proto::RECForwardRequest* request,
                       proto::RECForwardResponse* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    if (!request->has_packed_input() ||
        request->packed_input().payload().size() < 11 ||
        static_cast<uint8_t>(request->packed_input().payload()[10]) != 3) {
      controller->SetFailed("Unexpected packed input domain");
      return;
    }
    response->add_outputs()->add_tokens()->set_id(3);
  }

  void ExecuteDITModel(google::protobuf::RpcController* controller,
                       const proto::DITForwardRequest* request,
                       proto::DITForwardResponse* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    if (!request->has_packed_input() ||
        request->packed_input().payload().size() < 11 ||
        static_cast<uint8_t>(request->packed_input().payload()[10]) != 4) {
      controller->SetFailed("Unexpected packed input domain");
      return;
    }
    response->mutable_dit_forward_output()->add_text_output("diffusion");
  }

  void GetLastLLMStepResult(google::protobuf::RpcController* controller,
                            const proto::Empty* /*request*/,
                            proto::LLMForwardResponse* /*response*/,
                            google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    controller->SetFailed("result unavailable");
  }

  void GetLastVLMStepResult(google::protobuf::RpcController* controller,
                            const proto::Empty* /*request*/,
                            proto::VLMForwardResponse* /*response*/,
                            google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    controller->SetFailed("result unavailable");
  }

 private:
  std::mutex restore_mutex_;
  uint64_t restore_batch_id_ = 0;
  uint32_t registration_count_ = 0;
  bool completion_allowed_ = false;
  std::promise<void> registration_requested_;
  std::promise<void> completion_requested_;
  proto::TransferStatus* registration_response_ = nullptr;
  google::protobuf::Closure* registration_done_ = nullptr;
  proto::Status* completion_response_ = nullptr;
  google::protobuf::Closure* completion_done_ = nullptr;
};

class CommChannelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(server_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE),
              0);
    ASSERT_EQ(server_.Start("127.0.0.1:0", nullptr), 0);
    ASSERT_TRUE(channel_.init_brpc(
        butil::endpoint2str(server_.listen_address()).c_str()));
  }

  void TearDown() override {
    server_.Stop(/*timeout_ms=*/0);
    server_.Join();
  }

  ModelRoutingService service_;
  brpc::Server server_;
  CommChannel channel_;
};

TEST_F(CommChannelTest, RoutesLLMInputToItsOwnRpc) {
  LlmForwardInput input;
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  channel_.execute_llm_model_async(input, promise);
  const auto output = std::move(future).get();
  ASSERT_TRUE(output.has_value());
  ASSERT_EQ(output->outputs.size(), 1);
  ASSERT_EQ(output->outputs[0].tokens.size(), 1);
  EXPECT_EQ(output->outputs[0].tokens[0].id, 1);
}

TEST_F(CommChannelTest, PreservesResidencyGenerationAcrossLifecycleRpcs) {
  ASSERT_TRUE(channel_.sleep(MasterStatus::DEEP_SLEEP, /*transition_id=*/41));
  WakeupOptions options;
  options.master_status = MasterStatus::DEEP_SLEEP;
  options.transition_id = 41;
  options.remote_addrs = {"source:1234"};
  options.src_weight_segments = {{{128, 256}}};
  EXPECT_TRUE(channel_.wakeup(options));
  options.transition_id = 40;
  EXPECT_FALSE(channel_.wakeup(options));
}

TEST_F(CommChannelTest, RestoreAcknowledgesRegistrationBeforeCopyCompletion) {
  const std::string address =
      butil::endpoint2str(server_.listen_address()).c_str();
  auto channel = std::make_unique<CommChannel>();
  ASSERT_TRUE(channel->init_brpc(address));
  RemoteWorker worker(/*global_rank=*/0,
                      address,
                      torch::Device(torch::kCPU),
                      std::move(channel));
  auto release_restore = folly::makeGuard([this]() {
    service_.allow_restore_registration();
    service_.allow_restore_completion();
  });
  auto registration_requested = service_.registration_requested();
  auto completion_requested = service_.completion_requested();
  const uint8_t hash_key[XXH3_128BITS_HASH_VALUE_LEN] = {};
  const BlockTransferInfo info(
      /*src_id=*/1, /*dst_id=*/2, hash_key, TransferType::H2D);
  auto restored = worker.transfer_kv_blocks(/*batch_id=*/73, {info});
  ASSERT_EQ(registration_requested.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  LlmForwardInput input;
  auto forward = worker.step_remote_async(input);
  EXPECT_EQ(completion_requested.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  EXPECT_FALSE(forward.isReady());

  service_.allow_restore_registration();
  ASSERT_EQ(completion_requested.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  const auto output = std::move(forward).get(std::chrono::seconds(5));
  ASSERT_TRUE(output.has_value());
  EXPECT_FALSE(restored.isReady());

  service_.allow_restore_completion();
  EXPECT_EQ(std::move(restored).get(std::chrono::seconds(5)), 1u);
}

TEST_F(CommChannelTest, RejectsCompletionForAnUnregisteredRestore) {
  EXPECT_FALSE(channel_.wait_for_kv_blocks_transfer(/*batch_id=*/73));
}

TEST_F(CommChannelTest, RoutesVLMInputToItsOwnRpc) {
  VlmForwardInput input;
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  channel_.execute_vlm_model_async(input, promise);
  const auto output = std::move(future).get();
  ASSERT_TRUE(output.has_value());
  ASSERT_EQ(output->outputs.size(), 1);
  ASSERT_EQ(output->outputs[0].tokens.size(), 1);
  EXPECT_EQ(output->outputs[0].tokens[0].id, 2);
}

TEST_F(CommChannelTest, RoutesRECInputToItsOwnRpc) {
  RecForwardInput input;
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  channel_.execute_rec_model_async(input, promise);
  const auto output = std::move(future).get();
  ASSERT_TRUE(output.has_value());
  ASSERT_EQ(output->outputs.size(), 1);
  ASSERT_EQ(output->outputs[0].tokens.size(), 1);
  EXPECT_EQ(output->outputs[0].tokens[0].id, 3);
}

TEST_F(CommChannelTest, RoutesDITInputToItsOwnRpc) {
  DiTForwardInput input;
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  channel_.execute_dit_model_async(input, promise);
  const auto output = std::move(future).get();
  ASSERT_TRUE(output.has_value());
  ASSERT_EQ(output->dit_forward_output.text_output.size(), 1);
  EXPECT_EQ(output->dit_forward_output.text_output[0], "diffusion");
}

TEST_F(CommChannelTest, CompletesPromiseWhenLastLLMResultRpcFails) {
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  EXPECT_FALSE(channel_.get_last_llm_step_result_async(promise));
  ASSERT_TRUE(future.isReady());
  EXPECT_FALSE(std::move(future).get().has_value());
}

TEST_F(CommChannelTest, CompletesPromiseWhenLastVLMResultRpcFails) {
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  EXPECT_FALSE(channel_.get_last_vlm_step_result_async(promise));
  ASSERT_TRUE(future.isReady());
  EXPECT_FALSE(std::move(future).get().has_value());
}

}  // namespace
}  // namespace xllm
