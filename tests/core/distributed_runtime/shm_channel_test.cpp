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

#include "core/distributed_runtime/shm_channel.h"

#include <brpc/closure_guard.h>
#include <brpc/server.h>
#include <butil/endpoint.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/runtime/dit_forward_params.h"
#include "runtime/vlm_forward_params.h"
#include "util/net.h"

namespace xllm {
namespace {

class ShmFallbackService final : public proto::DistributeWorker {
 public:
  void ExecuteLLMModel(google::protobuf::RpcController* /*controller*/,
                       const proto::LLMForwardRequest* /*request*/,
                       proto::LLMForwardResponse* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    executions_.fetch_add(1);
    response->add_outputs()->add_tokens()->set_id(99);
  }

  void ExecuteVLMModel(google::protobuf::RpcController* /*controller*/,
                       const proto::VLMForwardRequest* /*request*/,
                       proto::VLMForwardResponse* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    executions_.fetch_add(1);
    response->add_outputs()->add_tokens()->set_id(99);
  }

  void ExecuteDITModel(google::protobuf::RpcController* /*controller*/,
                       const proto::DITForwardRequest* /*request*/,
                       proto::DITForwardResponse* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    executions_.fetch_add(1);
    response->mutable_dit_forward_output()->add_text_output("fallback");
  }

  int32_t execution_count() const { return executions_.load(); }

 private:
  std::atomic<int32_t> executions_{0};
};

class ShmChannelTest : public ::testing::Test {
 protected:
  using OutputPromise = folly::Promise<std::optional<RawForwardOutput>>;

  void SetUp() override {
    auto& execution_config = ExecutionConfig::get_instance();
    previous_contiguous_ = execution_config.use_contiguous_input_buffer();
    execution_config.use_contiguous_input_buffer(false);
    ASSERT_EQ(server_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE),
              0);
    ASSERT_EQ(server_.Start("127.0.0.1:0", nullptr), 0);
    address_ = butil::endpoint2str(server_.listen_address()).c_str();
    options_.enable_shm(true)
        .master_node_addr(address_)
        .input_shm_size(kInputBytes)
        .output_shm_size(kOutputBytes);
    channel_ = std::make_unique<ShmChannel>(
        /*dp_group=*/0, /*rank=*/0, /*is_driver=*/false, options_);
    ASSERT_TRUE(channel_->init_brpc(address_));
    const std::string name = ForwardSharedMemoryTransport::create_unique_name(
        "xllm_" + net::extract_port(address_),
        /*dp_group=*/0,
        ForwardType::RAW_OUTPUT,
        /*rank=*/0);
    bool is_creator = true;
    writer_ = std::make_unique<ForwardSharedMemoryTransport>(
        name, kOutputBytes, is_creator, ForwardType::RAW_OUTPUT);
    ASSERT_FALSE(is_creator);
  }

  void TearDown() override {
    channel_.reset();
    writer_.reset();
    server_.Stop(/*timeout_ms=*/0);
    server_.Join();
    ExecutionConfig::get_instance().use_contiguous_input_buffer(
        previous_contiguous_);
  }

  void use_driver_channel() {
    channel_ = std::make_unique<ShmChannel>(
        /*dp_group=*/0, /*rank=*/0, /*is_driver=*/true, options_);
    ASSERT_TRUE(channel_->init_brpc(address_));
  }

  template <typename Input>
  void expect_input_capacity(const Input& input) {
    proto::PackedForwardInput packed;
    ASSERT_TRUE(input.to_packed_proto(&packed));
    const size_t segment_size =
        sizeof(ControlMetadata) + sizeof(uint64_t) + packed.payload().size();
    EXPECT_TRUE(
        ForwardSharedMemoryTransport::can_write_input(input, segment_size));
    EXPECT_FALSE(
        ForwardSharedMemoryTransport::can_write_input(input, segment_size - 1));
    EXPECT_FALSE(ForwardSharedMemoryTransport::can_write_input(input, 0));
    bool is_creator = false;
    ForwardSharedMemoryTransport writer(writer_->name() + "_input_capacity",
                                        segment_size,
                                        is_creator,
                                        ForwardType::RAW_INPUT);
    ASSERT_TRUE(is_creator);
    EXPECT_TRUE(writer.input_write(input));
  }

  template <typename Input>
  void expect_group_input_fallback(
      const Input& oversized,
      const Input& small,
      void (ShmChannel::*execute)(const Input&, OutputPromise&)) {
    ASSERT_FALSE(
        ForwardSharedMemoryTransport::can_write_input(oversized, kInputBytes));
    ASSERT_TRUE(
        ForwardSharedMemoryTransport::can_write_input(small, kInputBytes));
    use_driver_channel();
    ShmChannel follower(
        /*dp_group=*/0, /*rank=*/1, /*is_driver=*/false, options_);
    ASSERT_TRUE(follower.init_brpc(address_));

    // A follower can be scheduled before the driver. It must choose BRPC
    // independently instead of waiting for an unpublished SHM input.
    const std::array<ShmChannel*, 2> channels = {&follower, channel_.get()};
    for (const Input* input : {&oversized, &small}) {
      for (ShmChannel* channel : channels) {
        OutputPromise promise;
        auto future = promise.getSemiFuture();
        (channel->*execute)(*input, promise);
        ASSERT_TRUE(std::move(future).get().has_value());
      }
    }
    EXPECT_EQ(service_.execution_count(), 4);
    const std::string name = ForwardSharedMemoryTransport::create_unique_name(
        "xllm_" + net::extract_port(address_),
        /*dp_group=*/0,
        ForwardType::RAW_INPUT,
        /*rank=*/0);
    bool is_creator = true;
    ForwardSharedMemoryTransport input_reader(
        name, kInputBytes, is_creator, ForwardType::RAW_INPUT);
    ASSERT_FALSE(is_creator);
    EXPECT_EQ(static_cast<const ControlMetadata*>(input_reader.base_address())
                  ->version,
              0);
  }

  static constexpr uint64_t kInputBytes = 4096;
  static constexpr uint64_t kOutputBytes = 4096;
  bool previous_contiguous_ = false;
  std::string address_;
  runtime::Options options_;
  ShmFallbackService service_;
  brpc::Server server_;
  std::unique_ptr<ShmChannel> channel_;
  std::unique_ptr<ForwardSharedMemoryTransport> writer_;
};

TEST_F(ShmChannelTest, ReadsLlmOutputWithoutRpcExecution) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({7}, torch::kInt32);
  ASSERT_TRUE(writer_->llm_output_write(source, {}));
  OutputPromise promise;
  auto future = promise.getSemiFuture();
  channel_->execute_llm_model_async(LlmForwardInput(), promise);
  const auto output = std::move(future).get();
  ASSERT_TRUE(output.has_value());
  ASSERT_EQ(output->outputs.size(), 1);
  EXPECT_EQ(output->outputs[0].tokens[0].id, 7);
  EXPECT_EQ(service_.execution_count(), 0);
}

TEST_F(ShmChannelTest, ReadsVlmOutputWithoutRpcExecution) {
  ForwardOutput source;
  source.sample_output.mm_embeddings = {{torch::tensor({1.0f, 2.0f})}};
  ASSERT_TRUE(writer_->vlm_output_write(source, {}));
  OutputPromise promise;
  auto future = promise.getSemiFuture();
  channel_->execute_vlm_model_async(VlmForwardInput(), promise);
  const auto output = std::move(future).get();
  ASSERT_TRUE(output.has_value());
  ASSERT_EQ(output->outputs.size(), 1);
  ASSERT_EQ(output->outputs[0].mm_embeddings.size(), 1);
  EXPECT_TRUE(torch::equal(output->outputs[0].mm_embeddings[0],
                           source.sample_output.mm_embeddings[0][0]));
  EXPECT_EQ(service_.execution_count(), 0);
}

TEST_F(ShmChannelTest, ReadsDitOutputWithoutRpcExecution) {
  DiTForwardOutput source;
  source.text_output = {"shared output"};
  ASSERT_TRUE(writer_->dit_output_write(source));
  OutputPromise promise;
  auto future = promise.getSemiFuture();
  channel_->execute_dit_model_async(DiTForwardInput(), promise);
  const auto output = std::move(future).get();
  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->dit_forward_output.text_output, source.text_output);
  EXPECT_EQ(service_.execution_count(), 0);
}

TEST_F(ShmChannelTest, DitFailureDoesNotReexecuteAndNextReadRecovers) {
  ASSERT_TRUE(
      writer_->dit_output_write({}, {StatusCode::RESOURCE_EXHAUSTED, "out"}));
  OutputPromise promise;
  auto future = promise.getSemiFuture();
  channel_->execute_dit_model_async(DiTForwardInput(), promise);
  ASSERT_TRUE(future.isReady());
  try {
    std::move(future).get();
    FAIL() << "expected published DiT failure";
  } catch (const DiTForwardExecutionError& error) {
    EXPECT_EQ(error.status().code(), StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_EQ(error.status().message(), "out");
  }
  EXPECT_EQ(service_.execution_count(), 0);
  DiTForwardOutput source;
  source.text_output = {"after failure"};
  ASSERT_TRUE(writer_->dit_output_write(source));
  OutputPromise next_promise;
  auto next = next_promise.getSemiFuture();
  channel_->execute_dit_model_async(DiTForwardInput(), next_promise);
  const auto output = std::move(next).get(std::chrono::seconds(5));
  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->dit_forward_output.text_output, source.text_output);
  EXPECT_TRUE(output->dit_execution_status.ok());
  EXPECT_EQ(service_.execution_count(), 0);
}

TEST_F(ShmChannelTest, DitMalformedFrameCompletesExceptionAndNextReadRecovers) {
  ASSERT_TRUE(writer_->dit_output_write({}));
  const uint64_t bad_magic = 0;
  char* payload =
      static_cast<char*>(writer_->base_address()) + sizeof(ControlMetadata);
  std::memcpy(payload, &bad_magic, sizeof(bad_magic));
  OutputPromise promise;
  auto future = promise.getSemiFuture();
  channel_->execute_dit_model_async(DiTForwardInput(), promise);
  ASSERT_TRUE(future.isReady());
  try {
    std::move(future).get();
    FAIL() << "expected malformed frame error";
  } catch (const DiTForwardExecutionError& error) {
    EXPECT_EQ(error.status().code(), StatusCode::UNKNOWN);
  }
  ASSERT_TRUE(writer_->dit_output_write({}));
  OutputPromise next_promise;
  auto next = next_promise.getSemiFuture();
  channel_->execute_dit_model_async(DiTForwardInput(), next_promise);
  ASSERT_TRUE(std::move(next).get(std::chrono::seconds(5)).has_value());
  EXPECT_EQ(service_.execution_count(), 0);
}

TEST_F(ShmChannelTest, DitRejectsOutputCapacityBelowFailureHeader) {
  options_.backend("dit").output_shm_size(
      ForwardSharedMemoryTransport::minimum_dit_output_size() - 1);
  EXPECT_DEATH(
      {
        ShmChannel too_small(
            /*dp_group=*/0, /*rank=*/2, /*is_driver=*/false, options_);
      },
      "minimum|DiT|DIT");
}

TEST_F(ShmChannelTest, InputWriteFailureFallsBackBeforePublishingRequest) {
  constexpr uint64_t kSmallInputBytes = 64;
  options_.input_shm_size(kSmallInputBytes);
  use_driver_channel();
  OutputPromise promise;
  auto future = promise.getSemiFuture();
  channel_->execute_llm_model_async(LlmForwardInput(), promise);
  const auto output = std::move(future).get();
  ASSERT_TRUE(output.has_value());
  ASSERT_EQ(output->outputs.size(), 1);
  EXPECT_EQ(output->outputs[0].tokens[0].id, 99);
  EXPECT_EQ(service_.execution_count(), 1);
  const std::string name = ForwardSharedMemoryTransport::create_unique_name(
      "xllm_" + net::extract_port(address_),
      /*dp_group=*/0,
      ForwardType::RAW_INPUT,
      /*rank=*/0);
  bool is_creator = true;
  ForwardSharedMemoryTransport input_reader(
      name, kSmallInputBytes, is_creator, ForwardType::RAW_INPUT);
  ASSERT_FALSE(is_creator);
  EXPECT_EQ(
      static_cast<const ControlMetadata*>(input_reader.base_address())->version,
      0);
}

TEST_F(ShmChannelTest, LlmOverflowFallsBackForEveryRankAndStaysOnBrpc) {
  LlmForwardInput oversized;
  oversized.token_ids = torch::zeros({kInputBytes * 2}, torch::kInt32);
  expect_group_input_fallback(
      oversized, LlmForwardInput(), &ShmChannel::execute_llm_model_async);
}

TEST_F(ShmChannelTest, VlmOverflowFallsBackForEveryRankAndStaysOnBrpc) {
  VlmForwardInput oversized;
  oversized.token_ids = torch::zeros({kInputBytes * 2}, torch::kInt32);
  expect_group_input_fallback(
      oversized, VlmForwardInput(), &ShmChannel::execute_vlm_model_async);
}

TEST_F(ShmChannelTest, DitOverflowFallsBackForEveryRankAndStaysOnBrpc) {
  DiTForwardInput oversized;
  oversized.model_params.prompts = {std::string(kInputBytes * 2, 'x')};
  expect_group_input_fallback(
      oversized, DiTForwardInput(), &ShmChannel::execute_dit_model_async);
}

TEST_F(ShmChannelTest, DecodeFailureDoesNotReexecuteAndNextReadRecovers) {
  const std::array<std::function<void(OutputPromise&)>, 2> execute = {
      [this](OutputPromise& promise) {
        channel_->execute_llm_model_async(LlmForwardInput(), promise);
      },
      [this](OutputPromise& promise) {
        channel_->execute_vlm_model_async(VlmForwardInput(), promise);
      }};
  for (const auto& execute_model : execute) {
    ASSERT_TRUE(writer_->llm_output_write(ForwardOutput(), {}));
    const uint64_t invalid_count = std::numeric_limits<uint64_t>::max();
    char* payload =
        static_cast<char*>(writer_->base_address()) + sizeof(ControlMetadata);
    std::memcpy(payload, &invalid_count, sizeof(invalid_count));
    OutputPromise promise;
    auto future = promise.getSemiFuture();
    execute_model(promise);
    ASSERT_TRUE(future.isReady());
    EXPECT_FALSE(std::move(future).get().has_value());
    EXPECT_EQ(service_.execution_count(), 0);
  }
  DiTForwardOutput source;
  source.text_output = {"recovered output"};
  ASSERT_TRUE(writer_->dit_output_write(source));
  OutputPromise promise;
  auto future = promise.getSemiFuture();
  channel_->execute_dit_model_async(DiTForwardInput(), promise);
  const auto output = std::move(future).get();
  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->dit_forward_output.text_output, source.text_output);
  EXPECT_EQ(service_.execution_count(), 0);
}

TEST_F(ShmChannelTest, InputPreflightMatchesExactPackedSize) {
  LlmForwardInput llm_input;
  llm_input.token_ids =
      torch::arange(8, torch::kInt32).reshape({2, 4}).transpose(0, 1);
  ASSERT_FALSE(llm_input.token_ids.is_contiguous());
  ASSERT_NO_FATAL_FAILURE(expect_input_capacity(llm_input));
  VlmForwardInput vlm_input;
  vlm_input.positions =
      torch::arange(8, torch::kInt64).reshape({2, 4}).transpose(0, 1);
  ASSERT_NO_FATAL_FAILURE(expect_input_capacity(vlm_input));
  DiTForwardInput dit_input;
  dit_input.model_params.prompts = {"capacity boundary"};
  ASSERT_NO_FATAL_FAILURE(expect_input_capacity(dit_input));
}

}  // namespace
}  // namespace xllm
