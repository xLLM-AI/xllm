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

#include "core/distributed_runtime/worker_service.h"

#include <brpc/controller.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/runtime/options.h"
#include "core/runtime/worker_residency_state.h"

namespace xllm {

class WorkerServiceOutputTest : public ::testing::Test {
 protected:
  void collect(const ForwardOutput& source, ForwardOutput& destination) {
    runtime::Options options;
    options.backend("llm");
    WorkerService service(options, torch::Device("npu:0"), nullptr);
    std::vector<SpeculativeTokenStats> stats;
    service.collect_llm_output(
        source, destination, /*cpu_ready_result=*/true, stats);
  }
};

namespace {

TEST_F(WorkerServiceOutputTest, PreservesRetiredDsparkFeedbackBeforeEncoding) {
  ForwardOutput source;
  source.sample_output.next_tokens = torch::tensor({{7, 8}}, torch::kInt32);
  source.dspark_feedback.confidence = {0.9F, 0.8F, 0.7F, 0.6F, 0.5F};
  source.dspark_feedback.rounds_since_full_width = {3};
  ForwardOutput collected;
  collect(source, collected);
  EXPECT_EQ(collected.dspark_feedback.confidence,
            source.dspark_feedback.confidence);
  EXPECT_EQ(collected.dspark_feedback.rounds_since_full_width,
            source.dspark_feedback.rounds_since_full_width);

  source.dspark_feedback = {};
  collect(source, collected);
  EXPECT_TRUE(collected.dspark_feedback.confidence.empty());
  EXPECT_TRUE(collected.dspark_feedback.rounds_since_full_width.empty());
}

TEST(WorkerResidencyStateTest, RejectsUninitializedAndInvalidTransitions) {
  WorkerResidencyState state;
  int32_t operation_count = 0;
  auto operation = [&operation_count]() {
    ++operation_count;
    return true;
  };
  WakeupOptions options;
  options.master_status = MasterStatus::LIGHT_SLEEP;
  EXPECT_FALSE(state.sleep(MasterStatus::LIGHT_SLEEP,
                           /*transition_id=*/1,
                           operation));
  EXPECT_FALSE(state.wakeup(options, operation));
  MasterStatus invalid(static_cast<MasterStatus::Value>(-1));
  EXPECT_FALSE(state.initialize(invalid, operation));
  ASSERT_TRUE(state.initialize(MasterStatus::WAKEUP, operation));
  EXPECT_FALSE(state.sleep(MasterStatus::WAKEUP,
                           /*transition_id=*/1,
                           operation));
  EXPECT_FALSE(state.sleep(invalid, /*transition_id=*/1, operation));
  EXPECT_FALSE(state.sleep(MasterStatus::LIGHT_SLEEP,
                           /*transition_id=*/0,
                           operation));
  EXPECT_FALSE(state.wakeup(options, operation));
  options.master_status = invalid;
  EXPECT_FALSE(state.wakeup(options, operation));
  EXPECT_EQ(operation_count, 1);
}

TEST(WorkerResidencyStateTest, InitialSleepUsesTransitionZero) {
  for (MasterStatus sleep_status :
       {MasterStatus::LIGHT_SLEEP, MasterStatus::DEEP_SLEEP}) {
    WorkerResidencyState state;
    ASSERT_TRUE(state.initialize(sleep_status, []() { return true; }));
    int32_t sleep_count = 0;
    EXPECT_TRUE(state.sleep(sleep_status, /*transition_id=*/0, [&]() {
      ++sleep_count;
      return true;
    }));
    WakeupOptions options;
    options.master_status = sleep_status;
    int32_t wakeup_count = 0;
    auto wakeup = [&]() {
      ++wakeup_count;
      return true;
    };
    EXPECT_TRUE(state.wakeup(options, wakeup));
    EXPECT_TRUE(state.wakeup(options, wakeup));
    EXPECT_EQ(sleep_count, 0);
    EXPECT_EQ(wakeup_count, 1);
  }
}

TEST(WorkerResidencyStateTest, LateSleepRetryDoesNotUndoCompletedWakeup) {
  WorkerResidencyState state;
  ASSERT_TRUE(state.initialize(MasterStatus::WAKEUP, []() { return true; }));
  int32_t sleep_count = 0;
  auto sleep = [&]() {
    ++sleep_count;
    return true;
  };
  ASSERT_TRUE(state.sleep(MasterStatus::LIGHT_SLEEP,
                          /*transition_id=*/1,
                          sleep));
  EXPECT_TRUE(state.sleep(MasterStatus::LIGHT_SLEEP,
                          /*transition_id=*/1,
                          sleep));
  EXPECT_FALSE(state.sleep(MasterStatus::DEEP_SLEEP,
                           /*transition_id=*/1,
                           sleep));
  EXPECT_FALSE(state.sleep(MasterStatus::LIGHT_SLEEP,
                           /*transition_id=*/2,
                           sleep));
  WakeupOptions options;
  options.master_status = MasterStatus::LIGHT_SLEEP;
  options.transition_id = 1;
  int32_t wakeup_count = 0;
  auto wakeup = [&]() {
    ++wakeup_count;
    return true;
  };
  ASSERT_TRUE(state.wakeup(options, wakeup));
  EXPECT_TRUE(state.sleep(MasterStatus::LIGHT_SLEEP,
                          /*transition_id=*/1,
                          sleep));
  EXPECT_TRUE(state.wakeup(options, wakeup));
  EXPECT_EQ(sleep_count, 1);
  EXPECT_EQ(wakeup_count, 1);

  ASSERT_TRUE(state.sleep(MasterStatus::DEEP_SLEEP,
                          /*transition_id=*/2,
                          sleep));
  EXPECT_FALSE(state.sleep(MasterStatus::LIGHT_SLEEP,
                           /*transition_id=*/1,
                           sleep));
  EXPECT_FALSE(state.wakeup(options, wakeup));
  options.master_status = MasterStatus::DEEP_SLEEP;
  options.transition_id = 2;
  EXPECT_TRUE(state.wakeup(options, wakeup));
  EXPECT_EQ(sleep_count, 2);
  EXPECT_EQ(wakeup_count, 2);
}

TEST(WorkerResidencyStateTest, FailedOperationsDoNotRecordCompletion) {
  WorkerResidencyState state;
  EXPECT_FALSE(state.initialize(MasterStatus::WAKEUP, []() { return false; }));
  int32_t sleep_count = 0;
  auto sleep = [&]() {
    ++sleep_count;
    return true;
  };
  EXPECT_FALSE(state.sleep(MasterStatus::LIGHT_SLEEP,
                           /*transition_id=*/1,
                           sleep));
  ASSERT_TRUE(state.initialize(MasterStatus::WAKEUP, []() { return true; }));
  EXPECT_FALSE(state.sleep(MasterStatus::LIGHT_SLEEP,
                           /*transition_id=*/1,
                           []() { return false; }));
  ASSERT_TRUE(state.sleep(MasterStatus::LIGHT_SLEEP,
                          /*transition_id=*/1,
                          sleep));
  WakeupOptions options;
  options.master_status = MasterStatus::LIGHT_SLEEP;
  options.transition_id = 1;
  EXPECT_FALSE(state.wakeup(options, []() { return false; }));
  int32_t wakeup_count = 0;
  EXPECT_TRUE(state.wakeup(options, [&]() {
    ++wakeup_count;
    return true;
  }));
  EXPECT_EQ(sleep_count, 1);
  EXPECT_EQ(wakeup_count, 1);
}

TEST(WorkerResidencyStateTest, ExceptionsPropagateWithoutAcknowledgingSuccess) {
  WorkerResidencyState state;
  ASSERT_TRUE(state.initialize(MasterStatus::WAKEUP, []() { return true; }));
  auto throw_operation = []() -> bool {
    throw std::runtime_error("injected residency failure");
  };
  EXPECT_THROW(state.sleep(MasterStatus::LIGHT_SLEEP,
                           /*transition_id=*/1,
                           throw_operation),
               std::runtime_error);
  ASSERT_TRUE(state.sleep(MasterStatus::LIGHT_SLEEP,
                          /*transition_id=*/1,
                          []() { return true; }));
  WakeupOptions options;
  options.master_status = MasterStatus::LIGHT_SLEEP;
  options.transition_id = 1;
  EXPECT_THROW(state.wakeup(options, throw_operation), std::runtime_error);
  EXPECT_TRUE(state.wakeup(options, []() { return true; }));
}

TEST(WorkerResidencyStateTest, RejectsWakeupRetriesWithChangedOptions) {
  WorkerResidencyState state;
  ASSERT_TRUE(
      state.initialize(MasterStatus::DEEP_SLEEP, []() { return true; }));
  WakeupOptions options;
  options.master_status = MasterStatus::DEEP_SLEEP;
  options.remote_addrs = {"worker:1234"};
  options.src_weight_segments = {{{0, 128}}};
  int32_t wakeup_count = 0;
  auto wakeup = [&]() {
    ++wakeup_count;
    return true;
  };
  ASSERT_TRUE(state.wakeup(options, wakeup));
  EXPECT_TRUE(state.wakeup(options, wakeup));
  WakeupOptions changed = options;
  changed.master_status = MasterStatus::LIGHT_SLEEP;
  EXPECT_FALSE(state.wakeup(changed, wakeup));
  changed = options;
  changed.transition_id = 1;
  EXPECT_FALSE(state.wakeup(changed, wakeup));
  changed = options;
  changed.remote_addrs[0] = "worker:5678";
  EXPECT_FALSE(state.wakeup(changed, wakeup));
  changed = options;
  changed.src_weight_segments[0][0].offset = 128;
  EXPECT_FALSE(state.wakeup(changed, wakeup));
  changed = options;
  changed.src_weight_segments[0][0].size = 256;
  EXPECT_FALSE(state.wakeup(changed, wakeup));
  changed = options;
  changed.src_weight_segments[0].push_back({128, 128});
  EXPECT_FALSE(state.wakeup(changed, wakeup));
  changed = options;
  changed.src_weight_segments.push_back({});
  EXPECT_FALSE(state.wakeup(changed, wakeup));
  EXPECT_EQ(wakeup_count, 1);
}

TEST(WorkerResidencyStateTest, ConcurrentRetriesSerializeWithWakeup) {
  WorkerResidencyState state;
  ASSERT_TRUE(state.initialize(MasterStatus::WAKEUP, []() { return true; }));
  std::atomic<int32_t> sleep_count{0};
  auto sleep = [&]() {
    sleep_count.fetch_add(1, std::memory_order_relaxed);
    return true;
  };
  ASSERT_TRUE(state.sleep(MasterStatus::LIGHT_SLEEP,
                          /*transition_id=*/1,
                          sleep));
  WakeupOptions options;
  options.master_status = MasterStatus::LIGHT_SLEEP;
  options.transition_id = 1;
  std::atomic<int32_t> wakeup_count{0};
  std::promise<void> wakeup_started;
  auto started = wakeup_started.get_future();
  std::promise<void> release_wakeup;
  auto released = release_wakeup.get_future().share();
  auto first_wakeup = std::async(std::launch::async, [&]() {
    return state.wakeup(options, [&]() {
      wakeup_count.fetch_add(1, std::memory_order_relaxed);
      wakeup_started.set_value();
      released.wait();
      return true;
    });
  });
  EXPECT_EQ(started.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  auto retry_wakeup = std::async(std::launch::async, [&]() {
    return state.wakeup(options, [&]() {
      wakeup_count.fetch_add(1, std::memory_order_relaxed);
      return true;
    });
  });
  auto late_sleep = std::async(std::launch::async, [&]() {
    return state.sleep(MasterStatus::LIGHT_SLEEP,
                       /*transition_id=*/1,
                       sleep);
  });
  release_wakeup.set_value();
  EXPECT_TRUE(first_wakeup.get());
  EXPECT_TRUE(retry_wakeup.get());
  EXPECT_TRUE(late_sleep.get());
  EXPECT_EQ(sleep_count.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(wakeup_count.load(std::memory_order_relaxed), 1);
}

TEST(WorkerResidencyStateTest, SuccessfulInitializationResetsGeneration) {
  WorkerResidencyState state;
  ASSERT_TRUE(state.initialize(MasterStatus::WAKEUP, []() { return true; }));
  ASSERT_TRUE(state.sleep(MasterStatus::LIGHT_SLEEP,
                          /*transition_id=*/7,
                          []() { return true; }));
  EXPECT_FALSE(
      state.initialize(MasterStatus::DEEP_SLEEP, []() { return false; }));
  WakeupOptions options;
  options.master_status = MasterStatus::LIGHT_SLEEP;
  options.transition_id = 7;
  EXPECT_TRUE(state.wakeup(options, []() { return true; }));
  ASSERT_TRUE(
      state.initialize(MasterStatus::DEEP_SLEEP, []() { return true; }));
  EXPECT_FALSE(state.wakeup(options, []() { return true; }));
  options.master_status = MasterStatus::DEEP_SLEEP;
  options.transition_id = 0;
  EXPECT_TRUE(state.wakeup(options, []() { return true; }));
  EXPECT_TRUE(state.sleep(MasterStatus::LIGHT_SLEEP,
                          /*transition_id=*/1,
                          []() { return true; }));
}

class CountingClosure final : public google::protobuf::Closure {
 public:
  void Run() override {
    if (run_count_.fetch_add(1, std::memory_order_relaxed) == 0) {
      completed_.set_value();
    }
  }

  std::future<void> get_future() { return completed_.get_future(); }

  int32_t run_count() const {
    return run_count_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<int32_t> run_count_{0};
  std::promise<void> completed_;
};

template <typename Request, typename Response>
using ModelRpc = void (WorkerService::*)(google::protobuf::RpcController*,
                                         const Request*,
                                         Response*,
                                         google::protobuf::Closure*);

template <typename Request, typename Response>
void expect_rpc_rejection(runtime::Options options,
                          bool initialized,
                          const Request& request,
                          ModelRpc<Request, Response> rpc,
                          const std::string& expected_error) {
  brpc::Controller controller;
  Response response;
  CountingClosure done;
  auto completed = done.get_future();
  // The injected constructor marks the service initialized. A null Worker
  // ensures rejected requests cannot accidentally enter model execution.
  auto service = initialized ? std::make_unique<WorkerService>(
                                   options, torch::Device("npu:0"), nullptr)
                             : std::make_unique<WorkerService>(
                                   options, torch::Device("npu:0"));
  (service.get()->*rpc)(&controller, &request, &response, &done);
  EXPECT_EQ(completed.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  // Destruction drains request tasks before the callback count is inspected.
  service.reset();
  EXPECT_EQ(done.run_count(), 1);
  EXPECT_TRUE(controller.Failed());
  EXPECT_NE(controller.ErrorText().find(expected_error), std::string::npos)
      << controller.ErrorText();
}

void make_packed_inputs(
    std::array<proto::PackedForwardInput, 4>& packed_inputs) {
  LlmForwardInput llm_input;
  VlmForwardInput vlm_input;
  RecForwardInput rec_input;
  DiTForwardInput dit_input;
  ASSERT_TRUE(llm_input.to_packed_proto(&packed_inputs[0]));
  ASSERT_TRUE(vlm_input.to_packed_proto(&packed_inputs[1]));
  ASSERT_TRUE(rec_input.to_packed_proto(&packed_inputs[2]));
  ASSERT_TRUE(dit_input.to_packed_proto(&packed_inputs[3]));
}

template <typename Request, typename Response>
void expect_execute_rejections(const std::string& backend,
                               size_t input_index,
                               ModelRpc<Request, Response> rpc) {
  runtime::Options options;
  options.backend(backend);
  Request request;
  expect_rpc_rejection(options,
                       /*initialized=*/true,
                       request,
                       rpc,
                       "requires a packed input payload");

  request.mutable_packed_input()->set_payload("invalid");
  expect_rpc_rejection(options,
                       /*initialized=*/true,
                       request,
                       rpc,
                       "input domain, schema, or layout");

  std::array<proto::PackedForwardInput, 4> packed_inputs;
  ASSERT_NO_FATAL_FAILURE(make_packed_inputs(packed_inputs));
  for (size_t index = 0; index < packed_inputs.size(); ++index) {
    if (index == input_index) {
      continue;
    }
    SCOPED_TRACE(index);
    *request.mutable_packed_input() = packed_inputs[index];
    expect_rpc_rejection(options,
                         /*initialized=*/true,
                         request,
                         rpc,
                         "input domain, schema, or layout");
  }

  expect_rpc_rejection(options,
                       /*initialized=*/false,
                       request,
                       rpc,
                       "Worker is not initialized");
  options.backend(backend == "llm" ? "dit" : "llm");
  expect_rpc_rejection(options,
                       /*initialized=*/true,
                       request,
                       rpc,
                       "RPC does not match the configured backend");
}

TEST(WorkerServiceModelRpcTest,
     ExecuteLLMRejectsInvalidRequestsBeforeExecution) {
  expect_execute_rejections("llm",
                            /*input_index=*/0,
                            &WorkerService::ExecuteLLMModel);
}

TEST(WorkerServiceModelRpcTest,
     ExecuteVLMRejectsInvalidRequestsBeforeExecution) {
  expect_execute_rejections("vlm",
                            /*input_index=*/1,
                            &WorkerService::ExecuteVLMModel);
}

TEST(WorkerServiceModelRpcTest,
     ExecuteRECRejectsInvalidRequestsBeforeExecution) {
  expect_execute_rejections("rec",
                            /*input_index=*/2,
                            &WorkerService::ExecuteRECModel);
}

TEST(WorkerServiceModelRpcTest,
     ExecuteDITRejectsInvalidRequestsBeforeExecution) {
  expect_execute_rejections("dit",
                            /*input_index=*/3,
                            &WorkerService::ExecuteDITModel);
}

template <typename Response>
void expect_last_step_rejections(const std::string& backend,
                                 ModelRpc<proto::Empty, Response> rpc) {
  runtime::Options options;
  options.backend(backend).enable_schedule_overlap(true);
  proto::Empty request;
  expect_rpc_rejection(options,
                       /*initialized=*/false,
                       request,
                       rpc,
                       "Worker is not initialized");

  options.backend(backend == "llm" ? "vlm" : "llm");
  expect_rpc_rejection(
      options,
      /*initialized=*/true,
      request,
      rpc,
      "last-step results require the matching overlap backend");

  options.backend(backend).enable_schedule_overlap(false);
  expect_rpc_rejection(
      options,
      /*initialized=*/true,
      request,
      rpc,
      "last-step results require the matching overlap backend");
}

TEST(WorkerServiceModelRpcTest, LastLLMStepRequiresInitializedOverlapBackend) {
  expect_last_step_rejections("llm", &WorkerService::GetLastLLMStepResult);
}

TEST(WorkerServiceModelRpcTest, LastVLMStepRequiresInitializedOverlapBackend) {
  expect_last_step_rejections("vlm", &WorkerService::GetLastVLMStepResult);
}

class WorkerServiceInputLifetimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto& config = ExecutionConfig::get_instance();
    previous_contiguous_ = config.use_contiguous_input_buffer();
    config.use_contiguous_input_buffer(false);
  }

  void TearDown() override {
    ExecutionConfig::get_instance().use_contiguous_input_buffer(
        previous_contiguous_);
  }

 private:
  bool previous_contiguous_ = false;
};

template <typename Input>
void expect_deferred_input_rebuilds_owned_host_views() {
  const torch::Tensor expected_tokens = torch::tensor({11, 12}, torch::kInt32);
  const torch::Tensor expected_positions = torch::tensor({3, 4}, torch::kInt32);
  const torch::Tensor expected_blocks = torch::tensor({{7, 8}}, torch::kInt32);
  Input input;
  {
    const std::string name =
        "worker_deferred_input_" + std::to_string(getpid()) + "_" +
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    bool is_creator = false;
    ForwardSharedMemoryManager writer(
        name, /*size=*/1 << 20, is_creator, ForwardType::RAW_INPUT);
    ASSERT_TRUE(is_creator);
    ForwardSharedMemoryManager reader(
        name, /*size=*/1 << 20, is_creator, ForwardType::RAW_INPUT);
    ASSERT_FALSE(is_creator);
    Input source;
    source.token_ids = expected_tokens;
    source.positions = expected_positions;
    source.input_params.attention.device.block_tables = expected_blocks;
    ASSERT_TRUE(writer.input_write(source));
    ASSERT_TRUE(reader.input_read(
        input,
        torch::Device(torch::kCPU),
        InputDeviceMaterializationPolicy::DEFER_TO_WORKER_PREPARE));
    ASSERT_TRUE(input.runtime.input_host_buffer.defined());
    ASSERT_TRUE(input.runtime.input_host_buffer_has_layout);
    ASSERT_FALSE(input.runtime.device_tensors_ready);

    // Packed preparation must reconstruct all three views. Stabilizing these
    // temporary fields would allocate copies that preparation discards.
    input.token_ids_host = torch::full_like(expected_tokens, -1);
    input.positions_host = torch::full_like(expected_positions, -1);
    input.input_params.attention.host.block_tables =
        torch::full_like(expected_blocks, -1);
    source.token_ids = torch::full_like(expected_tokens, 99);
    source.positions = torch::full_like(expected_positions, 99);
    source.input_params.attention.device.block_tables =
        torch::full_like(expected_blocks, 99);
    ASSERT_TRUE(writer.input_write(source));
  }

  // The next SHM request has overwritten the segment and both mappings have
  // gone away before the Worker prepares this input.
  const auto prepared = input.to(torch::Device(torch::kCPU), torch::kFloat32);
  input = Input();
  EXPECT_TRUE(torch::equal(prepared.token_ids_host, expected_tokens));
  EXPECT_TRUE(torch::equal(prepared.positions_host, expected_positions));
  EXPECT_TRUE(torch::equal(prepared.input_params.attention.host.block_tables,
                           expected_blocks));
}

TEST_F(WorkerServiceInputLifetimeTest,
       DeferredLlmPreparationRebuildsOwnedHostViewsAfterShmOverwrite) {
  expect_deferred_input_rebuilds_owned_host_views<LlmForwardInput>();
}

TEST_F(WorkerServiceInputLifetimeTest,
       DeferredVlmPreparationRebuildsOwnedHostViewsAfterShmOverwrite) {
  expect_deferred_input_rebuilds_owned_host_views<VlmForwardInput>();
}

TEST(WorkerServiceShutdownTest, JoinsIdleShmPollingThreadForEveryBackend) {
  for (const std::string& backend : {"llm", "vlm", "rec", "dit"}) {
    SCOPED_TRACE(backend);
    runtime::Options options;
    options.backend(backend);
    auto service =
        std::make_unique<WorkerService>(options, torch::Device("npu:0"));
    const std::string prefix =
        "worker_shutdown_" + std::to_string(getpid()) + "_" + backend;
    bool is_creator = false;
    auto input =
        std::make_unique<ForwardSharedMemoryManager>(prefix + "_input",
                                                     /*size=*/1 << 20,
                                                     is_creator,
                                                     ForwardType::RAW_INPUT);
    auto output =
        std::make_unique<ForwardSharedMemoryManager>(prefix + "_output",
                                                     /*size=*/1 << 20,
                                                     is_creator,
                                                     ForwardType::RAW_OUTPUT);
    service->create_polling_shm_thread(std::move(input), std::move(output));
    // No input is ever published: destruction must cancel the read and join,
    // including when shutdown races with the polling thread's first read.
    service.reset();
  }
}

TEST(WorkerServiceShutdownTest, StopsWithoutShmPollingThread) {
  runtime::Options options;
  WorkerService service(options, torch::Device("npu:0"));
}

}  // namespace
}  // namespace xllm
