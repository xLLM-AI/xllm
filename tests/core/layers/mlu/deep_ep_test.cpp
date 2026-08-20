/* Copyright 2025-2026 The xLLM Authors.

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

#include "layers/common/deep_ep.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "framework/parallel_state/parallel_args.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "util/tensor_helper.h"

#if defined(USE_MLU)
#include "framework/parallel_state/mlu_process_group.h"
#elif defined(USE_CUDA)
#include "framework/parallel_state/cuda_process_group.h"
#endif

namespace xllm {
namespace layer {
namespace test {

// Special exit code definition for skipping test
constexpr int32_t EXIT_CODE_SKIP = 77;

// Helper function to create ProcessGroup
std::unique_ptr<xllm::ProcessGroup> create_test_process_group(
    int32_t rank,
    int32_t world_size,
    int32_t port,
    const std::string& host,
    const torch::Device& device) {
  return xllm::create_process_group(rank,
                                    world_size,
                                    world_size,
                                    port,
                                    false,
                                    host,
                                    "deep_ep_test_group",
                                    device);
}

struct TestParams {
  int32_t rank;
  int32_t world_size;
  int32_t port;
  std::string host;
  int32_t device_index;
  int64_t token_size;
  int64_t max_tokens;
  int64_t num_experts;
};

// Child process test function
int32_t run_deep_ep_test_child(TestParams params) {
  try {
    // 1. Check devices
    int32_t dev_count = xllm::Platform::device_count();
    if (dev_count < params.world_size) {
      LOG(WARNING) << "Rank " << params.rank
                   << ": Insufficient devices. Skipping.";
      return EXIT_CODE_SKIP;
    }
    params.device_index = params.rank % dev_count;

    // 2. Set device
    xllm::Device xllm_device(params.device_index);
    xllm_device.set_device();
    torch::Device device = xllm_device.unwrap();

    // 3. Create ProcessGroup
    auto process_group = create_test_process_group(
        params.rank, params.world_size, params.port, params.host, device);

    CHECK(process_group) << "Rank " << params.rank
                         << ": Failed to create ProcessGroup";

    // 4. Create ParallelArgs
    ParallelArgs parallel_args(
        params.rank, params.world_size, process_group.get());
    parallel_args.moe_ep_group_ = process_group.get();
    parallel_args.ep_size_ = params.world_size;

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(device)
                       .requires_grad(false);

    // 5. Initialize DeepEPImpl
    DeepEPImpl deep_ep(params.token_size,
                       params.token_size,
                       params.max_tokens,
                       params.num_experts,
                       parallel_args,
                       options);

    LOG(INFO) << "Rank " << params.rank << ": DeepEP created successfully";

    const int64_t hidden_dim = params.token_size / sizeof(float);  // 32
    const int64_t num_tokens_sent = 4;

    // Prepare Layout
    torch::Tensor token_count_slice = torch::zeros(
        {params.num_experts},
        torch::TensorOptions().dtype(torch::kInt32).device(device));

    float base_val = 0.0f;

    if (params.rank == 0) {
      token_count_slice[0] = 2;
      token_count_slice[2] = 2;
      base_val = 10.0f;
    } else {
      token_count_slice[1] = 2;
      token_count_slice[3] = 2;
      base_val = 20.0f;
    }

    // Step 1: Prepare Input Data
    auto buffers = deep_ep.get_buffer();

    torch::Tensor src_data = torch::zeros(
        {num_tokens_sent, hidden_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    for (size_t i = 0; i < num_tokens_sent; ++i) {
      src_data[i].fill_(base_val + i);
    }

    torch::Tensor src_data_bytes =
        src_data.to(device).view({-1}).view(torch::kUInt8);
    CHECK(src_data_bytes.numel() <= buffers.dispatch_send_token_tensor.numel());
    buffers.dispatch_send_token_tensor.view({-1})
        .slice(0, 0, src_data_bytes.numel())
        .copy_(src_data_bytes);

    // Step 2: Dispatch Step
    deep_ep.dispatch_step(num_tokens_sent, token_count_slice);

    // Step 3: Process Dispatch Result
    int64_t num_experts_per_rank = params.num_experts / params.world_size;
    int64_t max_recv_capacity = params.max_tokens * params.world_size;

    // Allocate Expert Input as Float
    torch::Tensor expert_input =
        torch::zeros({max_recv_capacity, hidden_dim}, options);

    // Convert to Int8 View to match DeepEP internal buffer dtype
    // DeepEP buffer is Int8, so we must pass a Int8 tensor to gather_split
    torch::Tensor expert_input_byte = view_as_dtype(expert_input, torch::kInt8);

    auto meta = deep_ep.process_dispatch_result(
        num_experts_per_rank, expert_input_byte, std::nullopt);

    // Validate count
    torch::Tensor valid_token_num_cpu = meta.token_sum.to(torch::kCPU);
    int64_t received_count = valid_token_num_cpu.item<int64_t>();
    CHECK_EQ(received_count, 4)
        << "Rank " << params.rank << ": Expected 4 received tokens, got "
        << received_count;

    // Step 4: Simulate Computation (on Float tensor)
    torch::Tensor expert_output = expert_input + 1.0f;

    // Step 5: Combine Pack
    // Convert to Int8 View to match DeepEP internal buffer dtype
    torch::Tensor expert_output_byte =
        view_as_dtype(expert_output, torch::kInt8);

    torch::Tensor combine_layout =
        deep_ep.combine_step_pack(expert_output_byte,
                                  meta.gather_rank_index,
                                  meta.token_sum,
                                  params.token_size,
                                  torch::kInt8);

    // Step 6: Combine Comm
    torch::Tensor final_output = deep_ep.combine_step_comm(
        combine_layout, num_tokens_sent, hidden_dim, torch::kFloat32);

    // Step 7: Verification
    xllm_device.synchronize_default_stream();

    torch::Tensor final_cpu = final_output.to(torch::kCPU);
    auto final_acc = final_cpu.accessor<float, 2>();

    for (int64_t i = 0; i < num_tokens_sent; ++i) {
      float expected_val = base_val + i + 1.0f;
      for (int64_t j = 0; j < hidden_dim; ++j) {
        float got = final_acc[i][j];
        CHECK(std::abs(got - expected_val) <= 1e-4)
            << "Rank " << params.rank << " Verification Failed at [" << i
            << "][" << j << "]: Expected " << expected_val << ", Got " << got;
      }
    }

    LOG(INFO) << "Rank " << params.rank << ": E2E Test Passed!";
    return 0;

  } catch (const std::exception& e) {
    LOG(ERROR) << "Rank " << params.rank << ": Exception: " << e.what();
    return 1;
  }
}

// Multi-process test fixture
class DeepEPMultiDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    world_size_ = 2;
    port_ = 29500;
    host_ = "127.0.0.1";
    token_size_ = 128;
    max_tokens_ = 64;
    num_global_experts_ = 4;
  }

  void run_test() {
    std::vector<pid_t> child_pids;

    for (int32_t rank = 0; rank < world_size_; ++rank) {
      pid_t pid = fork();
      if (pid == 0) {
        TestParams params;
        params.rank = rank;
        params.world_size = world_size_;
        params.port = port_;
        params.host = host_;
        params.device_index = -1;
        params.token_size = token_size_;
        params.max_tokens = max_tokens_;
        params.num_experts = num_global_experts_;

        int32_t exit_code = run_deep_ep_test_child(params);
        _exit(exit_code);
      } else if (pid > 0) {
        child_pids.push_back(pid);
      } else {
        LOG(FATAL) << "Failed to fork rank " << rank;
      }
    }

    bool any_failed = false;
    bool any_skipped = false;

    for (size_t i = 0; i < child_pids.size(); ++i) {
      int32_t status;
      waitpid(child_pids[i], &status, 0);
      if (WIFEXITED(status)) {
        int32_t exit_code = WEXITSTATUS(status);
        if (exit_code == EXIT_CODE_SKIP) {
          any_skipped = true;
        } else if (exit_code != 0) {
          any_failed = true;
          LOG(ERROR) << "Rank " << i << " failed with code " << exit_code;
        }
      } else {
        any_failed = true;
        LOG(ERROR) << "Rank " << i << " crashed (signal).";
      }
    }

    if (any_skipped) {
      GTEST_SKIP() << "Test skipped due to insufficient devices.";
    } else {
      ASSERT_FALSE(any_failed) << "DeepEP End-to-End Test Failed.";
    }
  }

  int32_t world_size_;
  int32_t port_;
  std::string host_;
  int64_t token_size_;
  int64_t max_tokens_;
  int64_t num_global_experts_;
};

TEST_F(DeepEPMultiDeviceTest, EndToEndFlow) { run_test(); }

}  // namespace test
}  // namespace layer
}  // namespace xllm
