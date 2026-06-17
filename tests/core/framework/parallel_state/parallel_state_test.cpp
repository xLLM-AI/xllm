/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "framework/parallel_state/parallel_state.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <cstring>
#include <memory>
#include <vector>

#include "platform/device.h"

#if defined(USE_MLU)
#include "framework/parallel_state/mlu_process_group.h"
#elif defined(USE_CUDA) || defined(USE_DCU)
#include "framework/parallel_state/cuda_process_group.h"
#endif

namespace xllm {
namespace parallel_state {
namespace test {

// Helper function to create ProcessGroup for multi-device testing
std::unique_ptr<xllm::ProcessGroup> create_test_process_group(
    int rank,
    int world_size,
    int port,
    const std::string& host,
    const torch::Device& device) {
  return xllm::create_process_group(static_cast<int32_t>(rank),
                                    static_cast<int32_t>(world_size),
                                    static_cast<int32_t>(world_size),
                                    static_cast<int32_t>(port),
                                    false,
                                    host,
                                    "reduce_scatter_test_group",
                                    device);
}

// Test parameters structure to pass to child processes
struct TestParams {
  int32_t rank;
  int32_t world_size;
  int32_t port;
  std::string host;
  int32_t device_index;
  int64_t input_size;
  int64_t hidden_dim;
  bool test_padding;
};

struct AllGatherBaseTestParams {
  int32_t rank;
  int32_t world_size;
  int32_t port;
  std::string host;
  int32_t device_index;
  int64_t input_size;
  int64_t hidden_dim;
};
// Child process test function
int run_reduce_scatter_test_child(const TestParams& params) {
  try {
    // Set device
    xllm::Device xllm_device(params.device_index);
    xllm_device.set_device();
    torch::Device device = xllm_device.unwrap();

    // Create ProcessGroup
    auto process_group = create_test_process_group(
        params.rank, params.world_size, params.port, params.host, device);

    if (!process_group) {
      LOG(ERROR) << "Rank " << params.rank << ": Failed to create ProcessGroup";
      return 1;
    }

    LOG(INFO) << "Rank " << params.rank
              << ": ProcessGroup created successfully";

    // Create tensor options
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(device)
                       .requires_grad(false);

    // Create test input tensor
    // Each rank creates a tensor with values equal to (rank + 1)
    // This allows us to verify that reduce_scatter correctly sums the values
    torch::Tensor input = torch::full({params.input_size, params.hidden_dim},
                                      static_cast<float>(params.rank + 1),
                                      options);

    LOG(INFO) << "Rank " << params.rank << ": Input tensor created, shape: ["
              << params.input_size << ", " << params.hidden_dim << "]";

    // Perform reduce_scatter
    torch::Tensor output = reduce_scatter(input, process_group.get());

    // Synchronize device
    xllm_device.synchronize_default_stream();

    LOG(INFO) << "Rank " << params.rank
              << ": reduce_scatter completed, output shape: [" << output.size(0)
              << ", " << output.size(1) << "]";

    // Verify output
    // After reduce_scatter, each rank should receive a chunk of the reduced
    // tensor
    // The reduced value should be the sum across all ranks: (1 + 2 + ... +
    // world_size) = world_size * (world_size + 1) / 2
    float expected_value =
        static_cast<float>(params.world_size * (params.world_size + 1) / 2);

    // Calculate expected chunk size
    int64_t padded_size = params.input_size;
    if (params.test_padding && params.input_size % params.world_size != 0) {
      int64_t remainder = params.input_size % params.world_size;
      padded_size = params.input_size + params.world_size - remainder;
    }
    int64_t chunk_size = padded_size / params.world_size;
    int64_t expected_output_size = chunk_size;
    if (params.test_padding && params.input_size % params.world_size != 0) {
      int64_t global_start = params.rank * chunk_size;
      int64_t global_end = global_start + chunk_size;
      if (global_start >= params.input_size) {
        expected_output_size = 0;
      } else if (global_end > params.input_size) {
        expected_output_size = params.input_size - global_start;
      }
    }

    // Move output to CPU for verification
    torch::Tensor output_cpu = output.to(torch::kCPU);

    // Verify output size
    if (expected_output_size == 0) {
      if (output_cpu.size(0) != 0) {
        LOG(ERROR) << "Rank " << params.rank
                   << ": Expected empty output, but got size: "
                   << output_cpu.size(0);
        return 1;
      }
      LOG(INFO) << "Rank " << params.rank
                << ": Output is empty as expected (padding case)";
      return 0;
    }

    if (output_cpu.size(0) != expected_output_size) {
      LOG(ERROR) << "Rank " << params.rank
                 << ": Output size mismatch. Expected: " << expected_output_size
                 << ", got: " << output_cpu.size(0);
      return 1;
    }

    if (output_cpu.size(1) != params.hidden_dim) {
      LOG(ERROR) << "Rank " << params.rank
                 << ": Output hidden_dim mismatch. Expected: "
                 << params.hidden_dim << ", got: " << output_cpu.size(1);
      return 1;
    }

    // Verify output values
    auto output_values = output_cpu.accessor<float, 2>();
    bool all_match = true;
    for (int64_t i = 0; i < expected_output_size; ++i) {
      for (int64_t j = 0; j < params.hidden_dim; ++j) {
        if (std::abs(output_values[i][j] - expected_value) > 1e-5) {
          all_match = false;
          LOG(ERROR) << "Rank " << params.rank << ": Mismatch at [" << i << "]["
                     << j << "]: expected=" << expected_value
                     << ", got=" << output_values[i][j];
        }
      }
    }

    if (!all_match) {
      LOG(ERROR) << "Rank " << params.rank
                 << ": reduce_scatter test failed - output values don't match";
      return 1;
    }

    LOG(INFO) << "Rank " << params.rank << ": reduce_scatter test passed";
    return 0;

  } catch (const std::exception& e) {
    LOG(ERROR) << "Rank " << params.rank << ": Exception: " << e.what();
    return 1;
  }
}

int run_allgather_base_test_child(const AllGatherBaseTestParams& params) {
  try {
    xllm::Device xllm_device(params.device_index);
    xllm_device.set_device();
    torch::Device device = xllm_device.unwrap();

    auto process_group = create_test_process_group(
        params.rank, params.world_size, params.port, params.host, device);
    if (!process_group) {
      LOG(ERROR) << "Rank " << params.rank << ": Failed to create ProcessGroup";
      return 1;
    }

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(device)
                       .requires_grad(false);
    auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor input =
        torch::arange(params.input_size * params.hidden_dim, cpu_options)
            .reshape({params.input_size, params.hidden_dim})
            .to(device) +
        static_cast<float>(params.rank * 1000);

    torch::Tensor stacked = process_group->allgather_base_sync(input);
    torch::Tensor async_stacked = torch::empty_like(stacked);
    process_group->allgather_base_async(input, async_stacked)->wait();
    xllm_device.synchronize_default_stream();
    if (stacked.dim() != 3 || stacked.size(0) != params.world_size ||
        stacked.size(1) != params.input_size ||
        stacked.size(2) != params.hidden_dim) {
      LOG(ERROR) << "Rank " << params.rank
                 << ": allgather_base output shape mismatch";
      return 1;
    }

    torch::Tensor stacked_cpu = stacked.to(torch::kCPU);
    for (int32_t src_rank = 0; src_rank < params.world_size; ++src_rank) {
      torch::Tensor expected =
          torch::arange(params.input_size * params.hidden_dim, cpu_options)
              .reshape({params.input_size, params.hidden_dim}) +
          static_cast<float>(src_rank * 1000);
      if (!torch::equal(stacked_cpu[src_rank], expected)) {
        LOG(ERROR) << "Rank " << params.rank
                   << ": allgather_base slice mismatch for src_rank="
                   << src_rank;
        return 1;
      }
    }
    if (!torch::equal(async_stacked.to(torch::kCPU), stacked_cpu)) {
      LOG(ERROR) << "Rank " << params.rank
                 << ": allgather_base_async result mismatch";
      return 1;
    }

    torch::Tensor gathered = gather(input, process_group.get(), 0);
    xllm_device.synchronize_default_stream();
    std::vector<torch::Tensor> gathered_parts;
    gathered_parts.reserve(params.world_size);
    for (int32_t src_rank = 0; src_rank < params.world_size; ++src_rank) {
      gathered_parts.push_back(stacked_cpu[src_rank]);
    }
    torch::Tensor expected_gather = torch::cat(gathered_parts, 0);
    if (!torch::equal(gathered.to(torch::kCPU), expected_gather)) {
      LOG(ERROR) << "Rank " << params.rank
                 << ": gather result mismatch after allgather_base";
      return 1;
    }

    auto gather_ctx = launch_gather(
        input,
        process_group.get(),
        std::vector<int32_t>(params.world_size, params.input_size));
    torch::Tensor gathered_async = finish_gather(std::move(gather_ctx));
    xllm_device.synchronize_default_stream();
    if (!torch::equal(gathered_async.to(torch::kCPU), expected_gather)) {
      LOG(ERROR) << "Rank " << params.rank
                 << ": async gather result mismatch for equal token_num_list";
      return 1;
    }

    return 0;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Rank " << params.rank << ": Exception: " << e.what();
    return 1;
  }
}
// Multi-process test fixture
class ReduceScatterMultiDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize test parameters
    world_size_ = 2;
    port_ = 29501;
    host_ = "127.0.0.1";
    input_size_ = 8;
    hidden_dim_ = 128;
  }

  void TearDown() override {
    // Clean up if needed
  }

  // Run test with multiple processes
  void RunMultiProcessTest(int64_t input_size, bool test_padding) {
    std::vector<pid_t> child_pids;
    std::vector<int> child_statuses(world_size_);

    // Fork child processes
    for (int32_t rank = 0; rank < world_size_; ++rank) {
      pid_t pid = fork();
      if (pid == 0) {
        // Child process
        TestParams params;
        params.rank = rank;
        params.world_size = world_size_;
        params.port = port_;
        params.host = host_;
        params.device_index = rank % Device::device_count();
        params.input_size = input_size;
        params.hidden_dim = hidden_dim_;
        params.test_padding = test_padding;

        int exit_code = run_reduce_scatter_test_child(params);
        _exit(exit_code);
      } else if (pid > 0) {
        // Parent process
        child_pids.push_back(pid);
      } else {
        // Fork failed
        LOG(FATAL) << "Failed to fork child process for rank " << rank;
      }
    }

    // Wait for all child processes to complete
    bool all_passed = true;
    for (size_t i = 0; i < child_pids.size(); ++i) {
      int status;
      pid_t waited_pid = waitpid(child_pids[i], &status, 0);
      if (waited_pid == child_pids[i]) {
        if (WIFEXITED(status)) {
          child_statuses[i] = WEXITSTATUS(status);
          if (child_statuses[i] != 0) {
            all_passed = false;
            LOG(ERROR) << "Child process for rank " << i << " exited with code "
                       << child_statuses[i];
          }
        } else {
          all_passed = false;
          LOG(ERROR) << "Child process for rank " << i
                     << " did not exit normally";
        }
      } else {
        all_passed = false;
        LOG(ERROR) << "Failed to wait for child process for rank " << i;
      }
    }

    CHECK(all_passed) << "One or more child processes failed";
  }

  int32_t world_size_;
  int32_t port_;
  std::string host_;
  int64_t input_size_;
  int64_t hidden_dim_;
};

class AllGatherBaseMultiDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    world_size_ = 2;
    port_ = 29531;
    host_ = "127.0.0.1";
    input_size_ = 4;
    hidden_dim_ = 6;
  }

  void RunMultiProcessTest(int64_t input_size, int64_t hidden_dim) {
    std::vector<pid_t> child_pids;
    std::vector<int> child_statuses(world_size_);

    for (int32_t rank = 0; rank < world_size_; ++rank) {
      pid_t pid = fork();
      if (pid == 0) {
        AllGatherBaseTestParams params;
        params.rank = rank;
        params.world_size = world_size_;
        params.port = port_;
        params.host = host_;
        params.device_index = rank % Device::device_count();
        params.input_size = input_size;
        params.hidden_dim = hidden_dim;
        _exit(run_allgather_base_test_child(params));
      } else if (pid > 0) {
        child_pids.push_back(pid);
      } else {
        LOG(FATAL) << "Failed to fork child process for rank " << rank;
      }
    }

    bool all_passed = true;
    for (size_t i = 0; i < child_pids.size(); ++i) {
      int status;
      pid_t waited_pid = waitpid(child_pids[i], &status, 0);
      if (waited_pid == child_pids[i]) {
        if (WIFEXITED(status)) {
          child_statuses[i] = WEXITSTATUS(status);
          if (child_statuses[i] != 0) {
            all_passed = false;
            LOG(ERROR) << "Child process for rank " << i << " exited with code "
                       << child_statuses[i];
          }
        } else {
          all_passed = false;
          LOG(ERROR) << "Child process for rank " << i
                     << " did not exit normally";
        }
      } else {
        all_passed = false;
        LOG(ERROR) << "Failed to wait for child process for rank " << i;
      }
    }

    CHECK(all_passed) << "One or more child processes failed";
  }

  int32_t world_size_ = 0;
  int32_t port_ = 0;
  std::string host_;
  int64_t input_size_ = 0;
  int64_t hidden_dim_ = 0;
};
TEST_F(ReduceScatterMultiDeviceTest, BasicTest) {
  // Test with input size divisible by world_size
  RunMultiProcessTest(8, false);
}

TEST_F(ReduceScatterMultiDeviceTest, PaddingTest) {
  // Test with input size not divisible by world_size (requires padding)
  RunMultiProcessTest(7, true);
}

TEST_F(ReduceScatterMultiDeviceTest, LargeInputTest) {
  // Test with larger input
  RunMultiProcessTest(32, false);
}

TEST_F(AllGatherBaseMultiDeviceTest, BasicTest) {
  RunMultiProcessTest(input_size_, hidden_dim_);
}
}  // namespace test
}  // namespace parallel_state
}  // namespace xllm
