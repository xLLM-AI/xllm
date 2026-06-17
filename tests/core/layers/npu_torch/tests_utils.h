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

#pragma once

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"

namespace xllm {
namespace layer {
namespace test {

class CompletedWork : public c10d::Work {
 public:
  CompletedWork() { finish(); }
};

inline c10::intrusive_ptr<c10d::Work> make_completed_work() {
  return c10::make_intrusive<CompletedWork>();
}

// Mock Backend for testing - minimal implementation for tp=1 tests
class MockBackend final : public c10d::Backend {
 public:
  MockBackend(int64_t rank, int64_t world_size)
      : c10d::Backend(rank, world_size), rank_(rank), world_size_(world_size) {}

  c10::intrusive_ptr<c10d::Work> allreduce(
      std::vector<torch::Tensor>& tensors,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override {
    // Mock implementation - return a completed work
    return make_completed_work();
  }

  c10::intrusive_ptr<c10d::Work> allgather(
      std::vector<std::vector<torch::Tensor>>& outputTensors,
      std::vector<torch::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override {
    // Mock implementation - return a completed work
    return make_completed_work();
  }

  c10::intrusive_ptr<c10d::Work> barrier(
      const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override {
    return make_completed_work();
  }

  c10::intrusive_ptr<c10d::Work> broadcast(
      std::vector<torch::Tensor>& tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override {
    return make_completed_work();
  }

  c10::intrusive_ptr<c10d::Work> reduce(
      std::vector<torch::Tensor>& tensors,
      const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override {
    return make_completed_work();
  }

  c10::intrusive_ptr<c10d::Work> allgather_coalesced(
      std::vector<std::vector<torch::Tensor>>& outputTensorLists,
      std::vector<torch::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override {
    return make_completed_work();
  }

  c10::intrusive_ptr<c10d::Work> reduce_scatter(
      std::vector<torch::Tensor>& outputTensors,
      std::vector<std::vector<torch::Tensor>>& inputTensors,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override {
    return make_completed_work();
  }

  c10::intrusive_ptr<c10d::Work> alltoall_base(
      torch::Tensor& outputTensor,
      torch::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override {
    return make_completed_work();
  }

  c10::intrusive_ptr<c10d::Work> alltoall(
      std::vector<torch::Tensor>& outputTensors,
      std::vector<torch::Tensor>& inputTensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override {
    return make_completed_work();
  }

  c10::intrusive_ptr<c10d::Work> send(std::vector<torch::Tensor>& tensors,
                                      int dstRank,
                                      int tag) override {
    return make_completed_work();
  }

  c10::intrusive_ptr<c10d::Work> recv(std::vector<torch::Tensor>& tensors,
                                      int srcRank,
                                      int tag) override {
    return make_completed_work();
  }

  c10::intrusive_ptr<c10d::Work> recvAnysource(
      std::vector<torch::Tensor>& tensors,
      int tag) override {
    return make_completed_work();
  }

  int64_t getRank() const { return rank_; }

  int64_t getSize() const { return world_size_; }

  void shutdown() override {
    // Mock implementation - do nothing
  }

 private:
  int64_t rank_;
  int64_t world_size_;
};

// Mock ProcessGroup for testing
class MockProcessGroup : public xllm::ProcessGroup {
 public:
  MockProcessGroup(const torch::Device& device,
                   int64_t rank = 0,
                   int64_t world_size = 1)
      : xllm::ProcessGroup(rank, world_size, device) {
    // Initialize pg_ with a mock backend for testing
    pg_ = std::make_unique<MockBackend>(rank, world_size);
  }

  void allreduce(torch::Tensor& input) override {
    // Mock implementation - do nothing for testing
  }

  c10::intrusive_ptr<c10d::Work> allreduce_async(
      torch::Tensor& input) override {
    allreduce(input);
    return make_completed_work();
  }

  void allgather(const torch::Tensor& input,
                 std::vector<torch::Tensor>& outputs) override {
    outputs.resize(this->world_size());
    if (!allgather_outputs_.empty()) {
      CHECK_EQ(allgather_outputs_.size(), outputs.size())
          << "mock allgather outputs size mismatch";
      for (size_t i = 0; i < outputs.size(); ++i) {
        outputs[i] = allgather_outputs_[i].clone();
      }
      return;
    }

    // Mock implementation - just copy input to outputs
    for (size_t i = 0; i < this->world_size(); ++i) {
      outputs[i] = input.clone();
    }
  }

  c10::intrusive_ptr<c10d::Work> allgather_async(
      const torch::Tensor& input,
      std::vector<torch::Tensor>& outputs) override {
    allgather(input, outputs);
    return make_completed_work();
  }

  c10::intrusive_ptr<c10d::Work> allgather_base_async(
      const torch::Tensor& input,
      torch::Tensor& output) override {
    CHECK(output.defined()) << "mock allgather_base_async requires output";
    CHECK_EQ(output.size(0), this->world_size())
        << "mock allgather_base_async world_size mismatch";

    if (!allgather_outputs_.empty()) {
      CHECK_EQ(allgather_outputs_.size(), static_cast<size_t>(output.size(0)))
          << "mock allgather_base outputs size mismatch";
      for (int64_t i = 0; i < output.size(0); ++i) {
        output[i].copy_(allgather_outputs_[i]);
      }
      return make_completed_work();
    }

    for (int64_t i = 0; i < output.size(0); ++i) {
      output[i].copy_(input);
    }
    return make_completed_work();
  }

  void reduce_scatter(const torch::Tensor& input,
                      torch::Tensor& output) override {
    int64_t world_size = this->world_size();
    int64_t chunk_size = input.size(0) / world_size;
    int64_t start = this->rank() * chunk_size;
    output.copy_(input.slice(0, start, start + chunk_size));
  }

  void set_allgather_outputs(std::vector<torch::Tensor> outputs) {
    allgather_outputs_ = std::move(outputs);
  }

 private:
  std::vector<torch::Tensor> allgather_outputs_;
};

// Helper function to create custom input tensor for precision testing
torch::Tensor create_custom_input(const std::vector<int64_t>& shape,
                                  const std::vector<float>& values,
                                  const torch::TensorOptions& options);

// Helper function to verify tensor values are close to expected
void verify_tensor_close(const torch::Tensor& actual,
                         const torch::Tensor& expected,
                         double rtol = 1e-5,
                         double atol = 1e-8);

// Helper function to verify precision against expected output
void verify_precision(const torch::Tensor& actual_output,
                      const std::vector<float>& expected_values,
                      double rtol = 1e-3,
                      double atol = 1e-4);

// Expect tensor's min, max, sum (computed in fp32) to match expected values
// within tolerance. Uses atol + rtol * |expected| for each of min, max, sum.
void expect_tensor_stats(const torch::Tensor& t,
                         double expected_min,
                         double expected_max,
                         double expected_sum,
                         double rtol = 1e-2,
                         double atol = 1e-5);

// Helper function to create default model arguments for testing
ModelArgs create_default_model_args();

// Helper function to create default quantization arguments for testing
QuantArgs create_default_quant_args();

// Helper function to create default parallel arguments for testing
ParallelArgs create_default_parallel_args(
    std::unique_ptr<xllm::ProcessGroup>& mock_process_group);

// create a tensor with a seeded random number generator (based on key and
// shape) It is robust enough to generate the same tensor across any device or
// os
torch::Tensor seeded_tensor(const std::string& key,
                            torch::IntArrayRef shape,
                            torch::ScalarType dtype = torch::kFloat,
                            torch::Device device = torch::Device(torch::kCPU));

void append_w4a8_expert_weights(
    std::unordered_map<std::string, torch::Tensor>& weight_dict,
    const std::string& expert_prefix,
    const std::string& seed_prefix,
    int64_t hidden_size,
    int64_t gate_up_intermediate_size,
    int64_t down_qweight_intermediate_size,
    int64_t down_scale_intermediate_size,
    int64_t group_size,
    const torch::Device& device);

}  // namespace test
}  // namespace layer
}  // namespace xllm
