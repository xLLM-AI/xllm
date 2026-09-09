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
#pragma once
#include <ATen/cuda/CUDAGraph.h>
#include <glog/logging.h>

#include <memory>
#include <vector>

#include "attention_runner.h"

namespace xllm::runtime::cuda {

// Encapsulate multiple piecewise CUDA graphs, provide unified replay interface
class PiecewiseGraphs {
 public:
  // Instruction type for replay sequence
  enum class InstructionType { kGraph, kRunner };

  PiecewiseGraphs() = default;
  ~PiecewiseGraphs() = default;
  PiecewiseGraphs(PiecewiseGraphs&&) noexcept = default;
  PiecewiseGraphs& operator=(PiecewiseGraphs&&) noexcept = default;

  // Add a graph and record instruction
  void add_graph(std::unique_ptr<at::cuda::CUDAGraph>&& graph);
  // Add attention runner and record instruction
  void add_attention_runner(::xllm::kernel::cuda::AttentionRunner&& runner);
  // Replay all graphs and runners according to recorded instruction sequence
  void replay(const ::xllm::kernel::cuda::AttentionReplayParams& runner_params);
  // Get number of graphs
  size_t size() const { return graphs_.size(); }
  // Check if empty
  bool empty() const { return graphs_.empty(); }
  // Get runner count
  size_t num_runners() const;

 private:
  std::vector<std::unique_ptr<at::cuda::CUDAGraph>> graphs_;
  std::vector<std::unique_ptr<::xllm::kernel::cuda::AttentionRunner>>
      attention_runners_;
  // Recorded instruction sequence for replay
  std::vector<InstructionType> instructions_;
};

}  // namespace xllm::runtime::cuda
