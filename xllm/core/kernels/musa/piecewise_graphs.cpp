/* Copyright 2025-2026 The xLLM Authors.

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
#include "core/kernels/musa/piecewise_graphs.h"

#include "core/kernels/musa/attention_runner.h"

namespace xllm::runtime::cuda {

void PiecewiseGraphs::add_graph(std::unique_ptr<at::cuda::CUDAGraph>&& graph) {
  graphs_.emplace_back(std::move(graph));
  instructions_.emplace_back(InstructionType::kGraph);
}

void PiecewiseGraphs::add_attention_runner(
    ::xllm::kernel::cuda::AttentionRunner&& runner) {
  attention_runners_.emplace_back(
      std::make_unique<::xllm::kernel::cuda::AttentionRunner>(
          std::move(runner)));
  instructions_.emplace_back(InstructionType::kRunner);
}

size_t PiecewiseGraphs::num_runners() const {
  return attention_runners_.size();
}

void PiecewiseGraphs::replay(
    const ::xllm::kernel::cuda::AttentionReplayParams& runner_params) {
  if (instructions_.empty()) {
    return;
  }

  size_t graph_idx = 0;
  size_t runner_idx = 0;

  for (const auto& instruction : instructions_) {
    if (instruction == InstructionType::kGraph) {
      CHECK_LT(graph_idx, graphs_.size()) << "Graph index out of range";
      graphs_[graph_idx]->replay();
      ++graph_idx;
    } else {
      CHECK_LT(runner_idx, attention_runners_.size())
          << "Runner index out of range";
      attention_runners_[runner_idx]->run_replay(runner_params);
      ++runner_idx;
    }
  }

  CHECK_EQ(graph_idx, graphs_.size()) << "Not all graphs were replayed";
  CHECK_EQ(runner_idx, attention_runners_.size())
      << "Not all runners were replayed";
}

}  // namespace xllm::runtime::cuda
