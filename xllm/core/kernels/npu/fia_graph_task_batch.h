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

#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <torch/custom_class.h>
#include <torch/types.h>
#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <torch_npu/csrc/core/npu/NPUGraph.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace xllm::npu {

// One model graph owns its task handles, tensors and immutable operator
// arguments. Only the shared effective KV lengths change during replay.
class FiaGraphTaskBatch final : public torch::CustomClassHolder {
 public:
  FiaGraphTaskBatch(int64_t num_heads,
                    int64_t num_kv_heads,
                    double scale,
                    int64_t block_size,
                    bool use_v2,
                    std::vector<int64_t> query_ends,
                    std::vector<int64_t> kv_lengths);

  int64_t capture(torch::Tensor query,
                  torch::Tensor key,
                  torch::Tensor value,
                  torch::Tensor block_table,
                  torch::Tensor workspace,
                  torch::Tensor output,
                  torch::Tensor lse);

  void update(int64_t begin,
              int64_t end,
              const std::vector<int64_t>& kv_lengths);
  int64_t size() const;

 private:
  struct Task {
    torch::jit::Stack arguments;
    c10_npu::NPUTaskGroupHandle handle;
    std::shared_ptr<c10_npu::NPUEvent> event;
  };

  void invoke(const torch::jit::Stack& arguments) const;

  c10::OperatorHandle operation_;
  torch::jit::Stack defaults_;
  c10::List<int64_t> kv_lengths_;
  size_t query_index_;
  size_t key_index_;
  size_t value_index_;
  size_t block_table_index_;
  size_t workspace_index_;
  size_t output_index_;
  std::vector<Task> tasks_;
};

}  // namespace xllm::npu
