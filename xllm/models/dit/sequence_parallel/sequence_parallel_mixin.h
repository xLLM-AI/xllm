/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>

#include "core/framework/parallel_state/parallel_state.h"
#include "models/dit/utils/sequence_parallel_pad_manager.h"

namespace xllm::dit {

using SequenceParallelTensorMap =
    std::unordered_map<std::string, torch::Tensor>;
using SequenceParallelTensorDims = std::unordered_map<std::string, int64_t>;

class SequenceParallelMixin {
 public:
  SequenceParallelMixin(ProcessGroup* process_group,
                        SequenceParallelTensorDims input_sequence_dims,
                        SequenceParallelTensorDims output_sequence_dims)
      : process_group_(process_group),
        input_sequence_dims_(std::move(input_sequence_dims)),
        output_sequence_dims_(std::move(output_sequence_dims)) {}

  template <typename ForwardFn>
  SequenceParallelTensorMap sequence_parallel_forward(
      const SequenceParallelTensorMap& inputs,
      ForwardFn&& forward_fn) const {
    SequenceParallelTensorMap split_inputs =
        split_sequence_parallel_inputs(inputs);
    SequenceParallelTensorMap outputs =
        std::forward<ForwardFn>(forward_fn)(split_inputs);
    return gather_sequence_parallel_outputs(outputs);
  }

 private:
  SequenceParallelTensorMap split_sequence_parallel_inputs(
      const SequenceParallelTensorMap& inputs) const {
    SequenceParallelTensorMap split_inputs = inputs;
    if (!sequence_parallel_enabled()) {
      return split_inputs;
    }

    for (const auto& [tensor_name, sequence_dim] : input_sequence_dims_) {
      auto tensor_it = split_inputs.find(tensor_name);
      CHECK(tensor_it != split_inputs.end())
          << "Missing registered sequence-parallel input: " << tensor_name;
      if (!tensor_it->second.defined()) {
        continue;
      }

      torch::Tensor padded_tensor =
          SequenceParallelPadManager::get_instance().pad_tensor(
              tensor_it->second, tensor_name, sequence_dim);
      tensor_it->second = parallel_state::scatter(
          padded_tensor, process_group_, static_cast<int32_t>(sequence_dim));
    }
    return split_inputs;
  }

  SequenceParallelTensorMap gather_sequence_parallel_outputs(
      const SequenceParallelTensorMap& outputs) const {
    SequenceParallelTensorMap gathered_outputs = outputs;
    if (!sequence_parallel_enabled()) {
      return gathered_outputs;
    }

    for (const auto& [tensor_name, sequence_dim] : output_sequence_dims_) {
      auto tensor_it = gathered_outputs.find(tensor_name);
      CHECK(tensor_it != gathered_outputs.end())
          << "Missing registered sequence-parallel output: " << tensor_name;
      if (!tensor_it->second.defined()) {
        continue;
      }

      tensor_it->second =
          parallel_state::gather(tensor_it->second.contiguous(),
                                 process_group_,
                                 static_cast<int32_t>(sequence_dim));
      SequenceParallelPadManager::get_instance().unpad_tensor(
          tensor_it->second, tensor_name, sequence_dim);
    }
    return gathered_outputs;
  }

  bool sequence_parallel_enabled() const {
    return process_group_ != nullptr && process_group_->world_size() > 1;
  }

  ProcessGroup* process_group_{nullptr};
  SequenceParallelTensorDims input_sequence_dims_;
  SequenceParallelTensorDims output_sequence_dims_;
};

}  // namespace xllm::dit
