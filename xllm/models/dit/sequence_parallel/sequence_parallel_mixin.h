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

#include "models/dit/sequence_parallel/sequence_parallel_context.h"

namespace xllm::dit {

using SequenceParallelTensorMap =
    std::unordered_map<std::string, torch::Tensor>;
using SequenceParallelTensorDims = std::unordered_map<std::string, int64_t>;

class SequenceParallelMixin {
 protected:
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
    SequenceParallelContext context(process_group_);
    SequenceParallelTensorMap local_inputs = inputs;
    for (const auto& [tensor_name, sequence_dim] : input_sequence_dims_) {
      auto tensor_it = local_inputs.find(tensor_name);
      CHECK(tensor_it != local_inputs.end())
          << "Missing registered sequence-parallel input: " << tensor_name;
      tensor_it->second = context.scatter_sequence(
          tensor_it->second, tensor_name, sequence_dim);
    }

    SequenceParallelTensorMap outputs =
        std::forward<ForwardFn>(forward_fn)(local_inputs, context);
    for (const auto& [tensor_name, sequence_dim] : output_sequence_dims_) {
      auto tensor_it = outputs.find(tensor_name);
      CHECK(tensor_it != outputs.end())
          << "Missing registered sequence-parallel output: " << tensor_name;
      tensor_it->second =
          context.gather_sequence(tensor_it->second, tensor_name, sequence_dim);
    }
    return outputs;
  }

 private:
  ProcessGroup* process_group_{nullptr};
  SequenceParallelTensorDims input_sequence_dims_;
  SequenceParallelTensorDims output_sequence_dims_;
};

}  // namespace xllm::dit
