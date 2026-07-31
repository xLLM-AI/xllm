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
#include <vector>

#include "core/framework/parallel_state/parallel_state.h"

namespace xllm::dit {

using SequenceParallelTensorMap =
    std::unordered_map<std::string, torch::Tensor>;
using SequenceParallelTensorDims = std::unordered_map<std::string, int64_t>;

class SequenceParallelMixin {
 public:
  static torch::Tensor pad_tensor(const torch::Tensor& input,
                                  const std::string& tensor_name,
                                  int64_t dim) {
    if (!input.defined()) {
      return input;
    }

    return pad_right(input, padding_length(tensor_name), dim);
  }

  static torch::Tensor unpad_tensor(const torch::Tensor& input,
                                    const std::string& tensor_name,
                                    int64_t dim) {
    if (!input.defined()) {
      return input;
    }

    return unpad_right(input, padding_length(tensor_name), dim);
  }

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
      ForwardFn&& forward_fn) {
    padding_lengths_.clear();
    SequenceParallelTensorMap local_inputs = inputs;
    for (const auto& [tensor_name, sequence_dim] : input_sequence_dims_) {
      auto tensor_it = local_inputs.find(tensor_name);
      CHECK(tensor_it != local_inputs.end())
          << "Missing registered sequence-parallel input: " << tensor_name;
      tensor_it->second =
          scatter_sequence(tensor_it->second, tensor_name, sequence_dim);
    }

    SequenceParallelTensorMap outputs =
        std::forward<ForwardFn>(forward_fn)(local_inputs);
    for (const auto& [tensor_name, sequence_dim] : output_sequence_dims_) {
      auto tensor_it = outputs.find(tensor_name);
      CHECK(tensor_it != outputs.end())
          << "Missing registered sequence-parallel output: " << tensor_name;
      tensor_it->second =
          gather_sequence(tensor_it->second, tensor_name, sequence_dim);
    }
    return outputs;
  }

 private:
  int32_t world_size() const {
    return process_group_ == nullptr ? 1 : process_group_->world_size();
  }

  bool sequence_parallel_enabled() const { return world_size() > 1; }

  torch::Tensor scatter_sequence(const torch::Tensor& input,
                                 const std::string& tensor_name,
                                 int64_t sequence_dim) {
    if (!input.defined()) {
      return input;
    }

    const int64_t sequence_length = input.size(sequence_dim);
    const int64_t padding_length =
        (world_size() - sequence_length % world_size()) % world_size();
    padding_lengths_[tensor_name] = padding_length;
    if (!sequence_parallel_enabled()) {
      return input;
    }

    torch::Tensor padded_input = pad_tensor(input, tensor_name, sequence_dim);
    return parallel_state::scatter(
        padded_input, process_group_, static_cast<int32_t>(sequence_dim));
  }

  torch::Tensor gather_sequence(const torch::Tensor& input,
                                const std::string& tensor_name,
                                int64_t sequence_dim) const {
    if (!sequence_parallel_enabled() || !input.defined()) {
      return input;
    }

    torch::Tensor output = parallel_state::gather(
        input.contiguous(), process_group_, static_cast<int32_t>(sequence_dim));
    return unpad_tensor(output, tensor_name, sequence_dim);
  }

  static int64_t normalize_dim(const torch::Tensor& input, int64_t dim) {
    const int64_t normalized_dim = dim < 0 ? input.dim() + dim : dim;
    CHECK_GE(normalized_dim, 0) << "Invalid tensor dimension: " << dim;
    CHECK_LT(normalized_dim, input.dim())
        << "Invalid tensor dimension: " << dim;
    return normalized_dim;
  }

  static torch::Tensor pad_right(const torch::Tensor& input,
                                 int64_t padding_length,
                                 int64_t dim) {
    if (!input.defined() || padding_length == 0) {
      return input;
    }

    const int64_t normalized_dim = normalize_dim(input, dim);
    std::vector<int64_t> padding(static_cast<size_t>(input.dim() * 2), 0);
    const int64_t padding_index = 2 * (input.dim() - normalized_dim - 1) + 1;
    padding[static_cast<size_t>(padding_index)] = padding_length;
    return torch::pad(input, padding, "constant", 0);
  }

  static torch::Tensor unpad_right(const torch::Tensor& input,
                                   int64_t padding_length,
                                   int64_t dim) {
    if (!input.defined() || padding_length == 0) {
      return input;
    }

    const int64_t normalized_dim = normalize_dim(input, dim);
    CHECK_GE(input.size(normalized_dim), padding_length)
        << "Padding length exceeds tensor size";
    return input.narrow(normalized_dim,
                        /*start=*/0,
                        input.size(normalized_dim) - padding_length);
  }

  static int64_t padding_length(const std::string& tensor_name) {
    auto padding_it = padding_lengths_.find(tensor_name);
    CHECK(padding_it != padding_lengths_.end())
        << "Missing sequence-parallel padding metadata: " << tensor_name;
    return padding_it->second;
  }

  ProcessGroup* process_group_{nullptr};
  const SequenceParallelTensorDims input_sequence_dims_;
  const SequenceParallelTensorDims output_sequence_dims_;
  inline static SequenceParallelTensorDims padding_lengths_;
};

}  // namespace xllm::dit
