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
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/parallel_state.h"

namespace xllm::dit {

using SequenceParallelWork = std::function<torch::Tensor()>;

class SequenceParallelContext final {
 public:
  explicit SequenceParallelContext(ProcessGroup* process_group)
      : process_group_(process_group) {}

  int32_t world_size() const {
    return process_group_ == nullptr ? 1 : process_group_->world_size();
  }

  bool enabled() const { return world_size() > 1; }

  torch::Tensor scatter_sequence(const torch::Tensor& input,
                                 const std::string& tensor_name,
                                 int64_t sequence_dim) {
    if (!enabled() || !input.defined()) {
      return input;
    }

    const int64_t sequence_length = input.size(sequence_dim);
    const int64_t padding_length =
        (world_size() - sequence_length % world_size()) % world_size();
    padding_lengths_[tensor_name] = padding_length;
    torch::Tensor padded_input = pad_right(input, padding_length, sequence_dim);
    return parallel_state::scatter(
        padded_input, process_group_, static_cast<int32_t>(sequence_dim));
  }

  torch::Tensor gather_sequence(const torch::Tensor& input,
                                const std::string& tensor_name,
                                int64_t sequence_dim) const {
    if (!enabled() || !input.defined()) {
      return input;
    }

    torch::Tensor output = parallel_state::gather(
        input.contiguous(), process_group_, static_cast<int32_t>(sequence_dim));
    return unpad_right(output, padding_length(tensor_name), sequence_dim);
  }

  SequenceParallelWork launch_sequence_to_head(
      const torch::Tensor& input,
      const std::string& tensor_name) const {
    if (!enabled()) {
      return [input]() { return input; };
    }

    SequenceParallelWork work =
        parallel_state::all_to_all_4D(input,
                                      /*scatter_idx=*/2,
                                      /*gather_idx=*/1,
                                      /*async_ops=*/true,
                                      process_group_);
    const int64_t padding = padding_length(tensor_name);
    return [work = std::move(work), padding]() mutable {
      return unpad_right(work(), padding, /*dim=*/1);
    };
  }

  SequenceParallelWork launch_head_to_sequence(
      const torch::Tensor& input,
      const std::string& tensor_name) const {
    if (!enabled()) {
      return [input]() { return input; };
    }

    torch::Tensor padded_input =
        pad_right(input, padding_length(tensor_name), /*dim=*/1);
    return parallel_state::all_to_all_4D(padded_input,
                                         /*scatter_idx=*/1,
                                         /*gather_idx=*/2,
                                         /*async_ops=*/true,
                                         process_group_);
  }

 private:
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

  int64_t padding_length(const std::string& tensor_name) const {
    auto padding_it = padding_lengths_.find(tensor_name);
    CHECK(padding_it != padding_lengths_.end())
        << "Missing sequence-parallel padding metadata: " << tensor_name;
    return padding_it->second;
  }

  ProcessGroup* process_group_{nullptr};
  std::unordered_map<std::string, int64_t> padding_lengths_;
};

}  // namespace xllm::dit
