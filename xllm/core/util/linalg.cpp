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

#include "util/linalg.h"

#include <glog/logging.h>

#include <cmath>
#include <vector>

namespace {

bool is_power_of_two(int64_t dim) {
  return dim > 0 && ((dim & (dim - 1)) == 0);
}

int64_t next_power_of_two(int64_t dim) {
  CHECK_GT(dim, 0) << "next_power_of_two requires dim > 0";
  const double log_dim = std::ceil(std::log2(static_cast<double>(dim)));
  return static_cast<int64_t>(1ull << static_cast<uint64_t>(log_dim));
}

}  // namespace

namespace xllm::util {

torch::Tensor create_hadamard_matrix(int64_t dim,
                                     torch::Dtype dtype,
                                     const torch::Device& device,
                                     bool normalize) {
  CHECK(is_power_of_two(dim))
      << "hadamard dim must be a power of two, got " << dim;
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(dtype).device(device);
  torch::Tensor matrix = torch::ones({1, 1}, options);
  for (int64_t size = 1; size < dim; size <<= 1) {
    torch::Tensor top = torch::cat({matrix, matrix}, /*dim=*/1);
    torch::Tensor bottom = torch::cat({matrix, -matrix}, /*dim=*/1);
    matrix = torch::cat({top, bottom}, /*dim=*/0);
  }
  if (normalize) {
    matrix = matrix / std::sqrt(static_cast<double>(dim));
  }
  return matrix;
}

torch::Tensor hadamard_transform(const torch::Tensor& input,
                                 const torch::Tensor& hadamard_matrix) {
  CHECK(input.defined()) << "hadamard_transform requires input";
  CHECK(hadamard_matrix.defined()) << "hadamard_transform requires matrix";
  CHECK_GE(input.dim(), 1) << "hadamard_transform requires rank >= 1";
  const int64_t dim = input.size(-1);
  const int64_t padded_dim = next_power_of_two(dim);
  CHECK_EQ(hadamard_matrix.dim(), 2) << "hadamard matrix must be rank 2";
  CHECK_EQ(hadamard_matrix.size(0), padded_dim)
      << "hadamard matrix output dim mismatch";
  CHECK_EQ(hadamard_matrix.size(1), padded_dim)
      << "hadamard matrix input dim mismatch";

  std::vector<int64_t> input_shape(input.sizes().begin(), input.sizes().end());
  torch::Tensor flat = input.reshape({-1, dim});
  if (dim != padded_dim) {
    flat = torch::nn::functional::pad(
        flat,
        torch::nn::functional::PadFuncOptions({0, padded_dim - dim})
            .mode(torch::kConstant)
            .value(0));
  }

  torch::Tensor output = torch::nn::functional::linear(flat, hadamard_matrix);
  output = output.slice(/*dim=*/1, /*start=*/0, /*end=*/dim);
  return output.reshape(input_shape);
}

torch::Tensor rotate_activation(const torch::Tensor& input,
                                const torch::Tensor& hadamard_matrix) {
  CHECK_EQ(input.dtype(), torch::kBFloat16)
      << "rotate_activation requires bfloat16 input";
  return hadamard_transform(input, hadamard_matrix);
}

}  // namespace xllm::util
