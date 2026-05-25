/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "pretty_print.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <sstream>

#include "tensor_helper.h"

namespace xllm {
namespace {

torch::Tensor to_cpu_int_tensor_for_print(const torch::Tensor& values) {
  return safe_to(values.flatten(),
                 torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU),
                 false)
      .contiguous();
}

}  // namespace

std::string readable_size(size_t bytes) {
  static const std::array<const char*, 5> suffixes = {
      "B", "KB", "MB", "GB", "TB"};
  const size_t bytes_in_kb = 1024;
  double size = static_cast<double>(bytes);
  size_t suffix_index = 0;
  while (size >= bytes_in_kb && suffix_index < suffixes.size() - 1) {
    size /= bytes_in_kb;
    ++suffix_index;
  }
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << size << " "
         << suffixes.at(suffix_index);
  return stream.str();
}

std::string summarize_int32_values(const int32_t* values,
                                   size_t size,
                                   size_t limit) {
  std::ostringstream oss;
  oss << "[";
  const size_t print_size = std::min(size, limit);
  for (size_t i = 0; i < print_size; ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << values[i];
  }
  if (size > limit) {
    oss << ", ...";
  }
  oss << "]";
  return oss.str();
}

std::string summarize_int32_vector(const std::vector<int32_t>& values,
                                   size_t limit) {
  return summarize_int32_values(values.data(), values.size(), limit);
}

std::string summarize_string_vector(const std::vector<std::string>& values,
                                    size_t limit) {
  std::ostringstream oss;
  oss << "[";
  const size_t print_size = std::min(values.size(), limit);
  for (size_t i = 0; i < print_size; ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << values[i];
  }
  if (values.size() > limit) {
    oss << ", ...";
  }
  oss << "]";
  return oss.str();
}

std::string summarize_int_tensor(const torch::Tensor& tensor, size_t limit) {
  if (!tensor.defined()) {
    return "undefined";
  }
  torch::Tensor flat = to_cpu_int_tensor_for_print(tensor);
  std::ostringstream oss;
  oss << "sizes=" << tensor.sizes() << ", values="
      << summarize_int32_values(flat.const_data_ptr<int32_t>(),
                                static_cast<size_t>(flat.numel()),
                                limit);
  return oss.str();
}

std::string summarize_accepted_tokens(const torch::Tensor& accepted_tokens,
                                      size_t max_rows,
                                      size_t max_width) {
  if (!accepted_tokens.defined()) {
    return "undefined";
  }
  if (accepted_tokens.dim() != 2) {
    return summarize_int_tensor(accepted_tokens);
  }
  torch::Tensor flat = to_cpu_int_tensor_for_print(accepted_tokens);
  const int32_t* data = flat.const_data_ptr<int32_t>();
  const int64_t rows = accepted_tokens.size(0);
  const int64_t width = accepted_tokens.size(1);

  std::ostringstream oss;
  oss << "sizes=" << accepted_tokens.sizes() << ", rows=[";
  const int64_t print_rows =
      std::min<int64_t>(rows, static_cast<int64_t>(max_rows));
  for (int64_t row = 0; row < print_rows; ++row) {
    if (row > 0) {
      oss << ", ";
    }
    int32_t accepted_len = 0;
    int32_t last_token = -1;
    for (int64_t col = 0; col < width; ++col) {
      const int32_t token = data[row * width + col];
      if (token < 0) {
        break;
      }
      last_token = token;
      ++accepted_len;
    }
    oss << "{row=" << row << ", accepted_len=" << accepted_len
        << ", last_token=" << last_token << ", tokens=[";
    const int64_t print_width =
        std::min<int64_t>(width, static_cast<int64_t>(max_width));
    for (int64_t col = 0; col < print_width; ++col) {
      if (col > 0) {
        oss << ", ";
      }
      oss << data[row * width + col];
    }
    if (width > print_width) {
      oss << ", ...";
    }
    oss << "]}";
  }
  if (rows > print_rows) {
    oss << ", ...";
  }
  oss << "]";
  return oss.str();
}

}  // namespace xllm
