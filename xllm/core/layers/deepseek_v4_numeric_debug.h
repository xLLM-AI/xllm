/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace xllm {

inline bool deepseek_v4_numeric_debug_enabled() {
  const char* enabled = std::getenv("XLLM_DSV4_NUMERIC_DEBUG");
  if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
    return false;
  }
  const char* all_ranks = std::getenv("XLLM_DSV4_NUMERIC_DEBUG_ALL_RANKS");
  if (all_ranks != nullptr && std::strcmp(all_ranks, "1") == 0) {
    return true;
  }
  const char* visible_device = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
  return visible_device == nullptr || std::strcmp(visible_device, "0") == 0;
}

inline bool deepseek_v4_numeric_debug_layer(int64_t layer_id) {
  const char* selected = std::getenv("XLLM_DSV4_NUMERIC_LAYER");
  if (selected != nullptr) {
    char* end = nullptr;
    const long parsed = std::strtol(selected, &end, 10);
    return end != selected && *end == '\0' && parsed == layer_id;
  }
  return layer_id == 0 || layer_id == 2;
}

inline bool deepseek_v4_decode_detail_enabled(
    int64_t layer_id,
    const torch::Tensor& positions) {
  const char* dump_dir = std::getenv("XLLM_DSV4_DECODE_DETAIL_DIR");
  const char* target_position =
      std::getenv("XLLM_DSV4_DECODE_DETAIL_POSITION");
  if (dump_dir == nullptr || target_position == nullptr ||
      !positions.defined() || positions.numel() != 1) {
    return false;
  }
  const char* visible_device = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
  if (visible_device != nullptr && std::strcmp(visible_device, "0") != 0) {
    return false;
  }
  const char* target_layer = std::getenv("XLLM_DSV4_DECODE_DETAIL_LAYER");
  if (target_layer != nullptr) {
    char* end = nullptr;
    const long parsed = std::strtol(target_layer, &end, 10);
    if (end == target_layer || *end != '\0' || parsed != layer_id) {
      return false;
    }
  } else if (layer_id != 0) {
    return false;
  }
  char* end = nullptr;
  const long long parsed_position =
      std::strtoll(target_position, &end, 10);
  if (end == target_position || *end != '\0') {
    return false;
  }
  return positions.detach().to(torch::kCPU).reshape({-1})[0].item<int64_t>() ==
         parsed_position;
}

inline void deepseek_v4_save_decode_detail(const std::string& name,
                                           const torch::Tensor& tensor) {
  const char* dump_dir = std::getenv("XLLM_DSV4_DECODE_DETAIL_DIR");
  if (dump_dir == nullptr || !tensor.defined()) {
    return;
  }
  std::filesystem::create_directories(dump_dir);
  torch::save(tensor.detach().to(torch::kCPU),
              std::string(dump_dir) + "/cpp_" + name + ".pt");
}

inline void deepseek_v4_log_numeric_tensor(const char* name,
                                           const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    LOG(INFO) << "[DSV4_NUMERIC] " << name << " undefined";
    return;
  }

  std::ostringstream shape;
  shape << "[";
  for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
    if (dim > 0) {
      shape << ",";
    }
    shape << tensor.size(dim);
  }
  shape << "]";

  auto values = tensor.detach()
                    .to(torch::kCPU)
                    .to(torch::kFloat64)
                    .reshape({-1})
                    .contiguous();
  auto finite_mask = torch::isfinite(values);
  auto finite_values = values.index({finite_mask});
  const int64_t finite_count = finite_values.numel();

  double sum = 0.0;
  double abs_sum = 0.0;
  double square_sum = 0.0;
  double min = std::numeric_limits<double>::quiet_NaN();
  double max = std::numeric_limits<double>::quiet_NaN();
  if (finite_count > 0) {
    sum = finite_values.sum().item<double>();
    abs_sum = finite_values.abs().sum().item<double>();
    square_sum = finite_values.square().sum().item<double>();
    min = finite_values.min().item<double>();
    max = finite_values.max().item<double>();
  }

  std::ostringstream first;
  first << "[" << std::scientific << std::setprecision(9);
  const int64_t sample_count = std::min<int64_t>(8, values.numel());
  const auto* data = values.data_ptr<double>();
  for (int64_t i = 0; i < sample_count; ++i) {
    if (i > 0) {
      first << ",";
    }
    first << data[i];
  }
  first << "]";

  LOG(INFO) << "[DSV4_NUMERIC] " << name << " shape=" << shape.str()
            << " dtype=" << c10::toString(tensor.scalar_type())
            << " finite=" << finite_count << "/" << tensor.numel()
            << std::scientific << std::setprecision(12) << " sum=" << sum
            << " abs_sum=" << abs_sum << " square_sum=" << square_sum
            << " min=" << min << " max=" << max
            << " first=" << first.str();
}

inline bool deepseek_v4_cache_debug_enabled() {
  const char* enabled = std::getenv("XLLM_DSV4_CACHE_DEBUG");
  return enabled != nullptr && std::strcmp(enabled, "1") == 0;
}

}  // namespace xllm
