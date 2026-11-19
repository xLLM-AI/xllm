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

#include "core/kv_cache/storage/host_cache_layout.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "core/kv_cache/layout/kv_cache_layout_policy.h"

namespace xllm {

int64_t scale_host_block_count(int64_t block_count, double host_blocks_factor) {
  CHECK_GT(block_count, 0) << "block_count must be positive.";
  CHECK(std::isfinite(host_blocks_factor))
      << "host_blocks_factor must be finite.";
  CHECK_GE(host_blocks_factor, 0.0)
      << "host_blocks_factor must be non-negative.";
  const double factor = std::max(host_blocks_factor, 1.0);
  const long double scaled_block_count =
      static_cast<long double>(block_count) * static_cast<long double>(factor);
  CHECK_LE(scaled_block_count,
           static_cast<long double>(std::numeric_limits<int64_t>::max()))
      << "scaled host block count exceeds the int64 limit.";
  return std::max<int64_t>(block_count,
                           static_cast<int64_t>(scaled_block_count));
}

std::optional<std::string> validate_host_cache_options(
    const HostCacheValidationOptions& options) {
  if (!std::isfinite(options.host_blocks_factor) ||
      options.host_blocks_factor < 0.0) {
    std::ostringstream error;
    error << "Invalid host prefix-cache offload configuration "
          << "(host_blocks_factor=" << options.host_blocks_factor
          << "): --host_blocks_factor must be finite and non-negative.";
    return error.str();
  }

  if (options.host_blocks_factor <= 1.0) {
    return std::nullopt;
  }

  std::vector<std::string> violations;
  violations.reserve(10);
  const long double host_block_count =
      static_cast<long double>(options.device_block_count) *
      static_cast<long double>(options.host_blocks_factor);
  if (options.device_block_count <= 0 ||
      host_block_count >
          static_cast<long double>(std::numeric_limits<uint32_t>::max())) {
    violations.emplace_back(
        "requested host block count exceeds the uint32 block-manager limit; "
        "reduce --host_blocks_factor");
  }
  if (!options.supports_host_kv_offload) {
    violations.emplace_back(
        "the current platform has no host KV offload copy/synchronization "
        "provider");
  }
  const bool has_linear_cache_shape =
      options.has_conv_cache_shape && options.has_ssm_cache_shape;
  const bool enable_offload_only_decode =
      options.enable_disagg_pd && options.enable_kvcache_store &&
      options.instance_role == InstanceRole::DECODE && !has_linear_cache_shape;
  if (!options.enable_prefix_cache && !enable_offload_only_decode) {
    violations.emplace_back(
        "prefix caching is disabled for this engine; host offload requires "
        "an engine configuration that supports prefix caching");
  }
  if (!options.enable_disagg_pd &&
      options.instance_role != InstanceRole::DEFAULT) {
    violations.emplace_back(
        "host offload outside disaggregated serving requires the DEFAULT "
        "instance role");
  }

  // Quantized KV caches add scale tensors whose host offload and restore
  // lifecycle is not supported consistently by the common path yet.
  if (options.kv_cache_dtype != "auto") {
    violations.emplace_back(
        "quantized KV cache scales are not transferred to host memory; set "
        "--kv_cache_dtype=auto");
  }
  if (!options.has_key_cache_shape) {
    violations.emplace_back("KV cache has no key-cache tensor to offload");
  }
#if defined(USE_NPU)
  if (options.requires_indexer_row_folding &&
      use_npu_nz_kv_cache_layout(options.model_type)) {
    violations.emplace_back(
        "split indexer host offload requires ND cache storage");
  }
#endif
  if (options.has_grouped_cache_layout &&
      !options.supports_grouped_cache_offload) {
    std::ostringstream violation;
    violation << "model \"" << options.model_type
              << "\" uses a grouped cache layout (for example DeepSeek-V4 "
                 "SWA/C4/C128 or DeepSeek-V4.1 SWA/C1/C2) that host offload "
                 "does not support";
    violations.emplace_back(violation.str());
  }
  if (options.has_conv_cache_shape != options.has_ssm_cache_shape) {
    std::ostringstream violation;
    violation << "model \"" << options.model_type
              << "\" has a partial linear-attention cache layout; Host "
                 "offload requires both conv and SSM tensors";
    violations.emplace_back(violation.str());
  }

  if (violations.empty()) {
    return std::nullopt;
  }

  std::ostringstream error;
  error << "Invalid host prefix-cache offload configuration "
        << "(host_blocks_factor=" << options.host_blocks_factor << "): ";
  for (size_t i = 0; i < violations.size(); ++i) {
    if (i != 0) {
      error << "; ";
    }
    error << violations[i];
  }
  error << ". Fix the listed settings or disable offload with "
           "--host_blocks_factor=0.";
  return error.str();
}

std::vector<int64_t> build_host_tensor_shape(
    const std::vector<int64_t>& base_shape,
    double host_blocks_factor) {
  CHECK(!base_shape.empty()) << "base_shape must not be empty.";
  std::vector<int64_t> host_shape = base_shape;
  host_shape[0] = scale_host_block_count(host_shape[0], host_blocks_factor);
  return host_shape;
}

std::vector<int64_t> build_host_group_tensor_shape(
    const std::vector<int64_t>& base_shape,
    double host_blocks_factor,
    int64_t layer_count) {
  CHECK_GT(layer_count, 0) << "layer_count must be positive.";
  std::vector<int64_t> host_shape =
      build_host_tensor_shape(base_shape, host_blocks_factor);
  host_shape.insert(host_shape.begin() + 1, layer_count);
  return host_shape;
}

std::vector<int64_t> fold_shape_for_host_transfer(
    const std::vector<int64_t>& shape,
    int64_t logical_blocks) {
  CHECK_GE(shape.size(), 2);
  CHECK_GT(logical_blocks, 0);
  CHECK_EQ(shape[0] % logical_blocks, 0)
      << "indexer pages must form complete logical KV blocks";
  std::vector<int64_t> folded = shape;
  folded[0] = logical_blocks;
  folded[1] *= shape[0] / logical_blocks;
  return folded;
}

}  // namespace xllm
