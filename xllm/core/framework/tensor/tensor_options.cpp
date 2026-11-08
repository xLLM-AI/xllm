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

#include "core/framework/tensor/tensor_options.h"

#include <glog/logging.h>

#include <boost/algorithm/string.hpp>

#include "core/framework/tensor/tensor_helper.h"

namespace xllm::util {

torch::ScalarType parse_dtype(const std::string& dtype_str,
                              const std::optional<torch::Device>& device) {
  if (device.has_value() && device.value().is_cpu()) {
    // cpu only supports float32 for now
    return torch::kFloat32;
  }

  if (boost::iequals(dtype_str, "half") ||
      boost::iequals(dtype_str, "float16")) {
    return torch::kFloat16;
  }
  if (boost::iequals(dtype_str, "bfloat16")) {
    return torch::kBFloat16;
  }
  if (boost::iequals(dtype_str, "float") ||
      boost::iequals(dtype_str, "float32")) {
    return torch::kFloat;
  }

  if (dtype_str.empty() || boost::iequals(dtype_str, "auto")) {
    return torch::kFloat16;
  }
  CHECK(false) << "Unsupported dtype: " << dtype_str << " on device "
               << device.value();
}

}  // namespace xllm::util

namespace xllm {

torch::ScalarType resolve_ssm_dtype(const std::string& mamba_ssm_dtype_str,
                                    torch::ScalarType default_dtype) {
  if (mamba_ssm_dtype_str.empty()) {
    return default_dtype;
  }
  auto parsed = try_get_scalar_type_from_string(mamba_ssm_dtype_str);
  if (parsed) {
    return parsed.value();
  }
  LOG(WARNING) << "Failed to parse mamba_ssm_dtype='" << mamba_ssm_dtype_str
               << "', falling back to default_dtype: " << default_dtype;
  return default_dtype;
}

int64_t resolve_ssm_dtype_size(const std::string& mamba_ssm_dtype_str,
                               int64_t default_dtype_size) {
  if (mamba_ssm_dtype_str.empty()) {
    return default_dtype_size;
  }
  auto parsed = try_get_scalar_type_from_string(mamba_ssm_dtype_str);
  if (parsed) {
    return get_dtype_size(parsed.value());
  }
  LOG(WARNING) << "Failed to parse mamba_ssm_dtype='" << mamba_ssm_dtype_str
               << "', falling back to default dtype size";
  return default_dtype_size;
}

}  // namespace xllm
