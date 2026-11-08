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

#include <torch/types.h>

#include <cstdint>
#include <optional>
#include <string>

namespace xllm::util {

torch::ScalarType parse_dtype(const std::string& dtype_str,
                              const std::optional<torch::Device>& device);

}  // namespace xllm::util

namespace xllm {

torch::ScalarType resolve_ssm_dtype(const std::string& mamba_ssm_dtype_str,
                                    torch::ScalarType default_dtype);

int64_t resolve_ssm_dtype_size(const std::string& mamba_ssm_dtype_str,
                               int64_t default_dtype_size);

}  // namespace xllm
