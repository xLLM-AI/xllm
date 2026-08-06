/* Copyright 2025-2026 The xLLM Authors.

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

#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <cstdint>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "util/tensor_helper.h"

namespace xllm::kernel::npu {
struct type_info {
  static aclDataType get_acl_type(const torch::ScalarType& dtype);
};

void create_acltensor(aclTensor** tensor, const torch::Tensor& tensor_data);
void check_tensor(const torch::Tensor& t,
                  const std::string& name,
                  const std::string& func_name = "");
void check_tensor_shapes_equal(const torch::Tensor& a,
                               const torch::Tensor& b,
                               const std::string& func_name = "");
bool is_ascend950();
torch::Tensor expand_kv_heads(const torch::Tensor& tensor,
                              int64_t num_heads,
                              int64_t num_kv_heads);
}  // namespace xllm::kernel::npu
