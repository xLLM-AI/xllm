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

#include <glog/logging.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "acl/acl.h"

namespace xllm::kernel::npu::tilelang::detail {

constexpr int64_t kSharedMappingMaxOffset = std::numeric_limits<int32_t>::max();
constexpr int32_t kSharedMappingTile = 128;

inline void check_shared_mapping_tensor(const torch::Tensor& tensor,
                                        const torch::Device& device) {
  CHECK(tensor.defined());
  CHECK_EQ(tensor.device(), device);
  CHECK_EQ(tensor.scalar_type(), torch::kInt32);
  CHECK(tensor.is_contiguous());
  CHECK_LE(tensor.numel(), kSharedMappingMaxOffset);
}

inline void check_shared_mapping_rows(const torch::Tensor& rows,
                                      const torch::Device& device) {
  CHECK(rows.defined());
  CHECK_EQ(rows.device(), device);
  CHECK_EQ(rows.scalar_type(), torch::kInt32);
  CHECK_EQ(rows.dim(), 2);
  CHECK_GT(rows.size(0), 0);
  CHECK_GT(rows.size(1), 0);
  CHECK_LE(rows.size(1), kSharedMappingMaxOffset);
  CHECK_EQ(rows.stride(1), 1);
  CHECK_GE(rows.stride(0), rows.size(1));
  CHECK_LE(rows.stride(0), kSharedMappingMaxOffset);
  CHECK_LE(rows.size(0) - 1,
           (kSharedMappingMaxOffset - rows.size(1)) / rows.stride(0));
}

inline int32_t shared_mapping_launch_cores(const torch::Device& device,
                                           int64_t tile_count) {
  thread_local int32_t cached_device = -1;
  thread_local int32_t cached_core_limit = 0;
  const int32_t device_index = device.index();
  CHECK_GE(device_index, 0);
  if (cached_device != device_index) {
    int64_t vector_cores = 0;
    CHECK_EQ(aclrtGetDeviceInfo(static_cast<uint32_t>(device_index),
                                ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                &vector_cores),
             ACL_SUCCESS);
    CHECK_GT(vector_cores, 0);
    CHECK_EQ(vector_cores % 2, 0);
    CHECK_LE(vector_cores, kSharedMappingMaxOffset);
    cached_core_limit = static_cast<int32_t>(vector_cores / 2);
    cached_device = device_index;
  }
  return static_cast<int32_t>(
      std::min((tile_count + 1) / 2, static_cast<int64_t>(cached_core_limit)));
}

}  // namespace xllm::kernel::npu::tilelang::detail
