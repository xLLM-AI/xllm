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

#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/shared_cache_mapping_utils.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_SHARED_LINEAR_STATE_INDICES_REGISTRY_INC
#error "XLLM_TL_SHARED_LINEAR_STATE_INDICES_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_SHARED_LINEAR_STATE_INDICES_REGISTRY_INC

}  // namespace

void shared_linear_state_indices_out(const torch::Tensor& ids,
                                     const torch::Tensor& rows,
                                     torch::Tensor& out) {
  CHECK(ids.defined() && ids.device().is_privateuseone());
  const torch::Device device = ids.device();
  detail::check_shared_mapping_tensor(ids, device);
  CHECK_EQ(ids.dim(), 1);
  detail::check_shared_mapping_rows(rows, device);
  CHECK_EQ(rows.size(0) % 2, 0);
  detail::check_shared_mapping_tensor(out, device);
  const int64_t groups = rows.size(0) / 2;
  const int64_t batch_size = ids.numel();
  CHECK_EQ(out.dim(), 3);
  CHECK_EQ(out.size(0), groups);
  CHECK_EQ(out.size(1), batch_size);
  CHECK_EQ(out.size(2), 2);
  if (out.numel() == 0) {
    return;
  }

  const auto specialization = make_shared_linear_state_indices_specialization(
      SharedLinearStateIndicesTileSize{detail::kSharedMappingTile});
  const auto* entry =
      find_shared_linear_state_indices_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << available_shared_linear_state_indices_variant_keys();
  const int64_t tile_count = (out.numel() + detail::kSharedMappingTile - 1) /
                             detail::kSharedMappingTile;
  const int32_t launch_cores =
      detail::shared_mapping_launch_cores(device, tile_count);
  aclrtStream stream = c10_npu::getCurrentNPUStream(device.index()).stream();
  entry->fn(reinterpret_cast<uint8_t*>(const_cast<void*>(ids.data_ptr())),
            reinterpret_cast<uint8_t*>(const_cast<void*>(rows.data_ptr())),
            reinterpret_cast<uint8_t*>(out.data_ptr()),
            static_cast<int32_t>(batch_size),
            static_cast<int32_t>(groups),
            static_cast<int32_t>(rows.size(1)),
            static_cast<int32_t>(rows.stride(0)),
            launch_cores,
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
