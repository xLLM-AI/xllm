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

#ifndef XLLM_TL_SHARED_FULL_INDICES_REGISTRY_INC
#error "XLLM_TL_SHARED_FULL_INDICES_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_SHARED_FULL_INDICES_REGISTRY_INC

}  // namespace

void shared_full_indices_out(const torch::Tensor& block_table,
                             const torch::Tensor& slots,
                             const torch::Tensor& rows,
                             int64_t page_size,
                             int64_t padding_row,
                             torch::Tensor& out_tables,
                             torch::Tensor& out_slots) {
  CHECK(block_table.defined() && block_table.device().is_privateuseone());
  const torch::Device device = block_table.device();
  detail::check_shared_mapping_tensor(block_table, device);
  detail::check_shared_mapping_tensor(slots, device);
  detail::check_shared_mapping_rows(rows, device);
  CHECK_EQ(block_table.dim(), 2);
  CHECK_EQ(slots.dim(), 1);
  CHECK_GT(page_size, 0);
  CHECK_LE(page_size, detail::kSharedMappingMaxOffset);
  CHECK_GE(padding_row, 0);
  CHECK_LE(padding_row, detail::kSharedMappingMaxOffset);
  detail::check_shared_mapping_tensor(out_tables, device);
  detail::check_shared_mapping_tensor(out_slots, device);
  const int64_t layers = rows.size(0);
  CHECK_EQ(out_tables.dim(), 3);
  CHECK_EQ(out_tables.size(0), layers);
  CHECK_EQ(out_tables.size(1), block_table.size(0));
  CHECK_EQ(out_tables.size(2), block_table.size(1));
  CHECK_EQ(out_slots.dim(), 2);
  CHECK_EQ(out_slots.size(0), layers);
  CHECK_EQ(out_slots.size(1), slots.numel());
  if (out_tables.numel() == 0 && out_slots.numel() == 0) {
    return;
  }

  const auto specialization = make_shared_full_indices_specialization(
      SharedFullIndicesTileSize{detail::kSharedMappingTile});
  const auto* entry = find_shared_full_indices_kernel_entry(specialization);
  CHECK(entry != nullptr) << available_shared_full_indices_variant_keys();
  const int64_t tile_count =
      layers * ((block_table.numel() + detail::kSharedMappingTile - 1) /
                    detail::kSharedMappingTile +
                (slots.numel() + detail::kSharedMappingTile - 1) /
                    detail::kSharedMappingTile);
  CHECK_LE(tile_count, detail::kSharedMappingMaxOffset);
  const int32_t launch_cores =
      detail::shared_mapping_launch_cores(device, tile_count);
  aclrtStream stream = c10_npu::getCurrentNPUStream(device.index()).stream();
  entry->fn(
      reinterpret_cast<uint8_t*>(const_cast<void*>(block_table.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(slots.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(rows.data_ptr())),
      reinterpret_cast<uint8_t*>(out_tables.data_ptr()),
      reinterpret_cast<uint8_t*>(out_slots.data_ptr()),
      static_cast<int32_t>(block_table.numel()),
      static_cast<int32_t>(slots.numel()),
      static_cast<int32_t>(layers),
      static_cast<int32_t>(rows.size(1)),
      static_cast<int32_t>(rows.stride(0)),
      static_cast<int32_t>(page_size),
      static_cast<int32_t>(padding_row),
      launch_cores,
      stream);
}

}  // namespace xllm::kernel::npu::tilelang
