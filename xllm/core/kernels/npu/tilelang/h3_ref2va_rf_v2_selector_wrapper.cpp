/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <c10/core/DeviceType.h>
#include <glog/logging.h>

#include <cstdint>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/h3_rainfusion_runtime.h"

#ifndef XLLM_TL_H3_REF2VA_RF_V2_SELECTOR_REGISTRY_INC
#error "XLLM_TL_H3_REF2VA_RF_V2_SELECTOR_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_H3_REF2VA_RF_V2_SELECTOR_REGISTRY_INC

void check_tensor(const torch::Tensor& tensor,
                  const char* name,
                  c10::ScalarType dtype,
                  const torch::Device& device,
                  std::initializer_list<int64_t> shape) {
  CHECK(tensor.defined()) << "TileLang H3 rf_v2 selector: " << name
                          << " must be defined";
  CHECK_EQ(tensor.device(), device) << "TileLang H3 rf_v2 selector: " << name
                                    << " must be on the query device";
  CHECK_EQ(tensor.scalar_type(), dtype)
      << "TileLang H3 rf_v2 selector: " << name << " dtype mismatch";
  CHECK(tensor.is_contiguous())
      << "TileLang H3 rf_v2 selector: " << name << " must be contiguous";
  CHECK_EQ(tensor.sizes(), c10::IntArrayRef(shape))
      << "TileLang H3 rf_v2 selector: " << name << " shape mismatch";
}

uint8_t* as_bytes(const torch::Tensor& tensor) {
  return reinterpret_cast<uint8_t*>(const_cast<void*>(tensor.data_ptr()));
}

}  // namespace

namespace detail {

constexpr const char* kH3RainfusionSelectorCapacityKey =
    "cap_ctr61952_ctb483_crc5568_ctk97_abi2";

// The registry dispatch fields describe logical geometry. Capacity objects
// use variant keys so admission and launch select the same ABI.
inline const KernelEntry<H3Ref2vaRfV2SelectorSpecialization,
                         H3Ref2vaRfV2SelectorKernelFn>*
find_selector_entry_by_key(const char* key) {
  const std::string_view want(key);
  for (const auto& entry : kH3Ref2vaRfV2SelectorRegistry) {
    if (want == std::string_view(entry.variant_key)) {
      return &entry;
    }
  }
  return nullptr;
}

const char* h3_rainfusion_selected_selector_key() {
  const auto* entry =
      find_selector_entry_by_key(kH3RainfusionSelectorCapacityKey);
  return entry != nullptr ? entry->variant_key : "";
}

bool has_selector_capacity_specialization() {
  // A key string merely being listed proves nothing about what the runtime
  // selects, so availability resolves the SAME entry the launch will use.
  return find_selector_entry_by_key(kH3RainfusionSelectorCapacityKey) !=
         nullptr;
}

void validate_selector(const torch::Tensor& query,
                       const torch::Tensor& key,
                       const torch::Tensor& value,
                       const H3RainfusionGeometry& geometry,
                       const H3RainfusionDerived& derived,
                       const H3RainfusionWorkspace& workspace) {
  CHECK(query.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang H3 rf_v2 selector: tensors must be on NPU";
  const torch::Device device = query.device();
  const int64_t row_width = kH3RainfusionHeads * kH3RainfusionHeadDim;
  // Both record extents come from the workspace, resolved once at allocation.
  // record_count is a dispatch axis and drives the attention inverse epilogue,
  // so recomputing it here would rebuild the whole span table on every one of
  // the sparse block calls.

  const auto tnd = {geometry.total_rows, row_width};
  check_tensor(query, "query", torch::kBFloat16, device, tnd);
  check_tensor(key, "key", torch::kBFloat16, device, tnd);
  check_tensor(value, "value", torch::kBFloat16, device, tnd);
  check_tensor(workspace.reorder_records,
               "workspace.reorder_records",
               torch::kInt32,
               device,
               {workspace.records_padded, 4});
  check_tensor(workspace.plan_ids,
               "workspace.plan_ids",
               torch::kInt32,
               device,
               {kH3RainfusionQualifiedCapacity.num_blocks,
                kH3RainfusionHeads,
                kH3RainfusionQualifiedCapacity.plan_width});
  check_tensor(
      workspace.plan_count,
      "workspace.plan_count",
      torch::kInt32,
      device,
      {kH3RainfusionQualifiedCapacity.num_blocks, kH3RainfusionPlanAlign});
  check_tensor(workspace.reordered_q,
               "workspace.reordered_q",
               torch::kBFloat16,
               device,
               tnd);
  check_tensor(workspace.reordered_k,
               "workspace.reordered_k",
               torch::kBFloat16,
               device,
               tnd);
  check_tensor(workspace.reordered_v,
               "workspace.reordered_v",
               torch::kBFloat16,
               device,
               tnd);
  check_tensor(workspace.q_pool,
               "workspace.q_pool",
               torch::kBFloat16,
               device,
               {kH3RainfusionHeads,
                kH3RainfusionQualifiedCapacity.q_pool_pad,
                kH3RainfusionHeadDim});
  check_tensor(workspace.k_pool,
               "workspace.k_pool",
               torch::kBFloat16,
               device,
               {kH3RainfusionHeads,
                kH3RainfusionQualifiedCapacity.k_pool_pad,
                kH3RainfusionHeadDim});
  check_tensor(workspace.score_ring,
               "workspace.score_ring",
               torch::kFloat32,
               device,
               {kH3RainfusionNumCores,
                kH3RainfusionRingSlots,
                kH3RainfusionMTile,
                kH3RainfusionQualifiedCapacity.k_pool_pad});

  // The reordered triple is written here and read by the attention launch; the
  // pools and the ring are scratch. None may alias the read-only inputs,
  // otherwise the reorder would observe partially rewritten rows.
  CHECK_NE(workspace.reordered_q.data_ptr(), query.data_ptr())
      << "TileLang H3 rf_v2 selector: reordered_q aliases query";
  CHECK_NE(workspace.reordered_k.data_ptr(), key.data_ptr())
      << "TileLang H3 rf_v2 selector: reordered_k aliases key";
  CHECK_NE(workspace.reordered_v.data_ptr(), value.data_ptr())
      << "TileLang H3 rf_v2 selector: reordered_v aliases value";
}

void launch_selector(const torch::Tensor& query,
                     const torch::Tensor& key,
                     const torch::Tensor& value,
                     const H3RainfusionGeometry& geometry,
                     const H3RainfusionDerived& derived,
                     const H3RainfusionWorkspace& workspace,
                     aclrtStream stream) {
  // Dispatch on the CAPACITY tuple, not on the logical geometry: the abi2
  // object is compiled once per capacity and the logical extents travel as
  // runtime scalars. A key string merely being present in the registry proves
  // nothing, so the entry the runtime actually selected is asserted below and
  // recorded.
  const auto* entry =
      find_selector_entry_by_key(kH3RainfusionSelectorCapacityKey);
  CHECK(entry != nullptr)
      << "TileLang H3 rf_v2 selector: no compiled variant for the qualified "
      << "capacity. Available variants: "
      << available_h3_ref2va_rf_v2_selector_variant_keys();
  CHECK(entry->fn != nullptr)
      << "TileLang H3 rf_v2 selector: capacity entry has no kernel function";

  entry->fn(as_bytes(query),
            as_bytes(key),
            as_bytes(value),
            as_bytes(workspace.reorder_records),
            as_bytes(workspace.plan_ids),
            as_bytes(workspace.plan_count),
            as_bytes(workspace.reordered_q),
            as_bytes(workspace.reordered_k),
            as_bytes(workspace.reordered_v),
            as_bytes(workspace.q_pool),
            as_bytes(workspace.k_pool),
            as_bytes(workspace.score_ring),
            // abi2: the EIGHT runtime logical scalars between the GM handles
            // and the stream. The GM buffers are capacity-extent; these are the
            // ACTIVE extents the kernel walks, so the tail reduction, the
            // dense/sparse split and the record stride follow the real request
            // instead of a baked literal.
            static_cast<int>(derived.total_blocks),
            static_cast<int>(geometry.topk_r),
            static_cast<int>(workspace.record_count),
            static_cast<int>(workspace.records_padded / kH3RainfusionWorkers),
            static_cast<int>(derived.tail_rows),
            static_cast<int>(derived.ff_blocks),
            static_cast<int>(derived.video_full_blocks),
            static_cast<int>(derived.num_sq),
            stream);
}

}  // namespace detail

}  // namespace xllm::kernel::npu::tilelang
