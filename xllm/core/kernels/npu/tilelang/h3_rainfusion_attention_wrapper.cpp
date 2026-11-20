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
#include <stdexcept>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/h3_rainfusion_runtime.h"

#ifndef XLLM_TL_H3_RAINFUSION_ATTENTION_REGISTRY_INC
#error "XLLM_TL_H3_RAINFUSION_ATTENTION_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_H3_RAINFUSION_ATTENTION_REGISTRY_INC

void check_tensor(const torch::Tensor& tensor,
                  const char* name,
                  c10::ScalarType dtype,
                  const torch::Device& device,
                  std::initializer_list<int64_t> shape) {
  CHECK(tensor.defined()) << "TileLang H3 RainFusion attention: " << name
                          << " must be defined";
  CHECK_EQ(tensor.device(), device)
      << "TileLang H3 RainFusion attention: " << name
      << " must be on the output device";
  CHECK_EQ(tensor.scalar_type(), dtype)
      << "TileLang H3 RainFusion attention: " << name << " dtype mismatch";
  CHECK(tensor.is_contiguous())
      << "TileLang H3 RainFusion attention: " << name << " must be contiguous";
  CHECK_EQ(tensor.sizes(), c10::IntArrayRef(shape))
      << "TileLang H3 RainFusion attention: " << name << " shape mismatch";
}

uint8_t* as_bytes(const torch::Tensor& tensor) {
  return reinterpret_cast<uint8_t*>(const_cast<void*>(tensor.data_ptr()));
}

}  // namespace

namespace detail {

constexpr const char* kH3RainfusionAttentionCapacityKey =
    "cap_ctr61952_cnb483_crc5568_crp5568_bf16_abi2";

// The registry dispatch fields describe logical geometry. Capacity objects
// use variant keys so admission and launch select the same ABI.
inline const KernelEntry<H3RainfusionAttentionSpecialization,
                         H3RainfusionAttentionKernelFn>*
find_attention_entry_by_key(const char* key) {
  const std::string_view want(key);
  for (const auto& entry : kH3RainfusionAttentionRegistry) {
    if (want == std::string_view(entry.variant_key)) {
      return &entry;
    }
  }
  return nullptr;
}

const char* h3_rainfusion_selected_attention_key() {
  const auto* entry =
      find_attention_entry_by_key(kH3RainfusionAttentionCapacityKey);
  return entry != nullptr ? entry->variant_key : "";
}

bool has_attention_capacity_specialization() {
  return find_attention_entry_by_key(kH3RainfusionAttentionCapacityKey) !=
         nullptr;
}

void validate_attention(const H3RainfusionGeometry& geometry,
                        const H3RainfusionDerived& derived,
                        const H3RainfusionWorkspace& workspace,
                        torch::Tensor& output) {
  CHECK(output.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang H3 RainFusion attention: tensors must be on NPU";
  const torch::Device device = output.device();
  const int64_t row_width = kH3RainfusionHeads * kH3RainfusionHeadDim;
  const int64_t stack_width = kH3RainfusionStackBlocks * kH3RainfusionBlock;
  const int64_t records_padded = workspace.records_padded;

  const auto tnd = {geometry.total_rows, row_width};
  check_tensor(output, "output", torch::kBFloat16, device, tnd);
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
  check_tensor(workspace.reorder_records,
               "workspace.reorder_records",
               torch::kInt32,
               device,
               {records_padded, 4});
  check_tensor(workspace.workspace_s,
               "workspace.workspace_s",
               torch::kFloat32,
               device,
               {kH3RainfusionTaskSlots,
                kH3RainfusionNumCores,
                kH3RainfusionBlock,
                stack_width});
  check_tensor(workspace.workspace_p,
               "workspace.workspace_p",
               torch::kBFloat16,
               device,
               {kH3RainfusionTaskSlots,
                kH3RainfusionNumCores,
                kH3RainfusionBlock,
                stack_width});
  check_tensor(workspace.workspace_o,
               "workspace.workspace_o",
               torch::kFloat32,
               device,
               {kH3RainfusionTaskSlots,
                kH3RainfusionNumCores,
                kH3RainfusionBlock,
                kH3RainfusionHeadDim});
  check_tensor(workspace.packed_k,
               "workspace.packed_k",
               torch::kBFloat16,
               device,
               {kH3RainfusionHeads,
                kH3RainfusionQualifiedCapacity.total_rows,
                kH3RainfusionHeadDim});
  check_tensor(workspace.packed_v,
               "workspace.packed_v",
               torch::kBFloat16,
               device,
               {kH3RainfusionHeads,
                kH3RainfusionQualifiedCapacity.total_rows,
                kH3RainfusionHeadDim});
  check_tensor(workspace.reordered_output,
               "workspace.reordered_output",
               torch::kBFloat16,
               device,
               tnd);

  // The kernel writes `output` through the record-driven inverse epilogue and
  // uses `reordered_output` as its reordered-domain staging buffer, so the two
  // must be distinct allocations.
  CHECK_NE(output.data_ptr(), workspace.reordered_output.data_ptr())
      << "TileLang H3 RainFusion attention: output aliases reordered_output";
  CHECK_NE(output.data_ptr(), workspace.reordered_q.data_ptr())
      << "TileLang H3 RainFusion attention: output aliases reordered_q";
}

void launch_attention(const H3RainfusionGeometry& geometry,
                      const H3RainfusionDerived& derived,
                      const H3RainfusionWorkspace& workspace,
                      torch::Tensor& output,
                      aclrtStream stream) {
  // Dispatch on the CAPACITY tuple; the logical extents travel as runtime
  // scalars. A key string being present in the registry proves nothing, so the
  // entry the runtime actually selected is asserted against the required
  // capacity key.
  const auto* entry =
      find_attention_entry_by_key(kH3RainfusionAttentionCapacityKey);
  CHECK(entry != nullptr) << "TileLang H3 RainFusion attention: no compiled "
                             "variant for the qualified "
                          << "capacity. Available variants: "
                          << available_h3_rainfusion_attention_variant_keys();
  CHECK(entry->fn != nullptr) << "TileLang H3 RainFusion attention: capacity "
                                 "entry has no kernel function";

  entry->fn(as_bytes(workspace.reordered_q),
            as_bytes(workspace.reordered_k),
            as_bytes(workspace.reordered_v),
            as_bytes(workspace.plan_ids),
            as_bytes(workspace.plan_count),
            as_bytes(workspace.reorder_records),
            as_bytes(output),
            as_bytes(workspace.workspace_s),
            as_bytes(workspace.workspace_p),
            as_bytes(workspace.workspace_o),
            as_bytes(workspace.packed_k),
            as_bytes(workspace.packed_v),
            as_bytes(workspace.reordered_output),
            // abi2: the EIGHT runtime logical scalars. rt_records_per_worker is
            // the LOGICAL worker-major stride ceildiv(record_count,48), NOT the
            // capacity stride: the real reorder builder emits a compact
            // worker-major table that the host places in the prefix of the
            // capacity-extent buffer. Verified on device at 1440x736, where the
            // logical stride 84 differs from the capacity 116 and the inverse
            // reorder was exact on all 60891 valid rows.
            static_cast<int>(geometry.valid_rows),
            static_cast<int>(derived.total_blocks),
            static_cast<int>(derived.tail_rows),
            static_cast<int>(derived.ff_blocks),
            static_cast<int>(derived.video_full_blocks),
            static_cast<int>(workspace.record_count),
            static_cast<int>(
                h3_ceildiv(workspace.record_count, kH3RainfusionWorkers)),
            static_cast<int>(
                h3_ceildiv(geometry.valid_rows, kH3RainfusionKvPackRows)),
            stream);
}

}  // namespace detail

}  // namespace xllm::kernel::npu::tilelang
