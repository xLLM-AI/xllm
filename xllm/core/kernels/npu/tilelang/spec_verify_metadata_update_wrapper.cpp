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

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>

#include "dispatch_registry.h"
#include "tilelang_ops_api.h"

#ifndef XLLM_TL_SPEC_VERIFY_METADATA_UPDATE_REGISTRY_INC
#error "XLLM_TL_SPEC_VERIFY_METADATA_UPDATE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_SPEC_VERIFY_METADATA_UPDATE_REGISTRY_INC

void check_npu_tensor(const torch::Tensor& tensor,
                      torch::ScalarType dtype,
                      int64_t min_numel,
                      const char* name) {
  CHECK(tensor.defined() &&
        tensor.device().type() == c10::DeviceType::PrivateUse1)
      << name << " must be an NPU tensor";
  CHECK_EQ(tensor.scalar_type(), dtype) << name << " dtype mismatch";
  CHECK_GE(tensor.numel(), min_numel) << name << " is too small";
  CHECK(tensor.is_contiguous()) << name << " must be contiguous";
}

uint8_t* ptr(const torch::Tensor& tensor) {
  return reinterpret_cast<uint8_t*>(const_cast<void*>(tensor.data_ptr()));
}

}  // namespace

bool has_spec_verify_metadata_update_specialization(int64_t spec_width,
                                                    int64_t block_table_width) {
  const auto specialization = make_spec_verify_metadata_update_specialization(
      SpecVerifyMetadataUpdateSpecWidth{static_cast<int32_t>(spec_width)},
      SpecVerifyMetadataUpdateBlockTableWidth{
          static_cast<int32_t>(block_table_width)});
  return find_spec_verify_metadata_update_kernel_entry(specialization) !=
         nullptr;
}

void spec_verify_metadata_update(
    const torch::Tensor& positions,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& new_cache_slots,
    const torch::Tensor& block_table,
    const torch::Tensor& linear_state_index,
    const torch::Tensor& num_accepted,
    const std::vector<torch::Tensor>& persistent_position_rows,
    torch::Tensor& persistent_q_seq_lens,
    torch::Tensor& persistent_kv_seq_lens,
    torch::Tensor& persistent_new_cache_slots,
    torch::Tensor& persistent_block_table,
    torch::Tensor& persistent_linear_state_index,
    torch::Tensor& persistent_num_accepted,
    torch::Tensor& persistent_q_cu_seq_lens,
    torch::Tensor& persistent_expanded_kv_seq_lens,
    const std::vector<torch::Tensor>& persistent_expanded_block_rows) {
  const int64_t spec_width = positions.numel();
  constexpr int64_t kPaddedWidth = 8;
  CHECK(has_spec_verify_metadata_update_specialization(spec_width,
                                                       block_table.numel()))
      << "speculative verify metadata update has no compiled variant for width "
      << spec_width << " and block table width " << block_table.numel();
  check_npu_tensor(positions, torch::kInt32, spec_width, "positions");
  check_npu_tensor(kv_seq_lens, torch::kInt32, 1, "kv_seq_lens");
  check_npu_tensor(
      new_cache_slots, torch::kInt32, spec_width, "new_cache_slots");
  check_npu_tensor(block_table, torch::kInt32, 1, "block_table");
  check_npu_tensor(linear_state_index, torch::kInt32, 1, "linear_state_index");
  check_npu_tensor(num_accepted, torch::kInt32, 1, "num_accepted");

  CHECK_EQ(persistent_position_rows.size(), 3);
  CHECK_EQ(persistent_expanded_block_rows.size(), 6)
      << "fixed TileLang ABI requires six expanded block-table pointers";
  for (const auto& row : persistent_position_rows) {
    check_npu_tensor(
        row, torch::kInt32, kPaddedWidth, "persistent_position_row");
  }
  check_npu_tensor(
      persistent_q_seq_lens, torch::kInt32, 1, "persistent_q_seq_lens");
  check_npu_tensor(
      persistent_kv_seq_lens, torch::kInt32, 1, "persistent_kv_seq_lens");
  check_npu_tensor(persistent_new_cache_slots,
                   torch::kInt32,
                   kPaddedWidth,
                   "persistent_new_cache_slots");
  check_npu_tensor(persistent_block_table,
                   torch::kInt32,
                   block_table.numel(),
                   "persistent_block_table");
  check_npu_tensor(persistent_linear_state_index,
                   torch::kInt32,
                   1,
                   "persistent_linear_state_index");
  check_npu_tensor(
      persistent_num_accepted, torch::kInt32, 1, "persistent_num_accepted");
  check_npu_tensor(
      persistent_q_cu_seq_lens, torch::kInt32, 2, "persistent_q_cu_seq_lens");
  check_npu_tensor(persistent_expanded_kv_seq_lens,
                   torch::kInt32,
                   kPaddedWidth,
                   "persistent_expanded_kv_seq_lens");
  for (const auto& row : persistent_expanded_block_rows) {
    check_npu_tensor(row,
                     torch::kInt32,
                     block_table.numel(),
                     "persistent_expanded_block_row");
  }

  const auto specialization = make_spec_verify_metadata_update_specialization(
      SpecVerifyMetadataUpdateSpecWidth{static_cast<int32_t>(spec_width)},
      SpecVerifyMetadataUpdateBlockTableWidth{
          static_cast<int32_t>(block_table.numel())});
  const auto* entry =
      find_spec_verify_metadata_update_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "speculative verify metadata update has no compiled variant: "
      << available_spec_verify_metadata_update_variant_keys();

  aclrtStream stream =
      c10_npu::getCurrentNPUStream(positions.device().index()).stream();
  entry->fn(ptr(positions),
            ptr(kv_seq_lens),
            ptr(new_cache_slots),
            ptr(block_table),
            ptr(linear_state_index),
            ptr(num_accepted),
            ptr(persistent_position_rows[0]),
            ptr(persistent_position_rows[1]),
            ptr(persistent_position_rows[2]),
            ptr(persistent_q_seq_lens),
            ptr(persistent_kv_seq_lens),
            ptr(persistent_new_cache_slots),
            ptr(persistent_block_table),
            ptr(persistent_linear_state_index),
            ptr(persistent_num_accepted),
            ptr(persistent_q_cu_seq_lens),
            ptr(persistent_expanded_kv_seq_lens),
            ptr(persistent_expanded_block_rows[0]),
            ptr(persistent_expanded_block_rows[1]),
            ptr(persistent_expanded_block_rows[2]),
            ptr(persistent_expanded_block_rows[3]),
            ptr(persistent_expanded_block_rows[4]),
            ptr(persistent_expanded_block_rows[5]),
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
