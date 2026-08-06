/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <torch_npu/torch_npu.h>

#include <limits>
#include <utility>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_SPEC_VERIFY_ATTENTION_TILING_UPDATE_REGISTRY_INC
#error "XLLM_TL_SPEC_VERIFY_ATTENTION_TILING_UPDATE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {
#include XLLM_TL_SPEC_VERIFY_ATTENTION_TILING_UPDATE_REGISTRY_INC
}  // namespace

bool has_spec_verify_attention_tiling_update_specialization(
    int64_t spec_width,
    int64_t block_size) {
  if (spec_width <= 0 || block_size <= 0 ||
      !std::in_range<int32_t>(spec_width) ||
      !std::in_range<int32_t>(block_size)) {
    return false;
  }
  const auto specialization =
      make_spec_verify_attention_tiling_update_specialization(
          SpecVerifyAttentionTilingUpdateSpecWidth{
              static_cast<int32_t>(spec_width)});
  return find_spec_verify_attention_tiling_update_kernel_entry(
             specialization) != nullptr;
}

bool has_spec_verify_graph_update_specialization(int64_t spec_width,
                                                 int64_t block_size) {
  return block_size > 0 &&
         block_size % custom_paged_attention_block_alignment() == 0 &&
         has_spec_verify_token_update_specialization(spec_width) &&
         has_spec_verify_attention_tiling_update_specialization(spec_width,
                                                                block_size);
}

void spec_verify_attention_tiling_update(
    const torch::Tensor& src_kv_seq_lens,
    torch::Tensor& tiling_data,
    const PagedAttentionTilingLayout& layout,
    int64_t spec_width,
    int64_t block_size,
    int64_t max_kv_seq_len,
    int64_t kv_split_core_count) {
  CHECK_EQ(src_kv_seq_lens.device().type(), c10::DeviceType::PrivateUse1);
  CHECK_EQ(tiling_data.device().type(), c10::DeviceType::PrivateUse1);
  CHECK_EQ(tiling_data.device(), src_kv_seq_lens.device())
      << "attention tiling source and destination must be on the same NPU";
  CHECK_EQ(src_kv_seq_lens.scalar_type(), torch::kInt32);
  CHECK_EQ(tiling_data.scalar_type(), torch::kInt32);
  const int64_t num_rows = src_kv_seq_lens.numel();
  CHECK_GT(spec_width, 0);
  CHECK_EQ(num_rows % spec_width, 0)
      << "attention tiling rows must be divisible by verification width";
  const auto required_words =
      paged_attention_tiling_required_words(layout, num_rows);
  CHECK(required_words.has_value())
      << "attention tiling layout cannot represent " << num_rows << " rows";
  CHECK_GE(tiling_data.numel(), layout.buffer_words);
  for (const int64_t offset : {layout.max_kv_seq_len_offset,
                               layout.kv_split_length_offset,
                               layout.row_kv_seq_len_offset,
                               layout.row_stride_words}) {
    CHECK_LE(offset, std::numeric_limits<int32_t>::max());
  }
  CHECK_GT(block_size, 0);
  CHECK_GT(max_kv_seq_len, 0);
  CHECK_GT(kv_split_core_count, 0);
  CHECK_LE(num_rows, std::numeric_limits<int32_t>::max());
  CHECK_LE(block_size, std::numeric_limits<int32_t>::max());
  CHECK_LE(max_kv_seq_len, std::numeric_limits<int32_t>::max());
  const int64_t num_kv_blocks = (max_kv_seq_len + block_size - 1) / block_size;
  const int64_t kv_split_length =
      ((num_kv_blocks + kv_split_core_count - 1) / kv_split_core_count) *
      block_size;
  CHECK_LE(kv_split_length, std::numeric_limits<int32_t>::max());
  CHECK(src_kv_seq_lens.is_contiguous());
  CHECK(tiling_data.is_contiguous());
  const auto specialization =
      make_spec_verify_attention_tiling_update_specialization(
          SpecVerifyAttentionTilingUpdateSpecWidth{
              static_cast<int32_t>(spec_width)});
  const auto* entry =
      find_spec_verify_attention_tiling_update_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << available_spec_verify_attention_tiling_update_variant_keys();
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(src_kv_seq_lens.device().index()).stream();
  entry->fn(
      reinterpret_cast<uint8_t*>(const_cast<void*>(src_kv_seq_lens.data_ptr())),
      reinterpret_cast<uint8_t*>(tiling_data.data_ptr()),
      static_cast<int32_t>(max_kv_seq_len),
      static_cast<int32_t>(kv_split_length),
      static_cast<int32_t>(layout.max_kv_seq_len_offset),
      static_cast<int32_t>(layout.kv_split_length_offset),
      static_cast<int32_t>(layout.row_kv_seq_len_offset),
      static_cast<int32_t>(layout.row_stride_words),
      num_rows,
      tiling_data.numel(),
      stream);
}

}  // namespace xllm::kernel::npu::tilelang
