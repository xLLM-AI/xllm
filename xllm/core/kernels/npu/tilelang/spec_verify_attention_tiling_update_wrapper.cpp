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

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_SPEC_VERIFY_ATTENTION_TILING_UPDATE_REGISTRY_INC
#error "XLLM_TL_SPEC_VERIFY_ATTENTION_TILING_UPDATE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {
#include XLLM_TL_SPEC_VERIFY_ATTENTION_TILING_UPDATE_REGISTRY_INC
}

void spec_verify_attention_tiling_update(const torch::Tensor& src_kv_seq_lens,
                                         torch::Tensor& tiling_data,
                                         int64_t block_size) {
  CHECK_EQ(src_kv_seq_lens.device().type(), c10::DeviceType::PrivateUse1);
  CHECK_EQ(tiling_data.device().type(), c10::DeviceType::PrivateUse1);
  CHECK_EQ(src_kv_seq_lens.scalar_type(), torch::kInt32);
  CHECK_EQ(tiling_data.scalar_type(), torch::kInt32);
  const int64_t spec_width = src_kv_seq_lens.numel();
  CHECK_GE(spec_width, 4);
  CHECK_LE(spec_width, 6);
  CHECK_GE(tiling_data.numel(), 44 + spec_width * 17);
  CHECK(src_kv_seq_lens.is_contiguous());
  CHECK(tiling_data.is_contiguous());
  CHECK_EQ(block_size, 128);
  const auto specialization =
      make_spec_verify_attention_tiling_update_specialization(
          SpecVerifyAttentionTilingUpdateSpecWidth{
              static_cast<int32_t>(spec_width)},
          SpecVerifyAttentionTilingUpdateBlockSize{128});
  const auto* entry =
      find_spec_verify_attention_tiling_update_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << available_spec_verify_attention_tiling_update_variant_keys();
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(src_kv_seq_lens.device().index()).stream();
  entry->fn(
      reinterpret_cast<uint8_t*>(const_cast<void*>(src_kv_seq_lens.data_ptr())),
      reinterpret_cast<uint8_t*>(tiling_data.data_ptr()),
      stream);
}

}  // namespace xllm::kernel::npu::tilelang
