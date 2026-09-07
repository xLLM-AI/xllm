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
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <cstdint>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_SFA_DCP_REMAP_REGISTRY_INC
#error "XLLM_TL_SFA_DCP_REMAP_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kMaxTokens = 256;
constexpr int64_t kRemapTopk = 2048;

#include XLLM_TL_SFA_DCP_REMAP_REGISTRY_INC

}  // namespace

void sfa_dcp_remap_out(const torch::Tensor& topk_indices,
                       int64_t physical_block_size,
                       int64_t shard_size,
                       int64_t shard_rank,
                       torch::Tensor& out,
                       torch::Tensor& idx_scratch) {
  CHECK(topk_indices.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang sfa_dcp_remap: tensors must be on NPU";
  CHECK(out.device() == topk_indices.device() &&
        idx_scratch.device() == topk_indices.device());
  CHECK_EQ(topk_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(out.scalar_type(), torch::kInt32);
  CHECK_EQ(idx_scratch.scalar_type(), torch::kInt32);
  CHECK_EQ(out.sizes(), topk_indices.sizes());
  CHECK(topk_indices.is_contiguous() && out.is_contiguous() &&
        idx_scratch.is_contiguous());

  const int64_t topk = topk_indices.size(-1);
  CHECK_EQ(topk, kRemapTopk);
  CHECK_GT(physical_block_size, 0);
  CHECK_GT(shard_size, 0);
  const int64_t num_tokens = topk_indices.numel() / topk;
  CHECK_GT(num_tokens, 0);
  CHECK_LE(num_tokens, kMaxTokens);
  CHECK_GE(shard_rank, 0);
  CHECK_LT(shard_rank, shard_size);
  CHECK_GE(idx_scratch.numel(), num_tokens * topk);

  const SfaDcpRemapSpecialization specialization =
      make_sfa_dcp_remap_specialization(
          SfaDcpRemapTopk{static_cast<int32_t>(topk)});
  const auto* entry = find_sfa_dcp_remap_kernel_entry(specialization);
  CHECK(entry != nullptr) << "TileLang sfa_dcp_remap: no compiled variant "
                          << available_sfa_dcp_remap_variant_keys();

  const int32_t device_id = topk_indices.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  uint8_t* scratch_ptr = reinterpret_cast<uint8_t*>(idx_scratch.data_ptr());
  entry->fn(
      reinterpret_cast<uint8_t*>(const_cast<void*>(topk_indices.data_ptr())),
      reinterpret_cast<uint8_t*>(out.data_ptr()),
      scratch_ptr,
      scratch_ptr,
      static_cast<int32_t>(num_tokens),
      static_cast<int32_t>(physical_block_size),
      static_cast<int32_t>(shard_size),
      static_cast<int32_t>(shard_rank),
      stream);
}

}  // namespace xllm::kernel::npu::tilelang
