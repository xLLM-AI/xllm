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

#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kRopeWidth = 64;
constexpr int64_t kMaxTokens = 4096;
constexpr int64_t kMaxHeads = 128;
constexpr int32_t kTileHeads = 8;

#include XLLM_TL_V41_PARTIAL_ROPE_REGISTRY_INC

}  // namespace

void v41_partial_rope_(torch::Tensor& value,
                       const torch::Tensor& cos,
                       const torch::Tensor& signed_sin,
                       const torch::Tensor& swap_offsets,
                       bool inverse) {
  CHECK(value.dim() == 2 || value.dim() == 3);
  const int64_t tokens = value.size(0);
  const int64_t heads = value.dim() == 3 ? value.size(1) : 1;
  CHECK_LE(tokens, kMaxTokens);
  CHECK_GT(heads, 0);
  CHECK_LE(heads, kMaxHeads);
  CHECK_EQ(value.scalar_type(), torch::kBFloat16);
  CHECK_EQ(cos.scalar_type(), torch::kFloat32);
  CHECK_EQ(signed_sin.scalar_type(), torch::kFloat32);
  CHECK_EQ(swap_offsets.scalar_type(), torch::kInt32);
  CHECK_EQ(cos.dim(), 2);
  CHECK_EQ(cos.size(0), tokens);
  CHECK_EQ(cos.size(1), kRopeWidth);
  CHECK(cos.sizes() == signed_sin.sizes());
  CHECK_EQ(swap_offsets.numel(), kTileHeads * kRopeWidth);
  CHECK(value.is_contiguous() && cos.is_contiguous() &&
        signed_sin.is_contiguous() && swap_offsets.is_contiguous());
  CHECK(value.device() == cos.device() &&
        value.device() == signed_sin.device() &&
        value.device() == swap_offsets.device());
  if (tokens == 0) {
    return;
  }

  const int32_t tile_heads = heads % kTileHeads == 0 ? kTileHeads : 1;
  const auto spec = make_v41_partial_rope_specialization(
      V41PartialRopeHeadDim{static_cast<int32_t>(value.size(-1))},
      V41PartialRopeTileHeads{tile_heads});
  const auto* entry = find_v41_partial_rope_kernel_entry(spec);
  CHECK(entry != nullptr) << "Missing V4.1 partial RoPE specialization";
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(value.device().index()).stream();
  entry->fn(static_cast<uint8_t*>(value.data_ptr()),
            static_cast<uint8_t*>(cos.data_ptr()),
            static_cast<uint8_t*>(signed_sin.data_ptr()),
            static_cast<uint8_t*>(swap_offsets.data_ptr()),
            static_cast<int32_t>(tokens),
            static_cast<int32_t>(heads),
            static_cast<int32_t>(inverse),
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
