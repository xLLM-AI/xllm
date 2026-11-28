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

constexpr int32_t kHeadDim = 512;
constexpr int64_t kRopeWidth = 64;
constexpr int64_t kMaxTokens = 4096;
constexpr int64_t kMaxCacheRows = 1 << 21;

#include XLLM_TL_V41_SWA_ROPE_CACHE_REGISTRY_INC

}  // namespace

void v41_swa_rope_cache_(torch::Tensor& value,
                         const torch::Tensor& cos,
                         const torch::Tensor& signed_sin,
                         const torch::Tensor& swap_offsets,
                         const torch::Tensor& slots,
                         torch::Tensor& cache) {
  CHECK_EQ(value.dim(), 2);
  CHECK_EQ(value.size(1), kHeadDim);
  CHECK_EQ(cache.dim(), 2);
  CHECK_EQ(cache.size(1), kHeadDim);
  CHECK_EQ(value.scalar_type(), torch::kBFloat16);
  CHECK_EQ(cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(slots.scalar_type(), torch::kInt32);
  CHECK_EQ(swap_offsets.scalar_type(), torch::kInt32);
  CHECK_EQ(cos.scalar_type(), torch::kFloat32);
  CHECK_EQ(signed_sin.scalar_type(), torch::kFloat32);
  const int64_t tokens = value.size(0);
  const int64_t cache_rows = cache.size(0);
  CHECK_LE(tokens, kMaxTokens);
  CHECK_GT(cache_rows, 0);
  CHECK_LE(cache_rows, kMaxCacheRows);
  CHECK_EQ(slots.numel(), tokens);
  CHECK_GE(swap_offsets.numel(), kRopeWidth);
  CHECK_EQ(cos.dim(), 2);
  CHECK_EQ(cos.size(0), tokens);
  CHECK_EQ(cos.size(1), kRopeWidth);
  CHECK(cos.sizes() == signed_sin.sizes());
  const torch::Device device = value.device();
  for (const auto& tensor :
       {value, cos, signed_sin, swap_offsets, slots, cache}) {
    CHECK(tensor.is_contiguous() && tensor.device() == device);
  }
  if (tokens == 0) {
    return;
  }

  const auto spec =
      make_v41_swa_rope_cache_specialization(V41SwaRopeCacheHeadDim{kHeadDim});
  const auto* entry = find_v41_swa_rope_cache_kernel_entry(spec);
  CHECK(entry != nullptr) << "Missing V4.1 SWA RoPE cache specialization";
  aclrtStream stream = c10_npu::getCurrentNPUStream(device.index()).stream();
  entry->fn(static_cast<uint8_t*>(value.data_ptr()),
            static_cast<uint8_t*>(cos.data_ptr()),
            static_cast<uint8_t*>(signed_sin.data_ptr()),
            static_cast<uint8_t*>(swap_offsets.data_ptr()),
            static_cast<uint8_t*>(slots.data_ptr()),
            static_cast<uint8_t*>(cache.data_ptr()),
            static_cast<int32_t>(tokens),
            static_cast<int32_t>(cache_rows),
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
