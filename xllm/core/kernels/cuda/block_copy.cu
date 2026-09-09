/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

#include "cuda_ops_api.h"
#include "utils.h"

namespace xllm::kernel::cuda {
namespace {

template <typename scalar_t>
struct VecType;

template <>
struct VecType<c10::Half> {
  using type = uint4;
  static constexpr int32_t vec_width = 8;
};

template <>
struct VecType<c10::BFloat16> {
  using type = uint4;
  static constexpr int32_t vec_width = 8;
};

template <>
struct VecType<float> {
  using type = float4;
  static constexpr int32_t vec_width = 4;
};

DEVICE_INLINE int32_t find_group_idx(const int32_t* __restrict__ cum_sum,
                                     const int32_t num_groups,
                                     const int32_t dst_idx) {
  int32_t left = 0;
  int32_t right = num_groups - 1;
  while (left < right) {
    const int32_t mid = left + ((right - left) >> 1);
    const bool move_left = dst_idx < cum_sum[mid];
    right = move_left ? mid : right;
    left = move_left ? left : mid + 1;
  }
  return left;
}

template <typename scalar_t, bool kVectorized>
__global__ void block_copy_kernel(const int64_t* __restrict__ key_cache_ptrs,
                                  const int64_t* __restrict__ value_cache_ptrs,
                                  const int32_t* __restrict__ src_block_indices,
                                  const int32_t* __restrict__ dst_block_indices,
                                  const int32_t* __restrict__ cum_sum,
                                  const int32_t num_groups,
                                  const int64_t numel_per_block) {
  const int64_t layer_idx = static_cast<int64_t>(blockIdx.x);
  const int32_t dst_linear_idx = static_cast<int32_t>(blockIdx.y);
  const int64_t tile_idx = static_cast<int64_t>(blockIdx.z);

  scalar_t* __restrict__ key_cache = reinterpret_cast<scalar_t*>(
      static_cast<uintptr_t>(key_cache_ptrs[layer_idx]));
  scalar_t* __restrict__ value_cache = reinterpret_cast<scalar_t*>(
      static_cast<uintptr_t>(value_cache_ptrs[layer_idx]));

  const int32_t group_idx = find_group_idx(cum_sum, num_groups, dst_linear_idx);
  const int32_t src_block = src_block_indices[group_idx];
  const int32_t dst_block = dst_block_indices[dst_linear_idx];
  const int64_t src_offset = static_cast<int64_t>(src_block) * numel_per_block;
  const int64_t dst_offset = static_cast<int64_t>(dst_block) * numel_per_block;

  if constexpr (kVectorized) {
    using VecTypeT = typename VecType<scalar_t>::type;
    constexpr int32_t kVecWidth = VecType<scalar_t>::vec_width;
    const int64_t num_vecs_per_block = numel_per_block / kVecWidth;
    const int64_t vec_idx = tile_idx * static_cast<int64_t>(blockDim.x) +
                            static_cast<int64_t>(threadIdx.x);
    if (vec_idx >= num_vecs_per_block) {
      return;
    }

    const int64_t elem_offset = vec_idx * kVecWidth;
    const auto* key_src_vec =
        reinterpret_cast<const VecTypeT*>(key_cache + src_offset + elem_offset);
    const auto* value_src_vec = reinterpret_cast<const VecTypeT*>(
        value_cache + src_offset + elem_offset);
    auto* key_dst_vec =
        reinterpret_cast<VecTypeT*>(key_cache + dst_offset + elem_offset);
    auto* value_dst_vec =
        reinterpret_cast<VecTypeT*>(value_cache + dst_offset + elem_offset);
    *key_dst_vec = *key_src_vec;
    *value_dst_vec = *value_src_vec;
  } else {
    const int64_t elem_idx = tile_idx * static_cast<int64_t>(blockDim.x) +
                             static_cast<int64_t>(threadIdx.x);
    if (elem_idx >= numel_per_block) {
      return;
    }

    key_cache[dst_offset + elem_idx] = key_cache[src_offset + elem_idx];
    value_cache[dst_offset + elem_idx] = value_cache[src_offset + elem_idx];
  }
}

}  // namespace

void block_copy(torch::Tensor key_cache_ptrs,
                torch::Tensor value_cache_ptrs,
                torch::Tensor src_block_indices,
                torch::Tensor dst_block_indices,
                torch::Tensor cum_sum,
                int64_t numel_per_block,
                torch::ScalarType cache_dtype) {
  if (src_block_indices.numel() == 0) {
    return;
  }

  CHECK(key_cache_ptrs.is_cuda());
  CHECK(value_cache_ptrs.is_cuda());
  CHECK(src_block_indices.is_cuda());
  CHECK(dst_block_indices.is_cuda());
  CHECK(cum_sum.is_cuda());
  CHECK_EQ(key_cache_ptrs.scalar_type(), torch::kInt64);
  CHECK_EQ(value_cache_ptrs.scalar_type(), torch::kInt64);
  CHECK_EQ(src_block_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(dst_block_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(cum_sum.scalar_type(), torch::kInt32);
  CHECK_EQ(key_cache_ptrs.dim(), 1);
  CHECK_EQ(value_cache_ptrs.dim(), 1);
  CHECK_EQ(src_block_indices.dim(), 1);
  CHECK_EQ(dst_block_indices.dim(), 1);
  CHECK_EQ(cum_sum.dim(), 1);
  CHECK(key_cache_ptrs.is_contiguous());
  CHECK(value_cache_ptrs.is_contiguous());
  CHECK(src_block_indices.is_contiguous());
  CHECK(dst_block_indices.is_contiguous());
  CHECK(cum_sum.is_contiguous());
  CHECK_EQ(key_cache_ptrs.size(0), value_cache_ptrs.size(0));
  CHECK_EQ(src_block_indices.size(0), cum_sum.size(0));
  CHECK_GT(numel_per_block, 0);

  const at::cuda::OptionalCUDAGuard device_guard(key_cache_ptrs.device());
  constexpr int32_t kThreadsPerBlock = 256;
  const int32_t num_layers = static_cast<int32_t>(key_cache_ptrs.size(0));
  const int32_t num_groups = static_cast<int32_t>(src_block_indices.size(0));
  const int32_t num_dst_blocks =
      static_cast<int32_t>(dst_block_indices.size(0));
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(key_cache_ptrs.get_device());

  DISPATCH_FLOATING_TYPES(cache_dtype, "block_copy_kernel", [&] {
    constexpr bool kHasVecType = std::is_same_v<scalar_t, float> ||
                                 std::is_same_v<scalar_t, c10::Half> ||
                                 std::is_same_v<scalar_t, c10::BFloat16>;

    if constexpr (kHasVecType) {
      constexpr int32_t kVecWidth = VecType<scalar_t>::vec_width;
      if (numel_per_block % kVecWidth == 0) {
        const int64_t tiles_per_block =
            ceil_div<int64_t>(numel_per_block / kVecWidth, kThreadsPerBlock);
        const dim3 grid(num_layers, num_dst_blocks, tiles_per_block);
        block_copy_kernel<scalar_t, true>
            <<<grid, kThreadsPerBlock, 0, stream>>>(
                key_cache_ptrs.data_ptr<int64_t>(),
                value_cache_ptrs.data_ptr<int64_t>(),
                src_block_indices.data_ptr<int32_t>(),
                dst_block_indices.data_ptr<int32_t>(),
                cum_sum.data_ptr<int32_t>(),
                num_groups,
                numel_per_block);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
      }
    }

    const int64_t tiles_per_block =
        ceil_div<int64_t>(numel_per_block, kThreadsPerBlock);
    const dim3 grid(num_layers, num_dst_blocks, tiles_per_block);
    block_copy_kernel<scalar_t, false><<<grid, kThreadsPerBlock, 0, stream>>>(
        key_cache_ptrs.data_ptr<int64_t>(),
        value_cache_ptrs.data_ptr<int64_t>(),
        src_block_indices.data_ptr<int32_t>(),
        dst_block_indices.data_ptr<int32_t>(),
        cum_sum.data_ptr<int32_t>(),
        num_groups,
        numel_per_block);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

}  // namespace xllm::kernel::cuda
