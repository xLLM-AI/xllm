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
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <torch/cuda.h>

#include <cstdint>
#include <type_traits>

#include "kernels/cuda/cuda_ops_api.h"
#include "kernels/cuda/utils.h"
using at::device_of;

namespace {

template <typename scalar_t>
struct VecType;

template <>
struct VecType<c10::Half> {
  using type = uint4;  // 8 elements * 2 bytes = 16 bytes
  static constexpr int32_t vec_width = 8;
};

template <>
struct VecType<c10::BFloat16> {
  using type = uint4;  // 8 elements * 2 bytes = 16 bytes
  static constexpr int32_t vec_width = 8;
};

template <>
struct VecType<float> {
  using type = float4;  // 4 elements * 4 bytes = 16 bytes
  static constexpr int32_t vec_width = 4;
};

template <typename scalar_t>
__global__ void prefill_reshape_and_cache_kernel(
    const scalar_t* __restrict__ proj_k,    // [shared_len, kv_heads, head_dim]
    const scalar_t* __restrict__ proj_v,    // [shared_len, kv_heads, head_dim]
    scalar_t* __restrict__ shared_k_cache,  // [shared_len, kv_heads, head_dim]
    scalar_t* __restrict__ shared_v_cache,  // [shared_len, kv_heads, head_dim]
    const int64_t shared_len,
    const int64_t kv_heads,
    const int64_t head_dim,
    const int64_t k_stride0,    // proj_k.stride(0)
    const int64_t v_stride0,    // proj_v.stride(0)
    const int64_t v_stride1) {  // proj_v.stride(1), same as head_dim
  using VecTypeT = typename VecType<scalar_t>::type;
  constexpr int32_t VEC_WIDTH = VecType<scalar_t>::vec_width;
  const int64_t token_idx = static_cast<int64_t>(blockIdx.y);
  if (token_idx >= shared_len) {
    return;
  }

  const int64_t vecs_per_head = head_dim / VEC_WIDTH;
  const int64_t total_vecs = kv_heads * vecs_per_head;
  const int64_t k_token_base = token_idx * k_stride0;
  const int64_t v_token_base = token_idx * v_stride0;
  const int64_t dst_token_base = token_idx * kv_heads * head_dim;

  for (int64_t linear_idx = threadIdx.x; linear_idx < total_vecs;
       linear_idx += blockDim.x) {
    const int64_t head_idx = linear_idx / vecs_per_head;
    const int64_t vec_idx = linear_idx - head_idx * vecs_per_head;
    const int64_t head_offset = head_idx * head_dim;
    const int64_t vec_offset = vec_idx * VEC_WIDTH;

    const auto* k_src_vec = reinterpret_cast<const VecTypeT*>(
        proj_k + k_token_base + head_offset + vec_offset);
    const auto* v_src_vec = reinterpret_cast<const VecTypeT*>(
        proj_v + v_token_base + head_idx * v_stride1 + vec_offset);
    auto* k_dst_vec = reinterpret_cast<VecTypeT*>(
        shared_k_cache + dst_token_base + head_offset + vec_offset);
    auto* v_dst_vec = reinterpret_cast<VecTypeT*>(
        shared_v_cache + dst_token_base + head_offset + vec_offset);

    *k_dst_vec = *k_src_vec;
    *v_dst_vec = *v_src_vec;
  }
}

}  // namespace

namespace xllm::kernel::cuda {
void prefill_reshape_and_cache(
    torch::Tensor proj_k,  // [shared_len, kv_heads, head_dim]
    torch::Tensor proj_v,  // [shared_len, kv_heads, head_dim]
    torch::Tensor
        shared_k_cache,  // [num_shared_kv_seq_len, kv_heads, head_dim]
    torch::Tensor shared_v_cache) {
  CHECK(proj_k.dim() == 3) << "proj_k must be 3-dimensional";
  CHECK(proj_v.dim() == 3) << "proj_v must be 3-dimensional";
  CHECK(shared_k_cache.dim() == 3) << "shared_k_cache must be 3-dimensional";
  CHECK(shared_v_cache.dim() == 3) << "shared_v_cache must be 3-dimensional";
  CHECK(proj_k.is_cuda() && proj_v.is_cuda() && shared_k_cache.is_cuda() &&
        shared_v_cache.is_cuda())
      << "all tensors must be CUDA tensors";

  const int64_t shared_len = proj_k.size(0);
  const int64_t kv_heads = proj_k.size(1);
  const int64_t head_dim = proj_k.size(2);
  CHECK(proj_v.sizes() == proj_k.sizes())
      << "proj_v and proj_k must have same shape";
  CHECK(shared_k_cache.size(0) >= shared_len &&
        shared_k_cache.size(1) == kv_heads &&
        shared_k_cache.size(2) == head_dim)
      << "shared_k_cache shape mismatch";
  CHECK(shared_v_cache.size(0) >= shared_len &&
        shared_v_cache.size(1) == kv_heads &&
        shared_v_cache.size(2) == head_dim)
      << "shared_v_cache shape mismatch";

  shared_k_cache = shared_k_cache.slice(0, 0, shared_len);
  shared_v_cache = shared_v_cache.slice(0, 0, shared_len);

  // This kernel is specialized for qkv-slice layouts:
  // last dim contiguous and head stride tightly packed by head_dim.
  CHECK(proj_k.stride(2) == 1 && proj_v.stride(2) == 1)
      << "proj_k/proj_v must be contiguous on head_dim (stride(2)=1)";
  CHECK(proj_k.stride(1) == head_dim && proj_v.stride(1) == head_dim)
      << "proj_k/proj_v must satisfy stride(1)=head_dim for qkv-slice layout";
  CHECK(shared_k_cache.stride(2) == 1 && shared_v_cache.stride(2) == 1)
      << "shared caches must be contiguous on head_dim (stride(2)=1)";
  CHECK(shared_k_cache.stride(1) == head_dim &&
        shared_v_cache.stride(1) == head_dim)
      << "shared caches must satisfy stride(1)=head_dim";
  CHECK(shared_k_cache.stride(0) == kv_heads * head_dim &&
        shared_v_cache.stride(0) == kv_heads * head_dim)
      << "shared caches must be contiguous on token stride";

  const at::cuda::OptionalCUDAGuard device_guard(device_of(proj_k));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t k_stride0 = proj_k.stride(0);
  const int64_t v_stride0 = proj_v.stride(0);
  const int64_t v_stride1 = proj_v.stride(1);
  dim3 grid_dim(1, static_cast<unsigned int>(shared_len), 1);

  DISPATCH_FLOATING_TYPES(
      proj_k.scalar_type(), "prefill_reshape_and_cache_kernel", [&] {
        constexpr int32_t VEC_WIDTH = (std::is_same_v<scalar_t, c10::Half> ||
                                       std::is_same_v<scalar_t, c10::BFloat16>)
                                          ? 8
                                          : 4;  // FP16/BF16: 8, Float: 4
        constexpr int32_t kWarpSize = 32;
        constexpr int32_t kMaxThreadsPerBlock = 256;

        CHECK(head_dim % VEC_WIDTH == 0)
            << "head_dim must be divisible by vector width: " << VEC_WIDTH;
        const int64_t vecs_per_head = head_dim / VEC_WIDTH;
        const int64_t total_vecs = kv_heads * vecs_per_head;
        CHECK(total_vecs > 0) << "total_vecs must be > 0";

        int32_t threads_per_block = static_cast<int32_t>(
            total_vecs > kMaxThreadsPerBlock ? kMaxThreadsPerBlock
                                             : total_vecs);
        threads_per_block =
            ((threads_per_block + kWarpSize - 1) / kWarpSize) * kWarpSize;
        if (threads_per_block < kWarpSize) {
          threads_per_block = kWarpSize;
        }
        dim3 block_dim(threads_per_block, 1, 1);

        const auto proj_k_ptr =
            reinterpret_cast<std::uintptr_t>(proj_k.data_ptr<scalar_t>());
        const auto proj_v_ptr =
            reinterpret_cast<std::uintptr_t>(proj_v.data_ptr<scalar_t>());
        const auto k_cache_ptr = reinterpret_cast<std::uintptr_t>(
            shared_k_cache.data_ptr<scalar_t>());
        const auto v_cache_ptr = reinterpret_cast<std::uintptr_t>(
            shared_v_cache.data_ptr<scalar_t>());

        constexpr int32_t alignment_bytes = 16;  // 128-bit alignment
        CHECK(proj_k_ptr % alignment_bytes == 0)
            << "proj_k data_ptr must be 16-byte aligned";
        CHECK(proj_v_ptr % alignment_bytes == 0)
            << "proj_v data_ptr must be 16-byte aligned";
        CHECK(k_cache_ptr % alignment_bytes == 0)
            << "shared_k_cache data_ptr must be 16-byte aligned";
        CHECK(v_cache_ptr % alignment_bytes == 0)
            << "shared_v_cache data_ptr must be 16-byte aligned";

        const int64_t scalar_bytes = static_cast<int64_t>(sizeof(scalar_t));
        CHECK((k_stride0 * scalar_bytes) % alignment_bytes == 0)
            << "proj_k stride(0) bytes must be 16-byte aligned";
        CHECK((v_stride0 * scalar_bytes) % alignment_bytes == 0)
            << "proj_v stride(0) bytes must be 16-byte aligned";
        CHECK((v_stride1 * scalar_bytes) % alignment_bytes == 0)
            << "proj_v stride(1) bytes must be 16-byte aligned";

        prefill_reshape_and_cache_kernel<scalar_t>
            <<<grid_dim, block_dim, 0, stream>>>(
                proj_k.data_ptr<scalar_t>(),
                proj_v.data_ptr<scalar_t>(),
                shared_k_cache.data_ptr<scalar_t>(),
                shared_v_cache.data_ptr<scalar_t>(),
                shared_len,
                kv_heads,
                head_dim,
                k_stride0,
                v_stride0,
                v_stride1);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace xllm::kernel::cuda
